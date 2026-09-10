import time
import json
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from tests.ci.ci_register import register_rocm_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

register_rocm_ci(est_time=90, suite="nightly-stage-c-4-gpu-mi350", labels=["opd"])

_DEVICE = torch.device("cuda")
_PROMPT_LENGTH = 2
_RESPONSE_LENGTH = 2
_TOTAL_LENGTH = _PROMPT_LENGTH + _RESPONSE_LENGTH
_VOCAB_SIZE = 16
_SELECTED_IDS = [[2, 5, 9], [3, 6, 10]]
_SELECTED_WEIGHTS = [[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]]
_SAMPLED_IDS = [1, 4]


def _seeded_logits() -> torch.Tensor:
    generator = torch.Generator(device=_DEVICE).manual_seed(2352)
    return torch.randn(_TOTAL_LENGTH, _VOCAB_SIZE, device=_DEVICE, dtype=torch.float32, generator=generator)


def _opd_args(*, differentiable: bool) -> Namespace:
    return make_args(
        advantage_estimator="grpo",
        entropy_coef=0.0,
        kl_coef=0.0,
        kl_loss_coef=0.0,
        loss_type="policy_loss",
        observe_training_entropy=False,
        opd_differentiable_top_k_loss=differentiable,
        opd_kl_coef=0.7,
        opd_log_prob_top_k=3,
        opd_type="sglang",
        use_kl_loss=False,
        use_opd=True,
        use_rollout_logprobs=False,
        vocab_size=_VOCAB_SIZE,
    )


def _teacher_log_probs(token_ids: list[int]) -> list[float]:
    teacher_logits = torch.tensor([2.0, 1.0, 0.5, -0.2, -0.4, -0.6, -0.8, -1.0], device=_DEVICE)
    teacher_logits = torch.cat([teacher_logits, torch.tensor([-1.2, -1.4, -1.6, -1.8, -2.0, -2.2, -2.4, -2.6], device=_DEVICE)])
    return teacher_logits.log_softmax(dim=-1)[token_ids].tolist()


def _batch(
    args: Namespace,
    *,
    advantages: list[torch.Tensor],
    selected_ids: list[list[int]] | None = None,
    selected_weights: list[list[float]] | None = None,
    precomputed_reverse_kl: list[torch.Tensor] | None = None,
) -> dict:
    if selected_ids is None:
        selected_ids = _SELECTED_IDS
    if selected_weights is None:
        selected_weights = _SELECTED_WEIGHTS

    batch = {
        "unconcat_tokens": [torch.arange(_TOTAL_LENGTH, device=_DEVICE)],
        "response_lengths": [_RESPONSE_LENGTH],
        "total_lengths": [_TOTAL_LENGTH],
        "loss_masks": [torch.ones(_RESPONSE_LENGTH, device=_DEVICE)],
        "log_probs": [torch.zeros(_RESPONSE_LENGTH, device=_DEVICE)],
        "advantages": advantages,
        "returns": [torch.zeros(_RESPONSE_LENGTH, device=_DEVICE)],
    }
    if precomputed_reverse_kl is not None:
        batch["opd_reverse_kl"] = precomputed_reverse_kl
    if args.opd_differentiable_top_k_loss:
        counts = [len(ids) for ids in selected_ids]
        batch.update(
            {
                "opd_topk_counts": [counts],
                "opd_topk_token_ids": [[token_id for ids in selected_ids for token_id in ids]],
                "opd_topk_teacher_log_probs": [
                    [log_prob for ids in selected_ids for log_prob in _teacher_log_probs(ids)]
                ],
                "opd_topk_weights": [[weight for weights in selected_weights for weight in weights]],
            }
        )
    return batch


def _reducer(args: Namespace, batch: dict):
    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )


def _policy_loss(args: Namespace, logits: torch.Tensor, batch: dict):
    return policy_loss_function(args, batch, logits.unsqueeze(0), _reducer(args, batch))


def _topk_reverse_kl(logits: torch.Tensor) -> torch.Tensor:
    student_log_probs = logits[_PROMPT_LENGTH - 1 : _TOTAL_LENGTH - 1].log_softmax(dim=-1)
    estimates = []
    for position, (token_ids, weights) in enumerate(zip(_SELECTED_IDS, _SELECTED_WEIGHTS, strict=True)):
        teacher_log_probs = torch.tensor(_teacher_log_probs(token_ids), device=_DEVICE)
        estimates.append(
            torch.tensor(weights, device=_DEVICE)
            * (student_log_probs[position, token_ids] - teacher_log_probs)
        )
    return sum([estimate.sum() for estimate in estimates]) / _RESPONSE_LENGTH


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required")
def test_differentiable_topk_gradients_match_analysis_and_finite_differences():
    make_parallel_state()
    args = _opd_args(differentiable=True)
    logits = _seeded_logits().requires_grad_()
    batch = _batch(args, advantages=[torch.zeros(_RESPONSE_LENGTH, device=_DEVICE)])

    loss, metrics = _policy_loss(args, logits, batch)
    loss.backward()

    probabilities = logits.detach()[_PROMPT_LENGTH - 1 : _TOTAL_LENGTH - 1].softmax(dim=-1)
    expected = torch.zeros_like(logits.grad)
    for response_position, (token_ids, weights) in enumerate(zip(_SELECTED_IDS, _SELECTED_WEIGHTS, strict=True)):
        position = _PROMPT_LENGTH - 1 + response_position
        weight_sum = sum(weights)
        expected[position] -= args.opd_kl_coef * weight_sum * probabilities[response_position] / _RESPONSE_LENGTH
        expected[position, token_ids] += args.opd_kl_coef * torch.tensor(weights, device=_DEVICE) / _RESPONSE_LENGTH

    torch.testing.assert_close(logits.grad, expected, rtol=2e-5, atol=2e-6)
    assert metrics["opd_reverse_kl"].requires_grad is False

    epsilon = 1e-3
    finite_differences = torch.zeros_like(logits)
    with torch.no_grad():
        for position in range(_PROMPT_LENGTH - 1, _TOTAL_LENGTH - 1):
            for vocab_index in range(_VOCAB_SIZE):
                plus = logits.detach().clone()
                minus = logits.detach().clone()
                plus[position, vocab_index] += epsilon
                minus[position, vocab_index] -= epsilon
                plus_loss, _ = _policy_loss(args, plus, batch)
                minus_loss, _ = _policy_loss(args, minus, batch)
                finite_differences[position, vocab_index] = (plus_loss - minus_loss) / (2 * epsilon)

    torch.testing.assert_close(logits.grad, finite_differences, rtol=2e-3, atol=2e-4)
    print(
        json.dumps(
            {
                "analytical_max_abs_error": (logits.grad - expected).abs().max().item(),
                "finite_difference_max_abs_error": (logits.grad - finite_differences).abs().max().item(),
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required")
def test_omitted_tail_is_not_silently_included_and_still_couples_through_softmax():
    make_parallel_state()
    args = _opd_args(differentiable=True)
    logits = _seeded_logits().requires_grad_()
    omitted_batch = _batch(
        args,
        advantages=[torch.zeros(_RESPONSE_LENGTH, device=_DEVICE)],
        selected_ids=[[2, 5], [3, 6]],
        selected_weights=[[0.6, 0.4], [0.55, 0.45]],
    )
    full_batch = _batch(
        args,
        advantages=[torch.zeros(_RESPONSE_LENGTH, device=_DEVICE)],
        selected_ids=[[2, 5, 15], [3, 6, 15]],
        selected_weights=[[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]],
    )

    omitted_loss, _ = _policy_loss(args, logits, omitted_batch)
    full_loss, _ = _policy_loss(args, logits, full_batch)
    omitted_loss.backward()

    assert 15 not in omitted_batch["opd_topk_token_ids"][0]
    assert not torch.allclose(omitted_loss.detach(), full_loss.detach(), atol=1e-4)
    assert logits.grad[:, 15].abs().max() > 0
    print(
        json.dumps(
            {
                "full_set_loss": full_loss.detach().item(),
                "omitted_tail_loss": omitted_loss.detach().item(),
                "omitted_tail_max_gradient": logits.grad[:, 15].abs().max().item(),
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required")
def test_256_optimizer_steps_compare_differentiable_loss_to_detached_shaping():
    make_parallel_state()
    initial_logits = _seeded_logits()
    initial_sampled_log_probs = initial_logits[_PROMPT_LENGTH - 1 : _TOTAL_LENGTH - 1].log_softmax(dim=-1)[
        torch.arange(_RESPONSE_LENGTH), torch.tensor(_SAMPLED_IDS, device=_DEVICE)
    ]
    initial_topk_reverse_kl = _topk_reverse_kl(initial_logits).detach()

    def run(mode: str) -> dict[str, float]:
        logits = initial_logits.clone().requires_grad_()
        optimizer = torch.optim.Adam([logits], lr=0.02)
        if mode == "differentiable":
            args = _opd_args(differentiable=True)
            advantages = [torch.zeros(_RESPONSE_LENGTH, device=_DEVICE)]
            batch = _batch(args, advantages=advantages)
        else:
            args = _opd_args(differentiable=False)
            precomputed = initial_topk_reverse_kl.clone().detach().unsqueeze(0).repeat(_RESPONSE_LENGTH)
            advantages = [-args.opd_kl_coef * precomputed]
            batch = _batch(
                args,
                advantages=advantages,
                precomputed_reverse_kl=[precomputed],
            )
        batch["log_probs"] = [initial_sampled_log_probs.detach().clone()]

        gradient_norms = []
        start_time = time.perf_counter()
        for _ in range(256):
            optimizer.zero_grad(set_to_none=False)
            loss, _ = _policy_loss(args, logits, batch)
            loss.backward()
            gradient_norms.append(logits.grad.norm().detach())
            optimizer.step()
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time
        return {
            "final_topk_reverse_kl": _topk_reverse_kl(logits.detach()).item(),
            "mean_gradient_norm": torch.stack(gradient_norms).mean().item(),
            "elapsed_seconds": elapsed_seconds,
        }

    differentiable = run("differentiable")
    baseline = run("baseline")

    assert not torch.allclose(
        torch.tensor(differentiable["final_topk_reverse_kl"], device=_DEVICE),
        torch.tensor(baseline["final_topk_reverse_kl"], device=_DEVICE),
        rtol=1e-5,
        atol=1e-6,
    )
    assert differentiable["mean_gradient_norm"] > 0
    assert baseline["mean_gradient_norm"] > 0
    print(
        json.dumps(
            {
                "initial_topk_reverse_kl": initial_topk_reverse_kl.item(),
                **differentiable,
                **{f"baseline_{key}": value for key, value in baseline.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
