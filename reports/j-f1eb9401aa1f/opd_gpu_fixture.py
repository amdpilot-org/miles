#!/usr/bin/env python3
"""Reproduce the MI350X top-k OPD gradient diagnosis and correction."""

from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.rollout.on_policy_distillation import _compute_topk_distillation
from miles.utils.types import Sample


VOCAB_SIZE = 4
PROMPT_TOKEN = 7
STEPS = 30
LEARNING_RATE = 0.05


def install_single_process_parallel_state() -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def make_args(*, use_opd: bool, entropy_coef: float = 0.0) -> Namespace:
    return Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        allgather_cp=False,
        log_probs_chunk_size=-1,
        true_on_policy_mode=True,
        vocab_size=VOCAB_SIZE,
        bf16=False,
        fp16=False,
        advantage_estimator="grpo",
        use_rollout_logprobs=False,
        skip_actor_forward_only=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        tis_clip=1.5,
        tis_clip_low=0.5,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        calculate_per_token_loss=True,
        entropy_coef=entropy_coef,
        observe_training_entropy=False,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        use_opd=use_opd,
        opd_type="sglang",
        opd_kl_coef=1.0,
        opd_log_prob_top_k=2 if use_opd else 0,
        dump_details=None,
    )


def entry(probability: float, token_id: int) -> list[float | int]:
    return [math.log(probability), token_id]


def build_targets(device: torch.device):
    student_probabilities = [0.20, 0.79, 0.009999, 0.000001]
    teacher_probabilities = [0.80, 0.10, 0.099999, 0.000001]
    sample = Sample(
        tokens=[PROMPT_TOKEN, 0, 0],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [entry(student_probabilities[0], 0), entry(student_probabilities[1], 1)],
                [entry(student_probabilities[0], 0), entry(student_probabilities[1], 1)],
            ]
        },
    )
    reward_payload = {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [entry(teacher_probabilities[0], 0), entry(teacher_probabilities[2], 2)],
                    [entry(teacher_probabilities[0], 0), entry(teacher_probabilities[2], 2)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [entry(teacher_probabilities[0], 0), entry(teacher_probabilities[1], 1)],
                    [entry(teacher_probabilities[0], 0), entry(teacher_probabilities[1], 1)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [entry(student_probabilities[0], 0), entry(student_probabilities[2], 2)],
                    [entry(student_probabilities[0], 0), entry(student_probabilities[2], 2)],
                ]
            }
        },
    }
    args = Namespace(opd_top_k_strategy="union", opd_reward_weight_mode="student_p")
    targets = _compute_topk_distillation(args, sample, reward_payload)
    return targets, student_probabilities, teacher_probabilities


def make_batch(targets, logits: torch.Tensor, *, use_opd: bool) -> dict:
    action_log_prob = torch.log_softmax(logits[0, 0], dim=-1)[0]
    return {
        "unconcat_tokens": [torch.tensor([PROMPT_TOKEN, 0, 0], device=logits.device)],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.ones(2, device=logits.device)],
        "advantages": [torch.zeros(2, device=logits.device)],
        "log_probs": [action_log_prob.detach().reshape(1).repeat(2)],
        "opd_topk_token_ids": [torch.tensor(targets.token_ids, device=logits.device)],
        "opd_topk_teacher_log_probs": [
            torch.tensor(targets.teacher_log_probs, device=logits.device, dtype=torch.float32)
        ],
        "opd_topk_weights": [torch.tensor(targets.weights, device=logits.device, dtype=torch.float32)],
        "_use_opd": use_opd,
    }


def reducer(batch: dict):
    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        calculate_per_token_loss=True,
        qkv_format="thd",
        max_seq_lens=None,
    )


def distribution(logits: torch.Tensor) -> list[float]:
    return torch.softmax(logits[0, 0], dim=-1).detach().cpu().tolist()


def run_steps(targets, initial_logits: torch.Tensor, *, use_opd: bool) -> tuple[list[float], float]:
    logits = initial_logits.detach().clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=LEARNING_RATE)
    args = make_args(use_opd=use_opd, entropy_coef=0.0 if use_opd else 1.0)
    batch = make_batch(targets, logits, use_opd=use_opd)
    final_loss = None

    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy_loss_function(args, batch, logits, reducer(batch))
        final_loss = loss.detach().cpu().item()
        loss.backward()
        optimizer.step()

    return distribution(logits), float(final_loss)


def miles_gradient(targets, initial_logits: torch.Tensor) -> torch.Tensor:
    logits = initial_logits.detach().clone().detach().requires_grad_(True)
    args = make_args(use_opd=True)
    batch = make_batch(targets, logits, use_opd=True)
    loss, _ = policy_loss_function(args, batch, logits, reducer(batch))
    loss.backward()
    torch.cuda.synchronize()
    return logits.grad[0, 0].detach().clone()


def independent_reference_gradient(targets, initial_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = initial_logits.detach().clone().detach().requires_grad_(True)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = torch.tensor(targets.token_ids, device=logits.device)
    selected = log_probs[:, :2, :].gather(-1, token_ids.view(1, 2, -1)).squeeze(0)
    torch.cuda.synchronize()
    teacher = torch.tensor(targets.teacher_log_probs, device=logits.device, dtype=torch.float32)
    weights = torch.tensor(targets.weights, device=logits.device, dtype=torch.float32)
    current_subset = torch.log_softmax(selected, dim=-1)
    teacher_subset = torch.log_softmax(teacher, dim=-1)
    loss = (current_subset.exp() * (current_subset - teacher_subset)).sum()
    loss.backward()
    return loss.detach(), logits.grad[0, 0].detach().clone()


def analytic_gradient(targets, initial_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(initial_logits[0, 0], dim=-1)
    token_ids = torch.tensor(targets.token_ids, device=initial_logits.device)
    teacher = torch.tensor(targets.teacher_log_probs, device=initial_logits.device, dtype=torch.float32)
    current_selected = initial_logits[0, 0][token_ids[0]]
    current_subset = torch.log_softmax(current_selected, dim=-1)
    teacher_subset = torch.log_softmax(teacher[0], dim=-1)
    subset_probabilities = current_subset.exp()
    log_ratio = current_subset - teacher_subset
    selected_gradient = subset_probabilities * (log_ratio + 1.0)
    selected_gradient -= subset_probabilities * selected_gradient.sum()
    gradient = torch.zeros(VOCAB_SIZE, device=initial_logits.device)
    gradient[token_ids[0]] += selected_gradient
    return gradient - probabilities * selected_gradient.sum()


def old_action_independent_expected_gradient(targets, initial_logits: torch.Tensor) -> torch.Tensor:
    logits = initial_logits.detach().clone().detach().requires_grad_(True)
    probabilities = torch.softmax(logits[0, 0], dim=-1)
    advantages = [torch.zeros(2, device=logits.device)]
    rollout_data = {"opd_reverse_kl": [targets.reverse_kl]}
    apply_opd_kl_to_advantages(
        Namespace(opd_type="sglang", opd_kl_coef=1.0, opd_log_prob_top_k=0),
        rollout_data,
        advantages,
        [torch.zeros(2, device=logits.device)],
    )
    expected_gradient = torch.zeros_like(logits.grad if logits.grad is not None else logits)
    args = make_args(use_opd=False)

    for action in range(VOCAB_SIZE):
        batch = {
            "unconcat_tokens": [torch.tensor([PROMPT_TOKEN, action, action], device=logits.device)],
            "response_lengths": [2],
            "total_lengths": [3],
            "loss_masks": [torch.ones(2, device=logits.device)],
            "advantages": [advantages[0].clone()],
            "log_probs": [torch.log_softmax(logits[0, 0], dim=-1)[action].detach().reshape(1).repeat(2)],
        }
        if logits.grad is not None:
            logits.grad.zero_()
        loss, _ = policy_loss_function(args, batch, logits, reducer(batch))
        loss.backward()
        expected_gradient += probabilities[action].detach() * logits.grad

    return expected_gradient[0, :2].detach().cpu()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This fixture requires exactly one assigned CUDA/ROCm device.")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    if properties.major != 9 or properties.minor != 5:
        raise RuntimeError(f"Expected gfx950, got sm_{properties.major}{properties.minor}")

    install_single_process_parallel_state()
    targets, student_probabilities, teacher_probabilities = build_targets(device)
    response_logits = torch.log(torch.tensor(student_probabilities, device=device, dtype=torch.float32))
    initial_logits = torch.tensor(
        [[
            response_logits.tolist(),
            response_logits.tolist(),
            response_logits.tolist(),
        ]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    actual_gradient = miles_gradient(targets, initial_logits)
    reference_loss, reference_gradient = independent_reference_gradient(targets, initial_logits)
    analytic = analytic_gradient(targets, initial_logits)
    old_expected_gradient = old_action_independent_expected_gradient(targets, initial_logits)

    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0.0, atol=2e-6)
    torch.testing.assert_close(actual_gradient, analytic, rtol=0.0, atol=2e-6)
    assert actual_gradient.abs().max().item() > 1e-3
    assert old_expected_gradient.abs().max().item() < 1e-5

    opd_final, opd_final_loss = run_steps(targets, initial_logits, use_opd=True)
    entropy_final, entropy_final_loss = run_steps(targets, initial_logits, use_opd=False)
    teacher_tensor = torch.tensor(teacher_probabilities, device=device)
    opd_tensor = torch.tensor(opd_final, device=device)
    entropy_tensor = torch.tensor(entropy_final, device=device)
    initial_tensor = torch.tensor(student_probabilities, device=device)
    def kl_to_teacher(student: torch.Tensor) -> torch.Tensor:
        support = teacher_tensor > 0
        return torch.sum(
            teacher_tensor[support] * (torch.log(teacher_tensor[support]) - torch.log(student[support]))
        )

    assert opd_final[0] > entropy_final[0]
    assert kl_to_teacher(opd_tensor).item() < kl_to_teacher(entropy_tensor).item()

    result = {
        "gpu": {
            "count": torch.cuda.device_count(),
            "name": properties.name,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
            "capability": f"sm_{properties.major}{properties.minor}",
        },
        "runtime": {
            "torch": torch.__version__,
            "device": str(device),
        },
        "fixture": {
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "student_initial": student_probabilities,
            "teacher": teacher_probabilities,
            "topk_strategy": "union",
            "selected_token_ids": targets.token_ids,
            "selected_teacher_log_probs": targets.teacher_log_probs,
            "selected_weights": targets.weights,
            "rollout_reverse_kl": targets.reverse_kl.tolist(),
        },
        "gradient_checks": {
            "miles_max_abs": actual_gradient.abs().max().item(),
            "reference_loss": reference_loss.item(),
            "miles_vs_reference_max_abs": (actual_gradient - reference_gradient).abs().max().item(),
            "miles_vs_analytic_max_abs": (actual_gradient - analytic).abs().max().item(),
            "old_action_independent_expected_max_abs": old_expected_gradient.abs().max().item(),
        },
        "optimizer_trajectories": {
            "opd_final_distribution": opd_final,
            "opd_final_loss": opd_final_loss,
            "opd_final_kl_to_teacher": kl_to_teacher(opd_tensor).item(),
            "entropy_only_final_distribution": entropy_final,
            "entropy_only_final_loss": entropy_final_loss,
            "entropy_only_final_kl_to_teacher": kl_to_teacher(entropy_tensor).item(),
            "initial_kl_to_teacher": kl_to_teacher(initial_tensor).item(),
        },
    }
    output = Path(__file__).with_name("gpu_fixture_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
