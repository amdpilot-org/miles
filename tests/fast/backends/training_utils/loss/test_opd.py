"""Unit tests for the decoupled on-policy-distillation (OPD) loss path.

`apply_opd_kl_to_advantages` is orthogonal to the advantage estimator: it adds a
reverse-KL penalty (student_logp - teacher_logp) to per-token advantages. These
tests cover the math and the guard rails without needing the external loss
snapshot artifacts.
"""

from argparse import Namespace

import math

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils import loss as loss_utils
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.opd import (
    apply_opd_kl_to_advantages,
    compute_opd_topk_reverse_kl,
)

from .loss_test_utils import make_args, make_parallel_state

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


def _args(opd_kl_coef: float = 1.0) -> Namespace:
    return Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=opd_kl_coef)


def test_subtracts_weighted_reverse_kl_and_stores_metric():
    args = _args(opd_kl_coef=0.5)
    student = [torch.tensor([0.0, 1.0], requires_grad=True)]
    teacher = [torch.tensor([0.0, 0.0], requires_grad=True)]
    advantages = [torch.tensor([2.0, 2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # reverse_kl = student - teacher = [0, 1]; adv - 0.5 * reverse_kl = [2.0, 1.5]
    assert torch.allclose(advantages[0], torch.tensor([2.0, 1.5]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 1.0]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.2, -0.3], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert student[0].grad is None
    assert teacher[0].grad is None


def test_precomputed_reverse_kl_is_detached_before_weighting_advantages():
    args = _args(opd_kl_coef=0.25)
    precomputed = torch.tensor([0.4, -0.2], requires_grad=True)
    advantages = [torch.tensor([0.0, 0.0])]
    rollout_data = {"opd_reverse_kl": [precomputed]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs=[torch.zeros(2)])

    torch.testing.assert_close(advantages[0], torch.tensor([-0.1, 0.05]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.3, -0.1], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert precomputed.grad is None


def test_topk_reverse_kl_is_not_applied_to_advantages():
    args = _args()
    args.opd_log_prob_top_k = 2
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"opd_reverse_kl": [torch.tensor([0.5, -0.5])]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, [torch.zeros(2)])

    torch.testing.assert_close(advantages[0], torch.tensor([1.0, 2.0]))


def test_topk_policy_loss_matches_subset_kl_reference_gradient():
    make_parallel_state()
    args = make_args(
        use_opd=True,
        opd_log_prob_top_k=2,
        opd_kl_coef=1.0,
        entropy_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=True,
        vocab_size=4,
    )
    logits = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, math.log(4.0), 0.0, 0.0],
            [0.0, math.log(4.0), 0.0, 0.0],
        ]],
        requires_grad=True,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([0, 1, 2])],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.ones(2)],
        "advantages": [torch.zeros(2)],
        "log_probs": [torch.zeros(2)],
        "opd_topk_token_ids": [torch.tensor([[1, 2], [1, 2]])],
        "opd_topk_teacher_log_probs": [torch.log(torch.tensor([[0.8, 0.2], [0.8, 0.2]]))],
        "opd_topk_weights": [torch.tensor([[0.8, 0.2], [0.8, 0.2]])],
    }
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        qkv_format=args.qkv_format,
        max_seq_lens=None,
        calculate_per_token_loss=args.calculate_per_token_loss,
    )

    loss, metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    loss.backward()

    reference_logits = logits.detach().clone().detach().requires_grad_(True)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    selected = reference_log_probs[:, :2, :].gather(
        dim=-1,
        index=batch["opd_topk_token_ids"][0].unsqueeze(0).expand(1, 2, 2),
    ).squeeze(0)
    current_subset = torch.log_softmax(selected, dim=-1)
    teacher_subset = torch.log_softmax(batch["opd_topk_teacher_log_probs"][0], dim=-1)
    reference_loss = (
        current_subset.exp() * (current_subset - teacher_subset)
    ).sum() / batch["loss_masks"][0].sum()
    reference_loss.backward()

    torch.testing.assert_close(loss.detach(), reference_loss.detach())
    torch.testing.assert_close(metrics["opd_kl_loss"], reference_loss.detach())
    torch.testing.assert_close(logits.grad[:, :2, :], reference_logits.grad[:, :2, :])
    assert torch.count_nonzero(logits.grad[:, 2:, :]).item() == 0


def test_topk_policy_loss_requires_target_fields():
    make_parallel_state()
    args = make_args(use_opd=True, opd_log_prob_top_k=2)
    batch = {
        "unconcat_tokens": [torch.tensor([0, 1])],
        "response_lengths": [1],
        "total_lengths": [2],
    }

    with pytest.raises(ValueError, match="Top-k OPD requires"):
        compute_opd_topk_reverse_kl(args, batch, torch.zeros(1, 2, 2))


def test_fixed_opd_inputs_are_detached_in_persistent_rollout_data(monkeypatch):
    make_parallel_state()
    old_source = torch.tensor([0.2, 0.4], requires_grad=True)
    rollout_source = torch.tensor([0.3, 0.5], requires_grad=True)
    reference_source = torch.tensor([0.4, 0.6], requires_grad=True)
    teacher_source = torch.tensor([0.1, 0.2], requires_grad=True)
    rollout_data = {
        "log_probs": [old_source.sin()],
        "rollout_log_probs": [rollout_source.cos()],
        "ref_log_probs": [reference_source.exp()],
        "teacher_log_probs": [teacher_source.square()],
        "rewards": [0.0],
        "values": None,
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
    }
    args = Namespace(
        skip_actor_forward_only=False,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=0.5,
        normalize_advantages=False,
    )

    def fake_compute_advantages(**kwargs):
        assert kwargs["log_probs"][0].grad_fn is None
        zeros = torch.zeros_like(kwargs["log_probs"][0])
        return [zeros], [zeros.clone()]

    monkeypatch.setattr(loss_utils, "compute_advantages", fake_compute_advantages)

    loss_utils.compute_advantages_and_returns(args, rollout_data)

    for key in (
        "log_probs",
        "rollout_log_probs",
        "ref_log_probs",
        "teacher_log_probs",
        "opd_reverse_kl",
        "advantages",
    ):
        assert rollout_data[key][0].grad_fn is None
        assert rollout_data[key][0].requires_grad is False

    assert old_source.grad is None
    assert rollout_source.grad is None
    assert reference_source.grad is None
    assert teacher_source.grad is None


def test_noop_when_student_log_probs_none():
    args = _args()
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0, 0.0])]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, None)

    assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0]))
    assert "opd_reverse_kl" not in rollout_data


def test_raises_when_teacher_log_probs_missing():
    args = _args()
    with pytest.raises(ValueError, match="requires teacher_log_probs"):
        apply_opd_kl_to_advantages(args, {}, [torch.tensor([1.0])], [torch.tensor([1.0])])


def test_raises_on_length_mismatch():
    args = _args()
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0])]}  # 1 sample
    advantages = [torch.tensor([1.0]), torch.tensor([1.0])]  # 2 samples
    student = [torch.tensor([1.0]), torch.tensor([1.0])]

    with pytest.raises(ValueError, match="OPD length mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


def test_raises_on_scalar_advantage_broadcast_trap():
    # GRPO-style per-sample scalar advantage must be expanded to per-token first.
    args = _args()
    student = [torch.tensor([0.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0])]
    advantages = [torch.tensor([2.0])]  # shape (1,) != student shape (2,)
    rollout_data = {"teacher_log_probs": teacher}

    with pytest.raises(ValueError, match="OPD shape mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)
