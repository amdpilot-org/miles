#!/usr/bin/env python3
"""Reproduce the Miles top-k OPD gradient diagnosis on one assigned GPU."""

from __future__ import annotations

from argparse import Namespace
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.rollout.on_policy_distillation import _compute_topk_reverse_kl
from miles.utils.types import Sample


VOCAB_SIZE = 8
TOP_K = 3
STEPS = 5
LEARNING_RATE = 0.2


def module_path(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return str(spec.origin) if spec else None


def initialize_distributed() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda:0"),
    )


def metrics(logits: torch.Tensor, teacher_logps: torch.Tensor) -> tuple[float, float]:
    student_logps = torch.log_softmax(logits, dim=-1)
    student_probs = student_logps.exp()
    teacher_probs = teacher_logps.exp()
    teacher_kl = (teacher_probs * (teacher_logps - student_logps)).sum()
    entropy = -(student_probs * student_logps).sum()
    return teacher_kl.item(), entropy.item()


def reverse_topk_loss(logits: torch.Tensor, teacher_logps: torch.Tensor, top_ids: list[int]) -> torch.Tensor:
    student_logps = torch.log_softmax(logits, dim=-1)[top_ids]
    weights = torch.softmax(student_logps, dim=-1)
    return (weights * (student_logps - teacher_logps[top_ids])).sum()


def forward_topk_loss(logits: torch.Tensor, teacher_logps: torch.Tensor, top_ids: list[int]) -> torch.Tensor:
    student_logps = torch.log_softmax(logits, dim=-1)[top_ids]
    teacher_probs = teacher_logps[top_ids].exp()
    return (teacher_probs * (teacher_logps[top_ids] - student_logps)).sum()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    student_logps = torch.log_softmax(logits, dim=-1)
    return (student_logps.exp() * student_logps).sum()


def gradient(loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(loss, logits, retain_graph=True)[0]


def numerical_gradient(
    loss_function,
    logits: torch.Tensor,
    top_ids: list[int],
    teacher_logps: torch.Tensor,
    epsilon: float = 1e-2,
) -> torch.Tensor:
    values = []
    for index in range(logits.numel()):
        plus = logits.detach().clone()
        minus = logits.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        values.append(
            (
                loss_function(plus, teacher_logps, top_ids)
                - loss_function(minus, teacher_logps, top_ids)
            ).item()
            / (2.0 * epsilon)
        )
    return torch.tensor(values, device=logits.device)


def optimize_reference(
    loss_function,
    initial_logits: torch.Tensor,
    teacher_logps: torch.Tensor,
    top_ids: list[int],
) -> tuple[torch.Tensor, list[tuple[float, float]], torch.Tensor]:
    logits = initial_logits.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([logits], lr=LEARNING_RATE)
    trajectory = [metrics(logits, teacher_logps)]
    first_gradient = None
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(logits, teacher_logps, top_ids)
        loss.backward()
        if first_gradient is None:
            first_gradient = logits.grad.detach().clone()
        optimizer.step()
        trajectory.append(metrics(logits, teacher_logps))
    return logits, trajectory, first_gradient


def expected_miles_gradient(
    logits: torch.Tensor,
    old_logps: torch.Tensor,
    advantage: torch.Tensor,
) -> torch.Tensor:
    current_logps = torch.log_softmax(logits, dim=-1)
    probabilities = current_logps.exp()
    action_gradients = []
    for action in range(VOCAB_SIZE):
        ppo_kl = old_logps[action].view(1, 1) - current_logps[action].view(1, 1)
        policy_loss, _ = compute_policy_loss(
            ppo_kl,
            advantage.view(1, 1),
            eps_clip=0.2,
            eps_clip_high=0.2,
        )
        action_gradients.append(gradient(policy_loss.sum(), logits))
    return sum(probability * action_gradient for probability, action_gradient in zip(probabilities, action_gradients))


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This fixture requires exactly one assigned CUDA/HIP GPU")

    initialize_distributed()
    try:
        device = torch.device("cuda:0")
        initial_logits = torch.tensor(
            [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0],
            dtype=torch.float32,
            device=device,
        )
        teacher_logits = torch.tensor(
            [1.8, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            dtype=torch.float32,
            device=device,
        )
        student_logps = torch.log_softmax(initial_logits, dim=-1)
        teacher_logps = torch.log_softmax(teacher_logits, dim=-1)
        top_ids = student_logps.topk(TOP_K).indices.tolist()

        sample = Sample(
            tokens=[VOCAB_SIZE - 1, top_ids[0]],
            response_length=1,
            metadata={
                "opd_student_top_logprobs": [
                    [[student_logps[token_id].item(), token_id] for token_id in top_ids]
                ]
            },
        )
        teacher_entries = [[teacher_logps[token_id].item(), token_id] for token_id in top_ids]
        response = {"meta_info": {"input_token_ids_logprobs": [None, teacher_entries]}}
        reward_payload = {"teacher": response, "student_on_teacher": response}
        args = Namespace(
            opd_log_prob_top_k=TOP_K,
            opd_top_k_strategy="only-student",
            opd_reward_weight_mode="student_p",
        )

        miles_reverse_kl = _compute_topk_reverse_kl(args, sample, reward_payload).to(device)
        selected_student_logps = student_logps[top_ids]
        selected_teacher_logps = teacher_logps[top_ids]
        reference_weights = torch.softmax(selected_student_logps, dim=-1)
        reference_reverse_kl = (
            reference_weights * (selected_student_logps - selected_teacher_logps)
        ).sum()
        torch.testing.assert_close(miles_reverse_kl, reference_reverse_kl.view(1))

        student_tail_mass = 1.0 - selected_student_logps.exp().sum()
        teacher_tail_mass = 1.0 - selected_teacher_logps.exp().sum()
        assert student_tail_mass > 0.1
        assert teacher_tail_mass > 0.1

        advantages = [torch.zeros(1, device=device)]
        rollout_data = {"opd_reverse_kl": [miles_reverse_kl]}
        opd_args = Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=1.0)
        apply_opd_kl_to_advantages(
            opd_args,
            rollout_data,
            advantages,
            [student_logps.detach()],
        )

        miles_logits = initial_logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.SGD([miles_logits], lr=LEARNING_RATE)
        for _ in range(STEPS):
            expected_gradient = expected_miles_gradient(
                miles_logits,
                student_logps.detach(),
                advantages[0],
            )
            optimizer.zero_grad(set_to_none=True)
            miles_logits.grad = expected_gradient
            optimizer.step()
        torch.testing.assert_close(miles_logits, initial_logits, atol=1e-6, rtol=1e-6)

        reverse_logits, reverse_trajectory, reverse_gradient = optimize_reference(
            reverse_topk_loss,
            initial_logits,
            teacher_logps,
            top_ids,
        )
        forward_logits, forward_trajectory, forward_gradient = optimize_reference(
            forward_topk_loss,
            initial_logits,
            teacher_logps,
            top_ids,
        )
        entropy_logits, entropy_trajectory, entropy_gradient = optimize_reference(
            lambda logits, teacher_logps, top_ids: entropy_loss(logits),
            initial_logits,
            teacher_logps,
            top_ids,
        )

        reverse_numerical = numerical_gradient(
            reverse_topk_loss,
            initial_logits,
            top_ids,
            teacher_logps,
        )
        torch.testing.assert_close(reverse_gradient, reverse_numerical, atol=2e-3, rtol=2e-3)

        teacher_top_mass = selected_teacher_logps.exp().sum()
        forward_analytic = initial_logits.softmax(dim=-1) * teacher_top_mass
        forward_analytic[top_ids] -= teacher_logps[top_ids].exp()
        torch.testing.assert_close(forward_gradient, forward_analytic, atol=2e-6, rtol=2e-6)

        alternate_teacher_logits = torch.tensor(
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )
        alternate_teacher_logps = torch.log_softmax(alternate_teacher_logits, dim=-1)
        alternate_logits = initial_logits.detach().clone().requires_grad_(True)
        alternate_reverse_loss = reverse_topk_loss(
            alternate_logits,
            alternate_teacher_logps,
            top_ids,
        )
        alternate_reverse_gradient = gradient(alternate_reverse_loss, alternate_logits)
        assert (reverse_gradient - alternate_reverse_gradient).abs().max() > 0.1

        cosine = torch.nn.functional.cosine_similarity(
            reverse_gradient,
            entropy_gradient,
            dim=0,
        )
        assert cosine.abs() < 0.99
        assert reverse_trajectory[-1][0] < reverse_trajectory[0][0]
        assert reverse_trajectory[-1][1] < reverse_trajectory[0][1]
        assert forward_trajectory[-1][0] < forward_trajectory[0][0]
        assert forward_trajectory[-1][1] < forward_trajectory[0][1]
        assert entropy_trajectory[-1][0] > entropy_trajectory[0][0]
        assert entropy_trajectory[-1][1] > entropy_trajectory[0][1]

        print(f"device={torch.cuda.get_device_name(0)}")
        print(f"torch={torch.__version__} hip={torch.version.hip}")
        print(f"miles_module={module_path('miles')}")
        print(f"sglang_module={module_path('sglang')}")
        print(f"megatron_module={module_path('megatron')}")
        print(f"torch_native_module={module_path('torch._C')}")
        print(f"top_ids={top_ids}")
        print(f"student_tail_mass={student_tail_mass.item():.6f}")
        print(f"teacher_tail_mass={teacher_tail_mass.item():.6f}")
        print(f"miles_reverse_kl={miles_reverse_kl.item():.6f}")
        print(f"miles_expected_gradient_max_abs={expected_miles_gradient(miles_logits, student_logps.detach(), advantages[0]).abs().max().item():.6e}")
        print(f"miles_logits_max_change={(miles_logits - initial_logits).abs().max().item():.6e}")
        print(f"reverse_teacher_kl={reverse_trajectory[0][0]:.6f}->{reverse_trajectory[-1][0]:.6f}")
        print(f"reverse_entropy={reverse_trajectory[0][1]:.6f}->{reverse_trajectory[-1][1]:.6f}")
        print(f"forward_teacher_kl={forward_trajectory[0][0]:.6f}->{forward_trajectory[-1][0]:.6f}")
        print(f"forward_entropy={forward_trajectory[0][1]:.6f}->{forward_trajectory[-1][1]:.6f}")
        print(f"entropy_teacher_kl={entropy_trajectory[0][0]:.6f}->{entropy_trajectory[-1][0]:.6f}")
        print(f"entropy_entropy={entropy_trajectory[0][1]:.6f}->{entropy_trajectory[-1][1]:.6f}")
        print(f"reverse_entropy_gradient_cosine={cosine.item():.6f}")
        print("PASS: Miles top-k OPD has zero expected on-policy gradient")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
