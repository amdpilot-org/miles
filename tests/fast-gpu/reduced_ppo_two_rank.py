from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from miles.backends.fsdp_utils.checkpoint import ModelState, OptimizerState
from miles.backends.training_utils.loss_hub.advantages import normalize_advantages
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss, get_advantages_and_returns_batch
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    resume_split: int
    batch_size: int
    input_size: int
    hidden_size: int
    vocab_size: int
    gamma: float
    lambd: float
    eps_clip: float
    eps_clip_high: float
    learning_rate: float


@dataclass
class PhaseTimings:
    setup: float = 0.0
    gae_and_reference: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0
    checkpoint_save: float = 0.0
    checkpoint_load: float = 0.0


@dataclass
class UninterruptedResult:
    initial_actor: list[torch.Tensor]
    initial_critic: list[torch.Tensor]
    intermediate_actor: list[torch.Tensor]
    intermediate_critic: list[torch.Tensor]
    final_actor: list[torch.Tensor]
    final_critic: list[torch.Tensor]
    checks: dict[str, Any]
    actor_delta: dict[str, float]
    critic_delta: dict[str, float]


@dataclass
class ResumeResult:
    split_intermediate_actor_difference: dict[str, float]
    split_intermediate_critic_difference: dict[str, float]
    loaded_actor_difference: dict[str, float]
    loaded_critic_difference: dict[str, float]
    actor_resume_difference: dict[str, float]
    critic_resume_difference: dict[str, float]
    checks: dict[str, Any]


class TinyActor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class TinyCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def initialize_process_group() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=180))


def initialize_parallel_state() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    trivial = GroupInfo(rank=0, size=1, group=None)
    data_parallel = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
    set_parallel_state(
        ParallelState(
            intra_dp=data_parallel,
            intra_dp_cp=data_parallel,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def create_training_pair(
    config: TrainingConfig,
    seed: int,
) -> tuple[DDP, DDP, torch.optim.Adam, torch.optim.Adam]:
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    actor = TinyActor(config.input_size, config.hidden_size, config.vocab_size).to(device)
    critic = TinyCritic(config.input_size, config.hidden_size).to(device)
    actor_ddp = DDP(actor, device_ids=[device.index], gradient_as_bucket_view=True)
    critic_ddp = DDP(critic, device_ids=[device.index], gradient_as_bucket_view=True)
    actor_optimizer = torch.optim.Adam(actor_ddp.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic_ddp.parameters(), lr=config.learning_rate)
    return actor_ddp, critic_ddp, actor_optimizer, critic_optimizer


def make_rollout(
    config: TrainingConfig,
    step: int,
    rank: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5EED + 1009 * step + 31 * rank)
    response_lengths = [7, 11, 13, 9]
    masks = []
    for sample_index, response_length in enumerate(response_lengths):
        mask = torch.ones(response_length, dtype=torch.float32)
        if sample_index == 1:
            mask[2] = 0.0
            mask[-1] = 0.0
        if sample_index == 2:
            mask[4] = 0.0
        masks.append(mask)

    max_response_length = max(response_lengths)
    features = torch.randn(
        (config.batch_size, max_response_length, config.input_size),
        generator=generator,
        dtype=torch.float32,
    )
    actions = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, max_response_length),
        generator=generator,
        dtype=torch.int64,
    )
    token_rewards = torch.randn(
        (config.batch_size, max_response_length),
        generator=generator,
        dtype=torch.float32,
    )
    terminal_rewards = [float(value) for value in torch.randn(config.batch_size, generator=generator)]
    prompt_lengths = [3, 5, 2, 4]
    return {
        "features": features.cuda(non_blocking=False),
        "actions": actions.cuda(non_blocking=False),
        "masks": [mask.cuda(non_blocking=False) for mask in masks],
        "token_rewards": token_rewards.cuda(non_blocking=False),
        "terminal_rewards": terminal_rewards,
        "response_lengths": response_lengths,
        "prompt_lengths": prompt_lengths,
        "total_lengths": [
            prompt + response for prompt, response in zip(prompt_lengths, response_lengths, strict=True)
        ],
    }


def reference_gae(
    values: list[torch.Tensor],
    token_rewards: list[torch.Tensor],
    terminal_rewards: list[float],
    masks: list[torch.Tensor],
    gamma: float,
    lambd: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for sample_values, sample_rewards, terminal_reward, mask in zip(
        values,
        token_rewards,
        terminal_rewards,
        masks,
        strict=True,
    ):
        active_indices = mask.nonzero(as_tuple=True)[0]
        active_count = active_indices.numel()
        sample_advantages = torch.zeros_like(sample_values)
        sample_returns = torch.zeros_like(sample_values)
        if active_count == 0:
            advantages.append(sample_advantages)
            returns.append(sample_returns)
            continue

        active_values = sample_values[active_indices]
        active_rewards = sample_rewards[active_indices].clone()
        active_rewards[-1] += terminal_reward
        active_advantages = torch.zeros_like(active_values)
        gae_carry = torch.zeros((), dtype=active_values.dtype, device=active_values.device)
        for local_index in reversed(range(active_count)):
            next_value = active_values[local_index + 1] if local_index + 1 < active_count else 0.0
            delta = active_rewards[local_index] + gamma * next_value - active_values[local_index]
            gae_carry = delta + gamma * lambd * gae_carry
            active_advantages[local_index] = gae_carry
        active_returns = active_advantages + active_values
        sample_advantages[active_indices] = active_advantages
        sample_returns[active_indices] = active_returns
        advantages.append(sample_advantages)
        returns.append(sample_returns)
    return advantages, returns


def assert_gae_reference(
    rollout: dict[str, Any],
    values: list[torch.Tensor],
    miles_advantages: list[torch.Tensor],
    miles_returns: list[torch.Tensor],
    config: TrainingConfig,
) -> None:
    token_rewards = [
        rollout["token_rewards"][sample_index, :response_length]
        for sample_index, response_length in enumerate(rollout["response_lengths"])
    ]
    reference_advantages, reference_returns = reference_gae(
        values,
        token_rewards,
        rollout["terminal_rewards"],
        rollout["masks"],
        config.gamma,
        config.lambd,
    )
    torch.testing.assert_close(miles_advantages, reference_advantages, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(miles_returns, reference_returns, rtol=2e-5, atol=2e-5)
    assert rollout["masks"][1][-1].item() == 0.0
    assert miles_returns[1][-1].item() == 0.0


def global_mask_sum(masks: list[torch.Tensor]) -> int:
    local_count = torch.tensor([sum(mask.sum().item() for mask in masks)], device="cuda")
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    return int(local_count.item())


def assert_sample_accounting(rollout: dict[str, Any], expected_global_tokens: int) -> None:
    local_samples = torch.tensor([len(rollout["response_lengths"])], device="cuda")
    dist.all_reduce(local_samples, op=dist.ReduceOp.SUM)
    assert int(local_samples.item()) == 8
    assert global_mask_sum(rollout["masks"]) == expected_global_tokens


def current_log_probs(actor: DDP, rollout: dict[str, Any]) -> list[torch.Tensor]:
    logits = actor(rollout["features"])
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(-1, rollout["actions"].unsqueeze(-1)).squeeze(-1)
    return [
        gathered[sample_index, :response_length]
        for sample_index, response_length in enumerate(rollout["response_lengths"])
    ]


def old_log_probs(log_probs: list[torch.Tensor], step: int, rank: int) -> list[torch.Tensor]:
    offsets = [-0.05, 0.35, -0.45, 0.15]
    return [
        sample_log_probs.detach() + offsets[(sample_index + step + rank) % len(offsets)]
        for sample_index, sample_log_probs in enumerate(log_probs)
    ]


def assert_clipping_reference(
    old_log_probs: list[torch.Tensor],
    current_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    ppo_kl = torch.cat([old - current for old, current in zip(old_log_probs, current_log_probs, strict=True)])
    packed_advantages = torch.cat(advantages)
    miles_loss, miles_clip_fraction = compute_policy_loss(
        ppo_kl,
        packed_advantages,
        config.eps_clip,
        config.eps_clip_high,
    )
    ratio = torch.exp(-ppo_kl)
    unclipped_loss = -ratio * packed_advantages
    clipped_ratio = ratio.clamp(1 - config.eps_clip, 1 + config.eps_clip_high)
    clipped_loss = -clipped_ratio * packed_advantages
    reference_loss = torch.maximum(unclipped_loss, clipped_loss)
    torch.testing.assert_close(miles_loss, reference_loss, rtol=2e-5, atol=2e-5)
    return miles_loss, miles_clip_fraction


def masked_policy_mean(
    policy_loss: torch.Tensor,
    response_lengths: list[int],
    masks: list[torch.Tensor],
    global_tokens: int,
) -> torch.Tensor:
    sample_losses = torch.split(policy_loss, response_lengths)
    local_sum = sum((sample_loss * mask).sum() for sample_loss, mask in zip(sample_losses, masks, strict=True))
    return local_sum * dist.get_world_size() / global_tokens


def save_checkpoint(
    actor: DDP,
    critic: DDP,
    actor_optimizer: torch.optim.Adam,
    critic_optimizer: torch.optim.Adam,
    checkpoint_dir: Path,
) -> float:
    started = time.perf_counter()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dcp.save(
        {
            "actor_model": ModelState(actor),
            "actor_optimizer": OptimizerState(actor, actor_optimizer),
            "critic_model": ModelState(critic),
            "critic_optimizer": OptimizerState(critic, critic_optimizer),
        },
        checkpoint_id=str(checkpoint_dir),
    )
    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter() - started


def load_checkpoint(
    actor: DDP,
    critic: DDP,
    actor_optimizer: torch.optim.Adam,
    critic_optimizer: torch.optim.Adam,
    checkpoint_dir: Path,
) -> float:
    started = time.perf_counter()
    torch.cuda.synchronize()
    dcp.load(
        {
            "actor_model": ModelState(actor),
            "actor_optimizer": OptimizerState(actor, actor_optimizer),
            "critic_model": ModelState(critic),
            "critic_optimizer": OptimizerState(critic, critic_optimizer),
        },
        checkpoint_id=str(checkpoint_dir),
    )
    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter() - started


def prepare_gae(
    config: TrainingConfig,
    critic: DDP,
    rollout: dict[str, Any],
    timings: PhaseTimings,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    phase_started = time.perf_counter()
    torch.cuda.synchronize()
    values = [
        value[:response_length]
        for value, response_length in zip(critic(rollout["features"]), rollout["response_lengths"], strict=True)
    ]
    token_rewards = [
        rollout["token_rewards"][sample_index, :response_length]
        for sample_index, response_length in enumerate(rollout["response_lengths"])
    ]
    miles_advantages, miles_returns = get_advantages_and_returns_batch(
        total_lengths=rollout["total_lengths"],
        response_lengths=rollout["response_lengths"],
        values_list=values,
        rewards_list=token_rewards,
        terminal_rewards=rollout["terminal_rewards"],
        qkv_format="thd",
        max_seq_lens=None,
        loss_masks=rollout["masks"],
        gamma=config.gamma,
        lambd=config.lambd,
    )
    assert_gae_reference(rollout, values, miles_advantages, miles_returns, config)
    args = SimpleNamespace(qkv_format="thd", gamma=config.gamma, lambd=config.lambd)
    normalized_advantages = normalize_advantages(
        args,
        miles_advantages,
        rollout["masks"],
        rollout["total_lengths"],
        rollout["response_lengths"],
        max_seq_lens=None,
    )
    torch.cuda.synchronize()
    timings.gae_and_reference += time.perf_counter() - phase_started
    return values, miles_returns, normalized_advantages


def compute_losses(
    config: TrainingConfig,
    rollout: dict[str, Any],
    values: list[torch.Tensor],
    returns: list[torch.Tensor],
    advantages: list[torch.Tensor],
    old_policy_log_probs: list[torch.Tensor],
    current_policy_log_probs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    policy_loss, clip_fraction = assert_clipping_reference(
        old_policy_log_probs,
        current_policy_log_probs,
        advantages,
        config,
    )
    global_tokens = global_mask_sum(rollout["masks"])
    actor_loss = masked_policy_mean(
        policy_loss,
        rollout["response_lengths"],
        rollout["masks"],
        global_tokens,
    )
    critic_values = torch.cat(values)
    packed_returns = torch.cat(returns)
    packed_masks = torch.cat(rollout["masks"])
    critic_loss = ((critic_values - packed_returns) ** 2 * packed_masks).sum()
    critic_loss = critic_loss * dist.get_world_size() / global_tokens
    return actor_loss, critic_loss, clip_fraction, global_tokens


def run_training_segment(
    config: TrainingConfig,
    actor: DDP,
    critic: DDP,
    actor_optimizer: torch.optim.Adam,
    critic_optimizer: torch.optim.Adam,
    start_step: int,
    stop_step: int,
    timings: PhaseTimings,
) -> dict[str, Any]:
    rank = dist.get_rank()
    final_step_checks: dict[str, Any] = {}
    for step in range(start_step, stop_step):
        rollout = make_rollout(config, step, rank)
        assert_sample_accounting(rollout, expected_global_tokens=74)
        values, returns, advantages = prepare_gae(config, critic, rollout, timings)

        phase_started = time.perf_counter()
        torch.cuda.synchronize()
        current_policy_log_probs = current_log_probs(actor, rollout)
        old_policy_log_probs = old_log_probs(current_policy_log_probs, step, rank)
        torch.cuda.synchronize()
        timings.forward += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        torch.cuda.synchronize()
        actor_loss, critic_loss, clip_fraction, global_tokens = compute_losses(
            config,
            rollout,
            values,
            returns,
            advantages,
            old_policy_log_probs,
            current_policy_log_probs,
        )
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        critic_loss.backward()
        torch.cuda.synchronize()
        timings.backward += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        torch.cuda.synchronize()
        actor_optimizer.step()
        critic_optimizer.step()
        torch.cuda.synchronize()
        timings.optimizer += time.perf_counter() - phase_started

        if step == stop_step - 1:
            final_step_checks = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "clip_fraction_mean": float(clip_fraction.detach().mean().cpu()),
                "global_tokens": global_tokens,
            }
    return final_step_checks


def parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def parameter_delta_report(initial: list[torch.Tensor], final: list[torch.Tensor]) -> dict[str, float]:
    deltas = [final_value - initial_value for initial_value, final_value in zip(initial, final, strict=True)]
    flat_delta = torch.cat([delta.reshape(-1) for delta in deltas])
    return {
        "max_abs": float(flat_delta.abs().max().cpu()),
        "l2_norm": float(flat_delta.norm().cpu()),
        "nonzero_count": int(flat_delta.count_nonzero().cpu()),
        "finite": bool(torch.isfinite(flat_delta).all().cpu()),
    }


def compare_parameters(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    differences = [right_value - left_value for left_value, right_value in zip(left, right, strict=True)]
    flat_difference = torch.cat([difference.reshape(-1) for difference in differences])
    return {
        "max_abs": float(flat_difference.abs().max().cpu()),
        "l2_norm": float(flat_difference.norm().cpu()),
        "exact_equal": all(
            torch.equal(left_value, right_value) for left_value, right_value in zip(left, right, strict=True)
        ),
    }


def assert_optimizer_state_equal(
    expected: dict[int, dict[str, torch.Tensor]],
    actual: dict[int, dict[str, torch.Tensor]],
) -> None:
    assert expected.keys() == actual.keys()
    for parameter_index, expected_state in expected.items():
        actual_state = actual[parameter_index]
        assert expected_state.keys() == actual_state.keys()
        for state_name, expected_value in expected_state.items():
            torch.testing.assert_close(expected_value, actual_state[state_name], rtol=0, atol=0)


def parse_arguments() -> tuple[TrainingConfig, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--resume-split", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/job/reduced_ppo_two_rank_results.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/job/reduced-ppo-checkpoint"))
    arguments = parser.parse_args()
    config = TrainingConfig(
        steps=arguments.steps,
        resume_split=arguments.resume_split,
        batch_size=4,
        input_size=32,
        hidden_size=64,
        vocab_size=128,
        gamma=0.99,
        lambd=0.95,
        eps_clip=0.2,
        eps_clip_high=0.2,
        learning_rate=2e-3,
    )
    return config, arguments.output, arguments.checkpoint_dir


def run_uninterrupted(config: TrainingConfig, timings: PhaseTimings) -> UninterruptedResult:
    setup_started = time.perf_counter()
    torch.cuda.synchronize()
    actor, critic, actor_optimizer, critic_optimizer = create_training_pair(config, 1234)
    torch.cuda.synchronize()
    timings.setup += time.perf_counter() - setup_started
    initial_actor = parameter_snapshot(actor.module)
    initial_critic = parameter_snapshot(critic.module)

    run_training_segment(
        config,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        start_step=0,
        stop_step=config.resume_split,
        timings=timings,
    )
    intermediate_actor = parameter_snapshot(actor.module)
    intermediate_critic = parameter_snapshot(critic.module)
    checks = run_training_segment(
        config,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        start_step=config.resume_split,
        stop_step=config.steps,
        timings=timings,
    )
    final_actor = parameter_snapshot(actor.module)
    final_critic = parameter_snapshot(critic.module)
    return UninterruptedResult(
        initial_actor=initial_actor,
        initial_critic=initial_critic,
        intermediate_actor=intermediate_actor,
        intermediate_critic=intermediate_critic,
        final_actor=final_actor,
        final_critic=final_critic,
        checks=checks,
        actor_delta=parameter_delta_report(initial_actor, final_actor),
        critic_delta=parameter_delta_report(initial_critic, final_critic),
    )


def run_checkpoint_resume(
    config: TrainingConfig,
    checkpoint_dir: Path,
    timings: PhaseTimings,
    uninterrupted: UninterruptedResult,
) -> ResumeResult:
    split_actor, split_critic, split_actor_optimizer, split_critic_optimizer = create_training_pair(config, 1234)
    split_initial_actor = parameter_snapshot(split_actor.module)
    split_initial_critic = parameter_snapshot(split_critic.module)
    split_initial_actor_difference = compare_parameters(uninterrupted.initial_actor, split_initial_actor)
    split_initial_critic_difference = compare_parameters(uninterrupted.initial_critic, split_initial_critic)
    assert split_initial_actor_difference["max_abs"] <= 1e-6, split_initial_actor_difference
    assert split_initial_critic_difference["max_abs"] <= 1e-6, split_initial_critic_difference
    run_training_segment(
        config,
        split_actor,
        split_critic,
        split_actor_optimizer,
        split_critic_optimizer,
        start_step=0,
        stop_step=config.resume_split,
        timings=timings,
    )
    timings.checkpoint_save = save_checkpoint(
        split_actor,
        split_critic,
        split_actor_optimizer,
        split_critic_optimizer,
        checkpoint_dir,
    )
    split_final_actor = parameter_snapshot(split_actor.module)
    split_final_critic = parameter_snapshot(split_critic.module)
    split_intermediate_actor_difference = compare_parameters(
        uninterrupted.intermediate_actor,
        split_final_actor,
    )
    split_intermediate_critic_difference = compare_parameters(
        uninterrupted.intermediate_critic,
        split_final_critic,
    )
    assert split_intermediate_actor_difference["max_abs"] <= 1e-6, split_intermediate_actor_difference
    assert split_intermediate_critic_difference["max_abs"] <= 1e-6, split_intermediate_critic_difference
    split_actor_optimizer_state = split_actor_optimizer.state_dict()["state"]
    split_critic_optimizer_state = split_critic_optimizer.state_dict()["state"]

    resumed_actor, resumed_critic, resumed_actor_optimizer, resumed_critic_optimizer = create_training_pair(
        config, 1234
    )
    timings.checkpoint_load = load_checkpoint(
        resumed_actor,
        resumed_critic,
        resumed_actor_optimizer,
        resumed_critic_optimizer,
        checkpoint_dir,
    )
    loaded_actor = parameter_snapshot(resumed_actor.module)
    loaded_critic = parameter_snapshot(resumed_critic.module)
    assert_optimizer_state_equal(
        split_actor_optimizer_state,
        resumed_actor_optimizer.state_dict()["state"],
    )
    assert_optimizer_state_equal(
        split_critic_optimizer_state,
        resumed_critic_optimizer.state_dict()["state"],
    )
    loaded_actor_difference = compare_parameters(split_final_actor, loaded_actor)
    loaded_critic_difference = compare_parameters(split_final_critic, loaded_critic)
    assert loaded_actor_difference["max_abs"] <= 1e-6, loaded_actor_difference
    assert loaded_critic_difference["max_abs"] <= 1e-6, loaded_critic_difference
    checks = run_training_segment(
        config,
        resumed_actor,
        resumed_critic,
        resumed_actor_optimizer,
        resumed_critic_optimizer,
        start_step=config.resume_split,
        stop_step=config.steps,
        timings=timings,
    )
    resumed_final_actor = parameter_snapshot(resumed_actor.module)
    resumed_final_critic = parameter_snapshot(resumed_critic.module)
    actor_resume_difference = compare_parameters(uninterrupted.final_actor, resumed_final_actor)
    critic_resume_difference = compare_parameters(uninterrupted.final_critic, resumed_final_critic)
    assert actor_resume_difference["max_abs"] <= 1e-6, actor_resume_difference
    assert critic_resume_difference["max_abs"] <= 1e-6, critic_resume_difference
    for key in uninterrupted.checks:
        assert abs(uninterrupted.checks[key] - checks[key]) <= 1e-6, key
    return ResumeResult(
        split_intermediate_actor_difference=split_intermediate_actor_difference,
        split_intermediate_critic_difference=split_intermediate_critic_difference,
        loaded_actor_difference=loaded_actor_difference,
        loaded_critic_difference=loaded_critic_difference,
        actor_resume_difference=actor_resume_difference,
        critic_resume_difference=critic_resume_difference,
        checks=checks,
    )


def build_result(
    config: TrainingConfig,
    output: Path,
    checkpoint_dir: Path,
    timings: PhaseTimings,
    total_seconds: float,
    uninterrupted: UninterruptedResult,
    resume: ResumeResult,
) -> dict[str, Any]:
    return {
        "config": vars(config) | {"checkpoint_dir": str(checkpoint_dir)},
        "world_size": dist.get_world_size(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "process_group_backend": dist.get_backend(),
        "process_group_timeout_seconds": 180,
        "optimizer_steps_uninterrupted": config.steps,
        "optimizer_steps_before_checkpoint": config.resume_split,
        "optimizer_steps_after_checkpoint": config.steps - config.resume_split,
        "uninterrupted_final_checks": uninterrupted.checks,
        "resumed_final_checks": resume.checks,
        "actor_parameter_delta": uninterrupted.actor_delta,
        "critic_parameter_delta": uninterrupted.critic_delta,
        "actor_resume_difference": resume.actor_resume_difference,
        "critic_resume_difference": resume.critic_resume_difference,
        "actor_checkpoint_load_difference": resume.loaded_actor_difference,
        "critic_checkpoint_load_difference": resume.loaded_critic_difference,
        "actor_split_intermediate_difference": resume.split_intermediate_actor_difference,
        "critic_split_intermediate_difference": resume.split_intermediate_critic_difference,
        "timings_seconds": vars(timings) | {"total": total_seconds},
    }


def main() -> None:
    config, output, checkpoint_dir = parse_arguments()
    run_started = time.perf_counter()
    initialize_process_group()
    initialize_parallel_state()
    timings = PhaseTimings()
    uninterrupted = run_uninterrupted(config, timings)
    resume = run_checkpoint_resume(config, checkpoint_dir, timings, uninterrupted)
    total_seconds = time.perf_counter() - run_started
    result = build_result(
        config,
        output,
        checkpoint_dir,
        timings,
        total_seconds,
        uninterrupted,
        resume,
    )
    if dist.get_rank() == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
