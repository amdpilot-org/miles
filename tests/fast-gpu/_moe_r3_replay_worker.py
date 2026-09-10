"""Two-rank MoE R3 replay fixture for real NCCL all-to-all and backward."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from statistics import fmean

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.mappings import all_to_all

from miles.utils.reloadable_process_group import ReloadableProcessGroup, monkey_patch_torch_dist
from miles.utils.replay_base import routing_replay_manager


CYCLES = 64
TOKENS_PER_RANK = 32
HIDDEN = 16
NUM_EXPERTS = 4
TOPK = 2
SEED = 1649
TIMEOUT_SECONDS = 60
EXPERT_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
)


def _git_commit(path: str) -> str:
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def _make_scores(rank: int, cycle: int) -> torch.Tensor:
    token_ids = torch.arange(
        rank * TOKENS_PER_RANK,
        (rank + 1) * TOKENS_PER_RANK,
        dtype=torch.int64,
        device="cpu",
    )
    scores = torch.zeros((TOKENS_PER_RANK, NUM_EXPERTS), dtype=torch.float32)
    for token_index, token_id in enumerate(token_ids.tolist()):
        if token_index == 0:
            digit = cycle % 12
        elif token_index == 1:
            digit = (cycle // 12) % 12
        else:
            digit = (token_index + cycle) % 12
        first_expert, second_expert = EXPERT_PAIRS[digit]
        scores[token_index, first_expert] = 2.0
        scores[token_index, second_expert] = 1.0
    return scores.to(torch.cuda.current_device())


def _topk(scores: torch.Tensor, topk: int) -> torch.Tensor:
    return torch.topk(scores, topk, dim=-1).indices


class _ReplayTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, topk: int, topk_fn):
        ctx.topk = topk
        ctx.topk_fn = topk_fn
        ctx.save_for_backward(scores)
        return topk_fn(scores, topk)

    @staticmethod
    def backward(ctx, _grad_output):
        scores, = ctx.saved_tensors
        ctx.topk_fn(scores, ctx.topk)
        return None, None, None


def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    outputs = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(outputs, tensor, group=group)
    return outputs


def _validate_splits(
    input_rows: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    group: dist.ProcessGroup,
) -> dict[str, object]:
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    expected_shape = (world_size,)
    assert input_splits.shape == expected_shape, (
        f"rank {rank}: input_splits shape {tuple(input_splits.shape)} != {expected_shape}; "
        f"input_splits={input_splits.tolist()}"
    )
    assert output_splits.shape == expected_shape, (
        f"rank {rank}: output_splits shape {tuple(output_splits.shape)} != {expected_shape}; "
        f"output_splits={output_splits.tolist()}"
    )
    assert (input_splits >= 0).all(), (
        f"rank {rank}: negative input split; input_splits={input_splits.tolist()}"
    )
    assert (output_splits >= 0).all(), (
        f"rank {rank}: negative output split; output_splits={output_splits.tolist()}"
    )

    gathered_inputs = torch.stack(_all_gather_tensor(input_splits, group)).cpu()
    gathered_outputs = torch.stack(_all_gather_tensor(output_splits, group)).cpu()
    actual_input_rows = int(input_splits.sum().item())
    expected_output_rows = int(output_splits.sum().item())
    diagnostics = {
        "rank": rank,
        "world_size": world_size,
        "input_rows": input_rows,
        "input_splits": input_splits.tolist(),
        "output_splits": output_splits.tolist(),
        "gathered_inputs": gathered_inputs.tolist(),
        "gathered_outputs": gathered_outputs.tolist(),
        "expected_output_rows": expected_output_rows,
    }
    assert actual_input_rows == input_rows, (
        f"rank {rank}: local input split sum {actual_input_rows} != input rows {input_rows}; "
        f"diagnostics={diagnostics}"
    )
    assert gathered_outputs.equal(gathered_inputs.t()), (
        f"rank {rank}: cross-rank split matrices disagree; diagnostics={diagnostics}"
    )
    assert output_splits.equal(gathered_inputs[:, rank].to(output_splits.device)), (
        f"rank {rank}: local output_splits disagree with gathered input column; diagnostics={diagnostics}"
    )
    return diagnostics


def _make_hidden(
    rank: int,
    cycle: int,
    top_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.cuda.current_device()
    token_ids = torch.arange(
        rank * TOKENS_PER_RANK,
        (rank + 1) * TOKENS_PER_RANK,
        dtype=torch.int64,
        device=device,
    ).repeat_interleave(TOPK)
    expert_ids = top_indices.reshape(-1).to(torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1000 + rank * CYCLES + cycle)
    random_columns = torch.randn((TOKENS_PER_RANK * TOPK, HIDDEN - 2), generator=generator)
    hidden = torch.zeros((TOKENS_PER_RANK * TOPK, HIDDEN), device=device)
    hidden[:, 0] = token_ids
    hidden[:, 1] = expert_ids
    hidden[:, 2:] = random_columns.to(device)
    return hidden, token_ids, expert_ids


def _distributed_moe_forward(
    hidden: torch.Tensor,
    top_indices: torch.Tensor,
    local_weights: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, float]]:
    device = hidden.device
    world_size = dist.get_world_size(group=group)
    experts_per_rank = NUM_EXPERTS // world_size
    expert_ids = top_indices.reshape(-1).to(torch.int64)
    destination_ranks = torch.div(expert_ids, experts_per_rank, rounding_mode="floor")
    sort_order = torch.argsort(destination_ranks, stable=True)
    inverse_sort_order = torch.empty_like(sort_order)
    inverse_sort_order[sort_order] = torch.arange(sort_order.numel(), device=device)
    sorted_hidden = hidden[sort_order].contiguous()
    input_splits = torch.bincount(destination_ranks, minlength=world_size).to(torch.int64)
    gathered_inputs = torch.stack(_all_gather_tensor(input_splits, group))
    output_splits = gathered_inputs[:, rank].contiguous()

    dispatch_check = _validate_splits(
        sorted_hidden.shape[0], input_splits, output_splits, group
    )
    dispatch_start = torch.cuda.Event(enable_timing=True)
    dispatch_end = torch.cuda.Event(enable_timing=True)
    dispatch_start.record()
    dispatched = all_to_all(
        group, sorted_hidden, output_splits.tolist(), input_splits.tolist()
    )
    dispatch_end.record()
    dispatch_end.synchronize()

    expert_start = torch.cuda.Event(enable_timing=True)
    expert_end = torch.cuda.Event(enable_timing=True)
    expert_start.record()
    received_experts = dispatched[:, 1].to(torch.int64)
    local_expert_ids = received_experts - rank * experts_per_rank
    expert_output = dispatched * local_weights[local_expert_ids]
    expert_end.record()
    expert_end.synchronize()

    combine_check = _validate_splits(
        dispatched.shape[0], output_splits, input_splits, group
    )
    combine_start = torch.cuda.Event(enable_timing=True)
    combine_end = torch.cuda.Event(enable_timing=True)
    combine_start.record()
    combined = all_to_all(
        group, expert_output, input_splits.tolist(), output_splits.tolist()
    )
    combine_end.record()
    combine_end.synchronize()

    restored = combined[inverse_sort_order]
    output = restored.view(TOKENS_PER_RANK, TOPK, HIDDEN).sum(dim=1)
    timings = {
        "dispatch_ms": dispatch_start.elapsed_time(dispatch_end),
        "expert_ms": expert_start.elapsed_time(expert_end),
        "combine_ms": combine_start.elapsed_time(combine_end),
    }
    checks = {
        "dispatch_split_sum": dispatch_check["input_rows"],
        "combine_split_sum": combine_check["input_rows"],
    }
    return output, restored, checks, timings


def _reference_forward(
    hidden: torch.Tensor,
    top_indices: torch.Tensor,
    local_weights: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group=group)
    hidden_outs = _all_gather_tensor(hidden.detach(), group)
    topk_outs = _all_gather_tensor(top_indices, group)
    weight_outs = _all_gather_tensor(local_weights.detach(), group)
    global_hidden = torch.cat(hidden_outs, dim=0)
    global_topk = torch.cat(topk_outs, dim=0)
    global_weights = torch.cat(weight_outs, dim=0)
    global_experts = global_topk.reshape(-1).to(torch.int64)
    reference_hidden = global_hidden.detach().clone().requires_grad_()
    reference_output = (
        reference_hidden.view(-1, TOPK, HIDDEN) * global_weights[global_topk]
    ).sum(dim=1)
    reference_loss = reference_output.square().sum() / (TOKENS_PER_RANK * HIDDEN)
    reference_loss.backward()
    reference_grad_output = 2 * reference_output / (TOKENS_PER_RANK * HIDDEN)
    instance_grad_output = reference_grad_output.repeat_interleave(TOPK, dim=0)
    expected_weight_grad = torch.zeros_like(local_weights)
    for expert_id in range(rank * (NUM_EXPERTS // world_size), (rank + 1) * (NUM_EXPERTS // world_size)):
        mask = global_experts == expert_id
        expected_weight_grad[expert_id - rank * (NUM_EXPERTS // world_size)] += (
            instance_grad_output[mask] * global_hidden[mask]
        ).sum(dim=0)
    local_input_grad = reference_hidden.grad[
        rank * TOKENS_PER_RANK * TOPK : (rank + 1) * TOKENS_PER_RANK * TOPK
    ]
    local_reference_output = reference_output[
        rank * TOKENS_PER_RANK : (rank + 1) * TOKENS_PER_RANK
    ]
    local_reference_loss = local_reference_output.square().mean()
    return local_reference_output, local_input_grad, expected_weight_grad, local_reference_loss


def _aggregate_timings(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = samples[0].keys()
    return {
        key: {
            "min_ms": min(sample[key] for sample in samples),
            "max_ms": max(sample[key] for sample in samples),
            "mean_ms": fmean(sample[key] for sample in samples),
        }
        for key in keys
    }


def _run_control(control: str, group: dist.ProcessGroup) -> None:
    rank = dist.get_rank(group=group)
    rows = TOKENS_PER_RANK * TOPK
    if control == "local-input":
        input_splits = torch.tensor([rows + 1, 0], device=torch.cuda.current_device())
        output_splits = torch.tensor([0, rows + 1], device=torch.cuda.current_device())
    elif control == "cross-rank":
        input_splits = torch.tensor([rows // 2, rows // 2], device=torch.cuda.current_device())
        output_splits = (
            torch.tensor([0, rows], device=torch.cuda.current_device())
            if rank == 0
            else torch.tensor([rows, 0], device=torch.cuda.current_device())
        )
    else:
        raise ValueError(f"unknown control {control}")
    _validate_splits(rows, input_splits, output_splits, group)


def _write_result(
    output_dir: Path,
    rank: int,
    result: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"rank{rank}.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("none", "local-input", "cross-rank"), default="none")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=TIMEOUT_SECONDS))
    monkey_patch_torch_dist()
    group = dist.new_group(
        ranks=list(range(dist.get_world_size())),
        backend="nccl",
        timeout=timedelta(seconds=TIMEOUT_SECONDS),
    )
    assert isinstance(group, ReloadableProcessGroup)
    rank = dist.get_rank(group=group)

    if args.control != "none":
        _run_control(args.control, group)
        dist.destroy_process_group()
        return

    manager = routing_replay_manager
    manager.enabled = True
    manager.stage = "record"
    replay = manager.create_replay()
    manager.set_current(replay)
    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    record_start = torch.cuda.Event(enable_timing=True)
    record_end = torch.cuda.Event(enable_timing=True)
    record_start.record()
    recorded_indices = []
    for cycle in range(CYCLES):
        scores = _make_scores(rank, cycle)
        indices = topk_fn(scores, TOPK)
        recorded_indices.append(indices.detach().cpu())
    record_end.record()
    record_end.synchronize()
    manager.clear_all_forward()

    generator = torch.Generator(device="cpu").manual_seed(SEED + rank)
    local_weights = torch.randn((NUM_EXPERTS // dist.get_world_size(group=group), HIDDEN), generator=generator)
    local_weights[:, :2] = torch.tensor([[1.0, 1.0]])
    local_weights = local_weights.to(device).requires_grad_()
    optimizer = torch.optim.SGD([local_weights], lr=1e-3)

    phase_timings: list[dict[str, float]] = []
    forward_timings: list[float] = []
    backward_timings: list[float] = []
    max_output_error = 0.0
    max_input_grad_error = 0.0
    max_weight_grad_error = 0.0
    split_checks = 0
    token_identity_checks = 0
    gradient_checks = 0

    manager.stage = "replay_forward"
    for cycle in range(CYCLES):
        scores = _make_scores(rank, cycle)
        top_indices = _ReplayTopK.apply(scores, TOPK, topk_fn)
        hidden, token_ids, expert_ids = _make_hidden(rank, cycle, top_indices)
        hidden.requires_grad_()
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        actual_output, restored, checks, timings = _distributed_moe_forward(
            hidden, top_indices, local_weights, rank, group
        )
        forward_end.record()
        forward_end.synchronize()
        phase_timings.append(timings)
        forward_timings.append(forward_start.elapsed_time(forward_end))
        split_checks += len(checks)

        reference_output, reference_input_grad, expected_weight_grad, reference_loss = _reference_forward(
            hidden, top_indices, local_weights, rank, group
        )
        actual_loss = actual_output.square().mean()
        torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_output, reference_output, rtol=1e-5, atol=1e-6)
        max_output_error = max(
            max_output_error,
            float((actual_output - reference_output).abs().max().item()),
        )

        manager.stage = "replay_backward"
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        backward_start.record()
        actual_loss.backward()
        backward_end.record()
        backward_end.synchronize()
        backward_timings.append(backward_start.elapsed_time(backward_end))

        torch.testing.assert_close(hidden.grad, reference_input_grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(local_weights.grad, expected_weight_grad, rtol=1e-5, atol=1e-6)
        max_input_grad_error = max(
            max_input_grad_error,
            float((hidden.grad - reference_input_grad).abs().max().item()),
        )
        max_weight_grad_error = max(
            max_weight_grad_error,
            float((local_weights.grad - expected_weight_grad).abs().max().item()),
        )
        gradient_checks += 2

        restored_token_ids = restored[:, 0].to(torch.int64)
        restored_expert_ids = restored[:, 1].to(torch.int64)
        expected_token_ids = token_ids
        expected_expert_ids = expert_ids
        assert torch.equal(restored_token_ids, expected_token_ids), (
            f"rank {rank}: token identity mismatch at cycle {cycle}; "
            f"expected={expected_token_ids.tolist()}, actual={restored_token_ids.tolist()}"
        )
        assert torch.equal(restored_expert_ids, expected_expert_ids), (
            f"rank {rank}: expert identity mismatch at cycle {cycle}; "
            f"expected={expected_expert_ids.tolist()}, actual={restored_expert_ids.tolist()}"
        )
        token_identity_checks += 1

        local_weights.grad[:, :2] = 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        manager.stage = "replay_forward"

    unique_distributions = torch.unique(torch.stack(recorded_indices), dim=0).shape[0]
    timing_summary = _aggregate_timings(phase_timings)
    timing_summary["forward_ms"] = {
        "min_ms": min(forward_timings),
        "max_ms": max(forward_timings),
        "mean_ms": fmean(forward_timings),
    }
    timing_summary["backward_ms"] = {
        "min_ms": min(backward_timings),
        "max_ms": max(backward_timings),
        "mean_ms": fmean(backward_timings),
    }
    timing_summary["record_ms"] = {
        "min_ms": record_start.elapsed_time(record_end),
        "max_ms": record_start.elapsed_time(record_end),
        "mean_ms": record_start.elapsed_time(record_end),
    }
    result = {
        "rank": rank,
        "world_size": dist.get_world_size(group=group),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "miles_commit": _git_commit(str(Path(__file__).resolve().parents[2])),
        "megatron_commit": _git_commit(
            str(Path(inspect.getfile(all_to_all)).resolve().parents[3])
        ),
        "cycles": CYCLES,
        "unique_distributions": int(unique_distributions),
        "all_to_all_forward_calls": CYCLES * 2,
        "all_to_all_backward_calls": CYCLES * 2,
        "split_checks": split_checks,
        "token_identity_checks": token_identity_checks,
        "gradient_checks": gradient_checks,
        "malformed_controls": 2,
        "max_output_error": max_output_error,
        "max_input_gradient_error": max_input_grad_error,
        "max_weight_gradient_error": max_weight_grad_error,
        "timings": timing_summary,
    }
    _write_result(args.output, rank, result)
    manager.clear_all()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
