"""Two-rank GPU fixture for rollout step and dynamic-GBS accounting.

The fixture uses a one-hot parameter so each consumed sample contributes an
identifiable gradient weight.  It compares Miles' measured schedule with an
explicit reference derived only from DP size, active GBS, and arrival order.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import megatron
import sglang
import torch
import torch.distributed as dist

import miles
from miles.backends.training_utils.data import get_data_iterator
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.ray.rollout.rollout_data_conversion import _compute_dynamic_global_batch_size


class OneHotModel(torch.nn.Module):
    def __init__(self, sample_count: int, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(sample_count, device=device))

    def forward(self, sample_index: torch.Tensor) -> torch.Tensor:
        return self.weight[sample_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixed", "dynamic"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def module_paths() -> dict[str, object]:
    megatron_spec = importlib.util.find_spec("megatron")
    return {
        "miles": miles.__file__,
        "sglang": sglang.__file__,
        "megatron": list(megatron.__path__),
        "megatron_spec_origin": megatron_spec.origin if megatron_spec else None,
        "torch": torch.__file__,
        "torch_native": torch._C.__file__,
    }


def initialize_parallel_state(rank: int, world_size: int) -> None:
    group = dist.distributed_c10d._get_default_group()
    dp_group = GroupInfo(rank=rank, size=world_size, group=group)
    trivial = GroupInfo(rank=0, size=1, group=None)
    state = ParallelState(
        intra_dp=dp_group,
        intra_dp_cp=dp_group,
        cp=trivial,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
        meshes={},
        cp_comm_type=None,
        is_pp_last_stage=True,
        vpp_size=1,
        microbatch_group_size_per_vp_stage=1,
    )
    set_parallel_state(state)


def asynchronous_local_arrival(mode: str, rank: int) -> tuple[list[int], float]:
    delay_seconds = 0.08 if rank == 0 else 0.02
    started = time.monotonic()
    arrived = threading.Event()

    def produce() -> None:
        time.sleep(delay_seconds)
        arrived.set()

    producer = threading.Thread(target=produce)
    producer.start()
    assert arrived.wait(timeout=5.0)
    producer.join()
    if mode == "fixed":
        indices = [0, 2, 4, 1, 3, 5] if rank == 0 else [7, 6, 11, 9, 8, 10]
    else:
        indices = [0, 2, 4, 1, 3, 5, 8] if rank == 0 else [7, 6, 11, 9, 10, 12]
    return indices, time.monotonic() - started


def run_case(
    mode: str,
    rank: int,
    world_size: int,
    device: torch.device,
    arrival_by_rank: list[list[int]],
) -> dict[str, object]:
    total_samples = sum(len(indices) for indices in arrival_by_rank)
    local_indices = arrival_by_rank[rank]
    rollout_data = {
        "total_lengths": [1] * len(local_indices),
        "sample_index": local_indices,
    }
    args = SimpleNamespace(
        global_batch_size=4,
        use_dynamic_global_batch_size=mode == "dynamic",
        use_dynamic_batch_size=False,
        micro_batch_size=1,
        qkv_format="thd",
    )
    if mode == "dynamic":
        dynamic_gbs = _compute_dynamic_global_batch_size(
            args,
            train_parallel_config={"dp_size": world_size},
            num_samples=total_samples,
        )
        rollout_data["dynamic_global_batch_size"] = dynamic_gbs
    else:
        dynamic_gbs = None

    data_iterators, num_microbatches = get_data_iterator(args, model=None, rollout_data=rollout_data)
    data_iterator = data_iterators[0]
    active_gbs = dynamic_gbs if dynamic_gbs is not None else args.global_batch_size
    local_gbs = active_gbs // world_size
    optimizer_steps = len(num_microbatches)

    model = OneHotModel(total_samples, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    observed_weights = torch.zeros(total_samples, device=device)
    step_weights: list[list[float]] = []
    consumed_indices: list[int] = []

    for step_id in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=False)
        for _ in range(num_microbatches[step_id]):
            batch = data_iterator.get_next(["sample_index"])
            sample_index = torch.tensor(batch["sample_index"][0], device=device)
            loss = model(sample_index) / active_gbs
            loss.backward()
            consumed_indices.append(int(sample_index.item()))
        dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM)
        observed_weights += model.weight.grad.detach().clone()
        step_weights.append(model.weight.grad.detach().cpu().tolist())
        optimizer.step()

    reference_weights = torch.zeros(total_samples, device=device)
    consumed_per_rank = local_gbs * optimizer_steps
    for indices in arrival_by_rank:
        for sample_index in indices[:consumed_per_rank]:
            reference_weights[sample_index] = 1.0 / active_gbs

    max_error = float((observed_weights - reference_weights).abs().max().item())
    assert max_error == 0.0, f"gradient-weight mismatch: {max_error}"
    assert optimizer_steps == (3 if mode == "fixed" else 1)
    observed_count = int((observed_weights != 0).sum().item())
    assert observed_count == 12, f"expected 12 consumed samples, got {observed_count}"

    return {
        "mode": mode,
        "world_size": world_size,
        "total_arrived_samples": total_samples,
        "arrival_by_rank": arrival_by_rank,
        "active_global_batch_size": active_gbs,
        "dynamic_global_batch_size": dynamic_gbs,
        "local_global_batch_size": local_gbs,
        "num_microbatches": num_microbatches,
        "optimizer_steps": optimizer_steps,
        "consumed_samples": observed_count,
        "consumed_indices": [
            sample_index for sample_index, weight in enumerate(observed_weights.detach().cpu().tolist()) if weight != 0
        ],
        "local_consumed_indices": consumed_indices,
        "dropped_indices": [index for indices in arrival_by_rank for index in indices[consumed_per_rank:]],
        "effective_gradient_weights": observed_weights.detach().cpu().tolist(),
        "reference_gradient_weights": reference_weights.detach().cpu().tolist(),
        "per_step_gradient_weights": step_weights,
        "max_gradient_weight_error": max_error,
    }


def main() -> None:
    cli_args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, f"this fixture requires exactly 2 GPU ranks, got {world_size}"

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=cli_args.timeout_seconds))
    initialize_parallel_state(rank, world_size)

    local_indices, arrival_delay = asynchronous_local_arrival(cli_args.mode, rank)
    max_local_samples = 7 if cli_args.mode == "dynamic" else 6
    arrival_tensor = torch.full((max_local_samples,), -1, device=device, dtype=torch.int64)
    arrival_tensor[: len(local_indices)] = torch.tensor(local_indices, device=device, dtype=torch.int64)
    gathered_arrival = [torch.empty_like(arrival_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_arrival, arrival_tensor)
    arrival_by_rank = [[index for index in tensor.tolist() if index >= 0] for tensor in gathered_arrival]

    result = run_case(cli_args.mode, rank, world_size, device, arrival_by_rank)
    result.update(
        {
            "rank": rank,
            "local_rank": local_rank,
            "arrival_delay_seconds": arrival_delay,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "module_paths": module_paths(),
        }
    )

    if rank == 0:
        payload = json.dumps(result, indent=2, sort_keys=True)
        if cli_args.output is not None:
            cli_args.output.parent.mkdir(parents=True, exist_ok=True)
            cli_args.output.write_text(payload + "\n")
        else:
            print(payload)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
