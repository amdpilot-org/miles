from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from miles.backends.training_utils.data import DataIterator
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    split_train_data_by_dp_scheduled_raw,
)
from miles.utils.types import Sample


ARRIVAL_ORDER = [7, 2, 11, 0, 5, 9, 1, 8, 3, 10, 4, 6]
INITIAL_WEIGHT = 0.25
LEARNING_RATE = 0.03
PROCESS_GROUP_TIMEOUT_SECONDS = 120


def make_args(*, use_dynamic_global_batch_size: bool) -> SimpleNamespace:
    return SimpleNamespace(
        global_batch_size=4,
        num_steps_per_rollout=3,
        rollout_batch_size=12,
        n_samples_per_prompt=1,
        use_dynamic_global_batch_size=use_dynamic_global_batch_size,
        disable_rollout_trim_samples=False,
        multi_lora=False,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        micro_batch_size=1,
        balance_data=False,
        balance_by_flops=False,
        allow_partial_train_step=False,
        advantage_estimator="ppo",
        rewards_normalization=False,
        reward_key=None,
    )


def make_samples() -> list[Sample]:
    return [
        Sample(
            index=sample_index,
            rollout_id=None,
            tokens=[0, 1],
            response_length=2,
            loss_mask=[1, 1],
            reward=float(sample_index),
            status=Sample.Status.COMPLETED,
        )
        for sample_index in ARRIVAL_ORDER
    ]


def train_parallel_config() -> dict[str, int]:
    return {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }


def module_path(name: str) -> str | None:
    try:
        module = __import__(name, fromlist=["__path__"])
    except Exception:
        return None
    return getattr(module, "__file__", None)


def native_module_paths() -> dict[str, str | None]:
    torch_c = getattr(torch._C, "__file__", None)
    return {
        "torch": torch.__file__,
        "torch._C": torch_c,
        "torch_cuda_library": str(Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"),
    }


def run_case(
    case_name: str,
    *,
    use_dynamic_global_batch_size: bool,
    rank: int,
    device: torch.device,
) -> dict[str, Any]:
    args = make_args(use_dynamic_global_batch_size=use_dynamic_global_batch_size)
    samples = make_samples()
    rollout_samples, metadata = postprocess_rollout_data(args, samples, train_parallel_config())
    train_data = convert_samples_to_train_data(
        args,
        rollout_samples,
        metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    shards = split_train_data_by_dp_scheduled_raw(
        args,
        train_data,
        train_parallel_config=train_parallel_config(),
    )
    shard = shards[rank]
    global_batch_size = shard.get("dynamic_global_batch_size", args.global_batch_size)
    data_iterator = DataIterator(shard, micro_batch_indices=shard["micro_batch_indices"])

    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(INITIAL_WEIGHT)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    optimizer_steps = len(shard["num_microbatches"])
    consumed_indices_by_step: list[list[int]] = []
    for _step_id in range(optimizer_steps):
        model.zero_grad()
        local_indices: list[int] = []
        for _microbatch_id in range(shard["num_microbatches"][_step_id]):
            batch = data_iterator.get_next(["sample_indices"])
            local_indices.extend(batch["sample_indices"])
        for sample_index in local_indices:
            input_value = torch.tensor([1.0], device=device)
            target_value = float(sample_index)
            loss = (model(input_value) - target_value) ** 2
            loss.backward()
        dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM)
        model.weight.grad /= global_batch_size
        optimizer.step()
        gathered_indices: list[list[int]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_indices, local_indices)
        consumed_indices_by_step.append(sorted(index for indices in gathered_indices for index in indices))

    reference_weight = INITIAL_WEIGHT
    for step_indices in consumed_indices_by_step:
        gradient = sum(2 * (reference_weight - index) for index in step_indices) / global_batch_size
        reference_weight -= LEARNING_RATE * gradient

    final_weight = model.weight.item()
    consumed_samples = sum(len(step_indices) for step_indices in consumed_indices_by_step)
    return {
        "case": case_name,
        "arrival_order": ARRIVAL_ORDER,
        "arrived_samples": len(samples),
        "configured_global_batch_size": args.global_batch_size,
        "configured_num_steps_per_rollout": args.num_steps_per_rollout,
        "use_dynamic_global_batch_size": use_dynamic_global_batch_size,
        "dynamic_global_batch_size": global_batch_size,
        "optimizer_steps": optimizer_steps,
        "consumed_samples": consumed_samples,
        "dropped_samples": len(samples) - consumed_samples,
        "per_step": [
            {
                "step": step_id,
                "global_batch_size": global_batch_size,
                "consumed_sample_indices": step_indices,
                "effective_gradient_weight_per_sample": 1.0 / global_batch_size,
            }
            for step_id, step_indices in enumerate(consumed_indices_by_step)
        ],
        "final_weight": final_weight,
        "reference_final_weight": reference_weight,
        "max_abs_error": abs(final_weight - reference_weight),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        fixed_result = run_case(
            "num_steps_per_rollout_3",
            use_dynamic_global_batch_size=False,
            rank=rank,
            device=device,
        )
        dynamic_result = run_case(
            "dynamic_global_batch_overrides_num_steps",
            use_dynamic_global_batch_size=True,
            rank=rank,
            device=device,
        )
        if rank == 0:
            result = {
                "world_size": dist.get_world_size(),
                "device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "arrival_order": ARRIVAL_ORDER,
                "module_paths": {
                    "miles": module_path("miles"),
                    "miles.backends.training_utils.data": module_path("miles.backends.training_utils.data"),
                    "miles.ray.rollout.rollout_data_conversion": module_path(
                        "miles.ray.rollout.rollout_data_conversion"
                    ),
                    "miles.ray.rollout.train_data_conversion": module_path(
                        "miles.ray.rollout.train_data_conversion"
                    ),
                    "sglang": module_path("sglang"),
                    "megatron": module_path("megatron"),
                    "megatron.core": module_path("megatron.core"),
                    **native_module_paths(),
                },
                "cases": [fixed_result, dynamic_result],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
