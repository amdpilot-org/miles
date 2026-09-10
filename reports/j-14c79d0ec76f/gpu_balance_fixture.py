from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from statistics import fmean

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp_scheduled_raw,
)
from miles.utils.function_registry import function_registry
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import Sample


BATCHES = 72
DP_SIZE = 2
OUTPUT_PATH = Path(__file__).with_name("gpu_validation.json")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.output = torch.nn.Linear(8, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(tokens)).unsqueeze(0)


def fixture_loss(args, batch, logits, sum_of_sample_mean):
    values = logits.squeeze(0).squeeze(-1)
    loss = sum_of_sample_mean(values)
    return loss, {"loss": loss.detach()}


def make_args() -> object:
    return type(
        "FixtureArgs",
        (),
        {
            "global_batch_size": 64,
            "use_dynamic_global_batch_size": True,
            "disable_rollout_trim_samples": False,
            "balance_data": True,
            "balance_by_flops": False,
            "use_dynamic_batch_size": True,
            "max_tokens_per_gpu": 32,
            "micro_batch_size": 1,
            "allow_partial_train_step": False,
            "advantage_estimator": "reinforce",
            "rewards_normalization": False,
            "reward_key": None,
            "multi_lora": False,
            "loss_type": "custom_loss",
            "custom_loss_function_path": "fixture:loss",
            "recompute_loss_function": False,
            "calculate_per_token_loss": False,
            "qkv_format": "thd",
            "get_mismatch_metrics": False,
            "use_tis": False,
            "use_opsm": False,
            "use_rollout_logprobs": False,
            "skip_actor_forward_only": False,
            "entropy_coef": 0.0,
            "observe_training_entropy": False,
            "use_kl_loss": False,
            "use_unbiased_kl": False,
            "kl_coef": 0.0,
            "allgather_cp": False,
            "true_on_policy_mode": False,
            "rollout_temperature": 1.0,
            "bf16": False,
            "fp16": False,
            "use_opd": False,
            "opd_type": "megatron",
        },
    )()


def make_parallel_state(rank: int, dp_size: int) -> ParallelState:
    trivial = GroupInfo(rank=0, size=1, group=None)
    dp = GroupInfo(rank=rank, size=dp_size, group=None)
    return ParallelState(
        intra_dp=dp,
        intra_dp_cp=dp,
        cp=trivial,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
        meshes={},
    )


def make_samples(batch_index: int) -> tuple[list[Sample], list[Sample], list[int]]:
    raw_count = 12
    valid_count = 10 if batch_index % 2 == 0 else 11
    invalid_count = raw_count - valid_count
    raw: list[Sample] = []
    raw_lengths: list[int] = []
    for sample_index in range(raw_count):
        length = 3 + ((batch_index * 5 + sample_index * 3) % 12)
        raw_lengths.append(length)
        raw.append(
            Sample(
                index=sample_index,
                tokens=[(batch_index * 17 + sample_index * 5 + position) % 128 for position in range(length)],
                response_length=length,
                reward=float((sample_index + 1) % 7),
                loss_mask=[1] * length,
                status=Sample.Status.COMPLETED,
                metadata={},
                remove_sample=sample_index < invalid_count,
            )
        )
    valid = [sample for sample in raw if not sample.remove_sample]
    return raw, valid, raw_lengths


def convert_batch(args, valid_samples: list[Sample]) -> dict:
    valid_samples, metadata = postprocess_rollout_data(
        args,
        valid_samples,
        train_parallel_config={"dp_size": DP_SIZE},
    )
    data = convert_samples_to_train_data(
        args,
        valid_samples,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    return data


def make_reference_batch(data: dict, device: torch.device) -> dict:
    return {
        "tokens": data["tokens"],
        "response_lengths": data["response_lengths"],
        "loss_masks": [torch.tensor(mask, dtype=torch.float32, device=device) for mask in data["loss_masks"]],
        "total_lengths": data["total_lengths"],
        "rollout_mask_sums": [
            torch.tensor(value, dtype=torch.float32, device=device) for value in data["rollout_mask_sums"]
        ],
        "dynamic_global_batch_size": data["dynamic_global_batch_size"],
    }


def synchronize() -> None:
    torch.cuda.synchronize()


def timed(callback: Callable[[], object]) -> tuple[object, float]:
    synchronize()
    start = time.perf_counter()
    result = callback()
    synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))

    args = make_args()
    torch.manual_seed(14)
    model = TinyModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    reference_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    local_state = make_parallel_state(rank, DP_SIZE)
    reference_state = make_parallel_state(0, 1)
    records: list[dict] = []

    with function_registry.temporary("fixture:loss", fixture_loss):
        for batch_index in range(BATCHES):
            data = None
            shards = None
            if rank == 0:
                raw, valid, raw_lengths = make_samples(batch_index)
                data = convert_batch(args, valid)
                shards = split_train_data_by_dp_scheduled_raw(
                    args,
                    data,
                    train_parallel_config={
                        "dp_size": DP_SIZE,
                        "cp_size": 1,
                        "vpp_size": 1,
                        "microbatch_group_size_per_vp_stage": None,
                    },
                )
                consumed = sorted(index for shard in shards for index in shard["partition"])
                assert consumed == list(range(len(valid))), (batch_index, consumed, len(valid))
            payload = [data, shards]
            dist.broadcast_object_list(payload, src=0, device=device)
            data, shards = payload

            shard = shards[rank]
            local_partition = list(shard["partition"])
            process_rollout_data_shard(args, shard)
            global_batch_size = data["dynamic_global_batch_size"]
            local_effective_weight = len(local_partition) / global_batch_size

            optimizer.zero_grad(set_to_none=True)
            reference_model.load_state_dict(model.state_dict())
            reference_model.zero_grad(set_to_none=True)

            set_parallel_state(local_state)
            iterator = DataIterator(shard, micro_batch_indices=shard["micro_batch_indices"])
            forward_ms = 0.0
            backward_ms = 0.0
            for _ in range(sum(shard["num_microbatches"])):
                keys = ("tokens", "response_lengths", "loss_masks", "total_lengths", "rollout_mask_sums")
                values = iterator.get_next(keys)
                batch = {
                    "tokens": values["tokens"],
                    "response_lengths": values["response_lengths"],
                    "loss_masks": [torch.tensor(mask, device=device) for mask in values["loss_masks"]],
                    "total_lengths": values["total_lengths"],
                    "rollout_mask_sums": [
                        torch.tensor(value, dtype=torch.float32, device=device)
                        for value in values["rollout_mask_sums"]
                    ],
                    "dynamic_global_batch_size": global_batch_size,
                }
                tokens = torch.tensor([token for sample in batch["tokens"] for token in sample], device=device)

                def local_forward() -> tuple[torch.Tensor, dict]:
                    logits = ddp_model(tokens)
                    return loss_function(
                        args,
                        batch,
                        num_microbatches=sum(shard["num_microbatches"]),
                        logits=logits,
                        apply_megatron_loss_scaling=False,
                        num_rollouts=global_batch_size,
                    )

                (loss, _normalizer, _metrics), elapsed = timed(local_forward)
                forward_ms += elapsed

                def local_backward() -> None:
                    loss.backward()

                _, elapsed = timed(local_backward)
                backward_ms += elapsed

            count_tensor = torch.tensor([len(local_partition)], dtype=torch.float32, device=device)
            weight_tensor = torch.tensor([local_effective_weight], dtype=torch.float32, device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(weight_tensor, op=dist.ReduceOp.SUM)
            assert count_tensor.item() == global_batch_size
            assert abs(weight_tensor.item() - 1.0) < 1e-6

            set_parallel_state(reference_state)
            reference_batch = make_reference_batch(data, device)
            reference_tokens = torch.tensor(
                [token for sample in reference_batch["tokens"] for token in sample],
                device=device,
            )

            def reference_forward_backward() -> None:
                logits = reference_model(reference_tokens)
                loss, _normalizer, _metrics = loss_function(
                    args,
                    reference_batch,
                    num_microbatches=1,
                    logits=logits,
                    apply_megatron_loss_scaling=False,
                    num_rollouts=global_batch_size,
                )
                loss.backward()

            _, reference_ms = timed(reference_forward_backward)
            gradient_differences = [
                (parameter.grad - reference_parameter.grad).abs().max().item()
                for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters())
            ]
            max_gradient_difference = max(gradient_differences)
            assert max_gradient_difference < 2e-5, (batch_index, max_gradient_difference)

            def optimizer_step() -> None:
                optimizer.step()

            _, optimizer_ms = timed(optimizer_step)
            set_parallel_state(local_state)

            records.append(
                {
                    "batch": batch_index,
                    "raw_samples": 12,
                    "valid_samples": global_batch_size,
                    "divisible_by_dp": global_batch_size % DP_SIZE == 0,
                    "raw_token_lengths": raw_lengths if rank == 0 else None,
                    "valid_token_lengths": [len(tokens) for tokens in data["tokens"]] if rank == 0 else None,
                    "local_partition": local_partition,
                    "local_effective_weight": local_effective_weight,
                    "global_effective_weight": weight_tensor.item(),
                    "max_gradient_difference": max_gradient_difference,
                    "forward_ms": forward_ms,
                    "backward_collective_ms": backward_ms,
                    "reference_ms": reference_ms,
                    "optimizer_ms": optimizer_ms,
                }
            )

    if rank == 0:
        timing_keys = ("forward_ms", "backward_collective_ms", "reference_ms", "optimizer_ms")
        timing_summary = {
            key: {
                "mean": fmean(record[key] for record in records),
                "max": max(record[key] for record in records),
                "min": min(record[key] for record in records),
            }
            for key in timing_keys
        }
        result = {
            "passed": True,
            "batches": BATCHES,
            "dp_size": DP_SIZE,
            "divisible_batches": sum(record["divisible_by_dp"] for record in records),
            "non_divisible_batches": sum(not record["divisible_by_dp"] for record in records),
            "max_gradient_difference": max(record["max_gradient_difference"] for record in records),
            "timing_ms": timing_summary,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "distributed_backend": dist.get_backend(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_paths": [
                "miles/ray/rollout/rollout_data_conversion.py",
                "miles/ray/rollout/train_data_conversion.py",
                "miles/utils/dp_schedule.py",
                "miles/utils/seqlen_balancing.py",
                "miles/backends/training_utils/data.py",
                "miles/backends/training_utils/loss.py",
                "miles/backends/training_utils/cp_utils.py",
            ],
            "native_paths": ["torch", "torch.distributed", "nccl"],
            "records": records,
        }
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in result if key != "records"}, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
