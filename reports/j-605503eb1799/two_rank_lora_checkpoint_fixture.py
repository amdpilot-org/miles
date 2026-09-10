#!/usr/bin/env python3
"""Run a two-rank Miles LoRA checkpoint experiment on the assigned GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.training.global_vars import set_args
from transformers import Qwen2Config, Qwen2ForCausalLM

from miles.backends.megatron_utils.bridge_lora_helpers import _setup_lora_model_via_bridge
from miles.backends.megatron_utils.lora_utils import load_lora_adapter, save_lora_checkpoint
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state
from miles.backends.training_utils.parallel import set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo


class StaticScheduler:
    def __init__(self) -> None:
        self.step = 7

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "hf-error", "native-error"), default="baseline")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def module_paths():
    import miles
    import megatron.core
    import sglang
    import aiter

    return {
        "python": sys.executable,
        "torch": torch.__file__,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "miles": miles.__file__,
        "megatron_core": megatron.core.__file__,
        "sglang": sglang.__file__,
        "aiter": aiter.__file__,
        "platform": platform.platform(),
    }


def ensure_model(model_dir: Path):
    if model_dir.exists() and (model_dir / "config.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(config).save_pretrained(model_dir, safe_serialization=True)


def adapter_state(model):
    return {
        name: parameter.detach().clone()
        for chunk in model
        for name, parameter in chunk.named_parameters()
        if "lora_" in name
    }


def recursive_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(recursive_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(recursive_equal(a, b) for a, b in zip(left, right))
    return left == right


def all_ranks_true(value):
    marker = torch.tensor([int(value)], device=torch.cuda.current_device())
    dist.all_reduce(marker, op=dist.ReduceOp.MIN)
    return bool(marker.item())


def write_evidence(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    options = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise SystemExit(f"this fixture requires exactly two ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=options.timeout_seconds))
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_distributed_optimizer_instances=1,
        expert_tensor_parallel_size=1,
        distributed_timeout_minutes=max(1, math.ceil(options.timeout_seconds / 60)),
        create_gloo_process_groups=True,
    )

    if rank == 0:
        ensure_model(options.model_dir)
    dist.barrier()

    args = type("Args", (), {
            "rank": rank,
            "seed": 1234,
            "data_parallel_size": world_size,
            "data_parallel_random_init": False,
            "indep_dp": False,
            "hf_checkpoint": str(options.model_dir),
            "tokenizer_model": str(options.model_dir),
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "virtual_pipeline_model_parallel_size": None,
            "pipeline_model_parallel_comm_backend": None,
            "context_parallel_size": 1,
            "hierarchical_context_parallel_sizes": None,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "num_distributed_optimizer_instances": 1,
            "nccl_communicator_config_path": None,
            "distributed_timeout_minutes": max(1, math.ceil(options.timeout_seconds / 60)),
            "use_gloo_process_groups": True,
            "use_tp_pp_dp_mapping": False,
            "cp_comm_type": None,
            "sequence_parallel": False,
            "gradient_accumulation_fusion": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "recompute_modules": None,
            "distribute_saved_activations": False,
            "attention_backend": "flash",
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "dsa_attention_backend": "megatron",
            "optimizer": "muon",
            "accumulate_allreduce_grads_in_fp32": True,
            "offload_train": False,
            "multi_lora": False,
            "target_modules": ["linear_qkv"],
            "lora_type": "lora",
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "lora_A_init_method": "xavier",
            "lora_B_init_method": "zero",
            "exclude_modules": None,
            "experts_shared_outer_loras": False,
        })()
    set_args(args)
    set_parallel_state(
        create_megatron_parallel_state(indep_dp=GroupInfo(rank=0, size=1, group=None))
    )
    tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
    model = _setup_lora_model_via_bridge(args)

    for parameter in adapter_state(model).values():
        dist.broadcast(parameter.data, src=0)

    trainable = [
        parameter
        for chunk in model
        for parameter in chunk.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    scheduler = StaticScheduler()
    saved_adapters = adapter_state(model)
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()

    from megatron.bridge import AutoBridge

    if options.mode in ("hf-error", "native-error"):

        def fail_export(cls, path, **kwargs):
            raise RuntimeError("injected HF export failure")

        AutoBridge.from_hf_pretrained = classmethod(fail_export)

    if options.mode == "native-error":
        real_save = torch.save

        def fail_rank_one_native(obj, path, *save_args, **save_kwargs):
            if rank == 1 and "adapter_megatron_rank1.pt" in str(path):
                raise RuntimeError("injected rank-one native save failure")
            return real_save(obj, path, *save_args, **save_kwargs)

        torch.save = fail_rank_one_native

    options.save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    result = {
        "mode": options.mode,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "gpu": torch.cuda.get_device_name(local_rank),
        "timeout_seconds": options.timeout_seconds,
        "modules": module_paths(),
    }
    failed = False
    try:
        save_lora_checkpoint(
            model,
            args,
            str(options.save_dir),
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            iteration=7,
        )
        if options.mode == "baseline":
            for parameter in adapter_state(model).values():
                parameter.add_(float(rank + 1))
            for state in optimizer.state.values():
                state["step"] = state["step"] + 10
            scheduler.step = 99

            loaded, iteration = load_lora_adapter(
                model,
                str(options.save_dir),
                optimizer=optimizer,
                opt_param_scheduler=scheduler,
            )
            result.update(
                loaded=loaded,
                iteration=iteration,
                adapter_restore_equal=all(torch.equal(value, saved_adapters[name]) for name, value in adapter_state(model).items()),
                optimizer_restore_equal=recursive_equal(optimizer.state_dict(), saved_optimizer),
                scheduler_restore_equal=recursive_equal(scheduler.state_dict(), saved_scheduler),
            )
            result["all_rank_restore_equal"] = all_ranks_true(
                result["adapter_restore_equal"]
                and result["optimizer_restore_equal"]
                and result["scheduler_restore_equal"]
                and loaded
                and iteration == 7
            )
            if rank == 0:
                shard_zero = torch.load(options.save_dir / "adapter_megatron_rank0.pt", weights_only=True)
                shard_one = torch.load(options.save_dir / "adapter_megatron_rank1.pt", weights_only=True)
                result["dp_shards_identical"] = shard_zero.keys() == shard_one.keys() and all(
                    torch.equal(shard_zero[name], shard_one[name]) for name in shard_zero
                )
                result["files"] = sorted(path.name for path in options.save_dir.iterdir() if path.is_file())
    except Exception as error:
        failed = True
        result["failure"] = f"{type(error).__name__}: {error}"
    finally:
        result["failed"] = failed
        result["artifacts"] = sorted(path.name for path in options.save_dir.iterdir() if path.is_file())
        write_evidence(options.evidence_dir / f"{options.mode}_rank{rank}.json", result)

    if not failed and options.mode == "baseline":
        dist.destroy_process_group()
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
