#!/usr/bin/env python3
"""Bounded two-GPU FSDP/Megatron Core DDP update comparison."""

from __future__ import annotations

import argparse
import json
import os
import importlib.metadata
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("comparison.json"))
    return parser.parse_args()


def global_batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1234 + step)
    inputs = torch.randn((4, 8), generator=generator, dtype=torch.float32)
    targets = torch.randn((4, 4), generator=generator, dtype=torch.float32)
    return inputs, targets


def local_batch(step: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = global_batch(step)
    return inputs[rank * 2 : (rank + 1) * 2].cuda(), targets[rank * 2 : (rank + 1) * 2].cuda()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "min": detached.min().item(),
        "max": detached.max().item(),
        "mean": detached.mean().item(),
        "norm": detached.square().sum().sqrt().item(),
    }


def global_loss(loss: torch.Tensor) -> float:
    value = loss.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()


def named_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def named_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    return gradients


def save_checkpoint(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, path)
    dist.barrier()


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    dist.barrier()
    return state_dict


def fsdp_full_grads(fsdp_model: FullyShardedDataParallel) -> dict[str, torch.Tensor]:
    handle = fsdp_model._handle
    sharded_grad = handle.sharded_grad
    if sharded_grad is None:
        raise RuntimeError("FSDP handle has no sharded gradient")
    pieces = [torch.empty_like(sharded_grad) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, sharded_grad.contiguous())
    full_grad = torch.cat(pieces, dim=0)
    views = handle._get_unflat_views_aligned(full_grad)
    names = handle.flat_param._fqns
    if len(views) != len(names):
        raise RuntimeError(
            f"FSDP gradient view/name mismatch: {len(views)} views, {len(names)} names"
        )
    return {
        name: view.detach().cpu().clone() for name, view in zip(names, views)
    }


def tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left.float() - right.float()).abs()
    return {
        "max_abs": difference.max().item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def public_backend_result(result: dict[str, Any]) -> dict[str, Any]:
    public = dict(result)
    public["initial_parameters"] = {
        name: tensor_stats(value) for name, value in result["initial_parameters"].items()
    }
    public["steps"] = []
    for record in result["steps"]:
        public_record = dict(record)
        public_record["gradients"] = {
            name: tensor_stats(value) for name, value in record["gradients"].items()
        }
        public_record["parameters"] = {
            name: tensor_stats(value) for name, value in record["parameters"].items()
        }
        public["steps"].append(public_record)
    return public


def run_fsdp(steps: int, checkpoint_path: Path) -> dict[str, Any]:
    torch.manual_seed(1234)
    model = TinyModel().cuda()
    fsdp_model = FullyShardedDataParallel(
        model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
    )
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    initial = named_state(fsdp_model.state_dict())
    records = []
    for step in range(steps):
        inputs, targets = local_batch(step, dist.get_rank())
        optimizer.zero_grad(set_to_none=True)
        outputs = fsdp_model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        gradients = fsdp_full_grads(fsdp_model)
        optimizer.step()
        parameters = named_state(fsdp_model.state_dict())
        records.append(
            {
                "step": step,
                "loss": global_loss(loss),
                "gradients": gradients,
                "parameters": parameters,
            }
        )
    checkpoint = named_state(fsdp_model.state_dict())
    save_checkpoint(checkpoint, checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    perturbed_state = {
        name: value + 0.25 for name, value in checkpoint.items()
    }
    fsdp_model.load_state_dict(perturbed_state)
    perturbed = named_state(fsdp_model.state_dict())
    reloaded_state = load_checkpoint(checkpoint_path)
    fsdp_model.load_state_dict(reloaded_state)
    reloaded = named_state(fsdp_model.state_dict())
    reload_max_abs = max(
        (checkpoint[name] - reloaded[name]).abs().max().item() for name in checkpoint
    )
    perturb_max_abs = max(
        (checkpoint[name] - perturbed[name]).abs().max().item() for name in checkpoint
    )
    return {
        "backend": "torch.fsdp.v1.full_shard",
        "initial_parameters": initial,
        "steps": records,
        "checkpoint_reload_max_abs": reload_max_abs,
        "checkpoint_perturb_max_abs": perturb_max_abs,
    }


def run_megatron(steps: int, checkpoint_path: Path) -> dict[str, Any]:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer import TransformerConfig

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        distributed_timeout_minutes=5,
    )
    torch.manual_seed(1234)
    model = TinyModel().cuda()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        params_dtype=torch.float32,
    )
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=False,
    )
    ddp_model = DistributedDataParallel(config, ddp_config, model)
    ddp_model.broadcast_params()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial = named_state(model.state_dict())
    records = []
    for step in range(steps):
        inputs, targets = local_batch(step, dist.get_rank())
        optimizer.zero_grad(set_to_none=True)
        ddp_model.zero_grad_buffer()
        outputs = ddp_model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        ddp_model.finish_grad_sync()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.grad = parameter.main_grad.clone()
        gradients = named_grads(model)
        optimizer.step()
        parameters = named_state(model.state_dict())
        records.append(
            {
                "step": step,
                "loss": global_loss(loss),
                "gradients": gradients,
                "parameters": parameters,
            }
        )
    checkpoint = named_state(model.state_dict())
    save_checkpoint(checkpoint, checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    perturbed_state = {
        name: value + 0.25 for name, value in checkpoint.items()
    }
    reloaded_state = load_checkpoint(checkpoint_path)
    model.load_state_dict(reloaded_state)
    model.load_state_dict(perturbed_state)
    perturbed = named_state(model.state_dict())
    model.load_state_dict(checkpoint)
    reloaded = named_state(model.state_dict())
    reload_max_abs = max(
        (checkpoint[name] - reloaded[name]).abs().max().item() for name in checkpoint
    )
    perturb_max_abs = max(
        (checkpoint[name] - perturbed[name]).abs().max().item() for name in checkpoint
    )
    return {
        "backend": "megatron.core.distributed.DistributedDataParallel",
        "initial_parameters": initial,
        "steps": records,
        "checkpoint_reload_max_abs": reload_max_abs,
        "checkpoint_perturb_max_abs": perturb_max_abs,
    }


def compare(fsdp_result: dict[str, Any], megatron_result: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    initial = {
        name: tensor_comparison(value, megatron_result["initial_parameters"][name])
        for name, value in fsdp_result["initial_parameters"].items()
    }
    for fsdp_step, megatron_step in zip(fsdp_result["steps"], megatron_result["steps"]):
        loss_delta = abs(fsdp_step["loss"] - megatron_step["loss"])
        gradients = {
            name: tensor_comparison(value, megatron_step["gradients"][name])
            for name, value in fsdp_step["gradients"].items()
        }
        parameters = {
            name: tensor_comparison(value, megatron_step["parameters"][name])
            for name, value in fsdp_step["parameters"].items()
        }
        comparisons.append(
            {
                "step": fsdp_step["step"],
                "loss_abs_delta": loss_delta,
                "gradients": gradients,
                "parameters": parameters,
            }
        )
    return {
        "initial_parameters": initial,
        "per_step": comparisons,
        "loss_abs_delta_max": max(item["loss_abs_delta"] for item in comparisons),
        "gradient_max_abs_delta_max": max(
            metric["max_abs"]
            for item in comparisons
            for metric in item["gradients"].values()
        ),
        "parameter_max_abs_delta_max": max(
            metric["max_abs"]
            for item in comparisons
            for metric in item["parameters"].values()
        ),
    }


def module_paths() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    modules = {}
    for name in ("torch", "megatron.core", "transformer_engine", "sglang", "miles"):
        module = importlib.import_module(name)
        modules[name] = str(Path(module.__file__).resolve())
    distributions = {}
    for name in ("torch", "megatron-core", "transformer-engine", "sglang", "miles"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    torch_root = Path(torch.__file__).resolve().parent
    def source_revision(path: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "modules": modules,
        "distributions": distributions,
        "source_revisions": {
            "miles_clone": source_revision(str(repo_root)),
            "megatron_preinstalled_source": source_revision("/root/Megatron-LM"),
            "sglang_preinstalled_source": source_revision("/sgl-workspace/sglang"),
        },
        "native": {
            "torch_python": str(Path(torch._C.__file__).resolve()),
            "torch_hip": str((torch_root / "lib/libtorch_hip.so").resolve()),
            "rccl": str(Path("/opt/rocm/lib/librccl.so.1").resolve()),
        },
    }


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 assigned GPUs, got {torch.cuda.device_count()}")
    if os.environ.get("WORLD_SIZE") != "2":
        raise RuntimeError(f"expected WORLD_SIZE=2, got {os.environ.get('WORLD_SIZE')}")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=300),
    )
    torch.cuda.set_device(dist.get_rank())
    try:
        fsdp_checkpoint_path = args.output.with_name(f"{args.output.stem}-fsdp.pt")
        megatron_checkpoint_path = args.output.with_name(f"{args.output.stem}-megatron.pt")
        fsdp_result = run_fsdp(steps=2, checkpoint_path=fsdp_checkpoint_path)
        megatron_result = run_megatron(
            steps=2, checkpoint_path=megatron_checkpoint_path
        )
        comparison = compare(fsdp_result, megatron_result)
        result = {
            "environment": {
                "world_size": dist.get_world_size(),
                "rank": dist.get_rank(),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(),
                "device_capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
            },
            "settings": {
                "global_batch_size": 4,
                "local_batch_size": 2,
                "steps": 2,
                "parameter_dtype": "float32",
                "gradient_dtype": "float32",
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "loss": "MSE(mean)",
                "process_group_timeout_seconds": 300,
            },
            "module_paths": module_paths(),
            "supported_combinations": [
                "Torch FSDP1 FULL_SHARD with DP=2",
                "Megatron Core DistributedDataParallel with DP=2, TP=1, PP=1",
            ],
            "unsupported_combinations": [
                "Tensor or pipeline parallelism",
                "Mixed precision, FP8, distributed optimizer, or gradient accumulation",
                "Megatron FSDP, pipeline schedules, or production transformer models",
            ],
            "fsdp": public_backend_result(fsdp_result),
            "megatron": public_backend_result(megatron_result),
            "comparison": comparison,
            "consumed_samples": {"after_step_1": 4, "after_step_2": 8},
        }
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
