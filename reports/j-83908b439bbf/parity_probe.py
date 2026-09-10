#!/usr/bin/env python3
"""Run a bounded two-GPU FSDP2 versus Megatron-Core DDP parity probe."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from megatron.core.distributed import (
    DistributedDataParallel as MegatronDistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.parallel_state import (
    destroy_model_parallel,
    initialize_model_parallel,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig


SEED = 20250910
INPUT_SIZE = 16
OUTPUT_SIZE = 8
LOCAL_BATCH_SIZE = 8
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * 2
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-8
PROCESS_GROUP_TIMEOUT_SECONDS = 180
COMPUTE_DTYPE = torch.bfloat16
REDUCE_DTYPE = torch.float32
MASTER_DTYPE = torch.float32


@dataclass
class StepRecord:
    step: int
    loss: float
    gradients: dict[str, torch.Tensor]
    parameters: dict[str, torch.Tensor]
    consumed_samples: int


@dataclass
class BackendState:
    model: nn.Module
    optimizer: Any
    consumed_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("results.json"),
        help="Path for the rank-zero JSON result",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/job/.cache/j-83908b439bbf/checkpoints"),
        help="Directory outside the repository for native checkpoint files",
    )
    return parser.parse_args()


def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "full_tensor"):
        return tensor.full_tensor()
    return tensor


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    difference = left_cpu - right_cpu
    absolute = difference.abs()
    denominator = right_cpu.abs().clamp_min(1e-12)
    return {
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "max_rel": float((absolute / denominator).max().item()),
    }


def compare_named_tensors(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], tolerance: float
) -> dict[str, Any]:
    if left.keys() != right.keys():
        return {
            "status": "key_mismatch",
            "left_keys": sorted(left),
            "right_keys": sorted(right),
        }
    metrics = {
        name: tensor_metrics(left[name], right[name]) for name in sorted(left)
    }
    max_abs = max(item["max_abs"] for item in metrics.values())
    return {
        "status": "match" if max_abs <= tolerance else "mismatch",
        "tolerance": tolerance,
        "max_abs": max_abs,
        "metrics": metrics,
    }


def compare_losses(left: list[float], right: list[float], tolerance: float) -> dict[str, Any]:
    if len(left) != len(right):
        return {"status": "length_mismatch", "left": left, "right": right}
    metrics = [tensor_metrics(torch.tensor([a]), torch.tensor([b])) for a, b in zip(left, right)]
    max_abs = max(item["max_abs"] for item in metrics)
    return {
        "status": "match" if max_abs <= tolerance else "mismatch",
        "tolerance": tolerance,
        "max_abs": max_abs,
        "metrics": metrics,
    }


def reference_adam_update(
    parameters: dict[str, torch.Tensor], gradients: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    updated = {}
    for name, parameter in parameters.items():
        gradient = gradients[name]
        gradient = gradient.float() + WEIGHT_DECAY * parameter.float()
        exp_avg = (1.0 - ADAM_BETA_1) * gradient
        exp_avg_sq = (1.0 - ADAM_BETA_2) * gradient.square()
        bias_correction_1 = 1.0 - ADAM_BETA_1
        bias_correction_2 = 1.0 - ADAM_BETA_2
        denominator = exp_avg_sq.sqrt() / (bias_correction_2 ** 0.5) + ADAM_EPSILON
        step_size = LEARNING_RATE / bias_correction_1
        updated[name] = parameter.float() - step_size * exp_avg / denominator
    return updated


def parameter_deltas(
    initial: dict[str, torch.Tensor], updated: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        name: updated[name].float() - initial[name].float()
        for name in sorted(initial)
    }


def optimizer_state_entries(optimizer: Any) -> dict[str, torch.Tensor]:
    child_optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    entries = {}
    for child_index, child_optimizer in enumerate(child_optimizers):
        state = child_optimizer.state_dict()["optimizer"]["state"]
        for group_index, group in enumerate(child_optimizer.param_groups):
            if "step" in group:
                entries[f"child_{child_index}_group_{group_index}_step"] = torch.tensor(
                    [group["step"]]
                )
        for parameter_id, parameter_state in state.items():
            prefix = f"child_{child_index}_param_{parameter_id}"
            for state_name, state_value in parameter_state.items():
                if torch.is_tensor(state_value):
                    entries[f"{prefix}_{state_name}"] = state_value.detach().clone()
                elif isinstance(state_value, (int, float)):
                    entries[f"{prefix}_{state_name}"] = torch.tensor([state_value])
    return entries


def compare_optimizer_states(left: Any, right: Any) -> dict[str, Any]:
    left_entries = optimizer_state_entries(left)
    right_entries = optimizer_state_entries(right)
    if left_entries.keys() != right_entries.keys():
        return {
            "status": "key_mismatch",
            "left_keys": sorted(left_entries),
            "right_keys": sorted(right_entries),
        }
    metrics = {
        name: tensor_metrics(left_entries[name], right_entries[name])
        for name in sorted(left_entries)
    }
    max_abs = max(item["max_abs"] for item in metrics.values())
    return {
        "status": "match" if max_abs == 0.0 else "mismatch",
        "max_abs": max_abs,
        "metrics": metrics,
    }


def optimizer_group_metadata(optimizer: Any) -> list[dict[str, Any]]:
    child_optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    metadata = []
    for child_index, child_optimizer in enumerate(child_optimizers):
        for group_index, group in enumerate(child_optimizer.param_groups):
            step = group.get("step", None)
            if torch.is_tensor(step):
                step = step.item()
            metadata.append(
                {
                    "child_index": child_index,
                    "group_index": group_index,
                    "keys": sorted(group.keys()),
                    "step": step,
                }
            )
    return metadata


def make_batch(step: int, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 100_000 * (step + 1) + rank)
    inputs = torch.randn(
        LOCAL_BATCH_SIZE, INPUT_SIZE, generator=generator, dtype=torch.float32
    )
    targets = torch.randn(
        LOCAL_BATCH_SIZE, OUTPUT_SIZE, generator=generator, dtype=torch.float32
    )
    return inputs.to(device), targets.to(device)


def global_loss(loss: torch.Tensor, world_size: int) -> float:
    value = loss.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float((value / world_size).item())


def make_tiny_model(device: torch.device, rank: int) -> nn.Module:
    model = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True).to(device)
    if rank == 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator)
                    .bfloat16()
                    .float()
                )
    for parameter in model.parameters():
        dist.broadcast(parameter, src=0)
    return model


def create_fsdp_backend(device: torch.device, rank: int) -> BackendState:
    model = make_tiny_model(device, rank)
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=COMPUTE_DTYPE,
        reduce_dtype=REDUCE_DTYPE,
        output_dtype=REDUCE_DTYPE,
        cast_forward_inputs=True,
    )
    fully_shard(model, mp_policy=mixed_precision, reshard_after_forward=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA_1, ADAM_BETA_2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    return BackendState(model=model, optimizer=optimizer, consumed_samples=0)


def fsdp_named_tensors(state: BackendState, use_gradients: bool) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in state.model.named_parameters():
        tensor = parameter.grad if use_gradients else parameter
        if tensor is None:
            raise RuntimeError(f"FSDP tensor is missing for {name}")
        result[name] = full_tensor(tensor).detach().clone()
    return result


def run_fsdp_step(state: BackendState, step: int, rank: int, device: torch.device) -> StepRecord:
    inputs, targets = make_batch(step, rank, device)
    state.optimizer.zero_grad(set_to_none=True)
    outputs = state.model(inputs)
    loss = F.mse_loss(outputs.float(), targets.float())
    loss.backward()
    gradients = fsdp_named_tensors(state, use_gradients=True)
    state.optimizer.step()
    parameters = fsdp_named_tensors(state, use_gradients=False)
    state.consumed_samples += GLOBAL_BATCH_SIZE
    return StepRecord(
        step=step,
        loss=global_loss(loss, dist.get_world_size()),
        gradients=gradients,
        parameters=parameters,
        consumed_samples=state.consumed_samples,
    )


def save_fsdp_checkpoint(state: BackendState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_state = get_model_state_dict(state.model)
    optimizer_state = get_optimizer_state_dict(state.model, state.optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=str(path),
    )


def load_fsdp_checkpoint(state: BackendState, path: Path) -> None:
    model_state = get_model_state_dict(state.model)
    optimizer_state = get_optimizer_state_dict(state.model, state.optimizer)
    checkpoint_state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(checkpoint_state, checkpoint_id=str(path))
    set_model_state_dict(state.model, checkpoint_state["model"])
    set_optimizer_state_dict(state.model, state.optimizer, checkpoint_state["optimizer"])


def megatron_master_parameters(optimizer: Any) -> list[torch.Tensor]:
    optimizers = getattr(optimizer, "chained_optimizers", None)
    if optimizers is None:
        optimizers = [optimizer]
    parameters = []
    for wrapped_optimizer in optimizers:
        for group in getattr(wrapped_optimizer, "fp32_from_float16_groups", []):
            parameters.extend(group)
    return parameters


def create_megatron_backend(device: torch.device, rank: int) -> BackendState:
    model = make_tiny_model(device, rank)
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=INPUT_SIZE,
        num_attention_heads=1,
        bf16=True,
        pipeline_dtype=COMPUTE_DTYPE,
    )
    float16_model = Float16Module(transformer_config, model)
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=False,
    )
    model = MegatronDistributedDataParallel(
        transformer_config,
        ddp_config,
        float16_model,
        disable_bucketing=True,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETA_1,
        adam_beta2=ADAM_BETA_2,
        adam_eps=ADAM_EPSILON,
        bf16=True,
        params_dtype=COMPUTE_DTYPE,
        decoupled_weight_decay=False,
        use_distributed_optimizer=False,
        clip_grad=0.0,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    return BackendState(model=model, optimizer=optimizer, consumed_samples=0)


def megatron_base_model(state: BackendState) -> nn.Module:
    return state.model.module.module


def megatron_named_tensors(
    state: BackendState, use_gradients: bool, use_master_parameters: bool = False
) -> dict[str, torch.Tensor]:
    if use_master_parameters:
        named_parameters = list(megatron_base_model(state).named_parameters())
        master_parameters = megatron_master_parameters(state.optimizer)
        if len(named_parameters) != len(master_parameters):
            raise RuntimeError(
                "Megatron model and fp32 master parameter counts differ: "
                f"{len(named_parameters)} != {len(master_parameters)}"
            )
        return {
            name: parameter.detach().clone()
            for (name, _), parameter in zip(named_parameters, master_parameters)
        }
    result = {}
    for name, parameter in megatron_base_model(state).named_parameters():
        tensor = parameter.main_grad if use_gradients else parameter
        if tensor is None:
            raise RuntimeError(f"Megatron tensor is missing for {name}")
        result[name] = tensor.detach().clone()
    return result


def run_megatron_step(
    state: BackendState, step: int, rank: int, device: torch.device
) -> StepRecord:
    inputs, targets = make_batch(step, rank, device)
    state.model.zero_grad_buffer()
    outputs = state.model(inputs)
    loss = F.mse_loss(outputs.float(), targets.float())
    loss.backward()
    state.model.finish_grad_sync()
    gradients = megatron_named_tensors(state, use_gradients=True)
    state.optimizer.step()
    parameters = megatron_named_tensors(
        state, use_gradients=False, use_master_parameters=True
    )
    state.optimizer.zero_grad(set_to_none=True)
    state.consumed_samples += GLOBAL_BATCH_SIZE
    return StepRecord(
        step=step,
        loss=global_loss(loss, dist.get_world_size()),
        gradients=gradients,
        parameters=parameters,
        consumed_samples=state.consumed_samples,
    )


def save_megatron_checkpoint(state: BackendState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_state = state.model.state_dict()
    optimizer_state = state.optimizer.state_dict()
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=str(path),
    )


def load_megatron_checkpoint(state: BackendState, path: Path) -> None:
    model_state = state.model.state_dict()
    child_optimizers = getattr(state.optimizer, "chained_optimizers", [state.optimizer])
    if len(child_optimizers) == 1:
        optimizer_state = child_optimizers[0].state_dict(is_loading=True)
    else:
        optimizer_state = [
            child_optimizer.state_dict(is_loading=True) for child_optimizer in child_optimizers
        ]
    optimizer_states = (
        optimizer_state if isinstance(optimizer_state, list) else [optimizer_state]
    )
    for child_optimizer_state in optimizer_states:
        for group in child_optimizer_state["optimizer"]["param_groups"]:
            group.setdefault("step", 0)
    checkpoint_state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(checkpoint_state, checkpoint_id=str(path))
    state.model.load_state_dict(checkpoint_state["model"])
    state.optimizer.load_state_dict(checkpoint_state["optimizer"])


def git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict[str, Any]:
    modules = {}
    for name in ("miles", "sglang", "megatron.core"):
        module = importlib.import_module(name)
        modules[name] = {
            "file": getattr(module, "__file__", None),
            "path": list(getattr(module, "__path__", [])),
            "version": getattr(module, "__version__", None),
        }
    native_modules = {}
    for name in ("torch", "torch._C", "torch.cuda"):
        module = importlib.import_module(name)
        native_modules[name] = getattr(module, "__file__", None)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
            }
            for index in range(torch.cuda.device_count())
        ],
        "modules": modules,
        "native_modules": native_modules,
        "miles_commit": git_commit(Path(__file__).resolve().parents[2]),
        "megatron_source_commit": git_commit(Path("/root/Megatron-LM")),
        "sglang_source_commit": git_commit(Path("/sgl-workspace/sglang")),
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"This fixture requires exactly 2 ranks, got {world_size}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This fixture requires two visible CUDA/ROCm devices")

    torch.manual_seed(SEED + rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    fsdp = create_fsdp_backend(device, rank)
    megatron = create_megatron_backend(device, rank)
    initial_parameters = {
        "fsdp": fsdp_named_tensors(fsdp, use_gradients=False),
        "megatron_master": megatron_named_tensors(
            megatron, use_gradients=False, use_master_parameters=True
        ),
    }
    initial_comparison = compare_named_tensors(
        initial_parameters["fsdp"], initial_parameters["megatron_master"], tolerance=0.0
    )
    initial_visible_comparison = compare_named_tensors(
        {
            name: parameter.bfloat16()
            for name, parameter in initial_parameters["fsdp"].items()
        },
        megatron_named_tensors(
            megatron, use_gradients=False, use_master_parameters=False
        ),
        tolerance=0.0,
    )

    fsdp_steps = [run_fsdp_step(fsdp, step, rank, device) for step in range(2)]
    megatron_steps = [run_megatron_step(megatron, step, rank, device) for step in range(2)]

    step_comparisons = {}
    for index, (fsdp_step, megatron_step) in enumerate(zip(fsdp_steps, megatron_steps)):
        fsdp_parameter_delta = parameter_deltas(
            initial_parameters["fsdp"], fsdp_step.parameters
        )
        megatron_parameter_delta = parameter_deltas(
            initial_parameters["megatron_master"], megatron_step.parameters
        )
        step_comparisons[f"step_{index + 1}"] = {
            "loss": compare_losses([fsdp_step.loss], [megatron_step.loss], 1e-5),
            "gradients": compare_named_tensors(
                fsdp_step.gradients, megatron_step.gradients, 2e-5
            ),
            "parameters": compare_named_tensors(
                fsdp_step.parameters, megatron_step.parameters, 2e-5
            ),
            "parameter_deltas": compare_named_tensors(
                fsdp_parameter_delta, megatron_parameter_delta, 2e-5
            ),
            "megatron_model_parameters": compare_named_tensors(
                fsdp_step.parameters,
                megatron_named_tensors(
                    megatron, use_gradients=False, use_master_parameters=False
                ),
                2e-5,
            ),
            "visible_parameters": compare_named_tensors(
                {
                    name: parameter.bfloat16()
                    for name, parameter in fsdp_step.parameters.items()
                },
                megatron_named_tensors(
                    megatron, use_gradients=False, use_master_parameters=False
                ),
                0.02,
            ),
            "consumed_samples": {
                "fsdp": fsdp_step.consumed_samples,
                "megatron": megatron_step.consumed_samples,
                "status": (
                    "match"
                    if fsdp_step.consumed_samples == megatron_step.consumed_samples
                    else "mismatch"
                ),
            },
        }

    fsdp_reference_parameters = reference_adam_update(
        initial_parameters["fsdp"], fsdp_steps[0].gradients
    )
    megatron_reference_parameters = reference_adam_update(
        initial_parameters["megatron_master"], megatron_steps[0].gradients
    )
    optimizer_update_isolation = {
        "fsdp_step_1": compare_named_tensors(
            fsdp_steps[0].parameters, fsdp_reference_parameters, 1e-6
        ),
        "megatron_step_1": compare_named_tensors(
            megatron_steps[0].parameters, megatron_reference_parameters, 1e-6
        ),
    }

    fsdp_checkpoint = args.checkpoint_root / "fsdp"
    megatron_checkpoint = args.checkpoint_root / "megatron"
    save_fsdp_checkpoint(fsdp, fsdp_checkpoint)
    save_megatron_checkpoint(megatron, megatron_checkpoint)

    fsdp_reloaded = create_fsdp_backend(device, rank)
    megatron_reloaded = create_megatron_backend(device, rank)
    load_fsdp_checkpoint(fsdp_reloaded, fsdp_checkpoint)
    load_megatron_checkpoint(megatron_reloaded, megatron_checkpoint)
    fsdp_reloaded.consumed_samples = fsdp.consumed_samples
    megatron_reloaded.consumed_samples = megatron.consumed_samples

    checkpoint_load_comparisons = {
        "fsdp_parameters": compare_named_tensors(
            fsdp_named_tensors(fsdp_reloaded, use_gradients=False),
            fsdp_named_tensors(fsdp, use_gradients=False),
            1e-6,
        ),
        "megatron_model_parameters": compare_named_tensors(
            megatron_named_tensors(
                megatron_reloaded, use_gradients=False, use_master_parameters=False
            ),
            megatron_named_tensors(
                megatron, use_gradients=False, use_master_parameters=False
            ),
            1e-6,
        ),
        "megatron_master_parameters": compare_named_tensors(
            megatron_named_tensors(
                megatron_reloaded, use_gradients=False, use_master_parameters=True
            ),
            megatron_named_tensors(
                megatron, use_gradients=False, use_master_parameters=True
            ),
            1e-6,
        ),
        "megatron_optimizer_state": compare_optimizer_states(
            megatron_reloaded.optimizer, megatron.optimizer
        ),
        "megatron_optimizer_group_metadata": {
            "original": optimizer_group_metadata(megatron.optimizer),
            "reloaded": optimizer_group_metadata(megatron_reloaded.optimizer),
        },
    }

    fsdp_reloaded_step = run_fsdp_step(fsdp_reloaded, 2, rank, device)
    megatron_reloaded_step = run_megatron_step(megatron_reloaded, 2, rank, device)
    fsdp_uninterrupted_step = run_fsdp_step(fsdp, 2, rank, device)
    megatron_uninterrupted_step = run_megatron_step(megatron, 2, rank, device)

    reload_comparisons = {
        "fsdp": {
            "gradients": compare_named_tensors(
                fsdp_reloaded_step.gradients,
                fsdp_uninterrupted_step.gradients,
                1e-6,
            ),
            "parameters": compare_named_tensors(
                fsdp_reloaded_step.parameters,
                fsdp_uninterrupted_step.parameters,
                1e-6,
            ),
            "loss": compare_losses(
                [fsdp_reloaded_step.loss], [fsdp_uninterrupted_step.loss], 1e-6
            ),
            "consumed_samples": {
                "reloaded": fsdp_reloaded_step.consumed_samples,
                "uninterrupted": fsdp_uninterrupted_step.consumed_samples,
            },
        },
        "megatron": {
            "gradients": compare_named_tensors(
                megatron_reloaded_step.gradients,
                megatron_uninterrupted_step.gradients,
                1e-6,
            ),
            "parameters": compare_named_tensors(
                megatron_reloaded_step.parameters,
                megatron_uninterrupted_step.parameters,
                1e-6,
            ),
            "loss": compare_losses(
                [megatron_reloaded_step.loss], [megatron_uninterrupted_step.loss], 1e-6
            ),
            "consumed_samples": {
                "reloaded": megatron_reloaded_step.consumed_samples,
                "uninterrupted": megatron_uninterrupted_step.consumed_samples,
            },
        },
    }

    result = {
        "task": "bounded FSDP2 vs Megatron-Core DDP parity probe",
        "configuration": {
            "model": f"nn.Linear({INPUT_SIZE}, {OUTPUT_SIZE}, bias=True)",
            "parameter_count": sum(
                parameter.numel() for parameter in nn.Linear(INPUT_SIZE, OUTPUT_SIZE).parameters()
            ),
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "world_size": world_size,
            "steps_compared": 2,
            "reload_step": 3,
            "loss": "mean squared error over the full local batch",
            "compute_dtype": str(COMPUTE_DTYPE),
            "reduce_dtype": str(REDUCE_DTYPE),
            "master_dtype": str(MASTER_DTYPE),
            "optimizer": {
                "fsdp": "torch.optim.Adam",
                "megatron": "Megatron Float16Optimizer with Transformer Engine FusedAdam",
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "betas": [ADAM_BETA_1, ADAM_BETA_2],
                "eps": ADAM_EPSILON,
                "decoupled_weight_decay": False,
            },
            "process_group_timeout_seconds": PROCESS_GROUP_TIMEOUT_SECONDS,
            "checkpoint_root": str(args.checkpoint_root),
        },
        "environment": environment_record(),
        "initial_parameter_comparison": initial_comparison,
        "initial_visible_parameter_comparison": initial_visible_comparison,
        "optimizer_update_isolation": optimizer_update_isolation,
        "step_comparisons": step_comparisons,
        "checkpoint_reload_comparisons": reload_comparisons,
        "checkpoint_load_comparisons": checkpoint_load_comparisons,
        "scope": {
            "proves": [
                "Two-rank parameter, gradient, loss, consumed-sample, and reload behavior for the synthetic linear fixture.",
                "bf16 compute with fp32 reduction/master parameters under the installed Torch and Megatron stack.",
                "Native distributed checkpoint save/load round trips for both reduced backends.",
            ],
            "does_not_prove": [
                "Tensor, pipeline, context, expert, or sequence parallelism.",
                "Megatron distributed optimizer or Megatron FSDP.",
                "Transformer Engine fused kernels beyond the installed FusedAdam path.",
                "Real language-model numerics, tokenizer behavior, rollout integration, or multi-LoRA behavior.",
                "Performance, scaling, or production checkpoint compatibility.",
            ],
            "unsupported_combinations": [
                "FSDP2 fixture with tensor/pipeline/context/expert parallelism.",
                "Megatron DDP fixture with distributed optimizer or nontrivial parallelism.",
                "Cross-backend checkpoint interchange; each backend is reloaded into its own backend.",
                "CPU or Gloo execution; this probe requires NCCL on two gfx950 devices.",
            ],
        },
    }

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["step_comparisons"], indent=2, sort_keys=True))
        print(json.dumps(result["checkpoint_reload_comparisons"], indent=2, sort_keys=True))

    destroy_model_parallel()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
