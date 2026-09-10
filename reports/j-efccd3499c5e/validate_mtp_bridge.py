#!/usr/bin/env python3
"""Two-GPU Qwen3.5 target-plus-MTP bridge validation fixture."""

from __future__ import annotations

import argparse
import enum
import importlib.metadata
import inspect
import json
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from mbridge.core import util as mbridge_util
from mbridge.core.parallel_states import ParallelStates
from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig


class CompatibleModelType(enum.Enum):
    """Compatibility enum for the installed mbridge model-type contract."""

    encoder_or_decoder = 1
    encoder_and_decoder = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def git_commit(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def build_hf_config() -> Any:
    config = AutoConfig.for_model("qwen3_5")
    text_config = config.text_config
    text_values = {
        "num_hidden_layers": 1,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "tie_word_embeddings": False,
        "mtp_num_hidden_layers": 1,
    }
    for name, value in text_values.items():
        setattr(text_config, name, value)
    text_config.layer_types = ["full_attention"]

    top_level_values = {
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
    }
    for name, value in top_level_values.items():
        setattr(config, name, value)
    return config


def build_miles_hook(config: Any) -> Qwen3_5Bridge:
    hook = Qwen3_5Bridge.__new__(Qwen3_5Bridge)
    hook.hf_config = config
    hook.dtype = torch.float32
    hook.safetensor_io = None
    hook.make_vocab_size_divisible_by = 1
    hook.vocab_size = 32
    hook.padded_vocab_size = 32
    return hook


def build_model(config: Any) -> tuple[Any, Qwen3_5Bridge]:
    bridge = Qwen3_5Bridge(
        config,
        dtype=torch.float32,
        parallel_states=ParallelStates.get_parallel_state(),
        make_vocab_size_divisible_by=1,
    )
    model, = bridge.get_model(
        model_type=CompatibleModelType.encoder_or_decoder,
        bf16=False,
        fp16=False,
        wrap_with_ddp=False,
    )
    return model, bridge


def mapping_source(mapping: Any, hf_state: dict[str, torch.Tensor]) -> Any:
    if isinstance(mapping.hf_param, dict):
        return {
            role: hf_state[name]
            for role, name in mapping.hf_param.items()
        }
    return hf_state[str(mapping.hf_param)]


def export_hf(
    model: Any,
    registry: Any,
) -> tuple[dict[str, torch.Tensor], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    hf_state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        mapping = registry.megatron_to_hf_lookup(name)
        if mapping is None:
            raise RuntimeError(f"Missing Megatron bridge mapping: {name}")
        module, _ = get_module_and_param_from_name(model, name)
        exported = mapping.megatron_to_hf(parameter.detach(), module)
        for hf_name, tensor in exported.items():
            hf_state[str(hf_name)] = tensor.detach().clone()
    torch.cuda.synchronize()
    return hf_state, (time.perf_counter() - started) * 1000.0


def restore_megatron(
    model: Any,
    registry: Any,
    hf_state: dict[str, torch.Tensor],
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            mapping = registry.megatron_to_hf_lookup(name)
            if mapping is None:
                raise RuntimeError(f"Missing Megatron bridge mapping: {name}")
            module, _ = get_module_and_param_from_name(model, name)
            restored = mapping.hf_to_megatron(
                mapping_source(mapping, hf_state), module
            )
            parameter.copy_(restored)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def perturb_parameters(model: Any, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ).to(parameter.device)
            parameter.add_(noise * 0.01)
    torch.cuda.synchronize()


def gather_max_difference(tensor: torch.Tensor) -> float:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.detach().contiguous())
    return max(
        float((tensor.detach() - other).abs().max()) for other in gathered
    )


def flat_parameters(model: Any, gradients: bool = False) -> torch.Tensor:
    tensors = []
    for parameter in model.parameters():
        source = parameter.grad if gradients else parameter
        if source is None:
            raise RuntimeError("Missing gradient during distributed check")
        tensors.append(source.detach().reshape(-1))
    return torch.cat(tensors)


def make_batch(cycle: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = 8
    if dist.get_rank() == 0:
        generator = torch.Generator(device="cpu").manual_seed(1234 + cycle)
        input_ids = torch.randint(
            0, 32, (2, sequence_length), generator=generator, dtype=torch.int64
        )
        labels = torch.randint(
            0, 32, (2, sequence_length), generator=generator, dtype=torch.int64
        )
    else:
        input_ids = torch.empty((2, sequence_length), dtype=torch.int64)
        labels = torch.empty((2, sequence_length), dtype=torch.int64)
    input_ids = input_ids.cuda()
    labels = labels.cuda()
    dist.broadcast(input_ids, 0)
    dist.broadcast(labels, 0)

    position_ids = torch.arange(sequence_length, device=input_ids.device)
    position_ids = position_ids.expand(2, sequence_length).contiguous()
    attention_mask = torch.ones(
        (1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device
    ).tril()
    return input_ids, position_ids, attention_mask, labels


def run_cycle(
    cycle: int,
    model: Any,
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    registry: Any,
) -> dict[str, Any]:
    input_ids, position_ids, attention_mask, labels = make_batch(cycle)

    torch.cuda.synchronize()
    forward_started = time.perf_counter()
    output = ddp(
        input_ids,
        position_ids,
        attention_mask,
        labels=labels,
    )
    loss = output.mean()
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - forward_started) * 1000.0

    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - backward_started) * 1000.0

    gradient_difference = gather_max_difference(flat_parameters(model, gradients=True))
    loss_value = float(loss.detach())
    loss_tensor = torch.tensor([loss_value], device=loss.device)
    loss_difference = gather_max_difference(loss_tensor)

    before_update = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer_started = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0

    updated = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    parameter_updates = {
        name: float((parameter.detach() - before_update[name]).abs().max())
        for name, parameter in model.named_parameters()
    }
    minimum_update_name = min(parameter_updates, key=parameter_updates.get)
    minimum_update = parameter_updates[minimum_update_name]
    parameter_difference = gather_max_difference(flat_parameters(model))

    hf_state, export_ms = export_hf(model, registry)
    perturb_parameters(model, 10_000 + cycle)
    restore_ms = restore_megatron(model, registry, hf_state)
    restore_difference = max(
        float((parameter.detach() - updated[name]).abs().max())
        for name, parameter in model.named_parameters()
    )

    return {
        "cycle": cycle,
        "loss": loss_value,
        "loss_rank_max_difference": loss_difference,
        "gradient_rank_max_difference": gradient_difference,
        "parameter_rank_max_difference": parameter_difference,
        "minimum_parameter_update": minimum_update,
        "minimum_parameter_update_name": minimum_update_name,
        "zero_update_parameters": [
            name for name, update in parameter_updates.items() if update == 0.0
        ],
        "export_restore_max_difference": restore_difference,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "export_ms": export_ms,
        "restore_ms": restore_ms,
    }


def validate_mappings(
    model: Any,
    registry: Any,
    miles_hook: Qwen3_5Bridge,
) -> dict[str, Any]:
    current_names = [name for name, _ in model.named_parameters()]
    mtp_names = [name for name in current_names if name.startswith("mtp.")]
    target_names = [name for name in current_names if not name.startswith("mtp.")]

    missing_bridge = [
        name for name in current_names
        if registry.megatron_to_hf_lookup(name) is None
    ]
    missing_miles = [
        name for name in current_names
        if not miles_hook._weight_name_mapping_mcore_to_hf(name)
    ]

    legacy_absent = []
    legacy_name_mismatches = []
    for name in mtp_names:
        if "mtp_model_layer" not in name:
            continue
        legacy_name = name.replace("mtp_model_layer", "transformer_layer")
        if registry.megatron_to_hf_lookup(legacy_name) is None:
            legacy_absent.append(name)
        current_hf_names = miles_hook._weight_name_mapping_mcore_to_hf(name)
        legacy_hf_names = miles_hook._convert_mtp_param(legacy_name)
        if current_hf_names != legacy_hf_names:
            legacy_name_mismatches.append(name)

    if missing_bridge or missing_miles or legacy_name_mismatches:
        raise RuntimeError(
            "Mapping validation failed: "
            f"bridge={missing_bridge}, miles={missing_miles}, "
            f"legacy={legacy_name_mismatches}"
        )

    return {
        "target_parameter_count": len(target_names),
        "mtp_parameter_count": len(mtp_names),
        "all_parameters_have_bridge_mappings": True,
        "all_parameters_have_miles_name_mappings": True,
        "legacy_inner_mtp_parameters": len(legacy_absent),
        "legacy_inner_mtp_absent_from_megatron_bridge": legacy_absent,
        "legacy_inner_mtp_mapped_by_miles_hook": len(legacy_absent),
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"Expected exactly 2 GPUs, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=args.timeout_seconds)
    )
    try:
        mbridge_util.ModelType = CompatibleModelType
        mpu.initialize_model_parallel(1, 1)
        tensor_parallel.model_parallel_cuda_manual_seed(1234)

        config = build_hf_config()
        model, miles_model_bridge = build_model(config)
        for parameter in model.parameters():
            dist.broadcast(parameter.data, 0)
        model.train()
        ddp = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            bucket_cap_mb=1,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        registry = Qwen35Bridge.__new__(Qwen35Bridge).mapping_registry()
        miles_hook = build_miles_hook(config)
        mapping_report = validate_mappings(model, registry, miles_hook)

        cycles = [
            run_cycle(cycle, model, ddp, optimizer, registry)
            for cycle in range(args.cycles)
        ]
        dist.barrier()

        if rank == 0:
            report = {
                "status": "passed",
                "cycles_requested": args.cycles,
                "cycles_completed": len(cycles),
                "world_size": world_size,
                "rendezvous_timeout_seconds": args.timeout_seconds,
                "downloaded_bytes": 0,
                "model": "local synthetic Qwen3_5Config (1 target layer, 1 MTP layer)",
                "mapping_validation": mapping_report,
                "cycles": cycles,
                "summary": {
                    "maximum_loss_rank_difference": max(
                        cycle["loss_rank_max_difference"] for cycle in cycles
                    ),
                    "maximum_gradient_rank_difference": max(
                        cycle["gradient_rank_max_difference"] for cycle in cycles
                    ),
                    "maximum_parameter_rank_difference": max(
                        cycle["parameter_rank_max_difference"] for cycle in cycles
                    ),
                    "minimum_parameter_update": min(
                        cycle["minimum_parameter_update"] for cycle in cycles
                    ),
                    "maximum_export_restore_difference": max(
                        cycle["export_restore_max_difference"] for cycle in cycles
                    ),
                    "total_forward_ms": sum(cycle["forward_ms"] for cycle in cycles),
                    "total_backward_ms": sum(cycle["backward_ms"] for cycle in cycles),
                    "total_optimizer_ms": sum(cycle["optimizer_ms"] for cycle in cycles),
                    "total_export_ms": sum(cycle["export_ms"] for cycle in cycles),
                    "total_restore_ms": sum(cycle["restore_ms"] for cycle in cycles),
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "torch_hip_version": torch.version.hip,
                    "gpu_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                    "megatron_core_version": package_version("megatron-core"),
                    "megatron_bridge_version": package_version("megatron-bridge"),
                    "transformer_engine_version": package_version("transformer-engine"),
                    "miles_commit": git_commit("/job/miles"),
                    "megatron_lm_commit": git_commit("/root/Megatron-LM"),
                    "paths": {
                        "torch": torch.__file__,
                        "megatron_core": inspect.getfile(mpu),
                        "megatron_bridge": inspect.getfile(Qwen35Bridge),
                        "mbridge": mbridge_util.__file__,
                        "miles_qwen_hook": inspect.getfile(Qwen3_5Bridge),
                    },
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["summary"], indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
