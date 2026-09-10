#!/usr/bin/env python3
"""Two-GPU row-parallel reduction diagnostic for Miles issue 1485.

Run with:
  PYTHONPATH=/job/miles TRITON_CACHE_DIR=/job/cache/triton SGLANG_USE_AITER=0 \
    /opt/venv/bin/torchrun --standalone --nproc-per-node=2 \
    reports/j-d945708a0e48/row_parallel_fixture.py --output results.json

The fixture is diagnostic, not a pass/fail parity assertion: it records the
current numerical differences without claiming that either order is correct.
"""

from __future__ import annotations

import argparse
import ctypes.util
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TRITON_CACHE_DIR", "/job/cache/triton")
os.environ["SGLANG_USE_AITER"] = "0"

import torch
import torch.distributed as dist
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.tensor_parallel.layers import RowParallelLinear as MegatronRowParallelLinear
from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.utils.test_utils.det_process_group import _fold_gathered_sum, register_det_nccl_backend
from sglang.srt.layers.linear import RowParallelLinear as SGLangRowParallelLinear
from sglang.srt.tp_invariant_ops import matmul_tp_inv


WORLD_SIZE = 2
TOKENS = 2
OUTPUT_SIZE = 128
FULL_K = 9728
LOCAL_K = 4864
BLOCK_K = 128
BLOCKS_PER_RANK = 38
MEGATRON_NAME = "module.module.decoder.layers.0.mlp.linear_fc2.weight"
HF_NAME = "model.layers.0.mlp.down_proj.weight"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def _module_path(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return spec.origin


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_head(path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _native_libraries() -> dict[str, str | None]:
    names = [
        "torch_hip",
        "rccl",
        "rocblas",
        "hipblaslt",
        "amdhip64",
        "hiprtc",
        "rocm_smi64",
    ]
    cache = subprocess.run(
        ["ldconfig", "-p"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    resolved = {}
    for name in names:
        soname = ctypes.util.find_library(name)
        path = None
        if soname is not None:
            for line in cache.splitlines():
                if f"{soname} " in line and "=>" in line:
                    path = line.rsplit("=>", 1)[1].strip()
                    break
        resolved[name] = path
    return resolved


def _metadata() -> dict[str, object]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "gcn_arch_name": properties.gcnArchName,
                "major": properties.major,
                "minor": properties.minor,
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    return {
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "module_paths": {
            name: _module_path(name)
            for name in ["miles", "sglang", "megatron", "megatron.core", "triton", "transformer_engine"]
        },
        "package_versions": {
            name: _package_version(name)
            for name in ["miles", "sglang", "megatron-core", "transformer-engine", "triton", "torch"]
        },
        "source_revisions": {
            "miles": _git_head("/job/miles"),
            "sglang": _git_head("/sgl-workspace/sglang"),
            "megatron": _git_head("/root/Megatron-LM"),
            "aiter": _git_head("/sgl-workspace/aiter"),
            "triton": _git_head("/sgl-workspace/triton-custom"),
        },
        "native_libraries": _native_libraries(),
        "aiter_native_module": _module_path("aiter.jit.module_aiter_core"),
        "sglang_use_aiter_override": os.environ["SGLANG_USE_AITER"],
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
    }


def _make_partials(rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(rank)
    values = (torch.rand(BLOCKS_PER_RANK, generator=generator) * 2 - 1).to(torch.bfloat16)
    return values.unsqueeze(1).expand(BLOCKS_PER_RANK, OUTPUT_SIZE).contiguous().to(device)


def _make_operands(partials: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.zeros(TOKENS, LOCAL_K, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.zeros(OUTPUT_SIZE, LOCAL_K, device=device, dtype=torch.bfloat16)
    for block in range(BLOCKS_PER_RANK):
        column = block * BLOCK_K
        with torch.no_grad():
            inputs[:, column] = 1
            weight[:, column] = partials[block]
    return inputs, weight


def _sglang_two_level_fold(partials: torch.Tensor) -> torch.Tensor:
    first = partials[0].clone()
    for block in range(1, BLOCKS_PER_RANK // 2):
        first += partials[block]
    second = partials[BLOCKS_PER_RANK // 2].clone()
    for block in range(BLOCKS_PER_RANK // 2 + 1, BLOCKS_PER_RANK):
        second += partials[block]
    return first + second


def _stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    difference = left.detach().float() - right.detach().float()
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "mismatched_elements": int((left.view(torch.int16) != right.view(torch.int16)).sum()),
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
    }


def _all_partials(partials: torch.Tensor) -> list[torch.Tensor]:
    flat = partials.reshape(-1).contiguous()
    gathered = [torch.empty_like(flat) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, flat)
    blocks = [tensor.view(BLOCKS_PER_RANK, OUTPUT_SIZE) for tensor in gathered]
    return [block for rank_blocks in blocks for block in rank_blocks]


def _write_results(path: Path, payload: dict[str, object]) -> None:
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _worker() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    register_det_nccl_backend()
    dist.init_process_group(
        backend="det_nccl",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=WORLD_SIZE,
        pipeline_model_parallel_size=1,
        distributed_timeout_minutes=max(1, args.timeout_seconds // 60),
    )

    partials = _make_partials(rank, device)
    inputs, local_weight = _make_operands(partials, device)

    gathered_weights = [torch.empty_like(local_weight) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_weights, local_weight)
    full_weight = torch.cat(gathered_weights, dim=1).contiguous()
    converted = convert_to_hf(
        SimpleNamespace(
            vocab_size=0,
            hidden_size=OUTPUT_SIZE,
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=OUTPUT_SIZE,
        ),
        "qwen3",
        MEGATRON_NAME,
        full_weight,
    )
    assert len(converted) == 1, converted
    assert converted[0][0] == HF_NAME, converted[0][0]
    assert torch.equal(converted[0][1], full_weight)

    config = ModelParallelConfig(
        tensor_model_parallel_size=WORLD_SIZE,
        pipeline_model_parallel_size=1,
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
    )
    megatron_layer = MegatronRowParallelLinear(
        FULL_K,
        OUTPUT_SIZE,
        config=config,
        init_method=lambda tensor: tensor,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
    )
    with torch.no_grad():
        megatron_layer.weight.copy_(local_weight)

    sglang_layer = SGLangRowParallelLinear(
        FULL_K,
        OUTPUT_SIZE,
        bias=False,
        input_is_parallel=True,
        params_dtype=torch.bfloat16,
        reduce_results=False,
        tp_rank=rank,
        tp_size=WORLD_SIZE,
    ).to(device)
    sglang_layer.weight_loader(sglang_layer.weight, converted[0][1])
    assert torch.equal(sglang_layer.weight, local_weight)

    expected_local = _sglang_two_level_fold(partials).unsqueeze(0).expand(TOKENS, OUTPUT_SIZE).contiguous()
    with torch.no_grad():
        sglang_local = matmul_tp_inv(inputs.detach(), sglang_layer.weight.t())
    assert torch.equal(sglang_local, expected_local)

    miles_local_fold = _fold_gathered_sum([tensor.clone() for tensor in partials])
    miles_local = miles_local_fold.unsqueeze(0).expand(TOKENS, OUTPUT_SIZE)

    sglang_global = sglang_local.clone()
    dist.all_reduce(sglang_global, op=dist.ReduceOp.SUM)
    expected_global = expected_local.clone()
    dist.all_reduce(expected_global, op=dist.ReduceOp.SUM)
    miles_global_fold = _fold_gathered_sum(_all_partials(partials))
    miles_global = miles_global_fold.unsqueeze(0).expand(TOKENS, OUTPUT_SIZE)

    megatron_output, _ = megatron_layer(inputs)
    alternating = torch.where(torch.arange(OUTPUT_SIZE, device=device) % 2 == 0, 1, -1).to(torch.bfloat16)
    upstream = torch.stack([alternating, 0.5 * alternating]).contiguous()
    megatron_grad_input, megatron_grad_weight = torch.autograd.grad(
        megatron_output,
        (inputs, megatron_layer.weight),
        upstream,
        retain_graph=True,
    )
    reference_grad_input = upstream.matmul(megatron_layer.weight)
    reference_grad_weight = upstream.t().matmul(inputs.detach())

    payload = {
        "metadata": _metadata(),
        "configuration": {
            "world_size": WORLD_SIZE,
            "tokens": TOKENS,
            "output_size": OUTPUT_SIZE,
            "full_k": FULL_K,
            "local_k": LOCAL_K,
            "block_k": BLOCK_K,
            "blocks_per_rank": BLOCKS_PER_RANK,
            "dtype": str(torch.bfloat16),
            "process_group_timeout_seconds": args.timeout_seconds,
            "megatron_parallel_timeout_minutes": max(1, args.timeout_seconds // 60),
        },
        "weights": {
            "miles_converter_name": HF_NAME,
            "miles_converter_bitwise": True,
            "sglang_loader_bitwise": True,
        },
        "reduction_order": {
            "sglang_kernel_vs_two_level_fold_bitwise": True,
            "local_miles_fold_vs_sglang": _stats(miles_local, sglang_local),
            "global_miles_fold_vs_sglang": _stats(miles_global, sglang_global),
            "sglang_global_vs_two_level_reference": _stats(sglang_global, expected_global),
        },
        "outputs": {
            "megatron_vs_sglang": _stats(megatron_output, sglang_global),
            "sglang_kernel_requires_grad": bool(sglang_local.requires_grad),
        },
        "gradients": {
            "upstream_pattern": "row 0 alternates +1/-1; row 1 is 0.5 times row 0",
            "input_megatron_vs_reference": _stats(megatron_grad_input, reference_grad_input),
            "weight_megatron_vs_reference": _stats(megatron_grad_weight, reference_grad_weight),
            "sglang_backward_available": False,
        },
    }
    _write_results(args.output, payload)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _worker()
