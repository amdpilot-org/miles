from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault(
    "AITER_CONFIG_GEMM_BF16",
    "/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv",
)
os.environ.setdefault("USE_ROCM_AITER_ROPE_BACKEND", "0")

import torch
import torch.distributed as dist
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import RowParallelLinear as MegatronRowParallelLinear
from sglang.srt.layers.parameter import RowvLLMParameter
from sglang.srt.runtime_context import get_context
from sglang.srt.tp_invariant_ops import matmul_tp_inv, tree_all_reduce_sum
from sglang.srt.true_on_policy import should_use_tp_invariant_row_linear

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.utils.test_utils.det_process_group import DetProcessGroup


@dataclass
class FixtureServerArgs:
    tp_size: int = 2
    true_on_policy_contract: str = "qwen3_dense_true_on_policy_v1"
    enable_torch_compile: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def _fold_sequential(values: list[torch.Tensor]) -> torch.Tensor:
    result = values[0]
    for value in values[1:]:
        result = (result + value).to(torch.bfloat16)
    return result


def _fold_two_level(values: list[torch.Tensor]) -> torch.Tensor:
    left = _fold_sequential(values[:19])
    right = _fold_sequential(values[19:])
    return (left + right).to(torch.bfloat16)


def _fold_padded_pairwise(values: list[torch.Tensor]) -> torch.Tensor:
    size = 1
    while size < len(values):
        size *= 2
    padded = values + [torch.zeros_like(values[0]) for _ in range(size - len(values))]
    while len(padded) > 1:
        padded = [
            (padded[index] + padded[index + 1]).to(torch.bfloat16)
            for index in range(0, len(padded), 2)
        ]
    return padded[0]


def _bit_pattern(value: torch.Tensor) -> str:
    packed = struct.pack("<f", float(value.detach().cpu()))
    return f"0x{struct.unpack('<I', packed)[0]:08x}"


def _module_record(name: str, module: Any) -> dict[str, Any]:
    return {
        "name": name,
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(getattr(module, "__file__", "unknown")),
    }


def _build_case(device: torch.device, rank: int, world_size: int) -> dict[str, Any]:
    partial_values = [
        1.0,
        -1.0,
        -1.0,
        -2.0,
        0.5,
        2.0,
        2.0,
        -0.0078125,
        -0.5,
        -0.0078125,
        1.0,
        -0.0078125,
        0.5,
        0.00390625,
        0.00390625,
        0.0078125,
        -2.0,
        0.0078125,
        -0.00390625,
        0.0078125,
        2.0,
        1.0,
        1.0,
        -2.0,
        -0.00390625,
        -2.0,
        0.00390625,
        0.00390625,
        0.0078125,
        0.5,
        0.0078125,
        0.5,
        -0.5,
        -0.5,
        1.0,
        0.5,
        -2.0,
        0.5,
    ]
    if len(partial_values) != 38:
        raise RuntimeError("Expected exactly 38 controlled partials")

    local_k = len(partial_values) * 128
    full_k = local_k * world_size
    output_size = 128
    batch_size = 64

    full_weight = torch.zeros(
        output_size,
        full_k,
        device=device,
        dtype=torch.bfloat16,
    )
    local_weight = full_weight[:, rank * local_k : (rank + 1) * local_k]
    local_input = torch.zeros(
        batch_size,
        local_k,
        device=device,
        dtype=torch.bfloat16,
    )
    for index, value in enumerate(partial_values):
        local_weight[0, index * 128] = value
        local_input[:, index * 128] = 1.0

    partials = [
        torch.tensor(value, device=device, dtype=torch.bfloat16)
        for value in partial_values
    ]
    return {
        "full_weight": full_weight,
        "local_weight": local_weight.contiguous(),
        "local_input": local_input,
        "partials": partials,
        "local_k": local_k,
        "full_k": full_k,
        "output_size": output_size,
        "batch_size": batch_size,
    }


def _make_megatron_layer(
    case: dict[str, Any],
    group: dist.ProcessGroup,
) -> MegatronRowParallelLinear:
    config = ModelParallelConfig(
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        perform_initialization=False,
    )
    layer = MegatronRowParallelLinear(
        case["full_k"],
        case["output_size"],
        config=config,
        init_method=lambda tensor: tensor.zero_(),
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        tp_group=group,
    )
    layer.weight.data.copy_(case["local_weight"])
    return layer


def _run_megatron_forward_and_gradients(
    layer: MegatronRowParallelLinear,
    case: dict[str, Any],
) -> dict[str, Any]:
    layer.zero_grad(set_to_none=True)
    source_input = case["local_input"]
    input_tensor = source_input.detach().clone().requires_grad_(True)
    output, _ = layer(input_tensor)
    loss = (output * output).sum()
    loss.backward()
    return {
        "output": output.detach(),
        "input_grad": input_tensor.grad.detach(),
        "weight_grad": layer.weight.grad.detach(),
    }


def _tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "first_value": float(tensor.reshape(-1)[0]),
        "bit_pattern": _bit_pattern(tensor.reshape(-1)[0]),
    }


def _difference_record(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_float = left.float()
    right_float = right.float()
    absolute = (left_float - right_float).abs()
    scale = torch.maximum(left_float.abs(), right_float.abs()).clamp_min(1e-30)
    return {
        "absolute_max": float(absolute.max()),
        "relative_max": float((absolute / scale).max()),
    }


def main() -> None:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError(f"This fixture requires exactly 2 GPUs, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    native_group = dist.new_group(
        ranks=[0, 1],
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    native_backend = native_group._get_backend(device)
    det_group = DetProcessGroup(native_backend)

    get_context().set_server_args(FixtureServerArgs())
    case = _build_case(device, rank, world_size)

    converted = convert_to_hf(
        SimpleNamespace(
            vocab_size=0,
            kv_channels=None,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=8,
        ),
        "qwen2",
        "module.module.decoder.layers.0.mlp.linear_fc2.weight",
        case["full_weight"],
    )
    if len(converted) != 1:
        raise RuntimeError(f"Expected one converted tensor, got {len(converted)}")
    hf_name, hf_weight = converted[0]
    weight_conversion_bitwise = torch.equal(hf_weight, case["full_weight"])

    sglang_weight = RowvLLMParameter(
        input_dim=1,
        data=torch.empty_like(case["local_weight"]),
        weight_loader=lambda parameter, loaded_weight: parameter.load_row_parallel_weight(
            loaded_weight,
            tp_rank=rank,
        ),
    )
    sglang_weight.load_row_parallel_weight(case["full_weight"], tp_rank=rank)
    sglang_weight_bitwise = torch.equal(sglang_weight, case["local_weight"])
    sglang_policy_enabled = should_use_tp_invariant_row_linear(case["local_k"])

    megatron_native = _make_megatron_layer(case, native_group)
    megatron_det = _make_megatron_layer(case, det_group)
    native_result = _run_megatron_forward_and_gradients(megatron_native, case)
    det_result = _run_megatron_forward_and_gradients(megatron_det, case)

    megatron_local_reference = torch.nn.functional.linear(
        case["local_input"],
        case["local_weight"],
    )
    megatron_local_native_allreduce = megatron_local_reference.clone()
    dist.all_reduce(megatron_local_native_allreduce, group=native_group)

    sglang_local = matmul_tp_inv(
        case["local_input"],
        sglang_weight.t(),
        fp32_accum=False,
    )
    sglang_global = tree_all_reduce_sum(sglang_local, device_group=native_group)

    sequential_reference = _fold_sequential(case["partials"])
    two_level_reference = _fold_two_level(case["partials"])
    padded_pairwise_reference = _fold_padded_pairwise(case["partials"])

    native_vs_det_output = torch.equal(native_result["output"], det_result["output"])
    native_vs_det_input_grad = torch.equal(
        native_result["input_grad"],
        det_result["input_grad"],
    )
    native_vs_det_weight_grad = torch.equal(
        native_result["weight_grad"],
        det_result["weight_grad"],
    )
    sglang_matches_two_level = torch.equal(
        sglang_local[:, 0],
        two_level_reference.repeat(case["batch_size"]),
    )
    sglang_other_columns_zero = bool(
        torch.equal(sglang_local[:, 1:], torch.zeros_like(sglang_local[:, 1:]))
    )

    results = {
        "fixture": "two-gpu controlled row-parallel linear",
        "issue": "radixark/miles#1485",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "precision": {
            "dtype": "bfloat16",
            "fp32_accum": False,
            "block_k": 128,
            "partial_count_per_rank": 38,
            "first_level_block": 19,
            "level_k": 2,
        },
        "distributed": {
            "world_size": world_size,
            "tensor_parallel_size": world_size,
            "process_group_timeout_seconds": args.timeout_seconds,
            "gpu_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
        },
        "modules": [
            _module_record("torch", torch),
            _module_record("megatron.core", __import__("megatron.core", fromlist=["__version__"])),
            _module_record("sglang", __import__("sglang", fromlist=["__version__"])),
            _module_record("miles", __import__("miles", fromlist=["__version__"])),
        ],
        "native_paths": {
            "torch_c": str(torch._C.__file__),
            "torch_hip": str(torch.version.hip),
            "megatron_row_parallel": str(
                __import__(
                    "megatron.core.tensor_parallel.layers",
                    fromlist=["RowParallelLinear"],
                ).__file__
            ),
            "sglang_tp_invariant_ops": str(
                __import__(
                    "sglang.srt.tp_invariant_ops.tp_invariant_ops",
                    fromlist=["matmul_tp_inv"],
                ).__file__
            ),
            "sglang_row_parameter": str(
                __import__(
                    "sglang.srt.layers.parameter",
                    fromlist=["RowvLLMParameter"],
                ).__file__
            ),
            "miles_det_process_group": str(
                __import__(
                    "miles.utils.test_utils.det_process_group",
                    fromlist=["DetProcessGroup"],
                ).__file__
            ),
        },
        "environment_overrides": {
            "AITER_CONFIG_GEMM_BF16": os.environ["AITER_CONFIG_GEMM_BF16"],
            "USE_ROCM_AITER_ROPE_BACKEND": os.environ["USE_ROCM_AITER_ROPE_BACKEND"],
        },
        "weight_conversion": {
            "megatron_name": "module.module.decoder.layers.0.mlp.linear_fc2.weight",
            "hf_name": hf_name,
            "bitwise_equal": weight_conversion_bitwise,
            "difference": _difference_record(hf_weight, case["full_weight"]),
            "sglang_loader_bitwise_equal": sglang_weight_bitwise,
            "sglang_loader_difference": _difference_record(sglang_weight, case["local_weight"]),
            "sglang_policy_enabled": sglang_policy_enabled,
        },
        "reduction_order": {
            "sequential": _tensor_stats("sequential", sequential_reference),
            "sglang_two_level": _tensor_stats("sglang_two_level", two_level_reference),
            "padded_pairwise_tree": _tensor_stats("padded_pairwise_tree", padded_pairwise_reference),
            "sglang_local_matches_two_level": sglang_matches_two_level,
            "sglang_other_columns_zero": sglang_other_columns_zero,
            "sglang_local_vs_sequential": _difference_record(
                sglang_local[:, 0],
                sequential_reference.repeat(case["batch_size"]),
            ),
            "sglang_local_vs_padded_pairwise_tree": _difference_record(
                sglang_local[:, 0],
                padded_pairwise_reference.repeat(case["batch_size"]),
            ),
        },
        "outputs": {
            "megatron_local_reference": _tensor_stats("megatron_local_reference", megatron_local_reference),
            "megatron_native_global": _tensor_stats("megatron_native_global", native_result["output"]),
            "megatron_det_global": _tensor_stats("megatron_det_global", det_result["output"]),
            "sglang_local": _tensor_stats("sglang_local", sglang_local),
            "sglang_global": _tensor_stats("sglang_global", sglang_global),
            "megatron_native_vs_det_bitwise": native_vs_det_output,
            "megatron_native_vs_det_difference": _difference_record(
                native_result["output"],
                det_result["output"],
            ),
            "megatron_local_vs_sglang_local": _difference_record(
                megatron_local_reference,
                sglang_local,
            ),
            "megatron_global_vs_sglang_global": _difference_record(
                native_result["output"],
                sglang_global,
            ),
            "megatron_local_reference_allreduce_bitwise": torch.equal(
                megatron_local_native_allreduce,
                native_result["output"],
            ),
        },
        "gradients": {
            "loss": "sum(output * output)",
            "native_vs_det_input_grad_bitwise": native_vs_det_input_grad,
            "native_vs_det_weight_grad_bitwise": native_vs_det_weight_grad,
            "native_input_grad": _tensor_stats("native_input_grad", native_result["input_grad"]),
            "det_input_grad": _tensor_stats("det_input_grad", det_result["input_grad"]),
            "native_weight_grad": _tensor_stats("native_weight_grad", native_result["weight_grad"]),
            "det_weight_grad": _tensor_stats("det_weight_grad", det_result["weight_grad"]),
            "native_vs_det_input_grad_difference": _difference_record(
                native_result["input_grad"],
                det_result["input_grad"],
            ),
            "native_vs_det_weight_grad_difference": _difference_record(
                native_result["weight_grad"],
                det_result["weight_grad"],
            ),
            "sglang_autograd_available": False,
        },
        "limitations": [
            "Synthetic one-nonzero-per-block case isolates outer reduction order, not general GEMM numerics.",
            "TP=2 makes cross-rank native and deterministic folds identical; it does not test "
            "non-power-of-two rank folds.",
            "SGLang matmul_tp_inv is inference-only in this path and has no autograd implementation.",
            "This fixture does not measure throughput and does not prove end-to-end Qwen3-4B parity.",
        ],
    }

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
