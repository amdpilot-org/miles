#!/usr/bin/env python3
"""Run and verify a masked-tail Triton FP32 vector-add probe."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    element_count,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x_values + y_values, mask=mask)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/ROCm device is visible to Torch")

    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, saw {device_count}")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "")),
        "gcn_arch_name": properties.gcnArchName,
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "visible_device_count": device_count,
    }


def _make_nonzero_tensors(length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    indices = torch.arange(length, dtype=torch.float32, device="cpu")
    x = 1.0 + (seed % 997) * 0.01 + indices * 0.125
    y = 2.0 - (seed % 883) * 0.005 - indices * 0.0625
    if not bool(torch.all(x != 0)) or not bool(torch.all(y != 0)):
        raise AssertionError("Generated tensor unexpectedly contains zero values")
    return x, y


def _run_case(length: int, seed: int, block_size: int) -> dict[str, Any]:
    x_cpu, y_cpu = _make_nonzero_tensors(length, seed)
    expected = x_cpu + y_cpu
    device = torch.device("cuda", 0)
    x_gpu = x_cpu.to(device=device)
    y_gpu = y_cpu.to(device=device)
    output_gpu = torch.empty_like(x_gpu)
    grid = (triton.cdiv(length, block_size),)

    started = time.perf_counter()
    _vector_add_kernel[grid](
        x_gpu,
        y_gpu,
        output_gpu,
        length,
        BLOCK_SIZE=block_size,
    )
    torch.cuda.synchronize(device)
    observed = output_gpu.cpu()
    elapsed_seconds = time.perf_counter() - started

    mismatches = observed != expected
    mismatch_count = int(mismatches.sum().item())
    max_abs_error = float((observed - expected).abs().max().item())
    exact_match = bool(torch.equal(observed, expected))
    sample_indices = sorted({0, length // 2, length - 1})
    samples = [
        {
            "index": index,
            "x": float(x_cpu[index].item()),
            "y": float(y_cpu[index].item()),
            "expected": float(expected[index].item()),
            "observed": float(observed[index].item()),
        }
        for index in sample_indices
    ]

    return {
        "length": length,
        "seed": seed,
        "block_size": block_size,
        "exact_match": exact_match,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "elapsed_ms": elapsed_seconds * 1000.0,
        "samples": samples,
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_values = [case["elapsed_ms"] for case in cases]
    return {
        "case_count": len(cases),
        "exact_case_count": sum(case["exact_match"] for case in cases),
        "failed_case_count": sum(not case["exact_match"] for case in cases),
        "total_mismatch_count": sum(case["mismatch_count"] for case in cases),
        "max_abs_error": max((case["max_abs_error"] for case in cases), default=0.0),
        "elapsed_ms_mean": sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0,
        "elapsed_ms_min": min(elapsed_values) if elapsed_values else 0.0,
        "elapsed_ms_max": max(elapsed_values) if elapsed_values else 0.0,
    }


def _sample_cases(cases: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    if len(cases) <= sample_count * 2:
        return cases
    return [*cases[:sample_count], *cases[-sample_count:]]


def _run_initial(args: argparse.Namespace, device_metadata: dict[str, Any]) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    case = _run_case(args.length, args.seed, args.block_size)
    elapsed_seconds = time.perf_counter() - started
    return {
        "mode": "initial",
        "validation_start_utc": started_at,
        "validation_end_utc": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "device": device_metadata,
        "versions": _version_metadata(),
        "parameters": {
            "length": args.length,
            "seed": args.seed,
            "block_size": args.block_size,
        },
        "aggregate": _aggregate([case]),
        "all_exact": case["exact_match"],
        "case": case,
    }


def _run_extended(args: argparse.Namespace, device_metadata: dict[str, Any]) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    iteration = 0

    while (
        time.perf_counter() - started < args.min_seconds or iteration == 0
    ) and iteration < args.max_iterations:
        seed = args.seed + iteration
        tail_length = iteration % args.tail_count
        length = args.base_length + tail_length
        case = _run_case(length, seed, args.block_size)
        cases.append(case)
        if not case["exact_match"]:
            failure = dict(case)
            failure["iteration"] = iteration
            failures.append(failure)
        iteration += 1

    elapsed_seconds = time.perf_counter() - started
    aggregate = _aggregate(cases)
    return {
        "mode": "extended",
        "validation_start_utc": started_at,
        "validation_end_utc": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "target_min_seconds": args.min_seconds,
        "device": device_metadata,
        "versions": _version_metadata(),
        "parameters": {
            "base_length": args.base_length,
            "tail_lengths": list(range(args.tail_count)),
            "seed_start": args.seed,
            "block_size": args.block_size,
            "max_iterations": args.max_iterations,
        },
        "aggregate": aggregate,
        "all_exact": aggregate["failed_case_count"] == 0,
        "all_iterations_performed_and_verified": True,
        "sample_cases": _sample_cases(cases, args.sample_count),
        "failure_samples": failures[: args.failure_sample_limit],
    }


def _version_metadata() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initial", "extended"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1_048_583)
    parser.add_argument("--seed", type=int, default=51_125_129)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--min-seconds", type=float, default=60.0)
    parser.add_argument("--base-length", type=int, default=4096)
    parser.add_argument("--tail-count", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=100_000)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--failure-sample-limit", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "initial" and args.length <= 0:
        parser.error("--length must be positive")
    if args.mode == "extended" and (args.base_length <= 0 or args.tail_count <= 0):
        parser.error("--base-length and --tail-count must be positive")
    if args.min_seconds < 0 or args.max_iterations <= 0:
        parser.error("--min-seconds must be nonnegative and --max-iterations positive")
    if args.block_size <= 0 or args.block_size.bit_count() != 1:
        parser.error("--block-size must be a positive power of two")
    return args


def main() -> None:
    args = _parse_args()
    device_metadata = _device_metadata()
    if args.mode == "initial":
        result = _run_initial(args, device_metadata)
    else:
        result = _run_extended(args, device_metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
