#!/usr/bin/env python3
"""GEMM throughput on a single AMD Instinct GPU.

Operation measured
------------------
A single dense matrix multiply (GEMM), the kernel that dominates the transformer
forward pass.  `miles/utils/flops_utils.py` models the forward FLOPs of every
attention/MLP/LM-head projection as a sum of 2*M*N*K GEMM FLOPs, and
`miles/utils/device_flops.py` is a peak-BF16-TFLOPS table used to turn those
FLOPs into a GPU utilization number.  This script measures the achieved
TFLOP/s of that kernel directly.

Reimplementation note
---------------------
The repository does not ship its own GEMM: it delegates matmuls to Megatron-LM /
Transformer Engine, which cannot be imported without a from-source build of a
large training stack.  `torch.matmul` and `torch._scaled_mm` on ROCm call into
hipBLASLt -- the same low-level library that stack targets -- so they are a
faithful single-kernel proxy.  No miles code is imported.

Usage
-----
    python bench_gemm.py                       # defaults: sizes 4096/8192/16384, 10 warmup, 50 reps
    python bench_gemm.py --sizes 8192 --repeats 100 --warmup 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any

import torch

FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------- env
def smi(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"<{exc}>"


def _read_gfx_arch() -> str:
    """Read the ISA name (e.g. gfx950) from rocminfo rather than guessing."""
    out = smi(["rocminfo"])
    m = re.search(r"gfx(\d+[a-z]*)", out)
    return m.group(0) if m else "unknown"


def read_env() -> dict[str, Any]:
    p = torch.cuda.get_device_properties(0)
    env: dict[str, Any] = {
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device_name": p.name,
        "gfx_arch": _read_gfx_arch(),
        "compute_units": p.multi_processor_count,
        "total_memory_gib": round(p.total_memory / 2**30, 1),
        "cuda_available": torch.cuda.is_available(),
    }
    # pull the live engine clock + power straight off the card
    env["rocm_smi_summary"] = smi(["rocm-smi", "--showclocks", "--showpower", "--showtemp"]).strip()
    env["amdsmi_monitor_summary"] = smi(["amd-smi", "monitor"]).strip()
    return env


# ------------------------------------------------------------------------- stats
@dataclass
class Stats:
    n: int
    mean: float
    median: float
    stdev: float
    p5: float
    p95: float
    min: float
    max: float


def summarize(values: list[float]) -> Stats:
    vs = sorted(values)
    n = len(vs)

    def pct(frac: float) -> float:
        idx = max(0, min(n - 1, int(round((n - 1) * frac))))
        return vs[idx]

    return Stats(
        n=n,
        mean=statistics.fmean(vs),
        median=statistics.median(vs),
        stdev=statistics.pstdev(vs) if n > 1 else 0.0,
        p5=pct(0.05),
        p95=pct(0.95),
        min=min(vs),
        max=max(vs),
    )


# ------------------------------------------------------------------ timing core
def time_gemm(
    M: int,
    N: int,
    K: int,
    dtype: str,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Return a list of per-iteration achieved TFLOP/s for one GEMM config."""
    dev = "cuda"
    torch.manual_seed(0)
    scale = 0.05  # keep values FP8-safe without per-tile scaling gymnastics

    if dtype == "bf16":
        a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * scale).contiguous()
        b = (torch.randn(K, N, device=dev, dtype=torch.bfloat16) * scale).contiguous()

        def run() -> None:
            torch.matmul(a, b)

    elif dtype == "fp8":
        a = (torch.randn(M, K, device=dev, dtype=torch.float32) * scale).clamp(-448, 448).to(FP8).contiguous()
        # _scaled_mm needs a row-major @ col-major pair
        b = (torch.randn(N, K, device=dev, dtype=torch.float32) * scale).clamp(-448, 448).to(FP8).t().contiguous().t()
        sa = torch.ones(1, device=dev, dtype=torch.float32)
        sb = torch.ones(1, device=dev, dtype=torch.float32)

        def run() -> None:
            torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    else:
        raise ValueError(dtype)

    flops = 2 * M * N * K

    # warmup (also lets the clock governor boost)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    tflops: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        tflops.append(flops / (ms * 1e-3) / 1e12)
    return tflops


# ----------------------------------------------------------------------- report
def fmt_stats(s: Stats) -> str:
    return (
        f"mean={s.mean:7.1f}  median={s.median:7.1f}  "
        f"std={s.stdev:6.2f}  min={s.min:7.1f}  max={s.max:7.1f}  "
        f"p5={s.p5:7.1f}  p95={s.p95:7.1f}  (n={s.n})"
    )



def time_backward_gemm(
    M: int,
    N: int,
    K: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Backward of C = A @ B (BF16): two GEMMs, 4*M*N*K FLOPs total.

    A training step is forward + backward; ``flops_utils.py`` models only the
    forward pass, so the headline is forward GEMM. This companion measurement
    closes the "forward only" gap with the same kernel family on the same card.
    """
    dev = "cuda"
    torch.manual_seed(0)
    scale = 0.05
    a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * scale).contiguous()
    b = (torch.randn(K, N, device=dev, dtype=torch.bfloat16) * scale).contiguous()
    a.requires_grad_(True)
    b.requires_grad_(True)
    c = a @ b                       # graph built once
    g = torch.randn_like(c)
    flops = 4 * M * N * K           # grad_A = g @ B^T, grad_B = A^T @ g

    def run() -> None:
        a.grad = None
        b.grad = None
        torch.autograd.backward(c, g, retain_graph=True)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    tflops: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        tflops.append(flops / (ms * 1e-3) / 1e12)
    return tflops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 16384], help="square M=N=K dims")
    ap.add_argument("--dtypes", type=str, nargs="+", default=["bf16", "fp8"], choices=["bf16", "fp8"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", type=str, default="", help="write JSON results here")
    args = ap.parse_args()

    env = read_env()
    print("=" * 78)
    print("ENVIRONMENT (read from the machine)")
    print("=" * 78)
    for k in ("device_name", "gfx_arch", "compute_units", "total_memory_gib", "torch_version", "hip_version"):
        print(f"  {k:20s} {env[k]}")
    print("=" * 78)

    results: dict[str, Any] = {"env": {k: v for k, v in env.items() if k in {
        "device_name", "gfx_arch", "compute_units", "total_memory_gib",
        "torch_version", "hip_version"}}, "configs": []}

    print(f"\nGEMM throughput  (warmup={args.warmup}, repeats={args.repeats})\n")
    for dtype in args.dtypes:
        print(f"-- {dtype.upper()} --")
        for size in args.sizes:
            vals = time_gemm(size, size, size, dtype, args.warmup, args.repeats)
            st = summarize(vals)
            line = f"  {dtype:4s} {size:6d}^3  TFLOP/s  {fmt_stats(st)}"
            print(line)
            results["configs"].append({
                "dtype": dtype, "M": size, "N": size, "K": size,
                **asdict(st),
            })
        print()

    # backward GEMM (BF16 only: _scaled_mm has no autograd backward path)
    print("-- BF16 BACKWARD (2 GEMMs/step = 4*M*N*K FLOPs) --")
    for size in args.sizes:
        vals = time_backward_gemm(size, size, size, args.warmup, args.repeats)
        st = summarize(vals)
        print(f"  bwd  {size:6d}^3  TFLOP/s  {fmt_stats(st)}")
        results["configs"].append({
            "dtype": "bf16_backward", "M": size, "N": size, "K": size,
            **asdict(st),
        })
    print()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
