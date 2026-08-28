#!/usr/bin/env python3
"""Per-projection GEMM throughput at the shapes the repo's FLOPs model predicts.

``bench_gemm.py`` times square GEMMs. This companion measures the *rectangular*
GEMMs a real transformer layer actually issues -- QKV projection, output
projection, MLP up (SwiGLU gate+up counted as two), MLP down, and the LM head --
at the exact M/N/K that ``miles/utils/flops_utils.py`` would count FLOPs for.
It imports the repo's real ``flops_args_from_hf_config`` (no build needed) to
derive those shapes, so the measurement corresponds directly to the repo's own
forward-FLOPs accounting rather than to an abstract square proxy.

BF16 only.  Reimplementation note identical to ``bench_gemm.py``: ``torch.matmul``
-> hipBLASLt on ROCm; Miles' own matmuls live in Megatron-LM / Transformer Engine,
which need a from-source build.

Usage:
    python bench_layer_gemms.py [--seqlen 4096] [--warmup 10] [--repeats 50] [--out layer_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch  # noqa: E402

from miles.utils.flops_utils import flops_args_from_hf_config  # noqa: E402


# Representative dense model (Llama-3-8B-ish geometry, GQA).
MODEL_HF = SimpleNamespace(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=152064,
    num_hidden_layers=32,
    intermediate_size=14336,
    head_dim=128,
)


@dataclass
class ProjSpec:
    name: str
    M: int
    K: int
    N: int
    count_per_layer: int  # how many times this GEMM fires per layer


def layer_projections(args, seqlen: int) -> list[ProjSpec]:
    H = args.hidden_size
    nh = args.num_attention_heads
    ng = args.num_query_groups
    kc = args.kv_channels
    ffn = args.ffn_hidden_size
    V = args.vocab_size
    S = seqlen
    return [
        ProjSpec("Q_proj", S, H, nh * kc, 1),
        ProjSpec("KV_proj", S, H, 2 * ng * kc, 1),
        ProjSpec("Out_proj", S, H, H, 1),
        ProjSpec("MLP_up_gate", S, H, ffn, 2),   # SwiGLU: gate + up
        ProjSpec("MLP_down", S, ffn, H, 1),
        ProjSpec("LM_head", S, H, V, 1),         # once total, not per layer
    ]


def time_one_gemm(M: int, K: int, N: int, warmup: int, repeats: int) -> dict[str, float]:
    dev = "cuda"
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    b = (torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    flops = 2 * M * N * K

    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    tflops: list[float] = []
    for _ in range(repeats):
        start.record()
        a @ b
        end.record()
        torch.cuda.synchronize()
        tflops.append(flops / (start.elapsed_time(end) * 1e-3) / 1e12)
    tflops.sort()
    n = len(tflops)

    def pct(frac: float) -> float:
        return tflops[max(0, min(n - 1, int(round((n - 1) * frac))))]

    import statistics
    return {
        "mean": statistics.fmean(tflops),
        "median": statistics.median(tflops),
        "stdev": statistics.pstdev(tflops) if n > 1 else 0.0,
        "p5": pct(0.05),
        "p95": pct(0.95),
        "min": min(tflops),
        "max": max(tflops),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    hf_args = flops_args_from_hf_config(MODEL_HF)
    projs = layer_projections(hf_args, args.seqlen)

    print(f"Model: h={hf_args.hidden_size} L={hf_args.num_layers} heads={hf_args.num_attention_heads} "
          f"kv_groups={hf_args.num_query_groups} ffn={hf_args.ffn_hidden_size} vocab={hf_args.vocab_size}")
    print(f"Seqlen={args.seqlen}  warmup={args.warmup}  repeats={args.repeats}\n")
    print(f"{'projection':<14} {'M':>6} {'K':>6} {'N':>7} {'x/layer':>7}  {'TFLOP/s (median)':>18}  {'spread (p5-p95)':>20}")
    print("-" * 84)

    results: dict[str, Any] = {
        "env": {
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
            "seqlen": args.seqlen,
        },
        "projections": [],
    }

    layer_flops = 0.0
    layer_time_ms = 0.0
    for p in projs:
        st = time_one_gemm(p.M, p.K, p.N, args.warmup, args.repeats)
        per = p.count_per_layer
        is_lm_head = p.name == "LM_head"
        layers = 1 if is_lm_head else hf_args.num_layers
        proj_flops = 2 * p.M * p.K * p.N * per * layers
        proj_time_ms = proj_flops / (st["median"] * 1e12) * 1e3
        layer_flops += proj_flops
        layer_time_ms += proj_time_ms
        tag = "1x total" if is_lm_head else f"{per}x/layer"
        print(f"{p.name:<14} {p.M:>6} {p.K:>6} {p.N:>7} {tag:>7}  "
              f"{st['median']:>14.1f}  {st['p5']:>8.1f}-{st['p95']:.1f}")
        results["projections"].append({
            "name": p.name, "M": p.M, "K": p.K, "N": p.N,
            "count_per_layer": per, "layers": layers, **st,
        })

    print("-" * 84)
    print(f"{'TOTAL/forward':<14} {'':>6} {'':>6} {'':>7} {'':>7}  "
          f"{layer_flops / layer_time_ms / 1e9:>14.1f}  "
          f"{'(harmonic-ish across projections)':>20}")
    print(f"  total fwd FLOPs = {layer_flops:.3e} ({layer_flops/1e12:.1f} TFLOP)")
    print(f"  est. wall-clock = {layer_time_ms:.2f} ms at measured per-projection rates")

    results["total_forward_tflops"] = layer_flops / 1e12
    results["total_forward_est_ms"] = layer_time_ms

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
