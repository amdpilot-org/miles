#!/usr/bin/env python3
"""Cross-check the measured GEMM throughput against the repo's own FLOPs model.

This imports the *real* miles code (no build needed -- both modules are pure
Python) and exercises the exact path ``flops_utils.calculate_fwd_flops`` +
``device_flops.local_peak_bf16_tflops`` that a Miles run uses to convert forward
GEMM FLOPs into a GPU-utilization number. It demonstrates, on metal, that:

  * the repo's forward-FLOPs model produces a concrete TFLOP count for a
    representative dense transformer, and
  * ``local_peak_bf16_tflops()`` returns ``None`` on this MI355X, so the repo
    cannot currently turn that count (or any achieved TFLOP/s) into a
    utilization % on this GPU -- the gap named in REPORT.md.

No assumed vendor peaks are used: the wall-clock estimate divides the repo's
predicted forward TFLOPs by the *measured* sustained BF16 GEMM TFLOP/s from
``results.json``.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch  # noqa: E402

from miles.utils import device_flops  # noqa: E402
from miles.utils.flops_utils import calculate_fwd_flops, flops_args_from_hf_config  # noqa: E402


def main() -> int:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "results.json")) as fh:
        results = json.load(fh)

    # representative dense transformer (Llama-3-8B-ish geometry), GQA
    hf = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=152064,
        num_hidden_layers=32,
        intermediate_size=14336,
        head_dim=128,
    )
    args = flops_args_from_hf_config(hf)
    seqlen = 4096
    fwd_flops = calculate_fwd_flops([seqlen], args)
    fwd_tflops = fwd_flops / 1e12

    # measured sustained BF16 GEMM rate at 16384^3 (most compute-bound, stablest)
    bf16 = next(c for c in results["configs"] if c["dtype"] == "bf16" and c["M"] == 16384)
    measured_tflops = bf16["median"]

    print("=" * 72)
    print("CROSS-CHECK: repo FLOPs model  vs  measured GEMM throughput")
    print("=" * 72)
    print(f"  device name (torch)     {torch.cuda.get_device_name(0)}")
    print(f"  model geometry          h={hf.hidden_size} L={hf.num_hidden_layers} "
          f"heads={hf.num_attention_heads} kv={hf.num_key_value_heads} ffn={hf.intermediate_size}")
    print(f"  seqlen                  {seqlen}")
    print(f"  repo fwd FLOPs          {fwd_flops:.4e}  ({fwd_tflops:.2f} TFLOP)")
    print(f"  measured BF16 GEMM      {measured_tflops:.1f} TFLOP/s (median, 16384^3, n=50)")
    print(f"  est. fwd wall-clock     {fwd_tflops / measured_tflops * 1e3:.2f} ms "
          f"(at measured sustained GEMM rate)")
    print("-" * 72)
    peak = device_flops.local_peak_bf16_tflops()
    print(f"  local_peak_bf16_tflops() -> {peak}")
    if peak is None:
        print("  => repo CANNOT compute a utilization % on this GPU (no AMD peak entry).")
        print("     This is the gap named in REPORT.md and pinned by")
        print("     tests/fast/utils/test_device_flops_amd.py.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
