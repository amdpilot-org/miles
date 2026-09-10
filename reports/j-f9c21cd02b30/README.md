# Two-GPU row-parallel linear fixture

This directory contains a bounded reproduction for [radixark/miles#1485](https://github.com/radixark/miles/issues/1485).
The fixture uses two MI355X GPUs, preserves bfloat16 inputs and weights, keeps
`matmul_tp_inv` on `fp32_accum=False`, and uses a 120-second process-group timeout.

## What it exercises

- Miles Megatron-to-HF conversion for `mlp.linear_fc2.weight` to `mlp.down_proj.weight`.
- SGLang's `RowvLLMParameter` row-parallel weight loader.
- Megatron's actual `RowParallelLinear` forward and backward paths.
- SGLang's actual `matmul_tp_inv` kernel and deterministic tree all-reduce.
- Miles' `DetProcessGroup` fixed-order fold.

The synthetic case uses one nonzero per 128-wide K block. This makes each block
partial exactly controlled and isolates the outer reduction order from general
GEMM accumulation. Each rank has K=4864, or 38 partials, matching the issue's
`FIRST_LEVEL_BLOCK=19` and `LEVEL_K=2` case.

## Reproduce

From the repository root:

```bash
timeout 600s /opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  reports/j-f9c21cd02b30/row_parallel_fixture.py \
  --output reports/j-f9c21cd02b30/results.json \
  --timeout-seconds 120
```

The script sets two process-local environment defaults:

- `AITER_CONFIG_GEMM_BF16` points at the stock single-file Aiter config to avoid
  a shared `/tmp/aiter_configs` lock that is not writable in this container.
- `USE_ROCM_AITER_ROPE_BACKEND=0` avoids the lower-precision Aiter RoPE path;
  RoPE is not used by this fixture.

## Observed result

The generated `results.json` records the exact module versions and native paths.
On the tested stack:

- Miles Megatron-to-HF conversion and SGLang row-parallel loading are bitwise exact.
- SGLang's local output exactly matches the expected two-level fold.
- Megatron's local output differs from SGLang by `0.02734375` absolute
  (`0.05223880708217621` relative).
- After TP all-reduce, Megatron differs from SGLang by `0.0546875` absolute
  (`0.05223880708217621` relative).
- Native NCCL and Miles' deterministic process group produce bitwise-identical
  TP2 outputs and gradients for this case.
- SGLang's `matmul_tp_inv` path has no autograd implementation in this stack.

## What this does not prove

- It does not measure throughput.
- It does not prove general GEMM numerical equivalence beyond the controlled
  one-nonzero-per-block case.
- TP2 cannot distinguish cross-rank fold orders that differ at larger or
  non-power-of-two world sizes.
- It does not prove end-to-end Qwen3-4B train/inference parity.
