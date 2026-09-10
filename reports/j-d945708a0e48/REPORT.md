# Two-GPU row-parallel reduction diagnostic

## Scope

This is a bounded investigation for upstream Miles issue 1485 (“MILES
row-parallel fc2/down_proj reduction order does not match SGLang”). It uses a
synthetic Qwen-shaped `down_proj`/`linear_fc2` case, not a public-model
download and not a throughput benchmark.

The fixture ran on two assigned AMD Instinct MI350X devices. Both devices
reported `gfx950:sramecc+:xnack-`, 256 compute units, and 270,566,162,432 bytes
of memory. The run used bf16 throughout and did not change precision settings.

## What ran

The executable fixture is `row_parallel_fixture.py`. It:

1. Initializes a two-rank `det_nccl` process group with a 120-second timeout
   and initializes Megatron tensor parallelism with a two-minute timeout.
2. Creates 38 deterministic 128-wide block partials per rank, matching the
   Qwen3-4B TP2 local `K=4864` shape from issue 1485.
3. Gathers the two rank-local Megatron weights into the full `K=9728` weight.
4. Exercises the actual Miles Qwen Megatron-to-HF name/tensor conversion for
   `mlp.linear_fc2.weight` to `model.layers.0.mlp.down_proj.weight`.
5. Loads that converted HF weight through SGLang's actual
   `RowParallelLinear.weight_loader` into each TP rank.
6. Runs SGLang's actual `matmul_tp_inv` Triton kernel.
7. Runs the actual Megatron `RowParallelLinear` forward and backward over the
   same rank-local operands and weights.
8. Compares the SGLang kernel against its documented two-level fold, the
   current Miles `_fold_gathered_sum`, global outputs, and local gradient
   references.

The complete machine-readable evidence is in `results.json`.

## Reproduction

From the Miles checkout:

```bash
mkdir -p /job/cache/triton /job/cache/hf
PYTHONPATH=/job/miles \
TRITON_CACHE_DIR=/job/cache/triton \
HF_HOME=/job/cache/hf \
SGLANG_USE_AITER=0 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
/opt/venv/bin/torchrun --standalone --nproc-per-node=2 \
  reports/j-d945708a0e48/row_parallel_fixture.py \
  --output reports/j-d945708a0e48/results.json \
  --timeout-seconds 120
```

## Observed results

| Comparison | Bitwise | Maximum absolute difference | Mismatched elements |
| --- | ---: | ---: | ---: |
| Miles Megatron-to-HF converted weight | equal | 0 | 0 |
| SGLang TP weight loader | equal | 0 | 0 |
| SGLang kernel vs two-level fold | equal | 0 | 0 |
| Local Miles fold vs SGLang kernel | not equal | 0.0625 | 256 / 256 |
| Global Miles fold vs SGLang output | not equal | 0.09375 | 256 / 256 |
| Megatron output vs SGLang output | not equal | 0.015625 | 256 / 256 |
| Megatron input gradient vs reference | equal | 0 | 0 |
| Megatron weight gradient vs reference | equal | 0 | 0 |

The controlled upstream gradient used row 0 alternating `+1,-1` and row 1 equal
to `0.5` times row 0. This creates cancellation in the local input-gradient
reduction while keeping a nonzero weight gradient. Megatron's input and weight
gradients were bitwise equal to local `torch.matmul` references for that
upstream pattern.

## Source behavior found

- SGLang's installed `matmul_tp_inv` derives `T=38`, `LEVEL_K=2`, and
  `FIRST_LEVEL_BLOCK=19` from `K=4864` and `BLOCK_K=128`. The fixture's kernel
  output was bitwise equal to the explicit two-level fold:
  `(p0 + p1 + ... + p18) + (p19 + p20 + ... + p37)`.
- The current mirror's Miles `_fold_gathered_sum` uses a pairwise tree only when
  the gathered list length is a power of two. For the fixture's 38 local and 76
  global block partials, it takes the ascending sequential branch. This differs
  from issue 1485's description of a global pairwise tree over all partials.
  That discrepancy is an observation about this mirror's current source, not a
  claim that either issue description or implementation is correct.
- Megatron's `RowParallelLinear` performs its local GEMM and then reduces the
  rank-local outputs through the tensor-parallel group. Its global output did
  not match SGLang's global output in this controlled case.

## What this proves

This run proves, on two actual MI350X devices:

- The synthetic Megatron shard can be gathered, converted through the actual
  Miles Qwen mapping, and loaded through the actual SGLang row-parallel loader
  without changing any weight bits.
- SGLang's installed TP-invariant kernel implements the expected two-level
  19-block/19-block reduction for the tested `K=4864` bf16 case.
- The current Miles fold and SGLang kernel produce different bf16 results for
  controlled, cancellation-sensitive 38-block partials.
- The actual Megatron row-parallel layer and SGLang kernel produce different
  global bf16 outputs for the same rank-local operands and weights.
- Megatron's local input and weight gradients match local bf16 references for
  the controlled upstream pattern.

## What this does not prove

- It does not prove which reduction order is numerically preferable or correct.
- It does not prove full Qwen3-4B model parity; only the Qwen-shaped
  `down_proj` dimensions and conversion mapping were exercised.
- It does not prove SGLang backward behavior. `matmul_tp_inv` returned a tensor
  with `requires_grad=False`; no SGLang backward path was available to compare.
  The gradient comparison is therefore Megatron versus a local reference, not
  Miles versus SGLang.
- It does not establish throughput, latency, or performance. No such claim is
  made.
- It does not establish MI355X equivalence. Both assigned devices were MI350X;
  sharing gfx950 with MI355X is not a performance claim.
- It does not prove production all-reduce behavior beyond the tested two-rank
  `det_nccl` path and controlled shapes.

## Environment and limitations

The tested source revisions and imported module paths are recorded in
`results.json`. The run explicitly used `PYTHONPATH=/job/miles` so the mounted
Miles clone took precedence over the preinstalled editable `/root/miles`
source. Without that override, the first attempt imported `/root/miles`; that
failure is preserved here rather than hidden.

The run set `SGLANG_USE_AITER=0` before importing SGLang. This avoided an
unwritable, root-owned AITER cache lock under `/tmp/aiter_configs` and kept the
test on the explicit SGLang TP-invariant kernel. It does not change the bf16
precision or the explicit kernel under test, but it is still a runtime
integration override and is not a claim about AITER-enabled SGLang behavior.

The first executable attempt also failed because the Miles Qwen converter
requires model arguments such as `hidden_size` and `kv_channels` even for the
`down_proj` mapping. The fixture now supplies those arguments explicitly. No
model weights were downloaded.
