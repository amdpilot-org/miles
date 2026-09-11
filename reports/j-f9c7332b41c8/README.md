# Bounded FSDP/Megatron update comparison

## Result

The reduced two-GPU case found **no mismatch** between Torch FSDP1
`FULL_SHARD` and Megatron Core `DistributedDataParallel` for the tested
settings. Both backends produced identical fp32 losses, gradients, and
post-step parameters for two steps, and both file-backed state-dict reloads
restored the saved tensors exactly.

| Check | Result |
|---|---:|
| Initial parameter max delta | `0.0` |
| Loss max delta over 2 steps | `0.0` |
| Gradient max delta over 2 steps | `0.0` |
| Post-step parameter max delta over 2 steps | `0.0` |
| FSDP file-backed state-dict reload max delta | `0.0` |
| Megatron file-backed state-dict reload max delta | `0.0` |
| Consumed samples after steps 1 and 2 | `4`, then `8` |

The exact zero deltas are expected for this deterministic fp32 fixture. They do
not imply bitwise parity for production models, mixed precision, or a
different collective/optimizer implementation.

## Hardware

The run used both assigned GPUs:

- GPU 0: AMD Instinct MI350X, gfx950, serial `692517020513`
- GPU 1: AMD Instinct MI350X, gfx950, serial `692517020502`

The complete `rocm-smi` identity is in `gpu_identity.json`. This is an MI350X
run; sharing a gfx950 ISA family with MI355X does not establish equal
performance.

## Fixture

`compare_backends.py` uses a synthetic 108-parameter model
(`Linear(8, 8) -> GELU -> Linear(8, 4)`) with deterministic synthetic data.
It runs:

- Torch FSDP1 `FULL_SHARD`, `use_orig_params=True`, DP=2.
- Megatron Core `DistributedDataParallel`, DP=2, TP=1, PP=1.
- Global batch 4, local batch 2, two optimizer steps.
- fp32 parameters and gradients, SGD at learning rate `0.1`, and mean MSE.
- A 300-second process-group timeout and a five-minute Megatron timeout.

The fixture compares initial parameters, per-step global loss, full unsharded
gradients, post-step parameters, and consumed samples. It then perturbs each
backend state by `0.25`, reloads the file-backed saved state, and checks the
restoration.

For FSDP, full gradients are reconstructed by all-gathering the equal-sized
flat gradient shards and applying FSDP's own flat-parameter metadata. Megatron
gradients are read from its synchronized `main_grad` buffers.

## Environment provenance

The run records imported module paths, distribution versions, source revisions,
and native libraries in `comparison.json`:

- Miles clone: `/job/miles`, commit `e5125a97e1fd383f005f4de258a5985026e09425`
- Megatron Core import: `/root/Megatron-LM/megatron/core/__init__.py`
- SGLang import: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Megatron Core: `0.19.0+8c1e05747`
- Torch HIP library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- RCCL: `/opt/rocm-7.2.0/lib/librccl.so.1.0.70200`

The preinstalled Megatron and SGLang source trees are environment context.
The Miles commit above is the tested clone revision. SGLang is imported only
for provenance and is not exercised by this training fixture.

## Isolation notes

Two intermediate failures were isolated before the successful run:

1. A first collector attempted `all_gather` directly on unequal per-rank FSDP
   parameter shards and hung in the collective. The final fixture gathers the
   equal-sized FSDP flat-gradient shard and reconstructs named full gradients.
2. Megatron `TransformerConfig(num_layers=0)` raised `ZeroDivisionError`
   during initialization. The metadata-only config now uses `num_layers=1`;
   the synthetic module and compared tensors are unchanged.

Neither failure indicates a backend update mismatch. The final supported
combination shows no mismatch.

## What this does not prove

This reduced case does **not** cover:

- Tensor parallelism, pipeline parallelism, context parallelism, or MoE.
- BF16/FP16/FP8 mixed precision or low-precision collectives.
- Megatron's distributed optimizer, gradient accumulation, or Megatron FSDP.
- Production transformer layers, tokenizer data, async training, or rollout.
- Optimizer-state checkpointing or Megatron distributed checkpoint formats.
- Performance, scaling, or MI355X equivalence.

Accordingly, no fix is proposed. A production mismatch would require a
separate reduction that preserves the failing backend's precision, sharding,
optimizer, and parallelism configuration.

## Reproduction

From the Miles clone:

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
timeout 300 /opt/venv/bin/torchrun --standalone --nproc_per_node=2 \
  reports/j-f9c7332b41c8/compare_backends.py \
  --output /tmp/miles-fsdp-megatron-comparison.json
```

The command requires exactly two visible GPUs and creates only its own
torchrun workers. It does not change node-wide state or kill unrelated
processes.
