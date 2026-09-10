# Reduced MTP weight-sync fixture

This records the bounded investigation for `radixark/miles` issue `roadmap-mtp-sync` and coordination tracker `amdpilot-org/amdpilotv2#402`.

This job adds a two-GPU Torch/ROCm test for a reduced target model plus draft/MTP head. The fixture uses `DistributedDataParallel`, NCCL/RCCL collectives, and a bounded 60-second process-group timeout.

## What the fixture proves

- Target and draft parameters are synchronized across both ranks after a full optimizer update.
- Target and draft forward outputs match across both ranks after the full update.
- Target and draft update versions match across both ranks and correspond to the same optimizer step.
- A partial update on rank 0 desynchronizes both parameter sets, forward outputs, and update versions across ranks.
- Broadcasting rank 0’s target/draft parameters and update versions to rank 1 recovers synchronization and restores matching forward outputs and update versions.

## What the fixture does not prove

- It does not measure full-model speculative-decoding throughput.
- It does not exercise the production SGLang weight-update protocol, Megatron optimizer sharding, or a public checkpoint.
- It uses a synthetic two-layer linear fixture rather than a full target/draft transformer.
- It verifies synchronization semantics under a controlled partial-update failure, not general fault tolerance or crash recovery.

## Reproduction and result

Run the fixture with both assigned GPUs:

```bash
cd /job/miles
export PYTHONPATH=/job/miles
export HIP_VISIBLE_DEVICES=0,1
/opt/venv/bin/python -m pytest --noconftest -q -s tests/fast-gpu/test_mtp_weight_sync.py
```

Observed result on 2026-09-10:

```text
1 passed in 14.35s
```

The normal pytest invocation without `--noconftest` does not reach this fixture. Repository-wide test collection imports SGLang through `tests/conftest.py`, and AITER attempts to create `/tmp/aiter_configs/bf16_tuned_gemm.csv.lock`, failing with `PermissionError: [Errno 13] Permission denied`. This is an environment/conftest import failure, not evidence about the fixture. `--noconftest` is used only to isolate this bounded test from that unrelated import path.

## Environment record

- Hardware: 2 assigned AMD Instinct MI350X GPUs, `gfx950`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Miles source: `/job/miles/miles/__init__.py`.
- SGLang source: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Megatron core source: `/root/Megatron-LM/megatron/core/__init__.py`.

The preinstalled Miles, SGLang, and Megatron sources are environment context, not proof of the tested revision. The fixture itself depends only on Torch and Torch distributed.
