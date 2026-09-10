# Reduced two-GPU target/MTP synchronization fixture

## What this change does

- Adds `tests/e2e/amd/test_mtp_sync_two_gpu.py`.
- Uses two `gfx950` GPUs through `torchrun` and NCCL.
- Trains a reduced target and MTP head jointly for two optimizer steps.
- Synchronizes both parameter sets into a separate rollout replica.
- Compares target and MTP forward outputs after the full update.
- Injects a target-only partial update and verifies the MTP branch is stale.
- Performs a full resync and verifies both branches recover to the same optimizer step.
- Records imported `miles`, `sglang`, `megatron.core`, `torch`, and `torch._C` paths.

## Observed module provenance

The run printed:

```text
miles=/root/miles/miles/__init__.py
sglang=/sgl-workspace/sglang/python/sglang/__init__.py
megatron.core=/root/Megatron-LM/megatron/core/__init__.py
torch=/opt/venv/lib/python3.10/site-packages/torch/__init__.py
torch._C=/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so
```

These paths are environment context only. They do not by themselves prove which
source revision produced the imported modules.

## Reproduction

```bash
torchrun --nproc_per_node=2 tests/e2e/amd/test_mtp_sync_two_gpu.py
```

## What this proves

- Both target and MTP parameters can be synchronized together.
- The synchronized replica's target and MTP forward outputs match the training model.
- The replica's target and MTP step counters match the optimizer step.
- A target-only partial update is detectable through the MTP forward output.
- A full resync recovers both branches and advances the rollout weight version.

## What this does not prove

- It does not measure full-model speculative-decoding throughput.
- It does not exercise a production SGLang rollout engine, Megatron weight conversion, or the production `WeightUpdater`.
- It does not validate multi-step training stability or acceptance length.
- It uses a synthetic reduced model, not a public checkpoint.
