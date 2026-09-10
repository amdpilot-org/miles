# P2P weight-transfer failure investigation

## Result

The reduced two-GPU fixture confirms the fail-open boundary described in radixark/miles#1731 and verifies the corrected behavior:

- Rank 1 injects a bounded Mooncake-shaped failure by returning `-1` from `batch_transfer_sync_write`.
- Both trainer ranks observe the same failure through the Gloo consensus before either raises.
- The failed update reaches `pause` and `begin`, but never reaches `end`, `set_weight_version`, or `resume`.
- The consumer remains un-finalized and cannot claim weight version 1.
- The failed updater refuses reconnection and requires replacement.
- A fresh updater performs a successful synchronization; the consumer is finalized, ready version 1 is published, generation resumes, and the claim succeeds.

The successful run used both assigned AMD Instinct MI355X GPUs (gfx950, capability `(9, 5)`). Process groups were bounded to 20 seconds and the transfer manager to 2 seconds. No model weights were downloaded.
The focused CPU suite passes 9 tests.

## Reproduction

From the repository root:

```bash
/opt/venv/bin/python -m pytest -q --noconftest \
  tests/fast/backends/training_utils/weight_update/test_p2p_transfer.py

FIXTURE_OUTPUT=$PWD/reports/j-65a08c0eb467/gpu_result.json \
  /opt/venv/bin/python -m torch.distributed.run \
    --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
    reports/j-65a08c0eb467/gpu_p2p_failure_fixture.py
```

The GPU fixture drives the real `WeightUpdater.update_weights()` lifecycle and `UpdateWeightP2P` methods. Its tiny model has one four-element GPU tensor. The fake transfer engine returns the injected failure on rank 1 and performs a real GPU copy on successful recovery. The consumer is a deterministic state machine rather than an SGLang engine.

## Evidence

- `gpu_result.json` records both ranks, device identity, failure messages, lifecycle events, consumer state, claims, recovered tensor values, and imported module paths.
- `gpu_run.txt` records the successful two-process run.
- The focused CPU suite covers task exceptions, timeout ownership, immediate-batch timeout, submission refusal after failure, remote-rank consensus, stopping later writes, updater reuse refusal, successful cleanup, and suppression of finalization.

## Tested revision and imports

- Base commit: `e5125a97e1fd383f005f4de258a5985026e09425`; the working tree includes this PR's changes.
- Miles: `/job/miles/miles/__init__.py`
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron Core: `/root/Megatron-LM/megatron/core/__init__.py`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Aiter: `/sgl-workspace/aiter/aiter/__init__.py`
- Mooncake: `/opt/venv/lib/python3.10/site-packages/mooncake/__init__.py`

Preinstalled sources are environment context, not proof of the tested Miles revision. The fixture explicitly puts `/job/miles` first on `sys.path`.

## What this does and does not prove

This reduced case proves that the actual updater/protocol boundary is fail-closed under one trainer-rank transfer failure, that all ranks agree before raising, that finalization and ready-version publication are skipped, and that a fresh updater can recover.

It does not exercise real Mooncake RDMA, a live SGLang rollout engine, multi-node placement, engine-side reset/retry policy, or a GPU timeout. Those remain follow-up integration concerns. The timeout path is covered deterministically on CPU with unresolved futures.

## Preserved failed attempts

- The first CPU pytest invocation failed during collection because root-owned `/tmp/aiter_configs` prevented Aiter from creating a lock. The focused suite passes with `--noconftest`; the test and GPU fixture bypass Aiter's write-back for this tiny model, which does not use Aiter kernels.
- The first GPU run accidentally imported Miles from `/root/miles` and also exposed a fixture hook signature error. The fixture now inserts the working clone at the front of `sys.path`; the corrected run imports `/job/miles`.
- The first version of the finalization test omitted fixture-only lifecycle attributes. It was corrected and the complete focused suite passes.
- `ruff`, `isort`, and `black` are not installed in the runtime venv, so only syntax, diff whitespace, focused tests, and the GPU fixture were run.
