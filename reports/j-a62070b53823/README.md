# Issue 1697 investigation: async step and dynamic-GBS accounting

## Result

The reduced two-GPU fixture found **no accounting defect** in the tested Miles
revision. Both schedules matched an explicit reference exactly:

| Case | Arrived samples | Active GBS | Optimizer steps | Consumed samples | Dropped | Per-sample weight | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed GBS | 12 | 4 | 3 | 12 | 0 | `1/4` | 0 |
| Dynamic GBS | 13 | 12 | 1 | 12 | 1 | `1/12` | 0 |

The dynamic case uses 13 arrivals to exercise the rounding path: Miles rounds the
global batch down from 13 to the DP multiple 12, consumes 12 samples in one optimizer
step, and leaves one tail sample unused.

This PR therefore adds the bounded clarification requested by upstream issue 1697:

- Warn when fully async mode combines `--max-weight-staleness` with an effective
  multi-step rollout, because per-batch off-policyness compounds with inter-batch
  staleness.
- Log that `--use-dynamic-global-batch-size` is redundant for the fixed fully-async
  drain count, while still valid as a way to force one optimizer step.
- Document both behaviors and keep the warning non-fatal.

## Hardware and software

- Assigned GPUs: 2.
- Devices: 2x AMD Instinct MI350X, gfx950, 270,566,162,432 bytes each.
- ROCm SMI driver: `7.1.1.31500000`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP runtime:
  `7.2.26015-fc0010cf6a`.
- Miles base commit: `e5125a97e1fd383f005f4de258a5985026e09425`.
- Imported Miles: `/job/miles/miles/__init__.py`.
- Imported SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Imported Megatron namespace:
  `/opt/venv/lib/python3.10/site-packages/megatron`.
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.

The preinstalled SGLang and Megatron sources are environment context, not proof of
the tested Miles revision. The fixture records the imported paths in both JSON results.

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=. timeout 180s /opt/venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  reports/j-a62070b53823/gpu_fixture.py \
  --mode fixed --timeout-seconds 90 \
  --output /tmp/fixed.json

PYTHONPATH=. timeout 180s /opt/venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  reports/j-a62070b53823/gpu_fixture.py \
  --mode dynamic --timeout-seconds 90 \
  --output /tmp/dynamic.json
```

The committed `fixed.json` and `dynamic.json` are the measured outputs from those
commands on the two assigned MI350X devices.

Focused argument tests:

```bash
/opt/venv/bin/python -m pytest -q --noconftest \
  tests/fast/utils/test_arguments.py::test_fully_async_multi_step_with_staleness_warns \
  tests/fast/utils/test_arguments.py::test_fully_async_dynamic_global_batch_size_is_reported_redundant
```

Result: 2 passed.

## What the fixture proves

- It uses a real two-rank NCCL process group and both assigned GPUs.
- It uses Miles' real `ParallelState`, `get_data_iterator`, and dynamic-GBS resize
  helper, not a reimplementation of the arithmetic.
- It uses a tiny one-hot Torch parameter so each consumed sample has an identifiable
  gradient weight. The measured all-reduced weights are compared elementwise with an
  explicit reference derived from DP size, active GBS, and arrival order.
- Asynchronous arrival is represented by rank-local production threads with different
  delays and rank-local arrival orders, followed by an all-gather. The measured count,
  optimizer steps, and weights are independent of that arrival order.

## What it does not prove

- It does not run the full SGLang producer, Megatron training backend, weight-update
  path, or staleness filter end to end.
- It does not use or download a public model; the synthetic one-hot model is sufficient
  for counting and gradient-weight accounting.
- It does not measure performance, and MI350X results should not be treated as MI355X
  performance.
- It does not prove that multi-step async training is algorithmically desirable; the
  warning and documentation make the compounded off-policyness explicit.

## Negative result and environment note

No Miles accounting regression was found. The normal pytest conftest path was blocked
before test collection by a pre-existing root-owned `/tmp/aiter_configs` lock used by
Aiter (`PermissionError`). The focused tests pass with `--noconftest`, which avoids that
unrelated import path. No node-wide state was changed and no broad process kill was
used.
