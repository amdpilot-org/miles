# Bounded per-sample abort/refill GPU investigation

## Status and scope

This is an operator-scoped, opt-in investigation and regression fixture for read-only reference [radixark/miles#2800](https://github.com/radixark/miles/issues/2800), “Agentic rollout: drop infra-aborted samples per sample (refill) instead of per group.”

The upstream proposal remains parked. Its preferred shape is bounded per-sample refill with fallback to today's group drop after the retry cap is exhausted. This investigation does not change that status, does not change production defaults, and does not enable a global refill policy. It only measures the existing conversion, advantage, loss, collective, and optimizer paths under controlled policy simulations.

No candidate production change is assumed correct. The committed fixture is report-only and preserves current behavior.

## What was measured

The fixture ran five controlled cases for 128 real optimizer steps each, for 640 total optimizer steps:

- `no_abort`: smallest real GPU baseline, run first.
- `group_drop`: current group-level abort behavior.
- `sample_refill`: bounded per-sample retry, with fallback to group drop.
- `survivor_keep`: variable-size groups retaining at least two survivors.
- `remove_sample`: negative control that retains aborted samples but zeroes their loss/reward.

Each step used controlled prompt groups of sizes 2, 3, 4, and 8. Abort probability was length-dependent:

```text
0.01 + 0.01 * response_length
```

Sample retries were capped at two attempts per aborted slot. Group regeneration was capped at four attempts per prompt group.

The fixture used a synthetic locally initialized model, so downloads were zero bytes. It did not use idle loops, spin waits, or sleeps as substitutes for GPU work.

## Reproduction

From the repository root on one assigned CUDA/ROCm device:

```bash
/opt/venv/bin/python reports/j-ed983d53c34d/gpu_fixture.py \
  --output reports/j-ed983d53c34d/results.json
```

The committed `results.json` is the complete 4.7 MB evidence artifact. It records every step, retained identity, group baseline, gradient weighting record, retry account, timing, and parameter-norm checkpoint used by this report.

## Runtime and commits

- GPU: 1 assigned AMD Instinct MI350X, `gfx950` capability `(9, 5)`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.
- Python: `3.10.12`.
- Distributed: NCCL, world size 1, unique file rendezvous, 30-second process-group timeout.
- Model: synthetic local random initialization; zero downloads.
- PR base: `e5125a97e1fd383f005f4de258a5985026e09425`.
- GPU-tested repository HEAD: `40a10df38cacdcc48ca5d7ea1b8eff12da562d6b`.
- GPU run state: repository HEAD above with `gpu_fixture.py` and `results.json` as untracked working-tree artifacts; the final PR commit adds those artifacts and this report.

Only the one assigned MI350X was used. This is not a MI355X performance-equivalence claim and is not evidence that a larger distributed topology passed.

## Actual Miles paths exercised

The fixture imported and exercised these repository paths:

- `miles/ray/rollout/rollout_data_conversion.py` through `postprocess_rollout_data`.
- `miles/ray/rollout/train_data_conversion.py` through `convert_samples_to_train_data`.
- `miles/backends/training_utils/loss.py` through `compute_advantages_and_returns`.
- `miles/backends/training_utils/loss_hub/losses.py` through `policy_loss_function`.
- `miles/backends/training_utils/cp_utils.py` through `get_sum_of_sample_mean`.
- `miles/backends/training_utils/loss_hub/logit_processors.py` through `get_log_probs_and_entropy`.
- `miles/utils/types.py` through `Sample`.

The runtime also imported:

- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Torch native: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.

Each training step performed model forward, policy-loss backward, NCCL `all_reduce`, and AdamW optimization on the GPU. Repeated cycles tested state drift; parameter norms were recorded at optimizer-step checkpoints 0, 1, 33, 65, 97, and 128.

## Numerical checks

The fixture passed all of these checks:

- Every retained `rollout_id` was unique.
- Retained statuses were completed for every case except the `remove_sample` negative control.
- No aborted sample was retained by `group_drop`, `sample_refill`, or `survivor_keep`.
- Each group's normalized baseline was zero within `1e-5`.
- Each GRPO advantage equaled its normalized reward within `1e-6`.
- Rollout mask sums and per-sample loss weights were recorded and checked.
- Loss was finite at every step.
- NCCL `all_reduce` was the identity operation on world size 1.
- Parameter norm drift was greater than `1e-8` for every case.
- Retry and group-attempt accounting respected the configured bounds.

The `remove_sample` case intentionally retained aborted samples with zeroed loss masks. This negative control demonstrates the false-negative reward bias called out in the upstream issue; it is not a proposed policy.

## GPU phase timings

All timings below are mean milliseconds over the 128 recorded steps. The first `no_abort` step includes compile/initialization outliers, so the steady-state row excludes step 0. Complete min/max values remain in `results.json`.

| Case | Rollout conversion | Forward/backward | Collective | Optimizer |
|---|---:|---:|---:|---:|
| `no_abort` | 0.311 | 33.874 | 12.828 | 0.785 |
| `group_drop` | 0.355 | 7.727 | 0.222 | 0.382 |
| `sample_refill` | 0.320 | 7.882 | 0.233 | 0.401 |
| `survivor_keep` | 0.313 | 7.587 | 0.230 | 0.390 |
| `remove_sample` | 0.305 | 7.836 | 0.227 | 0.390 |

Excluding the first baseline step:

| Case | Rollout conversion | Forward/backward | Collective | Optimizer |
|---|---:|---:|---:|---:|
| `no_abort` | 0.308 | 7.941 | 0.208 | 0.348 |
| `group_drop` | 0.355 | 7.726 | 0.222 | 0.381 |
| `sample_refill` | 0.319 | 7.881 | 0.233 | 0.400 |
| `survivor_keep` | 0.312 | 7.586 | 0.230 | 0.390 |
| `remove_sample` | 0.305 | 7.836 | 0.227 | 0.389 |

These timings are evidence for this one-GPU fixture only. They are not a production performance result.

## Retry and retention accounting

| Case | Attempts | Aborts | Retained | Retry/refill result |
|---|---:|---:|---:|---|
| `no_abort` | 2,176 | 0 | 2,176 | No aborts. |
| `group_drop` | 3,215 | 224 | 2,112 | 197 groups discarded; 8 prompt groups unfilled after four attempts. |
| `sample_refill` | 2,288 | 112 | 2,176 | 106 successful refills; 6 retry aborts; no retry cap exhaustion. |
| `survivor_keep` | 2,201 | 126 | 2,065 | 111 aborted samples dropped; 12 groups discarded. |
| `remove_sample` | 2,176 | 143 | 2,176 | 143 aborted samples retained but zeroed. |

Per-step records retain `sample_index`, `rollout_id`, source (`initial`, `group_retry`, or `sample_retry`), response length, true reward, abort probability, status, retention, and loss-zeroing state. Group records retain raw rewards, raw baselines, normalized rewards, normalized baselines, sample indices, and rollout IDs.

## Length-dependent bias

Because abort probability increased with response length, aborted attempts were longer than all attempts. The measured retained-sample biases relative to all attempts were:

| Case | Retained length bias | Retained reward bias | Mean group raw-baseline delta vs `no_abort` |
|---|---:|---:|---:|
| `group_drop` | -0.049770 | -0.006287 | +0.010140 |
| `sample_refill` | -0.040885 | -0.005037 | +0.004044 |
| `survivor_keep` | -0.036804 | -0.004598 | -0.005669 |
| `remove_sample` | 0.000000 | 0.000000 | -0.057321 |

The `no_abort` baseline had zero selection bias by construction.

The negative length bias shows that all abort-aware retention shapes can shift retained trajectories toward shorter responses when infrastructure aborts depend on trajectory length. Bounded sample refill reduced rollout attempts relative to group drop (2,288 versus 3,215) and slightly reduced retained length/reward bias, but it did not eliminate the bias. Survivor retention also retained a measurable negative bias.

The `remove_sample` negative control had no retained-sample selection bias, but zeroing aborted rewards produced a large false-negative group-baseline shift and mean loss of `-0.637728`. This supports the upstream issue's requirement not to train infrastructure aborts as reward zero.

## Focused test validation

The following focused tests passed under `--noconftest`:

```bash
/opt/venv/bin/python -m pytest --noconftest -q \
  tests/fast/rollout/inference_rollout/test_sample_completion_backfill.py \
  tests/fast/backends/training_utils/test_rollout_mask_sums.py
```

Result: 27 passed.

The Ray conversion test was also run with a temporary local Ray fixture plugin:

```bash
PYTHONPATH=/tmp /opt/venv/bin/python -m pytest --noconftest -q \
  tests/fast/ray/rollout/test_train_data_conversion.py
```

Result: 53 passed and 4 errors when run in one process because of Ray object-store singleton reuse. The four affected split tests then passed in isolated processes. The root conftest import also encountered an environment-owned Aiter lock under `/tmp/aiter_configs`; no node-wide state was changed.

## Honest limitations

- This is a synthetic, locally initialized fixture, not a production model rollout.
- Only one assigned MI350X and world size 1 were tested.
- No multi-GPU, multi-node, fully-async buffer, or real inference-server abort path was tested.
- The fixture simulates abort decisions after sample creation; it does not implement or validate the upstream prerequisite that classifies infrastructure aborts at the generation layer.
- Timing includes a small synthetic model and is not a production performance benchmark.
- The GPU evidence was generated at draft commit `40a10df38cacdcc48ca5d7ea1b8eff12da562d6b` with the fixture and result artifact untracked; the final PR commit records and commits that evidence without rerunning or altering it.
- No production default, global refill policy, or upstream status was changed.
