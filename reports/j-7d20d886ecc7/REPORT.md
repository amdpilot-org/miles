# Multi-trajectory GRPO weighting investigation

Reference: radixark/miles issue 2378. Coordination: amdpilot-org/amdpilotv2 issue 402.

## Result

The existing fix is present and validated. No production-code correction was needed.

- Current `main`: `8d9826eacc8b5c279546f96711bb401b7f62c54c`
- Pre-fix parent: `34d01b7a3ee8407780aeba1012f0b464dfd01a31`
- Existing fix: `63d2190a` (`fix(rollout): normalize rewards per rollout (#2369)`)
- Follow-up invariant fix: `456340f8` (`fix: require shared rewards within rollouts (#2498)`)

The pre-fix path normalizes flattened sibling rows, so a logical rollout with more trajectories changes the group mean and receives excess statistical weight. Current `main` deduplicates by `rollout_id` within each prompt group, normalizes one reward per logical rollout, and broadcasts that advantage to all siblings.

## GPU fixture

`gpu_fixture.py` runs the actual Miles reward and loss path on one AMD Instinct MI355X (`cuda:0`, gfx950):

- `_post_process_rewards`
- `_compute_rollout_mask_sums`
- `compute_advantages_and_returns`
- `get_sum_of_sample_mean(..., denominators=rollout_mask_sums)`
- `policy_loss_function`

The reduced batch has three logical rollouts in one prompt group:

| Rollout | Reward | Sibling trajectories | Trainable tokens per trajectory |
|---|---:|---:|---:|
| 10 | 1.0 | 2 | 2 |
| 11 | 0.5 | 1 | 2 |
| 12 | 0.25 | 1 | 2 |

`rollout_mask_sums` is therefore `[4, 4, 2, 2]`. Each rollout-10 token has loss weight `2 / 4 = 0.25`; each rollout-11 or rollout-12 token has weight `2 / 2 = 0.5`. The synthetic policy has two logits per token, old and current log-probabilities are equal (PPO ratio 1), and entropy, KL loss, TIS, and OPSM are disabled.

## Hand calculation

### Current `main`

The logical-rollout mean is:

```text
(1.0 + 0.5 + 0.25) / 3 = 0.5833333333
```

Advantages are:

| Rollout | Advantage | Broadcast rows |
|---|---:|---:|
| 10 | `1.0 - 0.5833333333 = 0.4166666667` | 2 |
| 11 | `0.5 - 0.5833333333 = -0.0833333333` | 1 |
| 12 | `0.25 - 0.5833333333 = -0.3333333333` | 1 |

For a constant advantage and ratio 1, the policy loss is `sum_i weight_i * (-advantage_i)`. Each logical rollout contributes one token-weighted mean, so the current loss is:

```text
-0.4166666667 + 0.0833333333 + 0.3333333333 = 0
```

The GPU result is `-5.960464477539063e-08`, which is zero within float32 roundoff.

For target token class 0, `d log_softmax[0] / d logit[0] = 0.5`. Therefore:

```text
d loss / d logit[0] = -weight * advantage / 2
d loss / d logit[1] = +weight * advantage / 2
```

This gives the selected-class gradients:

| Rollout | Token weight | Advantage | Gradient `[logit[0], logit[1]]` |
|---|---:|---:|---:|
| 10 | 0.25 | 0.4166666667 | `[-0.0520833333, 0.0520833333]` |
| 11 | 0.5 | -0.0833333333 | `[0.0208333333, -0.0208333333]` |
| 12 | 0.5 | -0.3333333333 | `[0.0833333333, -0.0833333333]` |

### Pre-fix parent

The flattened mean counts rollout 10 twice:

```text
(1.0 + 1.0 + 0.5 + 0.25) / 4 = 0.6875
```

The resulting row advantages are `[0.3125, 0.3125, -0.1875, -0.4375]`. The loss is:

```text
-0.3125 + 0.1875 + 0.4375 = 0.3125
```

The selected-class gradients are:

| Rollout | Token weight | Advantage | Gradient `[logit[0], logit[1]]` |
|---|---:|---:|---:|
| 10 | 0.25 | 0.3125 | `[-0.0390625, 0.0390625]` |
| 11 | 0.5 | -0.1875 | `[0.046875, -0.046875]` |
| 12 | 0.5 | -0.4375 | `[0.109375, -0.109375]` |

Thus the pre-fix baseline is not merely a different scalar: every rollout's advantage and gradient is shifted because rollout 10 contributes twice to the reward statistics.

## Evidence

- Pre-fix GPU output: `before.json`
- Current GPU output: `current.json`
- Executable fixture: `gpu_fixture.py`

Both runs used:

- GPU: 1 assigned AMD Instinct MI355X, gfx950
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- No model download and no public-model weights

Recorded current-run module paths include:

- Miles reward path: `/job/miles/miles/ray/rollout/train_data_conversion.py`
- Miles loss path: `/job/miles/miles/backends/training_utils/loss.py`
- Miles policy loss path: `/job/miles/miles/backends/training_utils/loss_hub/losses.py`
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron namespace: `/opt/venv/lib/python3.10/site-packages/megatron`
- Megatron core: `/root/Megatron-LM/megatron/core/__init__.py`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`

The pre-fix JSON records the corresponding `/job/miles-before/miles/...` paths, proving that run imported the detached parent worktree rather than current `main`.

## Reproduction

From the Miles checkout:

```bash
export PYTHONPATH="$PWD"
export TORCHINDUCTOR_CACHE_DIR=/job/.cache/inductor
export TRITON_CACHE_DIR=/job/.cache/triton
export HF_HOME=/job/.cache/huggingface

/opt/venv/bin/python reports/j-7d20d886ecc7/gpu_fixture.py \
  --expectation current \
  --output reports/j-7d20d886ecc7/current.json

git worktree add --detach /job/miles-before 63d2190a^
PYTHONPATH=/job/miles-before /opt/venv/bin/python \
  /job/miles/reports/j-7d20d886ecc7/gpu_fixture.py \
  --expectation legacy \
  --output /job/miles/reports/j-7d20d886ecc7/before.json
```

The fixture asserts the hand-derived rewards, loss, and gradients with `atol=1e-6`; it exits nonzero if either revision drifts.

Existing targeted tests were also run:

```bash
export PYTHONPATH="$PWD"
export AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv
/opt/venv/bin/python -m pytest -q \
  tests/fast/ray/rollout/test_train_data_conversion.py \
  tests/fast/backends/training_utils/test_rollout_mask_sums.py
```

Result: **65 passed**.

The first normal pytest attempt failed during conftest import because Aiter attempted to create a lock under the root-owned `/tmp/aiter_configs` directory. Setting `AITER_CONFIG_GEMM_BF16` to one existing tuned CSV avoids the multi-file merge and lock without changing node-wide state.

## What this proves and does not prove

**Proves:**

- The actual pre-fix Miles reward path gives excess weight to a logical rollout with two sibling trajectories.
- The actual current reward path counts each logical rollout once and broadcasts one advantage to all siblings.
- The existing `rollout_mask_sums` loss reducer gives each logical rollout one token-weighted loss contribution.
- The resulting loss and selected-logit gradients match explicit hand calculations on gfx950.

**Does not prove:**

- Full-model training quality or convergence.
- Multi-rank DP/CP all-reduce behavior; this was a one-GPU, single-process fixture and created no process group.
- GRPO standard-deviation normalization; the fixture disables it to keep the arithmetic exact.
- KL, entropy, TIS, OPSM, clipping, or off-policy ratio behavior; all are disabled or held at ratio 1.
- Prompt-group handling for every rollout producer; the fixture uses explicit `group_index` and `rollout_id`.

The existing fast tests cover many of those adjacent paths, including prompt boundaries, shared-reward validation, standard-deviation normalization, and loss-mask aggregation.
