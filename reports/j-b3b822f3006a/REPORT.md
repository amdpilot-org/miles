# Multi-trajectory GRPO weighting investigation

Reference: radixark/miles issue 2378. Coordination: amdpilot-org/amdpilotv2 issue 402.

## Result

The existing production fix is present and validated. No production-code correction was needed.

- PR base (`main`): `e5125a97e1fd383f005f4de258a5985026e09425`
- Pre-fix parent: `34d01b7a3ee8407780aeba1012f0b464dfd01a31`
- Existing fix: `63d2190ae29f97e99dd2efbf5ffee2012661e71c` (`fix(rollout): normalize rewards per rollout (#2369)`)
- Shared-reward invariant: `456340f86873c3da37b2e6e87568156b64bf401b` (`fix: require shared rewards within rollouts (#2498)`)

The pre-fix path normalizes flattened sibling rows, so a logical rollout with more trajectories changes the group mean and receives excess statistical weight. Current `main` groups sibling rows by `rollout_id` inside each prompt group, requires siblings to share one reward, normalizes one value per logical rollout, and broadcasts the result to every sibling.

## Hardware and imports

This was a separate one-GPU run, not an MI355X result reinterpreted as MI350X.

- Assigned GPUs: 1
- GPU: AMD Instinct MI350X
- Architecture: `gfx950`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`

The fixture records the imported module paths in `before.json` and `current.json`:

- Current Miles reward path: `/job/miles/miles/ray/rollout/train_data_conversion.py`
- Current Miles advantage path: `/job/miles/miles/backends/training_utils/loss.py`
- Current Miles policy-loss path: `/job/miles/miles/backends/training_utils/loss_hub/losses.py`
- Pre-fix Miles paths: `/job/miles-before/miles/...`
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron namespace: `/opt/venv/lib/python3.10/site-packages/megatron`
- Megatron core: `/root/Megatron-LM/megatron/core/__init__.py`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`

The preinstalled sources are environment context; the recorded paths prove which checkout each subprocess imported.

## Reduced GPU fixture

`gpu_fixture.py` executes the actual Miles path on `cuda:0`:

1. `_post_process_rewards`
2. `_compute_rollout_mask_sums`
3. `compute_advantages_and_returns`
4. `get_sum_of_sample_mean(..., denominators=rollout_mask_sums)`
5. `policy_loss_function`
6. `Tensor.backward()`

The batch has three logical rollouts in one prompt group:

| Rollout | Reward | Sibling trajectories | Trainable tokens per trajectory |
|---:|---:|---:|---:|
| 10 | 1.0 | 2 | 2 |
| 11 | 0.5 | 1 | 2 |
| 12 | 0.25 | 1 | 2 |

Thus `rollout_ids` is `[10, 10, 11, 12]` and `rollout_mask_sums` is `[4, 4, 2, 2]`. Each rollout-10 token has loss weight `2 / 4 = 0.25`; each rollout-11 or rollout-12 token has weight `2 / 2 = 0.5`. The synthetic policy has two logits per token, old and current log-probabilities are equal (PPO ratio 1), and entropy, KL loss, TIS, and OPSM are disabled.

## Hand calculation

### Current `main`

The reward-normalization weights are one per logical rollout:

| Rollout | Reward | Group weight |
|---:|---:|---:|
| 10 | 1.00 | 1/3 |
| 11 | 0.50 | 1/3 |
| 12 | 0.25 | 1/3 |

The logical-rollout mean is:

```text
(1.0 + 0.5 + 0.25) / 3 = 0.5833333333
```

The normalized rewards and advantages are:

| Rollout | Advantage | Broadcast rows |
|---:|---:|---:|
| 10 | 1.0 - 0.5833333333 = 0.4166666667 | 2 |
| 11 | 0.5 - 0.5833333333 = -0.0833333333 | 1 |
| 12 | 0.25 - 0.5833333333 = -0.3333333333 | 1 |

For ratio 1, each token policy loss is `-advantage`. The per-rollout loss denominators make the two rollout-10 siblings jointly contribute one token-weighted rollout mean:

```text
loss = -0.4166666667 + 0.0833333333 + 0.3333333333 = 0
```

For selected token class 0, `d log_softmax[0] / d logit[0] = 0.5` and `d log_softmax[0] / d logit[1] = -0.5`. Therefore:

```text
d loss / d logit[0] = -weight * advantage / 2
d loss / d logit[1] = +weight * advantage / 2
```

The selected-class gradients are:

| Rollout | Token weight | Advantage | Gradient `[logit[0], logit[1]]` |
|---:|---:|---:|---:|
| 10 | 0.25 | 0.4166666667 | `[-0.0520833333, 0.0520833333]` |
| 11 | 0.50 | -0.0833333333 | `[0.0208333333, -0.0208333333]` |
| 12 | 0.50 | -0.3333333333 | `[0.0833333333, -0.0833333333]` |

The GPU result is loss `-5.960464477539063e-08`, which is zero within float32 roundoff. Every asserted reward and gradient matches the hand calculation within `1e-6`.

### Pre-fix parent

The flattened reward-normalization weights are one per sibling trajectory:

| Rollout | Reward | Flattened rows | Group weight |
|---:|---:|---:|---:|
| 10 | 1.00 | 2 | 2/4 = 0.50 |
| 11 | 0.50 | 1 | 1/4 = 0.25 |
| 12 | 0.25 | 1 | 1/4 = 0.25 |

The flattened mean is:

```text
(1.0 + 1.0 + 0.5 + 0.25) / 4 = 0.6875
```

The row advantages are `[0.3125, 0.3125, -0.1875, -0.4375]`. The same per-rollout loss denominators give:

```text
loss = -0.3125 + 0.1875 + 0.4375 = 0.3125
```

The selected-class gradients are:

| Rollout | Token weight | Advantage | Gradient `[logit[0], logit[1]]` |
|---:|---:|---:|---:|
| 10 | 0.25 | 0.3125 | `[-0.0390625, 0.0390625]` |
| 11 | 0.50 | -0.1875 | `[0.046875, -0.046875]` |
| 12 | 0.50 | -0.4375 | `[0.109375, -0.109375]` |

This is not merely a different scalar. Every rollout's advantage and gradient is shifted because rollout 10 contributes twice to the reward statistics.

## Evidence

- Pre-fix GPU output: `before.json`
- Current GPU output: `current.json`
- Executable fixture: `gpu_fixture.py`
- GPU identity: `gpu_identity.txt`
- Focused test output: `focused_tests.log`

The focused runs passed:

- 48 selected tests in `tests/fast/ray/rollout/test_train_data_conversion.py`
- 8 tests in `tests/fast/backends/training_utils/test_rollout_mask_sums.py`

## Reproduction

From this checkout:

```bash
export PYTHONPATH="$PWD"
export TORCHINDUCTOR_CACHE_DIR=/job/.cache/inductor
export TRITON_CACHE_DIR=/job/.cache/triton
export HF_HOME=/job/.cache/huggingface

/opt/venv/bin/python reports/j-b3b822f3006a/gpu_fixture.py \
  --expectation current \
  --output reports/j-b3b822f3006a/current.json
```

For the pre-fix run:

```bash
git worktree add --detach /job/miles-before \
  34d01b7a3ee8407780aeba1012f0b464dfd01a31
mkdir -p /job/miles-before/reports/j-b3b822f3006a
cp reports/j-b3b822f3006a/gpu_fixture.py \
  /job/miles-before/reports/j-b3b822f3006a/gpu_fixture.py

cd /job/miles-before
export PYTHONPATH="$PWD"
/opt/venv/bin/python reports/j-b3b822f3006a/gpu_fixture.py \
  --expectation legacy \
  --output reports/j-b3b822f3006a/before.json
```

The normal focused pytest command is:

```bash
/opt/venv/bin/python -m pytest \
  tests/fast/ray/rollout/test_train_data_conversion.py \
  tests/fast/backends/training_utils/test_rollout_mask_sums.py \
  -q
```

In this container, importing the root test conftest reached a root-owned preinstalled Aiter cache under `/tmp/aiter_configs` and failed with `PermissionError`. To avoid changing node-wide state, the recorded run used `pytest --noconftest`, imported the test module's direct helper conftest normally, and supplied its `ray_local_mode` fixture through a private plugin under `/job/.cache/pytest`. The unrelated `TestSplitTrainDataByDp` class was excluded because bypassing the root conftest leaves its object-store singleton fixture in an incompatible lifecycle state.

## Scope and limitations

This reduced case proves the accounting behavior of the actual Miles reward normalization, advantage construction, per-rollout loss denominator, policy loss, and backward path for unequal sibling counts. It does not prove:

- end-to-end training quality or convergence with a public model;
- every advantage estimator or optional loss term;
- multi-GPU distributed reduction (the assigned distributed case is one GPU, and that one GPU was used);
- behavior when sibling rewards differ; current `main` intentionally rejects that case through the shared-reward invariant.

No model weights were downloaded.
