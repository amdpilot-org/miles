# Opt-in differentiable top-k OPD investigation

## Result

Implemented an opt-in differentiable top-k reverse-KL loss and validated it through Miles' actual `policy_loss_function` training-forward path on one AMD Instinct MI350X (`gfx950`, capability `9:5`). The existing detached reward-shaping behavior remains the production default.

- Base commit: `e5125a97e1fd383f005f4de258a5985026e09425`
- Tested implementation commit: `589d6266cab3d7a9caea99461ffd56c52a310bcc`
- Branch: `amdpilot/j-b7ca3d727d3e`
- Runtime: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, Torch commit `7e1940d4b11fd6128be4c42ba41567ec5ab87102`
- HIP runtime: `7.2.26015-fc0010cf6a`

## Implementation

- `--opd-differentiable-top-k-loss` is default-off and requires SGLang OPD with `--opd-log-prob-top-k > 0`.
- Rollout stores the controlled selected token IDs, fixed teacher log-probabilities, fixed reward weights, and per-position counts.
- Training re-scores those IDs from current student logits using Miles' response-chunk and log-probability helpers, then adds `opd_kl_coef * reverse_kl` directly to the policy loss.
- In this mode, top-K reverse-KL is no longer subtracted from advantages. The default path still uses detached precomputed reward shaping.
- Teacher terms remain fixed training data; only the student side is differentiable.

## GPU validation

Reproduction from the repository root:

```bash
/opt/venv/bin/python -m pytest --noconftest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/backends/training_utils/loss/test_opd.py \
  tests/fast/rollout/session/test_samples_codec.py \
  tests/fast-gpu/test_opd_differentiable_topk.py -s
```

`--noconftest` avoids an unrelated root-owned Aiter cache lock under `/tmp/aiter_configs`; it does not alter the tested Miles code paths.

Final run: **58 passed**. GPU phase durations from pytest:

- Analytical plus finite-difference gradient check: `1.51 s`
- Omitted-tail control: `0.02 s`
- Two 256-step optimizer comparisons: `0.91 s` total

### Gradient checks

The fixture uses a deterministic local random logits tensor, two response positions, a 16-token vocabulary, controlled selected sets, and fixed teacher log-probabilities.

- Analytical versus autograd maximum absolute error: `1.4901161193847656e-08`
- Central finite-difference versus autograd maximum absolute error: `4.402920603752136e-05` (`epsilon=1e-3`)
- Analytical tolerance: `rtol=2e-5`, `atol=2e-6`
- Finite-difference tolerance: `rtol=2e-3`, `atol=2e-4`

### Omitted-tail control

The omitted set excludes teacher tail token `15`; the full set includes it.

- Full-set loss: `0.358527809381485`
- Omitted-tail loss: `0.11721649020910263`
- Maximum omitted-tail logit gradient: `0.01207735762000084`

This is a negative finding: omitting the tail changes the sparse estimate, and softmax normalization still couples gradient into the omitted logit. Differentiability does **not** recover the omitted tail or make the sparse estimate equal to full-vocabulary reverse KL.

### 256-step teacher/student comparison

Both modes used the same deterministic initial logits, fixed teacher terms, Adam optimizer, and 256 real forward/backward/optimizer steps. No idle loops, spin waits, or sleeps were used.

| Mode | Steps | Mean gradient norm | Final top-K estimate | Wall time |
|---|---:|---:|---:|---:|
| Differentiable direct loss | 256 | `0.3428746461868286` | `-10.153799057006836` | `0.5683727543801069 s` |
| Detached reward-shaping baseline | 256 | `0.13411492109298706` | `-0.5441506505012512` | `0.328286612406373 s` |

Initial top-K estimate: `0.23033106327056885`.

In this synthetic fixture, the direct loss produced a larger mean gradient and a lower final sparse top-K objective than detached shaping. This is a controlled comparison, not a general claim that direct loss is better for every model or dataset.

## Honest limitations

- Only the one assigned MI350X was available; no multi-GPU, tensor-parallel, context-parallel, or distributed collective case is claimed.
- Validation used synthetic/local random initialization and fixed teacher terms. No model weights were downloaded; total downloads were `0 GB`.
- No end-to-end SGLang teacher server rollout was run. The rollout term packaging and training-forward loss were validated separately with controlled fixtures.
- The weighted sparse reverse-KL estimate is not guaranteed non-negative; the observed final estimate is negative. It is the top-K objective estimate, not full-vocabulary KL.
- Multimodal expansion and distributed CP/TP behavior were not validated. The loss performs explicit count/length checks and should fail loudly rather than silently mis-slice.
- Sharing this MI350X is not a MI355X performance-equivalence claim.

## Imported paths

- Miles source: `/job/miles/miles`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP native library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- ROCm HIP native library: `/opt/rocm/lib/libamdhip64.so`
- Aiter native module observed by the environment: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Megatron source observed by the environment: `/root/Megatron-LM/megatron`
