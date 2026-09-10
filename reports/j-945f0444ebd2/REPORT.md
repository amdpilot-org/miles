# Two-GPU batch-semantics investigation

## Result

The reduced fixture confirms the current semantics rather than finding a batch-counting defect:

| Case | Arrived samples | Effective global batch | Optimizer steps | Consumed samples | Per-sample gradient weight | Final weight | Reference | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `num_steps_per_rollout=3`, fixed global batch | 12 | 4 | 3 | 12 | `1/4` | `1.1420259475708008` | `1.142026` | `5.2429199204340193e-08` |
| Same config plus dynamic global batch | 12 | 12 | 1 | 12 | `1/12` | `0.5649999976158142` | `0.565` | `2.38418573772492e-09` |

Both cases use the same deterministic asynchronous arrival order:
`[7, 2, 11, 0, 5, 9, 1, 8, 3, 10, 4, 6]`.

The fixed case splits the drained rollout into three sequential optimizer steps of four samples.
The dynamic-global-batch case recomputes the effective global batch from the twelve arrived
samples, overriding the global batch size derived from `num_steps_per_rollout=3`, and consumes all
twelve samples in one optimizer step. The fixture compares the resulting scalar weight against an
explicit closed-form SGD reference that applies the same per-sample gradient weights and step
boundaries.

## Correction

This PR adds:

- A warning when fully-async training combines `num_steps_per_rollout > 1` with
  `max_weight_staleness`, because later optimizer steps train samples generated before the first
  update in addition to the inter-batch staleness already filtered.
- An informational message when dynamic global batch sizing is enabled in fully-async mode.
- Documentation of the derived fixed batch size, the dynamic metadata override, and the resulting
  one-step-per-drain behavior.
- A focused regression for the warning and a two-GPU executable fixture for the measured semantics.

## Reproduction

From this checkout:

```bash
/opt/venv/bin/python -m pytest --noconftest -q tests/fast/utils/test_arguments.py \
  -k 'fully_async_batch_semantics or fully_async_rejects_abort_pause_mode'

TORCH_EXTENSIONS_DIR=/job/.cache/torch_extensions \
TRITON_CACHE_DIR=/job/.cache/triton \
HF_HOME=/job/.cache/hf \
XDG_CACHE_HOME=/job/.cache \
timeout --signal=TERM --kill-after=10s 180s \
/opt/venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  reports/j-945f0444ebd2/two_gpu_async_batch_fixture.py \
  --output reports/j-945f0444ebd2/two_gpu_fixture_result.json
```

The fixture initializes an NCCL process group with a 120-second timeout and uses both assigned
MI355X GPUs. It uses a synthetic one-parameter linear model, so no model weights are downloaded.

## Evidence paths

The tested fixture imports Miles from this checkout:

- `miles`: `/job/miles/miles/__init__.py`
- `miles.backends.training_utils.data`: `/job/miles/miles/backends/training_utils/data.py`
- `miles.ray.rollout.rollout_data_conversion`: `/job/miles/miles/ray/rollout/rollout_data_conversion.py`
- `miles.ray.rollout.train_data_conversion`: `/job/miles/miles/ray/rollout/train_data_conversion.py`

Environment context recorded by the fixture:

- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron Core: `/root/Megatron-LM/megatron/core/__init__.py`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native extension: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Torch CUDA library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so`
- Torch version: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP version: `7.2.26015-fc0010cf6a`

The complete machine-readable result is in
`reports/j-945f0444ebd2/two_gpu_fixture_result.json`.

## Scope and limitations

This reduced case proves:

- The real Miles rollout post-processing, train-data conversion, DP schedule, and `DataIterator`
  split a fixed twelve-sample asynchronous arrival into the expected optimizer steps.
- Dynamic global batch metadata overrides the global batch size derived from
  `num_steps_per_rollout` and forces one optimizer step for this fixed drain.
- Manual gradient all-reduce plus explicit `1/global_batch_size` scaling produces the measured
  per-sample gradient weights and matches the closed-form sequential SGD reference within float
  error on two MI355X ranks.

It does not prove:

- End-to-end SGLang generation or Megatron actor behavior.
- Numerical behavior of `max_weight_staleness` filtering; the fixture tests the warning condition,
  not the filter itself.
- Variable physical sample counts after a custom rollout filter, compact rollouts, or multi-LoRA.
- That multi-step fully-async training is invalid; the correction is a warning because overlapped
  async workloads may legitimately accept the additional off-policyness.

## Environment note

The first focused pytest invocation failed while loading the repository conftest because Aiter
attempted to lock `/tmp/aiter_configs/bf16_tuned_gemm.csv.lock`, which is root-owned in this image.
The focused test then passed with `--noconftest`, which avoids that unrelated environment import.
The two-GPU fixture itself completed normally with private cache directories under `/job/.cache`.
