# Multi-teacher OPD two-GPU validation

## Scope and result

This investigation validates the existing Miles multi-teacher OPD path on two assigned AMD Instinct MI350X (`gfx950`) GPUs. It does not claim that a larger topology passed.

- Base commit: `df0e677f6dd51fa551d48d37860812ece904cd8f`
- Fixture commit: `1862da9abd11a39afec66f286efb87859f8310f0`
- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python` (resolved to `/usr/bin/python3.10`)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- GPUs: 2× AMD Instinct MI350X, `gfx950`

The fixture ran 256 real optimizer steps with `world_size=2`, NCCL DDP, AdamW updates, and deterministic per-step inputs. It exercised the actual Miles hooks:

- `miles/rollout/on_policy_distillation.py`: `reward_func`, `post_process_rewards`, and teacher routing.
- `miles/backends/training_utils/loss_hub/opd.py`: `apply_opd_kl_to_advantages`.
- `miles/utils/types.py`: `Sample` OPD fields and validation.

The test replaces only the HTTP transport function with an in-process call that performs the selected teacher's real GPU forward. It does not replace the Miles OPD routing, scoring payload, reverse-KL, advantage, or optimizer logic.

## Numerical and state checks

- Teacher identity was checked on every step through the resolved URL and the URL seen by the scoring transport.
- The deterministic schedule used teacher A for 192 steps and teacher B for 64 steps.
- Routing switched from A to B at step 128 and recovered to A at step 192.
- Both teachers were frozen with `requires_grad=False`, no gradients, and exact snapshot equality at every 16-step checkpoint, both routing boundaries, and the final step.
- Student top-k selected-token log-probs and teacher log-probs for those selected tokens were checked on every step.
- Weighted reverse-KL from `post_process_rewards` matched an independent GPU calculation on every step.
- `apply_opd_kl_to_advantages` produced the expected weighted advantages on every step.
- DDP gradients matched an independent weighted-gradient calculation and were synchronized across both ranks at every 16-step checkpoint and both routing boundaries.
- Final student weights changed by `0.1836557388305664` and remained finite.

Final 256-step GPU run:

- Wall time: `3.246843376196921 s`
- OPD scoring phase: `1588.465569972992 ms`
- Backward phase: `1456.9003616273403 ms`
- Optimizer phase: `111.69643887877464 ms`
- Peak allocated GPU memory: `293698048` bytes

## Commands

The committed launcher uses a unique free TCP port, a UUID rendezvous ID, a unique torchrun log directory, a 120-second process-group timeout, and bounded c10d rendezvous timeouts:

```bash
PYTHONPATH=/job/miles /opt/venv/bin/python -m pytest --noconftest \
  -q tests/fast-gpu/test_opd_multi_teacher_two_gpu.py
```

Adjacent Miles OPD coverage also passed:

```bash
PYTHONPATH=/job/miles /opt/venv/bin/python -m pytest --noconftest \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/backends/training_utils/loss/test_opd.py
```

Result: 27 passed.

## Imported source and native paths

- Miles source: `/job/miles/miles/rollout/on_policy_distillation.py`
- Miles source: `/job/miles/miles/backends/training_utils/loss_hub/opd.py`
- Miles source: `/job/miles/miles/utils/types.py`
- Torch Python source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- ROCm root: `/opt/rocm` (resolved to `/opt/rocm-7.2.0`)
- HIP runtime: `/opt/rocm/lib/libamdhip64.so.7.2.70200`
- RCCL runtime: `/opt/rocm/lib/librccl.so.1.0.70200`

## Limitations

- The model is a small synthetic, locally initialized causal model; no model or dataset downloads were used.
- The scoring transport is in-process rather than an SGLang HTTP server. This isolates the Miles OPD hooks and routing logic from unrelated server startup behavior, but it does not validate the full SGLang serving stack.
- Normal pytest collection in this container was blocked by an unrelated Aiter lock permission under the global test conftest. The GPU launcher and adjacent OPD tests were therefore run with `--noconftest`; the fixture itself does not depend on that conftest.
- Only the assigned two-GPU topology was tested. No performance equivalence with MI355X or any larger topology is claimed.
