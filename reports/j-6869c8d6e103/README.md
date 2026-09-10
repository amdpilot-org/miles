# J-6869c8d6e103: frozen OPD teacher lifecycle

## Result

This change makes frozen SGLang engines restore their weights after a colocated
weight release/resume and blocks rollout admission until that restoration succeeds.
Updatable engines continue to wait for actor weight synchronization.

The reduced GPU fixture uses a tiny teacher/student pair and the real
`torch_memory_saver.pause()` / `resume()` hooks. It does not replace the lifecycle with a
simulation. On the assigned AMD Instinct MI355X:

- a teacher allocated in a CPU-backed memory-saver region survived three pause/resume
  cycles with bit-identical parameters and outputs;
- a teacher allocated without backup came back with all-zero parameters after one cycle;
- a missing checkpoint failed and left rollout admission blocked;
- a subsequent valid checkpoint load restored the teacher;
- a student update then produced the teacher output while the teacher remained unchanged.

## Lifecycle change

- Frozen models default to `weights_backup_mode: reload`, which calls SGLang's existing
  `update_weights_from_disk()` after weight memory is resumed.
- `weights_backup_mode: cpu` is accepted only when the group also sets the real SGLang
  `enable_weights_cpu_backup: true` override.
- `weights_backup_mode: none` fails closed for frozen engines and is the implicit actor-sync
  mode for updatable engines.
- `InferenceController.prepare_rollout()` refuses admission if any addressable cell has not
  restored its weights.
- The current SGLang adapter does not expose `torch_memory_saver`'s disk-backup region, so
  this change intentionally does not advertise a `disk` mode.

## Reproduction

Run the real memory-saver probe:

```bash
SGLANG_USE_AITER=0 /opt/venv/bin/python \
  tools/opd_teacher_memory_saver_probe.py --cycles 3
```

Run the focused lifecycle and configuration tests:

```bash
SGLANG_USE_AITER=0 /opt/venv/bin/python -m pytest -q \
  tests/fast/ray/rollout/test_config_matrix.py \
  tests/fast/ray/rollout/test_inference_controller.py \
  tests/fast/ray/rollout/test_rollout_server.py
```

The focused command passed 106 tests and the inference spec suite passed 67 tests. A broader
rollout run passed 510 of 511 tests; the one failure is the pre-existing Python 3.10
`asyncio.TimeoutError` versus builtin `TimeoutError` assertion in
`test_inference_controller_tick.py`, unrelated to this change.

## Environment and imported paths

- Base commit: `8d9826eacc8b5c279546f96711bb401b7f62c54c`
- GPU: one AMD Instinct MI355X, gfx950
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`,
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Miles under test: `/job/miles/miles/__init__.py`
- SGLang: `0.5.17.dev2157+ga8e5c632f`,
  `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron namespace: `/opt/venv/lib/python3.10/site-packages/megatron`
- Transformer Engine: `2.17.0`,
  `/opt/venv/lib/python3.10/site-packages/transformer_engine/__init__.py`
- Torch memory saver:
  `/opt/venv/lib/python3.10/site-packages/torch_memory_saver/__init__.py`
- Memory-saver preload:
  `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so`

These paths record the runtime actually imported by the probe and tests. Preinstalled source
trees are environment context, not proof of the Miles revision under test.

## What this does and does not prove

The probe proves the memory-saver boundary on gfx950: without an armed backup, pause/resume
returns zero tensors; with CPU backup, parameters and outputs survive repeated cycles. The
unit tests prove that Miles reloads frozen engines, blocks admission after a failed reload,
and admits after a valid reload.

The reduced case does not prove a full SGLang OPD training run, large-teacher host-memory
feasibility, checkpoint-format compatibility, or end-to-end quality. It also does not test a
multi-GPU process group because the assignment provides one GPU. No model weights were
downloaded, no node-wide state was changed, and no broad process kill was used.

## Roadmap fit and open questions

Upstream issue 1958 describes a live teacher endpoint returning uniform-vocabulary scores
after weight release/resume. This change closes that silent failure mode before rollout
admission. It is a correctness primitive for the true-on-policy and long-horizon recipe work
in upstream roadmap issue 2853, not a claim that those larger recipes are now qualified.

The AMD Q3 2025 roadmap was not independently re-fetched during this bounded run. Remaining
open questions are full SGLang OPD coverage, a real SGLang disk-backup adapter, large-model
host-memory sizing, and multi-GPU process-group behavior.
