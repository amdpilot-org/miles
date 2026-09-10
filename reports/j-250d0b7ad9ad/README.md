# OPD frozen-teacher resume investigation

## Result

The reduced fixture reproduced the real `torch_memory_saver` limitation on one assigned AMD Instinct MI350X (`gfx950`): with CPU backup disabled, `pause()` makes weight storage unreadable and `resume()` remaps zero-filled storage. A live endpoint is therefore not evidence that frozen teacher weights were restored.

The controller change keeps the real resume hook, then explicitly reloads every offloaded frozen server cell from its configured `model_path`. It raises before rollout can continue when the path is missing or SGLang reports that the reload failed. Updatable student cells are left for the existing actor weight synchronization.

## Reproduction

```bash
cd /job/miles
MILES_OPD_FIXTURE_DIR=/job/.cache/opd-resume-fixture \
  /opt/venv/bin/python reports/j-250d0b7ad9ad/gpu_opd_resume_fixture.py
```

The parent process configures the native preload hook and launches only its own child. The child creates two tiny `nn.Linear` teacher/student models on CUDA, uses `torch_memory_saver.region(..., enable_cpu_backup=False)`, and performs three complete cycles:

1. update and checkpoint the student;
2. pause and resume both native weight regions;
3. confirm the no-backup resume is zero/uniform;
4. reject a missing-checkpoint load;
5. reload the frozen teacher and current student checkpoints;
6. compare every parameter and both outputs;
7. admit the OPD request only after all comparisons pass.

Observed result for all three cycles:

- `resume_without_backup_zeroed: true`
- `failed_load_rejected: true`
- `teacher_parameters_restored: true`
- `student_parameters_restored: true`
- `teacher_and_student_outputs_restored: true`
- `opd_request_admitted: true`

## Scope

This proves the native pause/resume failure mode and the explicit reload/admission gate for a tiny single-process, single-GPU teacher/student pair. It does not prove multi-rank SGLang serving, tensor-parallel checkpoint sharding, large-model disk bandwidth, CPU-backup mode, quality of an OPD-trained model, or MI355X performance. No public weights were downloaded and no process group was created.

## Environment record

- Miles base: `e5125a97e1fd383f005f4de258a5985026e09425`
- Miles import: `/job/miles/miles/__init__.py`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP runtime modules: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so` and `libc10_hip.so`
- SGLang: `0.5.17.dev2157+ga8e5c632f`, `/sgl-workspace/sglang/python/sglang/__init__.py`, source revision `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`
- Megatron namespace: `/opt/venv/lib/python3.10/site-packages/megatron`; `megatron.core` resolves to `/root/Megatron-LM/megatron/core/__init__.py`, source revision `8c1e05747eb612b382df2632783df5c83a853646`
- Transformer Engine: `2.17.0`, `/opt/venv/lib/python3.10/site-packages/transformer_engine/__init__.py`
- Torch Memory Saver: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver/__init__.py`
- Native preload: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so`
- GPU count: 1; product: AMD Instinct MI350X, model `0x75a0`, GFX `gfx950`, VRAM `270566162432` bytes

Preinstalled source paths and revisions are environment context. The tested Miles revision is the mirror branch/base recorded above.

## Validation

- Native GPU fixture: 3/3 cycles passed on MI350X.
- Final touched/adjacent test selection: 144 passed.
- `tests/fast/ray/rollout/test_rollout_server.py`, `test_rollout_server_fanout.py`, and `test_rollout_server_locking.py`: 50 passed.
- `tests/fast/ray/rollout/test_server_cell_state_machine.py::TestMemoryOperations`: 5 passed.
- Adjacent controller/SGLang suite: 841 passed, 6 xfailed, and 1 unrelated pre-existing failure in `test_sglang_api_client.py` because Python 3.10 exceptions do not expose `__notes__`.

The first test attempt also hit a preinstalled AITER shared `/tmp/aiter_configs` lock permission error. Reruns used a private copy under `/job/.cache/aiter` and `AITER_CONFIG_GEMM_BF16=/job/.cache/aiter/bf16_tuned_gemm.csv`; no node-wide state was changed.

## Uncertainty

The controller currently uses checkpoint reload as the restoration policy for every offloaded frozen group. That is fail-closed and does not depend on CPU backup, but it does not add an explicit CPU/disk policy and may repeat a disk reload when a user has separately enabled SGLang CPU backup. A future policy should expose that tradeoff without treating a live endpoint or finiteness alone as proof of restored weights.
