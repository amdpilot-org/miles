# ROCm LoRA pause/resume investigation

## Result

The reduced case **passed**. Across three consecutive `torch_memory_saver`
pause/resume cycles on one assigned AMD Instinct MI355X (`gfx950`), the frozen
base weight, both trainable LoRA adapter parameters, the existing forward
output, a fresh forward output after resume, and both adapter gradients were
**bitwise equal** to their pre-pause CPU snapshots. The maximum absolute
difference was `0.0` for every checked tensor.

This is evidence for the reduced Torch/TMS path only. It does not prove the
full Miles rollout/training swap, SGLang engine integration, optimizer state,
multi-GPU collectives, or long-run leak behavior.

## Reproduction

```bash
cd /job/miles
/opt/venv/bin/python reports/j-8c54e0f31383/test_lora_pause_resume.py \
  --cycles 3 --width 16384 --rank 64 --batch 64 --device-index 0 \
  --json-output reports/j-8c54e0f31383/results.json
```

The parent process launches only its own child processes with TMS preload mode,
`TMS_INIT_ENABLE=1`, and CPU backup enabled. Each child has a 180-second
subprocess timeout. The case uses one assigned GPU and therefore creates no
process group. No public model weights are downloaded.

## Fixture

- Synthetic `bfloat16` base linear layer: `16384 x 16384`, frozen.
- LoRA adapters: rank `64`, both trainable.
- Input batch: `64 x 16384`.
- One initial real GPU forward/backward, then three pause/resume cycles.
- After every resume, the fixture compares parameters, the existing output,
  adapter gradients, and a newly executed forward output against CPU snapshots.

## Memory readings

The worker recorded both process-local Torch allocator counters and global
device free memory. TMS released backing device storage while leaving Torch's
allocator accounting unchanged, which is the expected preload/remap behavior.

| Cycle | Device free gain | Process RSS gain at pause | Torch allocated | Torch reserved |
|---:|---:|---:|---:|---:|
| 1 | 636.0 MiB | 634.0 MiB | 676.0 MiB | 708.0 MiB |
| 2 | 636.0 MiB | 634.0 MiB | 676.0 MiB | 708.0 MiB |
| 3 | 636.0 MiB | 634.0 MiB | 676.0 MiB | 708.0 MiB |

For every cycle, `torch.cuda.memory_allocated()` and
`torch.cuda.memory_reserved()` were unchanged from before pause through pause.
The tracked model/input/output/gradient tensors totaled 524.0 MiB; the larger
device-free and RSS deltas include allocator backing and other region storage.
After resume, process RSS returned to within roughly 3 MiB of its before-pause
value, and TMS reported `retain_cpu_backup=False`.

Global device-free deltas can include other processes on the node, so they are
reported as raw observations rather than attributed exclusively to this worker.
The repeated, stable 636.0 MiB gain is nevertheless consistent with TMS
releasing this region's backing storage.

## Unsupported allocator behavior

The fixture records two expected failures without replacing Torch, ROCm, TMS, or
the surrounding framework stack:

- `cpu_backup_backend="mmap"` raises `ValueError` on ROCm:
  `cpu_backup_backend='mmap' is not supported on ROCm`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` causes TMS initialization
  to raise `RuntimeError`:
  `TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet.`

The tested worker uses the supported pinned CPU backup backend and leaves
expandable segments disabled.

## Environment and provenance

- Miles source commit: `e5125a97e1fd383f005d4de258a5985026e09425`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Torch allocator backend: `native`.
- Torch Memory Saver: `0.0.10b1`, commit
  `06caa534822c5b980b61733a5d79ac731f9a5f9c`, preload hook.
- SGLang: `0.5.17.dev2157+ga8e5c632f`.
- Megatron Core: `0.19.0+8c1e05747`.
- Device: AMD Instinct MI355X, `gfx950:sramecc+:xnack-`, capability `(9, 5)`.

Imported paths recorded in `results.json` include:

- Miles: `/job/miles/miles/__init__.py`
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron: `/opt/venv/lib/python3.10/site-packages/megatron`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- TMS: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver/__init__.py`

Native module paths recorded in `results.json` include:

- Torch native: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- TMS preload: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so`
- TMS torch hook: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_torch.abi3.so`

Installed source paths are environment context, not proof of the tested Miles
revision; the source commit above identifies the checkout used for this run.

## Artifacts

- `test_lora_pause_resume.py`: executable fixture.
- `results.json`: complete machine-readable run record.
- `run.log`: captured stdout from the reported run.

## Limitations

- The base weight is frozen, so this does not validate base-weight gradient
  accumulation or optimizer state across pause/resume.
- The fresh forward after each resume does not perform another backward, so it
  checks output reproducibility rather than repeated gradient accumulation.
- The case is single-rank and creates no process group; it does not test NCCL,
  RCCL, multi-GPU collectives, or bounded process-group timeouts.
- It does not start an SGLang engine or exercise Miles's rollout/training actor
  swap, only the shared Torch/TMS allocation path used by that stack.
- It does not test disk backup, CUDA graph capture, or long-running leak growth.
- It does not replace or patch the framework stack.
