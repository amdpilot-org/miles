# LoRA pause/resume investigation — j-0c19ee5e6eeb

## Scope

This is a bounded, synthetic MI350X investigation for the AMD Q3 LoRA and
Torch Memory Saver roadmap. It mirrors the Miles Megatron actor lifecycle in
`miles/backends/megatron_utils/actor.py`: frozen base and trainable adapter
parameters are allocated in a CPU-backed `default` region, while backward
allocations are placed in an unbacked `grad_buffer` region. The fixture pauses
`grad_buffer`, pauses `default`, resumes `default`, and resumes `grad_buffer`
three times.

The literal upstream issue name `roadmap-lora-pause` was not found. The scope
follows the task description and the referenced roadmap issues
`radixark/miles#2853` and `radixark/miles#2025`. Coordination tracker:
`amdpilot-org/amdpilotv2#402`.

## Reproduction

From this checkout:

```bash
export LD_PRELOAD=/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so
/opt/venv/bin/python tests/fast-gpu/lora_pause_fixture.py
```

The pytest wrapper configures the same preload for its own child process:

```bash
/opt/venv/bin/python -m pytest --noconftest -q tests/fast-gpu/test_lora_pause_rocm.py -vv
```

The local run used `--noconftest` because the repository conftest imports the
preinstalled SGLang/Aiter stack and then failed on an existing
`/tmp/aiter_configs/bf16_tuned_gemm.csv.lock` owned by another user. That lock
is node-wide state and was not removed or modified. The wrapper itself uses one
owned subprocess with a 180-second timeout.

## Environment

- Runtime: `amdpilotv2/miles-job:gbt350-d957-20260909`
- PR base: `main` at `e5125a97e1fd383f005f4de258a5985026e09425`
- GPU: one assigned AMD Instinct MI350X, Device ID `0x75a0`, GUID `51966`,
  serial `692517020474`, `gfx950`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Torch Memory Saver: `0.0.10b1`
- Miles module: `/job/miles/miles/__init__.py`
- SGLang module: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron module: `/opt/venv/lib/python3.10/site-packages/megatron`
- Torch module: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch Memory Saver module:
  `/opt/venv/lib/python3.10/site-packages/torch_memory_saver/__init__.py`
- Native preload hook:
  `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so`
- Native torch hook:
  `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_torch.abi3.so`

The SGLang and Megatron paths are recorded as preinstalled environment context.
The reduced allocator fixture does not exercise those frameworks. The tested
Miles revision is the checkout commit named above, not merely the preinstalled
package metadata.

## Result

The synthetic model has a frozen `4096 x 8192` base linear layer and trainable
rank-64 LoRA `A`/`B` projections. It uses real bf16 HIP GEMMs, a real backward
pass, and the installed Torch Memory Saver preload hook.

All three cycles passed exact (`torch.equal`) checks for:

- frozen base parameters
- trainable adapter parameters
- adapter/base parameter values
- recomputed forward output
- adapter gradients

Measured physical release, using `torch.cuda.mem_get_info`, was repeatable:

| Cycle | Grad-buffer release | Default-region release | `empty_cache` release |
|---:|---:|---:|---:|
| 0 | 122.0 MiB | 66.0 MiB | 20.0 MiB |
| 1 | 122.0 MiB | 66.0 MiB | 20.0 MiB |
| 2 | 122.0 MiB | 66.0 MiB | 20.0 MiB |

The tracked parameter tensors total 65.5 MiB and adapter gradients total
1.5 MiB. The 122 MiB grad-buffer release is allocator-segment granularity, not
an exact gradient-byte measurement. RSS increased by roughly 311–327 MiB while
paused; that includes the CPU backup, fixture snapshots, Python/runtime
overhead, and allocator behavior, so it is not a pure parameter-backup cost.

Raw machine-readable evidence is in
`reports/j-0c19ee5e6eeb/lora-pause-results.json`.

## Allocator behavior and limitations

- Torch Memory Saver's `cpu_backup_backend="mmap"` is unsupported on ROCm. The
  observed error is exactly: `cpu_backup_backend='mmap' is not supported on
  ROCm`. The fixture therefore uses the default pinned host backup.
- While paused, Torch's `allocated_bytes` and `reserved_bytes` counters remained
  unchanged even though device free memory increased. Torch allocator counters
  do not describe this physical release and must not be used as the release
  metric.
- `torch.cuda.empty_cache()` released 20 MiB of an inactive segment in each
  cycle. That is additional allocator behavior, not a replacement for the
  Miles/Torch Memory Saver stack and not proof that active paused regions were
  discarded.
- Device free bytes and process RSS are reported alongside Torch counters. This
  avoids claiming release from an allocator statistic that does not observe it.

## What this does not prove

- It does not run a full Miles actor, Megatron training step, optimizer update,
  checkpoint, SGLang rollout, adapter synchronization, or multi-LoRA service.
- It does not use a public model or download weights; the synthetic case is
  deliberately reduced.
- It does not create a distributed process group, so no process-group timeout
  was applicable. The one assigned GPU is used directly.
- Exact bf16 reproducibility under this deterministic synthetic case does not
  establish performance or numerical equivalence on MI355X. Sharing `gfx950`
  does not make MI350X and MI355X performance comparable.
- It does not prove that every Miles allocation is tagged correctly; it tests
  the two tags and lifecycle used by the LoRA actor path.
- It does not merge or modify any workload PR.

No broad process kill or node-wide state change was used. Build and model cache
directories were kept outside the checkout under `/job/miles-build-cache` and
`/job/miles-model-cache`.
