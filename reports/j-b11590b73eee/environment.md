## Environment variables

| Variable | Value |
| --- | --- |
| `PYTORCH_ROCM_ARCH` | `gfx950` |
| `MILES_HARDWARE_PLATFORM` | `rocm` |
| `GPU_ARCH` | `gfx950` |
| `SGLANG_USE_AITER` | `1` |
| `HIP_FORCE_DEV_KERNARG` | `1` |

## Source trees

- All three requested trees are present.

| Tree | Commit | Writable |
| --- | --- | --- |
| `/root/miles` | `b1229404c` | Yes; created and deleted a file |
| `/sgl-workspace/sglang` | `4e230c3d85` | Yes; created and deleted a file |
| `/root/Megatron-LM` | `235952df6` | Yes; created and deleted a file |

## Python imports

- `miles` resolves to `/root/miles/miles/__init__.py`.
- `sglang` resolves to `/sgl-workspace/sglang/python/sglang/__init__.py`.

## Triton custom directory

- `/sgl-workspace/triton-custom` is readable.
- `/sgl-workspace/triton-custom` is not writable.
