## Environment variables

- `PYTORCH_ROCM_ARCH` is set to `gfx950`.
- `MILES_HARDWARE_PLATFORM` is set to `rocm`.
- `GPU_ARCH` is set to `gfx950`.
- `SGLANG_USE_AITER` is set to `1`.
- `HIP_FORCE_DEV_KERNARG` is set to `1`.

## Source trees

| Tree | Commit | Writable |
| --- | --- | --- |
| `/root/miles` | `b1229404c` | Yes; write-delete test passed |
| `/sgl-workspace/sglang` | `4e230c3d85` | Yes; write-delete test passed |
| `/root/Megatron-LM` | `235952df6` | Yes; write-delete test passed |

## Python imports

- `miles` resolves to `/root/miles/miles/__init__.py`.
- `sglang` resolves to `/sgl-workspace/sglang/python/sglang/__init__.py`.

## Triton custom directory

- `/sgl-workspace/triton-custom` is readable.
- `/sgl-workspace/triton-custom` is not writable.
