## Environment variables

| Variable | Status | Value |
| --- | --- | --- |
| `PYTORCH_ROCM_ARCH` | Set | `gfx950` |
| `MILES_HARDWARE_PLATFORM` | Set | `rocm` |
| `GPU_ARCH` | Set | `gfx950` |
| `SGLANG_USE_AITER` | Set | `1` |
| `HIP_FORCE_DEV_KERNARG` | Set | `1` |

## Source trees

| Tree | Present | Commit | Writable |
| --- | --- | --- | --- |
| `/root/miles` | Yes | `b1229404c` | Yes; created and deleted a file |
| `/sgl-workspace/sglang` | Yes | `4e230c3d85` | Yes; created and deleted a file |
| `/root/Megatron-LM` | Yes | `235952df6` | Yes; created and deleted a file |

## Python imports

| Package | Resolved path |
| --- | --- |
| `miles` | `/root/miles/miles/__init__.py` |
| `sglang` | `/sgl-workspace/sglang/python/sglang/__init__.py` |

## Triton custom directory

| Path | Readable | Writable |
| --- | --- | --- |
| `/sgl-workspace/triton-custom` | Yes | No |
