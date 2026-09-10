# ROCm custom all-reduce graph-capture investigation

## Scope

This investigation compares an actual standalone launch context against a Ray-colocated launch context on two assigned AMD Instinct MI350X (`gfx950`) GPUs. It does **not** claim a four-GPU NVIDIA reproduction, and it does not claim MI355X performance equivalence.

The fixture uses the qualified Miles environment and real GPU operations only. It performs no model downloads, uses no idle loops, no spin waits, and no sleeps as substitutes for GPU work.

## Environment

- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python`
- GPUs: 2 × AMD Instinct MI350X (`gfx950`)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Miles commit: `e5125a97e1fd383f005f4de258a5985026e09425`
- SGLang commit: `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`
- Aiter commit: `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`

Imported source and native paths recorded by the fixture:

- SGLang Python: `/sgl-workspace/sglang/python/sglang/__init__.py`
- QuickAllReduce Python: `/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- QuickAllReduce native source: `/sgl-workspace/sglang/python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce.hip`
- QuickAllReduce native header: `/sgl-workspace/sglang/python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce_hip.h`
- QuickAllReduce native extension: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Aiter Python: `/sgl-workspace/aiter/aiter/__init__.py`
- Torch Python: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`

## Path selection

The available ROCm custom all-reduce and graph-capture path on this fixture is SGLang `QuickAllReduce`, not the NVIDIA-only JIT `CustomAllReduceV2` path from issue #1176.

The fixture explicitly preserves the unsupported-path boundaries:

- `CustomAllReduceV2` is not dispatched on HIP; `sglang_is_cuda=false`, `sglang_is_hip=true`, and `sglang_v2_dispatched=false`.
- Aiter `CustomAllreduce` graph capture was probed diagnostically and produced `NaN` output on `gfx950`; it is not used for validation.
- `QuickAllReduce` is selected with `ROCM_QUICK_REDUCE_QUANTIZATION=FP` and `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=64` in both launch contexts.

The default environment uses `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`. This investigation does not claim INT8 numerical equivalence; FP mode is used for a lossless, bit-exact comparison.

## Fixture

The fixture performs 32 capture/replay cycles across 32 valid workload shapes. Each cycle includes:

1. A real model forward and backward pass.
2. A custom all-reduce of the captured graph input.
3. A CUDA/HIP graph capture and replay.
4. A gradient all-reduce and optimizer update.
5. A cross-rank weight equality check.

The standalone context uses two subprocesses, one per assigned GPU. The Ray-colocated context uses a placement group with two bundles and fractional GPU actors, with an additional real GPU workload colocated on each bundle.

The fixture uses a unique rendezvous port per run and bounded process-group timeouts:

- Process-group timeout: 300 seconds
- Ray timeout: 900 seconds
- Subprocess timeout: 900 seconds

Pause/resume is supported in this environment through `torch_memory_saver`. The fixture verifies that a tagged GPU allocation is preserved across an explicit pause/resume cycle.

## Results

All 32 cycles completed in both contexts.

| Metric | Standalone | Ray-colocated |
|---|---:|---:|
| Cycles | 32 | 32 |
| Output equal to NCCL | Yes | Yes |
| Maximum output difference | 0.0 | 0.0 |
| Weights equal across ranks | Yes | Yes |
| Capture mean | 1.007 ms | 1.045 ms |
| Capture max | 1.329 ms | 1.324 ms |
| Replay mean | 0.932 ms | 0.375 ms |
| Replay max | 5.608 ms | 0.663 ms |
| Forward/backward mean | 60.808 ms | 57.188 ms |
| Gradient all-reduce mean | 0.917 ms | 1.648 ms |
| Optimizer mean | 0.878 ms | 0.859 ms |
| Pause/resume passed | Yes | Yes |

Cross-context comparison:

- Maximum output checksum difference: `0.0`
- Final weight checksum difference: `0.0`
- Outputs match: `true`
- Weights match: `true`

Under the controlled FP QuickAllReduce path, the standalone and Ray-colocated launch contexts produce bit-identical outputs and final weights across all 32 cycles.

## Reproduction

From the Miles repository root:

```bash
/opt/venv/bin/python reports/j-211b8b17dedb/rocm_colocate_allreduce.py --mode standalone
```

This command runs both the standalone and Ray-colocated contexts and writes the full result set to:

```text
reports/j-211b8b17dedb/results.json
```

## Limitations

- This is a two-GPU ROCm investigation only.
- It does not reproduce or validate the four-GPU NVIDIA topology from issue #1176.
- It does not claim MI355X performance equivalence.
- It validates the existing ROCm `QuickAllReduce` FP path; no Miles code change is assumed or required from this investigation.
- Aiter `CustomAllreduce` graph capture remains unsupported on this `gfx950` fixture.
