# Triton vector-add control-arm report

## Scope

This is a small diagnostic report for job `j-5f11251b29b2`. It does not change Miles runtime code and does not claim to fix any platform issue.

## Initial GPU observation

- Repository base: `e5125a97e1fd383f005f4de258a5985026e09425` (`main`)
- GPU: AMD Instinct MI355X, UUID `38396538-6335-6364-3337-363738313161`, architecture `gfx950:sramecc+:xnack-`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP: `7.2.26015-fc0010cf6a`; Triton: `3.6.0`
- Visible Torch CUDA/ROCm devices: `1`
- Tensor shape: `[1048583]`, FP32, seed `51125129`, Triton block size `1024`
- Both inputs were checked to contain no zero values.
- The Triton kernel used masked loads/stores for the final partial block.
- Independent CPU Torch add comparison: exact match; `0` mismatches; maximum absolute error `0.0`.
- Initial measured case elapsed time: `436.9311 ms` (includes transfer, kernel, synchronization, and comparison download).

The full machine-readable initial result is in `initial_results.json`.

## Reproduction

From the repository root:

```bash
python3 reports/j-5f11251b29b2/triton_vector_add_probe.py initial \
  --output reports/j-5f11251b29b2/initial_results.json
```

The extended mode is:

```bash
python3 reports/j-5f11251b29b2/triton_vector_add_probe.py extended \
  --min-seconds 60 \
  --output reports/j-5f11251b29b2/extended_results.json
```

## Honest run history

Two pre-measurement tooling failures occurred and were corrected:

1. The first launch failed before GPU work because `math.is_power_of_two` is unavailable in the installed Python 3.10.
2. The second launch completed the GPU operation, but JSON serialization failed because Torch exposes the GPU UUID as a custom object.

Neither failure was a GPU numerical mismatch. The successful initial run above is the measured result.

## Status and uncertainties

- Extended validation has not yet run in this early draft.
- No model weights were downloaded, no framework stack was installed, and no upstream repository was contacted.
- The report does not establish whether any broader AMDPilot delivery issue is fixed; it records only this vector-add observation.
