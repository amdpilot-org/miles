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
- Initial measured case elapsed time: `430.8267 ms` (includes transfer, kernel, synchronization, and comparison download).

The full machine-readable initial result is in `initial_results.json`.

## Extended GPU validation

- Validation start (UTC): `2026-09-10T05:22:46.559268+00:00`
- Validation end (UTC): `2026-09-10T05:23:46.686200+00:00`
- Elapsed duration: `60.00011810194701 s`
- Iterations performed and verified: `380426`
- Seeds: deterministic sequence starting at `51125129` and ending at `51505554`
- Tail lengths: cycling `0` through `31`; vector lengths `4096` through `4127`
- Triton block size: `1024`
- Exact cases: `380426`; failed cases: `0`
- Total mismatched elements: `0`; maximum absolute error: `0.0`
- Per-case timing mean: `0.04440293427040291 ms`
- Per-case timing minimum: `0.036088982596993446 ms`
- Per-case timing maximum: `423.35962504148483 ms` (first case includes Triton compilation)

Per-case timing includes host-to-device transfer, kernel launch, synchronization, and device-to-host download. The aggregate elapsed duration also includes independent CPU expected-value generation and exact comparison overhead.

### Small sample

| Seed | Length | Expected | Observed |
|---:|---:|---:|---:|
| `51125129` | `4096` | `14.190000534057617` | `14.190000534057617` |
| `51125130` | `4097` | `14.204999923706055` | `14.204999923706055` |
| `51505554` | `4105` | `778.6600341796875` | `778.6600341796875` |

The full aggregate and six-case sample are in `extended_results.json`.

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
3. The first extended attempt stopped after about `1.8 s` before its first kernel launch because the selected seed produced an exactly zero input value. Input generation was changed to be provably positive for every seed.
4. The second extended attempt verified `100000` exact GPU operations but stopped after `16.812424905016087 s` because it reached the `100000` iteration safety cap. That result is preserved in `extended_attempt_1_results.json`.

The final extended run above then met the required `60 s` interval. None of the recorded failures was a GPU numerical mismatch.

## Deliverables

- Draft PR: `https://github.com/amdpilot-org/miles/pull/132`
- Branch: `amdpilot/j-5f11251b29b2`
- PR base: `e5125a97e1fd383f005f4de258a5985026e09425` (`main`)
- Recovery patch: `/job/recovery.patch`

## Status and uncertainties

- Extended validation is complete and all measured cases were exact.
- No model weights were downloaded, no framework stack was installed, and no upstream repository was contacted.
- The report does not establish whether any broader AMDPilot delivery issue is fixed; it records only this vector-add observation.
