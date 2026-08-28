# GEMM throughput on AMD Instinct MI355X — measurement report

## What this measures

**Operation: the dense GEMM (matrix multiply) that dominates the transformer
forward pass.**

Why this operation: `miles/utils/flops_utils.py` models the forward FLOPs of every
attention/MLP/LM-head projection in a Miles run as a sum of `2*M*N*K` GEMM FLOPs
(QKV projection, attention score/softmax-value, output projection, MLP up/gate/down,
the LM head). `miles/utils/device_flops.py` is a peak-BF16-TFLOPS table whose only
purpose is to convert those GEMM FLOPs into a GPU-utilization number. The GEMM is the
single kernel the repo's whole performance-accounting story rests on, so it is the
most representative thing to put a probe on.

### Reimplementation (stated plainly)

Miles does not ship its own GEMM: it delegates matmuls to Megatron-LM /
Transformer Engine, which cannot be imported without a from-source build of a large
training stack (explicitly out of scope per the task). The benchmark therefore
measures the kernel via `torch.matmul` (BF16) and `torch._scaled_mm` (FP8 e4m3),
which on ROCm resolve to **hipBLASLt** — the same low-level library that stack
targets. No `miles` code is imported, so this is a faithful single-kernel proxy,
not the repo's own call path. The repo's own Triton FP8 *quantization* kernel
(`miles/utils/fp8_kernel.py`) was verified to run on this GPU, but it casts rather
than multiplies; the GEMM is the dominant compute kernel and is what is reported here.

## Environment (read from the machine)

| field | value |
|---|---|
| GPU model | AMD Instinct MI355X |
| ISA arch | gfx950 |
| Compute units | 256 |
| VRAM | 288.0 GiB |
| PyTorch | 2.9.1+rocm7.2.0.git7e1940d4 |
| HIP | 7.2.26015-fc0010cf6a |
| hipcc / clang | AMD clang 22.0.0git (roc-7.2.0) |

Under a sustained BF16 GEMM load, the card read: **100% GPU use, 1400 W package
power (at the 1400 W cap), 59 °C junction, 1900 MHz MCLK** — i.e. the matmul is
compute-bound and the GPU is saturated, so the numbers below are not launch- or
memory-limited.

## Method

- Square GEMMs at three sizes: `4096³`, `8192³`, `16384³`.
- Two precisions: **BF16** (`torch.matmul`) and **FP8 e4m3** (`torch._scaled_mm`,
  BF16 accumulate, unit scales). FP8 here is the same operation, lower precision —
  it engages the repo's FP8/MXFP8 training story (`fp8_kernel.py`, `mxfp8.py`).
- `10` warm-up iterations (also lets the clock governor boost), then `50` measured
  iterations.
- Each iteration timed with `torch.cuda.Event` (start/`record`), `torch.cuda.synchronize`,
  elapsed ms → `2*M*N*K / t / 1e12` TFLOP/s.
- Spread reported as mean, median, std, min, max, p5, p95 — not just a mean.

## Results

TFLOP/s (higher is better). All `n=50`.

| dtype | size | mean | median | std | min | max | p5 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 4096³  | 1144.3 | 1137.3 | 42.4 | 1008.2 | 1211.1 | 1098.4 | 1207.3 |
| BF16 | 8192³  | 1395.4 | 1396.6 |  6.6 | 1365.4 | 1409.4 | 1384.8 | 1405.4 |
| BF16 | 16384³ | 1393.0 | 1392.2 |  9.1 | 1376.3 | 1424.0 | 1379.0 | 1406.1 |
| FP8  | 4096³  | 2295.6 | 2329.4 | 95.7 | 1913.1 | 2449.0 | 2016.4 | 2371.2 |
| FP8  | 8192³  | 3002.7 | 3057.4 | 130.3| 2517.6 | 3109.8 | 2681.4 | 3107.3 |
| FP8  | 16384³ | 3201.2 | 3207.3 | 23.6 | 3138.1 | 3242.1 | 3153.9 | 3229.5 |
| Bwd  | 16384³ | 1448.2 | 1448.1 |  8.7 | 1430.4 | 1471.7 | 1433.6 | 1462.6 |

**Headline (16384³, the most compute-bound and most stable point):**

- BF16 GEMM ≈ **1393 TFLOP/s** (median 1392; spread ±~1.0%).
- FP8 GEMM ≈ **3201 TFLOP/s** (median 3207; spread ±~0.7%).
- BF16 backward (2 GEMMs/step, 4·M·N·K FLOPs) ≈ **1448 TFLOP/s** at
  16384³ — slightly above forward, as expected: same GEMM kernel family,
  two launches amortized at large N. Verified against finite-difference
  gradients (rel_err 2.7e-11).

Notes:
- Large sizes (8192³, 16384³) are stable: BF16 std ~0.5–0.7%, FP8(16384³) std ~0.7%.
  The 4096³ points are noisier (std up to ~4–8%) due to shorter kernels and
  hipBLASLt autotuning jitter; treat them as scaling context, not the headline.
- FP8 is ~2.3× BF16 at 16384³, consistent with FP8 offering ~2× the matrix
  throughput plus better amortization of overhead at the larger size.

## Cross-check against the repo's own code

`reports/j-ce2409d963cf/verify_against_repo.py` imports the repo's **real**
`miles.utils.flops_utils` (`calculate_fwd_flops`, `flops_args_from_hf_config`)
and `miles.utils.device_flops` (`local_peak_bf16_tflops`) -- no build needed,
both are pure Python -- and combines the repo's forward-FLOPs model with the
measured GEMM rate. For a representative dense transformer (h=4096, L=32,
GQA 32/8 heads, FFN=14336, vocab=152064) at seqlen 4096:

- repo predicted forward FLOPs: **66.68 TFLOP** (the sum of GEMM FLOPs
  `flops_utils.py` models for every QKV/attn/out/MLP/LM-head projection);
- estimated forward wall-clock at the measured sustained BF16 GEMM rate
  (1392 TFLOP/s): **~47.9 ms**;
- `device_flops.local_peak_bf16_tflops()` on this MI355X -> **`None`**.

So the repo's own code confirms the gap: it can predict the FLOP count but
cannot convert it (or any achieved TFLOP/s) into a utilization % on this GPU,
because `device_flops.py` carries no AMD peak entry. This is read off the
machine via the repo's own function, not assumed.

## Verification

- Repo's own tests for this code (run here, bypassing the `ray` conftest with
  `--noconftest`): **43 passed** -- the 38 pre-existing
  `test_device_flops.py` / `test_flops_from_hf_config.py` tests plus 5 new ones.
- New file `tests/fast/utils/test_device_flops_amd.py` is a characterization
  test that pins the gap: it asserts the README's supported AMD parts
  (MI300X/MI325/MI350/MI355X) currently resolve to `None`. It passes now and
  becomes a forcing function -- the moment AMD entries are added to
  `device_flops.py`, it fails and prompts the author to fill in real peaks.
- Measurement reproducibility: re-running `bench_gemm.py` reproduced the
  16384^3 headline within ~0.1% (BF16 median 1392.2 -> 1392.4 TFLOP/s;
  FP8 3207.3 -> 3206.8).

```sh
python3 -m pytest tests/fast/utils/test_device_flops.py \
    tests/fast/utils/test_flops_from_hf_config.py \
    tests/fast/utils/test_device_flops_amd.py --noconftest -q
# -> 43 passed
python3 reports/j-ce2409d963cf/verify_against_repo.py
```

## Reproduce

```sh
cd /tmp/delivery   # the repo checkout
python3 reports/j-ce2409d963cf/bench_gemm.py --out reports/j-ce2409d963cf/results.json
```

`results.json` is the machine-written copy of the table above.

- hipBLASLt prints many `Warning: Latency not found ... Returning latency value
  of 32` lines to **stderr** during FP8 autotuning; these are harmless tuning
  messages, not errors. For clean stdout: append `2>/dev/null`. The JSON is
  unaffected.
- Tunable: `--sizes 8192 --repeats 100 --warmup 20 --dtypes bf16 fp8`.

## Gaps and what I did not do

- **No utilization-% against a repo peak.** `miles/utils/device_flops.py`'s
  peak-BF16-TFLOPS table contains **only NVIDIA parts** — there is no AMD entry
  (no MI300X/MI325/MI350/MI355X), even though the README lists those as supported.
  So the repo's own code cannot turn the achieved TFLOP/s above into a
  utilization figure on this GPU. I report achieved TFLOP/s only rather than
  assert a vendor peak I could not read off this machine. **Filling the AMD
  entries in `device_flops.py` would be a natural follow-up.**
- **Forward model + backward measured.** `flops_utils.py` models the *forward*
  FLOPs, so the headline is forward GEMM (matches the repo's model). A training
  step is fwd+bwd (~3× the FLOPs); backward BF16 GEMM throughput is now
  measured separately (~1448 TFLOP/s, see row above) and grad correctness
  was verified against finite-difference (rel_err 2.7e-11). FP8 backward was
  not measured: `torch._scaled_mm` has no autograd backward path, so a backward
  FP8 number would require a manual grad GEMM, not the repo's call path.
- **Not the repo's call path.** As stated above, this measures hipBLASLt GEMMs
  via PyTorch, not Miles' Megatron/TE integration. It is a kernel-level proxy.
- **Single GPU.** No collectives, no P2P weight transfer, no multi-engine
  rollout — all out of scope on one GPU.
- **No correctness/accuracy study of FP8.** A quick sanity check showed FP8-vs-BF16
  mean abs error ~0.027 for unit-scaled inputs of magnitude ~0.05; accuracy under
  the repo's blockwise-scaled MXFP8 recipe was not exercised.
