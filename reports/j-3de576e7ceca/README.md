# Four-rank follow-up verification for PR52

Status: **DRAFT — final report update pending**

This report continues the unresolved GPU verification for [amdpilot-org/miles PR52](https://github.com/amdpilot-org/miles/pull/52), related to [radixark/miles issue 2406](https://github.com/radixark/miles/issues/2406). It does not duplicate or merge PR52's runtime change.

## Scope and source control

- Pinned validation control: `03a7d4096ffa4fe93ed4c5e2555c46ce9771bb78`, clean, detached HEAD at `/job/miles-validation`.
- Separate BSHD candidate: same base commit, dirty at `/job/miles-validation-candidate`; only `raw/../bshd_fsdp_candidate.patch` is applied.
- Current-main delivery base: `8d9826eacc8b5c279546f96711bb401b7f62c54c`, clean `main` at `/job/miles-delivery`.
- Imported Miles source for every run is the checkout named above; no refreshed main was merged into the validation control.

## Environment

- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`; ROCm: `7.2.0`
- Compiler: GCC `11.4.0`; HIPcc `7.2.26015-fc0010cf6a`
- GPUs: exactly 4 visible AMD Instinct MI355X devices, `gfx950` / capability `(9, 5)`
- Checkpoint: `Qwen/Qwen3-VL-4B-Instruct` revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Checkpoint content SHA-256: `ff00ef4f85f8b7c0200270a8a8003db7aba2c901fdd73806c78dfd12605a0620`

The complete machine-readable manifest is in `raw/environment_manifest.json`.

## Deterministic asymmetric fixtures

The rollout uses fixed text, a fixed 64x64 PNG, exact prompt/full processor token-prefix checks, and a one-minute distributed timeout. Every run uses four real FSDP ranks, two rollouts, and two optimizer steps.

For `micro_batch_size=2`, rank-local rows are:

- Rank 0: text + image (mixed)
- Rank 1: text + text (text-only)
- Rank 2: image + image (image-containing)
- Rank 3: text + image (mixed)

For `micro_batch_size=4`, rank-local rows are:

- Rank 0: text, text, image, image (mixed)
- Rank 1: four text-only rows
- Rank 2: four image-containing rows
- Rank 3: text, image, text, image (mixed)

## Reproduction

From `/job/validation-harness`:

```bash
python make_fixtures.py

timeout --signal=TERM --kill-after=30s 900s \
  python run_four_rank.py --format thd --micro-batch-size 2

timeout --signal=TERM --kill-after=30s 900s \
  python run_four_rank.py --format thd --micro-batch-size 4
```

The exact pinned control rejects FSDP BSHD during argument validation:

```bash
timeout --signal=TERM --kill-after=15s 180s \
  python run_four_rank.py --format bshd --micro-batch-size 2
```

The BSHD runs therefore use the separate, clearly unmerged candidate checkout:

```bash
timeout --signal=TERM --kill-after=30s 900s \
  python run_four_rank.py --format bshd --micro-batch-size 2 \
  --miles-checkout /job/miles-validation-candidate

timeout --signal=TERM --kill-after=30s 900s \
  python run_four_rank.py --format bshd --micro-batch-size 4 \
  --miles-checkout /job/miles-validation-candidate
```

## Four-rank results

| Case | Source | Exit | Log gather per rank | Step 0 loss / grad norm | Step 1 loss / grad norm |
|---|---|---:|---:|---:|---:|
| THD, mbs 2 | exact pinned control | 0 | 2 successes each | 10.138484954833984 / 3849.5869140625 | 10.01611328125 / 3811.39501953125 |
| THD, mbs 4 | exact pinned control | 0 | 2 successes each | 7.972590923309326 / 7693.03076171875 | 7.814351558685303 / 7613.75 |
| BSHD, mbs 2 | unmerged candidate | 0 | 2 successes each | 8.922124862670898 / 3688.971435546875 | 8.81326961517334 / 3623.50634765625 |
| BSHD, mbs 4 | unmerged candidate | 0 | 2 successes each | 7.969666957855225 / 7664.60986328125 | 7.831358909606934 / 7897.009765625 |

All four ranks reported FSDP mesh shape `(4,)`. All listed losses and gradient norms are finite. Per-rank raw logs are preserved under `raw/*/ranks/`.

## Dummy-image learning-signal check

The full Qwen3-VL-4B model compared a same-sequence text-only control against the dummy-image path:

- Sequence length: 82 tokens, including 64 zero-loss image tokens
- Control loss: `9.752323150634766`
- Dummy-image loss: `9.752323150634766`
- Loss difference: `0.0`
- Maximum difference across all 398 language-model gradients: `0.0`
- All compared language gradients were finite
- Maximum difference across all 315 vision gradients: `0.0`
- Control and dummy-path vision gradient maxima: `0.0`

Raw output is in `raw/gradient_check/`.

## Candidate patch and limitations

`bshd_fsdp_candidate.patch` is **not part of current main and is not merged by this report**. Its exact base is `03a7d4096ffa4fe93ed4c5e2555c46ce9771bb78`. It contains only:

1. Allowing FSDP in the BSHD backend assertion.
2. A local fallback for Megatron-only `args.compress_ratios` when FSDP BSHD reaches rollout padding.

The first candidate attempt preserved in `raw/bshd_mbs2_candidate/attempt1_compress_ratios.log` failed on the missing `compress_ratios` attribute before the patched retry completed.

Remaining limitations:

- Pinned FSDP BSHD is explicitly unsupported by its argument assertion; the passing BSHD results require the unmerged candidate patch.
- Miles logs aggregate loss and gradient norm on rank 0. Completion of ranks 1-3 is evidenced by their successful per-rank `log_gather` events and successful FSDP collectives, not separate per-rank loss metrics.
- The gradient comparison is a full-model single-GPU check matching the previous two-rank report; the four-rank runs are separate training checks.
- These are functional verification runs, not controlled performance benchmarks.
