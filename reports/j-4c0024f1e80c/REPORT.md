# R3 and speculative-path GPU investigation

## Scope

This is an operator-scoped investigation of [radixark/miles issue 1186](https://github.com/radixark/miles/issues/1186), “R3 accuracy bug.” The issue reports these step-0 `train_rollout_logprob_abs_diff` values for `Qwen3.5-35B-A3B`: no-R3/no-spec `0.013`, no-R3/spec-v2 `0.014`, R3/no-spec `0.007`, and R3/spec-v2 `0.022`.

The work below exercises the supported Miles R3 recording/replay and training-logprob implementation on two AMD Instinct MI350X GPUs for 64 real optimizer/update cycles. It also exercises the available SGLang EAGLE speculative-v2 path on two GPUs for 64 matched generation cycles. No production Miles code was changed; this report and its fixtures validate the existing implementation on the tested topology.

**The original 35B model and its topology were not reproduced.** All numerical bounds below are measured bounds for the stated reduced fixtures. They are not copied thresholds and are not evidence that any larger model passed. Sharing `gfx950` is also not a claim of MI355X performance equivalence.

## Environment and provenance

- Runtime: `amdpilotv2/miles-job:gbt350-d957-20260909`, `/opt/venv/bin/python`
- GPUs: 2 assigned AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, approximately 270.6 GB each
- Torch/ROCm: PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, HIP runtime `7.2.26015-fc0010cf6a`
- Miles: commit `df0e677f6dd51fa551d48d37860812ece904cd8f`, path `/job/miles`
- SGLang: `0.5.17.dev2157+ga8e5c632f`, commit `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`, source `/sgl-workspace/sglang`
- Aiter: commit `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`, source `/sgl-workspace/aiter`, native module `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Triton: `3.6.0+git42270451`, commit `42270451990532c67e69d753fbd026f28fcc4840`, source `/sgl-workspace/triton-custom`
- Megatron Core: `0.19.0+8c1e05747`, commit `8c1e05747eb612b382df2632783df5c83a853646`, source `/root/Megatron-LM`
- Transformers: `5.12.1`, package `/opt/venv/lib/python3.10/site-packages/transformers`
- Torch native libraries: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so` and `libtorch_cuda.so`
- HF fixture revision: `tiny-random/qwen3-moe` at `10a349dcb488b10c27aa4a3c1dbefb74c41565c3`

The HF fixture directory is approximately 35 MB and the locally generated speculative fixture is approximately 2.5 MB. Total model downloads were far below the 8 GB limit. Caches and weights were kept outside the Git worktree under `/job/fixture-cache`.

## Fixtures and methods

### R3 training fixture

The R3 case uses the downloaded `tiny-random/qwen3-moe` checkpoint at the revision above. The local copy was augmented with MTP keys for speculative-fixture compatibility; the R3 Qwen3 MoE loader does not consume those keys. It has two hidden layers, eight local experts, and two experts per token. The fixture performs real distributed training with DDP world size 2:

- fixed token batch: shape `[2, 32]`
- fp16 autocast forward compute
- fp32 parameters and gradients
- AdamW learning rate `1e-8`
- 64 optimizer/update cycles
- NCCL process-group timeout: 120 seconds

The explicit reference is a frozen copy of the initial fp32-parameter model using the same fp16 autocast forward compute and token batch. R3-on and R3-off use identical initial parameters, tokens, compute precision, and optimizer settings. Routing is captured from the R3 record stream and from forward hooks on the non-R3 model. Per-token absolute logprob error, gradient absolute difference, and maximum parameter drift are measured against the explicit reference every cycle.

### Speculative fixture

The speculative case uses a locally initialized two-layer Qwen3 MoE fixture with vocabulary size 4096, eight experts, and two experts per token. A local MTP/EAGLE weight set is generated from the first decoder layer. SGLang runs TP2 on both assigned GPUs with:

- fp16 model compute
- deterministic temperature-0 sampling
- identical seeded token inputs for corresponding no-spec and spec cycles
- 64 generation cycles per path
- EAGLE speculative-v2, two draft steps, top-k 1, three draft tokens
- Triton attention and MoE runner backends
- NCCL port 29731 and distributed timeout 120 seconds

The installed SGLang source reports that `SGLANG_ENABLE_SPEC_V2` has been removed and speculative decoding always uses the V2 worker (`/sgl-workspace/sglang/python/sglang/srt/arg_groups/speculative_hook.py`). Therefore the available speculative path is EAGLE/V2; no separate V1 path was claimed.

## R3 measured results

All 64 cycles completed with finite values. R3-on and R3-off used exactly the same routing in every cycle.

| Measurement | R3-on | R3-off |
|---|---:|---:|
| Cycles | 64 | 64 |
| Tokens compared | 2048 | 2048 |
| Routing max absolute difference | 0 | 0 |
| Per-token mean absolute error vs reference | 0.0007398370653390884 | 0.0007398370653390884 |
| Per-token p95 absolute error vs reference | 0.0027303695678710938 | 0.0027303695678710938 |
| Per-token p99 absolute error vs reference | 0.00445556640625 | 0.00445556640625 |
| Per-token max absolute error vs reference | 0.005275726318359375 | 0.005275726318359375 |
| Gradient max absolute difference vs reference | 0.0010708272457122803 | 0.0010708272457122803 |
| Maximum parameter drift | 9.5367431640625e-07 | 9.5367431640625e-07 |

The direct R3-on versus R3-off comparison had maximum absolute logprob difference `0` and maximum absolute gradient difference `0` in every cycle. The measured equality bound for this fixture is therefore exactly zero for the R3 replay path relative to the non-R3 path. The nonzero reference errors are the measured fp16-autocast-versus-explicit-reference bounds for this fixture; they are not a pass/fail threshold and do not establish model-level accuracy.

Aggregate GPU phase timings were:

| Phase | Seconds |
|---|---:|
| Explicit reference forward/backward | 9.58403199352324 |
| R3-on logprob | 0.33796000201255083 |
| R3-on training forward/backward | 0.8149707857519388 |
| R3-off logprob | 0.27568379044532776 |
| R3-off training forward/backward | 0.6764954784885049 |
| Optimizer updates | 0.13281195983290672 |
| Total measured run | 12.48273606505245 |

## Speculative measured results

The no-spec and EAGLE/V2 paths used the same seeded inputs and fp16 precision. Their generated token IDs matched exactly across all 64 cycles: the measured token mismatch bound is `0`.

The routed-expert capture is not equivalent under speculative decoding. In every cycle the first 16 routing rows matched, but exactly 15 later rows differed. Across 64 cycles there were 960 mismatched routing rows and the maximum expert-ID difference was 1. Consequently, this run does **not** support a claim that speculative and non-speculative routed-expert capture is equivalent. It is a negative routing-capture result, despite exact token agreement.

Aggregate timings were:

| Path | Startup seconds | Generation seconds |
|---|---:|---:|
| No speculative | 44.0116289453581 | 8.65152402408421 |
| EAGLE/V2 | 33.99236696213484 | 9.21848998684436 |

Speculative metadata over 64 cycles recorded 320 verify calls, 640 proposed drafts, and 640 accepted drafts. These timings are fixture timings only and are not a performance-equivalence claim.

## Unsupported and negative paths

- SGLang TP2 `return_logprob` returned `NaN` for every output-token logprob tuple in this tiny speculative fixture. Both no-spec and spec paths returned the same `NaN` pattern, so no usable per-token speculative logprob comparison was possible. This path is unsupported here and is not reported as passing.
- The speculative routed-expert capture differs as described above, so expert-routing equivalence is unsupported for this fixture.
- An earlier custom tiny-vocabulary fixture reached ROCm `indexSelectSmallIndex` HIP error 719. That architecture/path was abandoned and is not generalized to MI350X or Qwen3 MoE as a whole.
- The installed SGLang build removes `SGLANG_ENABLE_SPEC_V2`; only the available EAGLE/V2 worker was exercised.
- No 35B training or inference run was attempted, and no claim is made about the issue’s original topology or its reported differences.

## Reproduction

The commands below assume the qualified runtime and two assigned MI350X GPUs. Use unique rendezvous ports if concurrent jobs are present.

```bash
cd /job/miles
export PYTHON=/opt/venv/bin/python

# Optional: fetch the approximately 35 MB HF fixture at the recorded revision.
$PYTHON -m huggingface_hub.commands.huggingface_cli download \
  tiny-random/qwen3-moe \
  --revision 10a349dcb488b10c27aa4a3c1dbefb74c41565c3 \
  --local-dir /job/fixture-cache/hf-tiny-qwen3-moe

# Generate the approximately 2.5 MB local speculative fixture.
$PYTHON reports/j-4c0024f1e80c/create_fixture_model.py \
  --output /job/fixture-cache/tiny-qwen3-moe-mtp

# R3 on/off training comparison: 64 DDP optimizer/update cycles.
$PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=2 --master_port=29732 \
  reports/j-4c0024f1e80c/train_r3_paths.py \
  --model /job/fixture-cache/hf-tiny-qwen3-moe \
  --output reports/j-4c0024f1e80c/results/r3_64_bounded.json \
  --cycles 64 --learning-rate 1e-8

# SGLang TP2 no-spec and EAGLE/V2 generation comparisons.
$PYTHON reports/j-4c0024f1e80c/run_spec_path.py \
  --model /job/fixture-cache/tiny-qwen3-moe-mtp \
  --output reports/j-4c0024f1e80c/results/spec_off_64_tp2.json \
  --cycles 64 --tp-size 2 --nccl-port 29731

$PYTHON reports/j-4c0024f1e80c/run_spec_path.py \
  --model /job/fixture-cache/tiny-qwen3-moe-mtp \
  --output reports/j-4c0024f1e80c/results/spec_on_64_tp2.json \
  --cycles 64 --spec --tp-size 2 --nccl-port 29731

# Derive the measured summary from the raw records.
$PYTHON reports/j-4c0024f1e80c/analyze_results.py \
  --r3 reports/j-4c0024f1e80c/results/r3_64_bounded.json \
  --spec-off reports/j-4c0024f1e80c/results/spec_off_64_tp2.json \
  --spec-on reports/j-4c0024f1e80c/results/spec_on_64_tp2.json \
  --output reports/j-4c0024f1e80c/results/summary.json
```

The checked-in raw evidence is under `reports/j-4c0024f1e80c/results/`. The analyzer derives every reported bound from those records; it does not apply an external accuracy threshold.

## Left undone

- No speculative training/gradient comparison was claimed because the exercised SGLang path is generation-only and its returned logprobs were unusable (`NaN`).
- No attempt was made to repair SGLang’s speculative routed-expert capture or its TP2 logprob path.
- No larger Qwen MoE topology, multi-node run, or 35B reproduction was attempted.
