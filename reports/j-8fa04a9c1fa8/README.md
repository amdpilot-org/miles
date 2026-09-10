# Two-GPU Miles training and weight-update investigation

## Result

The definitive run passed the bounded requirement on two assigned AMD Instinct MI350X (`gfx950`) GPUs:

- 64 cycles, rollout IDs 0 through 63, stopping immediately at cycle 63.
- 256 real FSDP forward/backward and optimizer steps, four per cycle.
- 64 successive distributed weight updates after an initial pre-loop update.
- Nonzero gradients for all 256 steps; gradient norm range was 3.6279 to 5.9076.
- 64 distinct engine parameter checksums and 64 matching fresh-load control checksums.
- Version checks passed for every cycle. The first post-loop version was 2 and the final version was 65.
- Fixed-token greedy outputs matched the fresh control after every update.
- Maximum absolute fixed-token log-probability difference was 0.00097847; the mean of per-cycle maxima was 0.00042789.

The run used real rollout generation on GPU 1, real FSDP training forward/backward and optimizer work on GPU 0, and actual Miles distributed weight-update communication. It did not use mocked weights, idle loops, spin waits, or sleeps.

## Topology and model

- Actor/training: one FSDP rank on GPU 0.
- Rollout: one SGLang engine on GPU 1.
- Devices: 2 × AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, about 251.98 GiB visible memory each.
- Model: locally random-initialized `Qwen2ForCausalLM`, 13,173,888 parameters, hidden size 128, two layers, vocabulary size 50304.
- Tokenizer: local GPT-2 tokenizer files. No pretrained model weights were downloaded; total download usage was far below 8 GB.
- Rollout batch: two prompts with two samples each. Four optimizer steps per cycle use the resulting four samples.

This is evidence for the tested small local model and two-GPU topology only. It is not a large-model, multi-node, MI355X, or performance-equivalence claim.

## Checks

Each cycle performed:

1. Real SGLang rollout generation.
2. Real FSDP log-probability, forward, backward, gradient clipping, and optimizer work.
3. Actual Miles distributed weight update.
4. Engine version verification.
5. SGLang engine checksum collection.
6. Actor checkpoint save.
7. Distributed checkpoint conversion and fresh control model load.
8. Engine-versus-control checksum comparison after fusing QKV and gate/up tensors and casting to the engine's FP16 representation.
9. Fixed-token greedy output and log-probability comparison.

The expected version after cycle `r` was `r + 2`: version 1 came from the initial pre-loop update. All 64 cycles passed version, checksum, fixed-token, optimizer-count, and update-count checks.

## Timings and memory

CUDA-event medians / p95 values were:

- Forward/backward: 13.4485 ms / 17.0685 ms over 256 events.
- Gradient clipping: 1.70814 ms / 2.13107 ms over 256 events.
- Optimizer: 1.55050 ms / 1.99350 ms over 256 events.
- Post-initial distributed weight update: 21.1038 ms / 22.7887 ms over 64 events.
- Fresh-control fixed-token forward: 8.20388 ms / 16.4081 ms over 64 events.

The first events include JIT/compile warmup and have large maxima, so median and p95 are the useful steady-state values. Wall-clock cycle median / p95 was 2.30963 s / 2.51828 s. The update RPC wall-time median / p95 was 0.611178 s / 0.633331 s.

Memory was stable rather than growing across cycles:

- GPU 0: 5.66211 GiB first cycle, 5.68164 GiB last cycle, 5.68164 GiB maximum.
- GPU 1: 59.07813 GiB first cycle, 59.09180 GiB last cycle, 59.09375 GiB maximum.

## Environment and tested paths

- Miles source: `/job/miles/miles`
- Miles HEAD during the run: `251f28dcee384391fbcb97c6ba82cba4f58a7b49`
- Tested code-diff SHA-256 before report finalization: `9e870b3770ba7625ac0664fa1b0fd8328c3fa028ed8da054088ee03e2ad87d79`
- SGLang source: `/sgl-workspace/sglang/python/sglang`
- SGLang commit: `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`
- Aiter source/native root: `/sgl-workspace/aiter/aiter`
- Aiter commit: `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`
- Imported Aiter native modules included `jit/module_aiter_core.so`, `jit/module_custom.so`, `jit/module_rmsnorm_quant.so`, and `jit/module_sample.so`.
- Triton source: `/sgl-workspace/triton-custom/python/triton`
- Triton commit: `42270451990532c67e69d753fbd026f28fcc4840`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Transformers source: `/opt/venv/lib/python3.10/site-packages/transformers`
- Ray source: `/opt/venv/lib/python3.10/site-packages/ray`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- SGLang: `0.5.17.dev2157+ga8e5c632f`
- Transformers: `5.12.1`
- Ray: `2.58.0`
- Triton: `3.6.0`

The run used a short unique `RAY_TMPDIR`, a unique run UUID, and framework-selected unique rendezvous ports. Bounded timeouts were: training process group 600 s, update process-group default 1800 s, SGLang distributed initialization 600 s, SGLang watchdog 300 s, NCCL heartbeat 600 s, and NCCL socket 600000 ms.

## Reproduction

From the repository root:

```bash
reports/j-8fa04a9c1fa8/run_two_gpu_update.sh smoke
reports/j-8fa04a9c1fa8/run_two_gpu_update.sh full
PYTHONPATH="$PWD" AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv \
  HIP_VISIBLE_DEVICES=0,1 /opt/venv/bin/python reports/j-8fa04a9c1fa8/analyze_results.py \
  /job/miles-artifacts/run/latest-driver-full --write reports/j-8fa04a9c1fa8/results.json
```

The full command creates `/job/miles-artifacts/model`, uses the checked-in prompt file, writes run artifacts under `/job/miles-artifacts/run`, and stops after rollout 63.

## Findings and limitations

- The FSDP checkpoint metadata fields `global_step` and `micro_step` remained zero even though the run executed 256 real optimizer steps. The driver therefore counts CUDA optimizer events rather than treating that metadata as step evidence. This is recorded as an observation; it was not silently normalized or fixed as part of the investigation.
- A preliminary one-sample-per-prompt configuration produced zero advantages, zero gradients, and unchanged checksums. It was rejected because it did not test cumulative weight drift. The definitive two-samples-per-prompt run has nonzero gradients and 64 distinct checksums.
- The actor logs that `qwen2` has no recorded FSDP validation and loads through the generic HF path. This local random model is supported enough for this fixture, but that warning limits claims about general Qwen2 correctness.
- The fixed-token sequence is intentionally tiny and does not exercise long-context, tensor-parallel, pipeline-parallel, multi-node, quantized-weight, or large-model behavior.
- CUDA graphs were disabled for deterministic small-batch execution. This run does not measure CUDA-graph-enabled throughput.
- No production behavior is changed when `cuda_event_timing_path` is unset. The instrumentation is opt-in; the optional `rollout_id` argument only labels update timing records.

## Artifacts

The definitive run artifacts are outside the repository:

- Output directory: `/job/miles-artifacts/run/driver-full-292452586`
- Log: `/job/miles-artifacts/run/driver-full-292452586.log`
- Cycle records: `/job/miles-artifacts/run/driver-full-292452586/cycles.jsonl`
- CUDA events: `/job/miles-artifacts/run/driver-full-292452586/cuda_events.jsonl`
- Metadata: `/job/miles-artifacts/run/driver-full-292452586/metadata.json`
- Final checkpoint: `/job/miles-artifacts/checkpoints-full/iter_0000064`

The checked-in summarized result is `reports/j-8fa04a9c1fa8/results.json`.
