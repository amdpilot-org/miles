# gfx950 train/rollout log-prob parity investigation

## Result

On one assigned **AMD Instinct MI350X** (`gfx950:sramecc+:xnack-`, 256 CUs, 258,032 MiB reported by Torch), Qwen3-0.6B in bfloat16 still has a nonzero train/rollout log-prob mismatch after the existing true-on-policy batch-invariant setup.

The primary run is `results_batch_invariant.json`. It uses identical weights, token sequences, temperature-1 log-prob normalization, deterministic SGLang Triton rollout, Miles' true-on-policy log-prob helper, and SGLang batch-invariant ops in the training process. Across controlled batch sizes 1, 2, 4, and 8:

| Training path | Tokens compared | Mean absolute difference | Maximum absolute difference | RMSE | Mean signed difference | Bitwise-equal tokens |
|---|---:|---:|---:|---:|---:|---:|
| FlashAttention-2 unpacked | 116 | 0.0319222 | 0.1664782 | 0.0451942 | -0.0054950 | 0 / 116 |
| FlashAttention-2 packed | 116 | 0.0319222 | 0.1664782 | 0.0451942 | -0.0054950 | 0 / 116 |
| Miles SGLang Triton bridge | 116 | 0.0521295 | 0.1981664 | 0.0703295 | +0.0116285 | 0 / 116 |

The largest controlled batch (8 sequences, 60 response-side token comparisons) has:

| Training path | Mean absolute difference | Maximum absolute difference | RMSE | Mean signed difference |
|---|---:|---:|---:|---:|
| FlashAttention-2 unpacked/packed | 0.0379357 | 0.1664782 | 0.0506691 | -0.0034067 |
| Miles SGLang Triton bridge | 0.0497815 | 0.1981664 | 0.0651907 | +0.0077648 |

With batch-invariant ops enabled, packed and unpacked training are exactly equal for all 116 sweep observations, and every training path is exactly batch-invariant relative to its individual-sequence baseline. SGLang rollout is also exactly batch-invariant. Therefore the remaining mismatch is cross-engine numerical drift, not packing or batch-size leakage.

No framework correction is justified by this evidence. The existing FSDP true-on-policy actor already enables the batch-invariant setup that removes packing and batch-size effects. The remaining mismatch would require deeper alignment of SGLang and HF/Miles forward kernels; changing a single Miles log-prob formula would not address it.

## Controls and negative results

- `results_without_batch_invariant_ops.json` is a control run before enabling SGLang batch-invariant ops in the training process. In that run, FlashAttention unpacked training was not fully batch-invariant (mean absolute difference 0.0121203, maximum 0.125), and packed versus unpacked differed (mean absolute difference 0.0224986, maximum 0.140625 over batch sizes 2, 4, and 8).
- `results_rope_control_without_batch_invariant_ops.json` sets `USE_ROCM_AITER_ROPE_BACKEND=0`. Its raw rollout and training values are byte-for-byte identical to the preceding control, so the observed mismatch is not explained by the Aiter/native Apex RoPE selection.
- The first packed attempt failed because this ROCm FlashAttention build requires `cu_seq_lens_q` and `cu_seq_lens_k` with dtype `int32`; the fixture now uses that dtype.
- SGLang's offline `Engine` cannot be launched from a stdin-only Python process because its scheduler uses multiprocessing. The executable repository fixture avoids that failure.

## Tested paths and identity

- Miles: `/job/miles/miles/__init__.py`, commit `3a666ee4f7d25098796d80730afeb289cd8925f1`.
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`, version `0.5.17.dev2157+ga8e5c632f`.
- Megatron core: `/root/Megatron-LM/megatron/core/__init__.py`, version `0.19.0+8c1e05747`.
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, version `2.9.1+rocm7.2.0.git7e1940d4`.
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Aiter native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- FlashAttention native module: `/opt/venv/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`.
- Model: Qwen3-0.6B, bfloat16, `model.safetensors` SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

The rollout path is SGLang's offline `Engine.generate(..., return_logprob=True, logprob_start_len=0)` with deterministic inference, Triton attention, CUDA graph and radix cache disabled. Training uses the HF Qwen3 forward, Miles' SGLang Triton attention bridge where stated, and `miles.backends.training_utils.loss_hub.math_utils.calculate_log_probs_and_entropy(..., true_on_policy=True)`.

## Reproduction

Download the 1.5 GiB model outside the working clone:

```bash
export HF_HOME=/job/cache/hf
mkdir -p /job/cache/models /job/cache/hf /job/cache/triton
hf download Qwen/Qwen3-0.6B --local-dir /job/cache/models/Qwen3-0.6B
```

Run the primary one-GPU fixture and summarize it:

```bash
export HF_HOME=/job/cache/hf
export TRITON_CACHE_DIR=/job/cache/triton
export SGLANG_USE_AITER=0
export TOKENIZERS_PARALLELISM=false
cd /job/miles
timeout 900 /opt/venv/bin/python reports/j-473f9c74680e/measure_gfx950_logprobs.py \
  --model-path /job/cache/models/Qwen3-0.6B \
  --output reports/j-473f9c74680e/results_batch_invariant.json \
  --batch-sizes 1,2,4,8 \
  --timeout-seconds 300
/opt/venv/bin/python reports/j-473f9c74680e/summarize_results.py \
  --results reports/j-473f9c74680e/results_batch_invariant.json \
  --output reports/j-473f9c74680e/summary.json
```

`SGLANG_USE_AITER=0` avoids Aiter's shared `/tmp/aiter_configs` lock in this container and selects an available SGLang path; it is recorded in the JSON. The fixture uses only the one assigned GPU, a 300-second engine/process-group timeout, and its own SGLang subprocesses.

## What this does and does not prove

This reduced case proves that, on this MI350X and software stack, the existing batch-invariant setup makes packed and unpacked training bit-identical and removes controlled batch-size leakage. It also proves that a nonzero cross-engine log-prob mismatch remains for a small dense representative model.

It does not prove full FSDP/Ray lifecycle parity, optimizer or gradient behavior, weight synchronization, decode-stage parity, long-sequence behavior, MoE or hybrid architectures, multi-GPU scaling, or performance. It does not isolate the remaining mismatch to a particular GEMM, normalization, RoPE, or attention kernel. It also does not test the default Aiter-selected SGLang path because Aiter was disabled to keep cache locking job-private.
