# Grouped SGLang rollout investigation

## Result

This draft batches the default SGLang rollout at the GRPO prompt-group boundary while retaining the per-sample path for semantics that cannot be represented by one batched request. The reduced GPU fixture exercised Miles' real `generate_and_rm_group` path against SGLang on one assigned AMD Instinct MI350X.

Functional parity was exact for sample identity, generated token IDs, response text, response length, terminal status, and per-token rollout logprobs. The measured latency result is reported separately and was not used as a correctness assertion.

## Hardware and runtime

- Assigned GPUs: 1.
- Device: AMD Instinct MI350X, gfx950, 256 compute units, serial `692517020434`, unique ID `0x6e4208b780600d88`.
- ROCm driver reported by `rocm-smi`: `7.1.1.31500000`.
- Torch: `2.9.1+rocm7.2.0.lw.git7e1940d4`, module `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Torch HIP native library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`.
- SGLang: `0.5.17.dev2157+ga8e5c632f`, module `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Megatron Core: `0.19.0+8c1e05747`, module `/root/Megatron-LM/megatron/core/__init__.py`.
- Triton: `3.6.0+git42270451`, module `/sgl-workspace/triton-custom/python/triton/__init__.py`.
- AITER module: `/sgl-workspace/aiter/aiter/__init__.py`.
- ROCm libraries included `libamdhip64.so`, `librocblas.so`, and `librccl.so` from `/opt/rocm-7.2.0/lib`.

The preinstalled source paths above identify the runtime used for this run; they are not treated as proof of the Miles revision. The tested Miles worktree is the mounted clone at `/job/miles`, based on mirror `main` commit `e5125a97`.

## Reduced fixture

The fixture creates a synthetic Llama model with seed `1567`, 4,227,392 parameters, two layers, hidden size 64, four attention heads, and a 16-byte attention head size. Its bfloat16 safetensors file is about 8.1 MB. Only the small public tokenizer files from `hf-internal-testing/tiny-random-LlamaForCausalLM` are downloaded (about 3.5 MB); no large model weights are used.

The server is launched with:

- one GPU (`--tp-size 1`);
- bfloat16;
- context length 128;
- at most 64 running requests and 4096 total tokens;
- radix cache and CUDA graphs disabled;
- SGLang distributed timeout 60 seconds;
- startup timeout 180 seconds;
- shutdown timeout 15 seconds;
- server random seed 1567.

The server subprocess uses `start_new_session=True`. Cleanup sends SIGTERM, then SIGKILL after the bounded wait, to that process group only. No broad or node-wide process operation is used.

## Comparison

Both modes use greedy sampling (`temperature=0.0`, `top_p=1.0`, `top_k=-1`), 64 prompt groups, four samples per group, and 16 new tokens. Individual mode issues 256 requests through Miles' `generate_and_rm`. Grouped mode issues 64 batched requests through Miles' `generate_and_rm_group`, each containing four input-ID sequences.

The final reproducible run measured three repetitions and used the median:

| Mode | Median latency |
|---|---:|
| Individual requests | 2.655740 s |
| Group-batched requests | 0.651002 s |
| Speedup | 4.079x |

Functional checks passed for all 256 samples:

- `(group_index, index)` keys were preserved;
- token IDs, response text, and response lengths matched;
- length termination produced `Sample.Status.TRUNCATED`;
- a dynamically selected stop token produced `Sample.Status.COMPLETED` and the same final stop token in both modes;
- rollout logprobs aligned position by position within `1e-6` (they were byte-identical in this run).

The end-to-end test passed in 93.36 seconds. Adjacent fast rollout coverage for the old SGLang path passed with 16 passed, 5 skipped, and 42 deselected.

## Failures and negative results

- The mirror had no issue 1567. The read-only upstream issue and PR 1568 were used for context; no upstream state was modified.
- The first public tiny Llama had a four-byte attention head size. AITER rejected it with `Batch prefill requires head size divisible by vector size`. The synthetic 64-hidden model avoids that incompatibility.
- Float32 decode was also unsupported by the AITER paged-attention kernel (`Unsupported data type: torch.float32`). The fixture uses bfloat16.
- AITER attempted to use a shared `/tmp/aiter_configs` lock owned by another user. The run avoids that shared mutable path by setting `AITER_CONFIG_GEMM_BF16` to the existing read-only AITER config file.
- The first request after server startup exceeded a 60-second client timeout while AITER JIT-compiled a decode kernel. After that one-time compile, requests completed normally. The fixture warms both request shapes before measuring.

## What this does not prove

This reduced case does not exercise the full training loop, reward models, a real SGLang router, multi-GPU TP/DP, multimodal inputs, custom generate functions, deterministic per-sample seeds, LoRA, or partial/multi-turn rollout on GPU. Those paths remain on the conservative per-sample fallback. The synthetic model is intentionally tiny, so its 4.079x latency result is evidence for request-granularity overhead on MI350X, not a prediction for every production model or topology. It does not establish MI355X performance.

## Reproduction

From the mounted clone:

```bash
export AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv
/opt/venv/bin/python -m pytest tests/e2e/sglang/test_grouped_generation_parity.py -vv -s
```

The test creates its model and HF cache under `/tmp` by default. For this investigation, the existing model and cache were kept outside the Git worktree at `/job/model-fixture` and `/job/hf-cache`.
