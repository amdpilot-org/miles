# J-69f76ab697d3: SGLang rollout grouping investigation

## Status

This is an **early draft, not a merge recommendation**. The reduced real-model run shows the expected request-count reduction and a latency win, but it fails exact token and logprob parity. The result is therefore a negative parity result, not proof that group batching is safe to enable by default.

Reference context: `radixark/miles#1567`, upstream draft `radixark/miles#1568`, H2 roadmap `radixark/miles#2853`, AMD Q3 roadmap `radixark/miles#2025`, and coordination tracker `amdpilot-org/amdpilotv2#402`.

## What changed

- Ported the open group-batching idea onto the current mirror implementation in `miles/rollout/sglang_rollout.py`.
- Kept the current `compute_routing_headers` behavior for individual requests.
- Added one batched `/generate` request per prompt group when custom generation, deterministic inference, routing-key policies, LoRA, adapters, multi-turn responses, or multimodal inputs are present.
- Added `tools/rollout_grouping_probe.py`, which runs the real `generate_and_rm_group` path in both modes, counts HTTP requests, and compares sample identity, termination, tokens, responses, and logprobs.
- Added focused tests for nested input IDs, output-to-sample order, and conservative fallbacks.

## Real fixture

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, about 966 MB in the local Hugging Face cache (well below the 8 GB cap).
- Hardware/runtime: one assigned AMD Instinct MI355X (`gfx950`), Torch `2.9.1+rocm7.2.0.git7e1940d4`, SGLang `0.5.17.dev2157+ga8e5c632f`.
- Server: direct local SGLang engine, no router; default AITER attention; CUDA graphs disabled; radix cache disabled; server `--random-seed 1567`.
- Probe: 16 groups × 4 samples, 8 new tokens, temperature 0, top-p 1, top-k 1, seed 1567, one same-shape warmup, three measured runs.

## Result

Across the three measured runs:

- Individual mode made 192 requests total; grouped mode made 48.
- Both modes produced 64 samples per run and preserved object and index order.
- All 192 samples per mode ended `TRUNCATED` at the 8-token cap; no sample exceeded the cap.
- Median latency was 0.41540 s individual versus 0.16596 s grouped, a 2.503× observational speedup.
- Exact functional parity failed in every run. Token/response mismatch counts were 11/64, 4/64, and 10/64.
- Maximum absolute logprob differences were 2.85689, 2.51545, and 2.61179; means were 0.11769, 0.04809, and 0.14808.

See `results.json` for the complete per-run measurements.

## What this proves

- The Miles grouping path really reaches SGLang's nested `/generate` input form and returns one output per input sample.
- Grouping reduces request count by the expected factor for this fixture and can improve wall-clock latency on one MI355X.
- The output-to-sample mapping and length termination remain intact for this reduced workload.

## What this does not prove

- It does **not** prove token identity or logprob alignment. Greedy decoding still diverged between individual and grouped requests, so this draft should not be treated as functionally equivalent.
- It does not test the SGLang router, multi-engine routing, DP/TP process groups, reward models, multi-turn continuation, multimodal inputs, LoRA, or a production-sized batch.
- Latency is observational only: it is a small direct-engine fixture on one GPU, not a controlled benchmark of the upstream 8×H200 workload.
- The run covers length termination, not a natural EOS/stop-string termination case.

## Negative and control results

- A 13 MB `hf-internal-testing/tiny-random-gpt2` fixture was attempted first. Stock SGLang rejected its missing `architectures` field; after adding `GPT2LMHeadModel` locally, both AITER and Triton attention crashed with HIP `unspecified launch failure` during real batch execution. It was not used as evidence.
- A `torch_native` attention control with radix cache disabled still failed token/logprob parity in all three runs and had only a 1.022× median latency speedup. This suggests the issue is not solely the AITER attention backend, but it does not isolate the exact SGLang scheduler or kernel source.

## Reproduction

Use private caches outside the checkout:

```bash
export HF_HOME=/job/hf-cache
export SGLANG_CACHE_DIR=/job/sglang-cache
export TMPDIR=/job/tmp
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv
export AITER_CONFIG_BF16_BATCHED_GEMM=/sgl-workspace/aiter/aiter/configs/bf16_tuned_batched_gemm.csv

hf download Qwen/Qwen2.5-0.5B-Instruct --cache-dir /job/hf-cache
MODEL=/job/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port 30128 \
  --tp-size 1 --trust-remote-code \
  --context-length 512 --mem-fraction-static 0.20 \
  --random-seed 1567 --disable-radix-cache \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled
```

Then, from this checkout:

```bash
export PYTHONPATH=/job/miles
python tools/rollout_grouping_probe.py \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port 30128 \
  --groups 16 --samples-per-group 4 \
  --max-new-tokens 8 --runs 3 --warmups 1 \
  --seed 1567 --output /tmp/rollout-grouping.json
```

## Environment evidence

- Miles import: `/job/miles/miles/__init__.py`
- SGLang import: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron Core import: `/root/Megatron-LM/megatron/core/__init__.py`
- Torch import: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP native module: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- SGLang kernel native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- SGLang router native module: `/opt/venv/lib/python3.10/site-packages/sglang_router/sglang_router_rs.abi3.so`
- Transformer Engine native module: `/opt/venv/lib/python3.10/site-packages/transformer_engine/transformer_engine_torch.cpython-310-x86_64-linux-gnu.so`

Preinstalled source paths are environment context; the tested Miles revision is this checkout.

## Validation

```bash
pytest tests/fast/rollout/test_sglang_rollout_grouping.py \
  tests/fast/rollout/inference_rollout/integration/test_basic.py -q
```

Result: 8 passed. `pre-commit` also passed ruff, autoflake, isort, and black for the changed Python files.
