# Issue 920: MI350X prefill/decode contention investigation

## Result

This run used one assigned AMD Instinct MI350X (`gfx950`) and the qualified Miles/SGLang/ROCm stack. It exercised the real Miles rollout `/generate` path and the real Miles session-server `/v1/chat/completions` proxy path against a live SGLang engine. It did not use sleeps, idle loops, or synthetic latency.

The bounded workload was:

- 32 controlled waves in each of three modes: prefill-only, decode-only, and concurrent prefill plus decode.
- One 16,384-token prefill request per prefill wave.
- Eight concurrent short chat-decode requests per decode wave.
- One 16,384-token prefill request plus eight concurrent short chat-decode requests per concurrent wave.
- 576 total inference requests.
- Qwen2.5-0.5B-Instruct, context length 32,768, 512-token chunked prefill, and `prefill-decode-interval=1`.

All 576 requests returned the expected deterministic token IDs and completion counts. The concurrent short-decode latency increase was dominated by scheduler queueing, not by a large increase in decode compute time. This supports the issue-920 observation that a long-context prefill burst can starve concurrent short decodes through queueing, while leaving the long prefill’s own compute time essentially unchanged.

This single-engine run does not validate a multi-shard routing-affinity fix and makes no claim about a larger topology.

## Environment

- GPU: 1 × AMD Instinct MI350X, `gfx950`, capability `(9, 5)`.
- Runtime: `amdpilotv2/miles-job:gbt350-d957-20260909`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.
- SGLang: `0.5.17.dev2157+ga8e5c632f`.
- Miles commit: `e5125a97e1fd383f005f4de258a5985026e09425`.
- SGLang commit: `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`.
- Aiter commit: `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, 954 MiB downloaded, below the 8 GiB limit.

Imported source and native paths used by the run:

- Miles: `/job/miles/miles`
- SGLang: `/sgl-workspace/sglang/python/sglang`
- Aiter Python package: `/sgl-workspace/aiter/aiter`
- Aiter native modules: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`, `/sgl-workspace/aiter/aiter/jit/module_sample.so`, and other `.so` files under the same `jit` directory.
- Torch: `/opt/venv/lib/python3.10/site-packages/torch`
- ROCm probe: `/opt/rocm-7.2.0/bin/rocminfo`

## Server configuration

SGLang was launched with:

- `--context-length 32768`
- `--chunked-prefill-size 512`
- `--prefill-decode-interval 1`
- `--schedule-policy fcfs`
- `--max-running-requests 64`
- `--max-queued-requests 256`
- `--disable-radix-cache`
- `--enable-metrics`
- `--attention-backend triton`
- `--mem-fraction-static 0.75`
- `--random-seed 920`

The Miles session server used:

- Backend URL: `http://127.0.0.1:31111`
- Bind address: `127.0.0.1:31113`
- Timeout: 600 seconds
- TITO model: `default`
- Message matcher: `strict`
- Pause generation mode: `in_place`

Unique rendezvous ports used:

- SGLang HTTP: `31111`
- SGLang NCCL: `31112`
- Miles session server: `31113`
- Training-fixture process group: `31114`

All process-group timeouts were bounded at 300 seconds.

## Inference results

Times are seconds unless stated otherwise. `p50`, `p90`, `p95`, and `p99` are percentiles.

### Long-context prefill

| Mode | Requests | Queue p50 | E2E p50 | Compute p50 | Prefill p50 | Decode p50 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill-only | 32 | 0.000124 | 0.573600 | 0.573489 | 0.571492 | 0.001356 |
| Concurrent | 32 | 0.000119 | 0.575718 | 0.575600 | 0.573730 | 0.001351 |

The concurrent prefill’s compute-time p50 changed by about 0.0021 seconds versus prefill-only. The long prefill itself was not materially slowed by the eight concurrent short decodes.

### Short chat decode

| Mode | Requests | Queue p50 | E2E p50 | Compute p50 | Prefill p50 | Decode p50 |
|---|---:|---:|---:|---:|---:|---:|
| Decode-only | 256 | 0.002388 | 0.030797 | 0.028948 | 0.015393 | 0.002847 |
| Concurrent | 256 | 0.561723 | 0.583618 | 0.021979 | 0.015361 | 0.003182 |

In the concurrent case, queueing accounted for roughly 96% of the short-decode end-to-end latency:

- Concurrent decode E2E p50: `0.583618`
- Concurrent decode queue p50: `0.561723`
- Concurrent decode compute p50: `0.021979`

The short decode’s compute time remained close to the decode-only baseline. The dominant change was scheduler queueing while the 16,384-token prefill was being processed.

The first decode-only wave included a cold-start outlier, visible in the raw results as a maximum E2E latency of `0.871541` seconds. This does not change the median or the concurrent-mode conclusion.

### Token correctness

| Path | Requests | Exact token correct | Completion-count correct |
|---|---:|---:|---:|
| Prefill-only rollout `/generate` | 32 | 32 | 32 |
| Decode-only session chat | 256 | 256 | 256 |
| Concurrent rollout `/generate` | 32 | 32 | 32 |
| Concurrent session chat | 256 | 256 | 256 |

Expected deterministic outputs:

- Prefill output token ID: `[1147]`
- Session chat content: `PASS`
- Session chat token IDs: `[49792, 151645]`
- Session chat completion tokens: `2`

## Training, collective, and update fixture

`training_collective_update_fixture.py` performs real GPU training work on the assigned MI350X:

- NCCL process group with world size 1.
- Unique TCP rendezvous: `tcp://127.0.0.1:31114`.
- Bounded process-group timeout: 300 seconds.
- Eight repeated optimizer cycles.
- A `1024 × 1024` float32 linear layer.
- Batch size 128.
- SGD with learning rate `0.01`.
- Real `barrier()` and `all_reduce(SUM)` calls on each cycle.

Numerical checks:

- Initial loss: `1.329919695854187`
- Final loss: `1.3269267082214355`
- Loss decreased: yes.
- All eight gradients finite: yes.
- Weight checksum drifted from `19.325218200683594` to `19.318851470997266`, confirming repeated updates.

## Reproduction

1. Download the model outside the Miles worktree:

   ```bash
   hf download Qwen/Qwen2.5-0.5B-Instruct \
     --local-dir /job/models/Qwen2.5-0.5B-Instruct
   ```

2. Launch SGLang on the assigned GPU:

   ```bash
   export AITER_CONFIG_GEMM_BF16=/job/cache/aiter_configs/bf16_tuned_gemm.csv
   /opt/venv/bin/python -m sglang.launch_server \
     --model-path /job/models/Qwen2.5-0.5B-Instruct \
     --host 127.0.0.1 --port 31111 \
     --nccl-port 31112 --dist-timeout 300 \
     --tp-size 1 --device cuda \
     --context-length 32768 \
     --chunked-prefill-size 512 \
     --prefill-decode-interval 1 \
     --schedule-policy fcfs \
     --max-running-requests 64 \
     --max-queued-requests 256 \
     --disable-radix-cache \
     --enable-metrics \
     --attention-backend triton \
     --mem-fraction-static 0.75 \
     --random-seed 920
   ```

3. Launch the Miles session server:

   ```bash
   /opt/venv/bin/python -m miles.rollout.session.server \
     --config-json '<contents of artifacts/session-config.json>'
   ```

4. Run the training/collective/update fixture:

   ```bash
   export PYTHONPATH=/job/miles
   /opt/venv/bin/python reports/j-d6bfff6c1917/training_collective_update_fixture.py
   ```

5. Run the 32-wave inference benchmark:

   ```bash
   export PYTHONPATH=/job/miles
   /opt/venv/bin/python reports/j-d6bfff6c1917/bench_issue_920.py \
     --waves 32 \
     --long-context-tokens 16384 \
     --short-decodes-per-wave 8 \
     --short-max-tokens 8 \
     --output /job/artifacts/issue-920/results.json
   ```

## Artifacts

- `artifacts/results.json` — raw per-request records, metadata, aggregate latency distributions, and path-level aggregates.
- `artifacts/session-config.json` — exact Miles session-server configuration used for the run.
- `artifacts/training_collective_update.json` — training/collective/update numerical checks.
- `artifacts/sglang-server.log` — SGLang startup, prefill, decode, and scheduler logs.
- `artifacts/session-server.log` — Miles session-server startup and proxy logs.

## Limitations

- Only one assigned GPU was available, so no multi-GPU distributed inference topology is claimed.
- The training fixture uses world size 1; its NCCL calls are real but do not test multi-rank communication.
- The benchmark uses one SGLang engine and one Miles session server, not a multi-shard router fleet.
- The model is small and the context is bounded at 32,768 tokens; this is not a claim about larger models or contexts.
- The first decode-only wave contains a cold-start outlier; the report uses medians and percentiles rather than only means.
