# Job `j-06a291616028`: two-GPU actor/rollout update investigation

This is a reduced, real-stack investigation fixture for read-only reference
`radixark/miles#921`. It validates the existing implementation in the
qualified Miles environment; it does not assume a candidate change is correct
and does not claim that a larger topology passed.

## Fixture

- Two `torchrun` ranks perform real bf16 forward/backward and DDP gradient
  all-reduce on the two assigned AMD Instinct MI350X (`gfx950`) GPUs.
- Rank 0 streams the full Qwen3-0.6B state dict through a real NCCL process
  group to a TP1 SGLang engine. The engine also uses GPU 1, so both assigned
  devices participate in the distributed case.
- Every cycle performs real training, two pre-update requests, eight concurrent
  requests around an in-place pause/update/resume, and two post-update requests.
- The update sequence is pause, begin update, NCCL broadcast, end update,
  version publish, and generation resume. Process-group timeout is 120 seconds
  and HTTP request timeout is 30 seconds.
- The engine checksum is compared with the actor checksum after every update.
  SGLang stores Qwen3 attention and MLP shards fused, so the comparison hashes
  the actor's `q/k/v` and `gate/up` shards in the engine's fused layout.
- Cycle 16 injects a bogus LoRA checksum manifest through the real
  `end_weight_update` API. SGLang returns HTTP 400; cycle 17 performs a normal
  successful update and proves recovery.

## Actual result

The completed run used 32 cycles, concurrency 8, 64 new tokens per request, and
failure cycle 16. `results.json` contains the aggregate and per-cycle evidence;
the original full summary and logs remain under
`/job/artifacts/j-06a291616028/`.

| Measurement | Result |
| --- | --- |
| Real weight-update events | 32 |
| Successful updates | 31 |
| Bounded failed updates | 1 (cycle 16, `HTTPStatusError`, HTTP 400) |
| Successful recovery | cycle 17 |
| Admitted requests | 384 |
| Completed requests | 384 |
| Failed requests | 0 |
| Before-update requests | 64 |
| Concurrent during-update requests | 256 |
| After-update requests | 64 |
| Request timeout causes | none in any phase |
| Engine/version consistency | 32/32 cycles |
| Requests with mixed weight-version spans | 0/384 |
| Checksum mismatches | 0/32 cycles |
| All 226 engine tensors checked | 32/32 cycles |
| Expected/engine overall checksum equality | 32/32 cycles |
| Unique actor overall checksums | 32 |
| Unique engine overall checksums | 32 |
| Unique training losses | 32 |

The actor state dict has 311 named tensors. The TP1 engine exposes 226 tensors:
170 name-matched tensors, 56 fused tensors constructed from actor shards, and
no unexpected engine tensors. The tied `lm_head` and split actor shards account
for the 141 actor-only names. Every cycle checked all 226 engine tensors.

The recorded scalar weight-norm delta was 0.0, but state drift is demonstrated
by 32 distinct actor checksums, 32 matching engine checksums, and 32 distinct
measured losses.

## GPU phase timings

Each update transfers 1,503,264,768 bytes. Timings below are mean/min/max
seconds over 32 real updates. Training timing is mean/min/max over 32 DDP
steps. The one-time update-group connection took 0.03635 seconds.

| GPU phase | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| DDP training step | 0.76589 | 0.55822 | 6.20887 |
| Pause generation | 0.00314 | 0.00218 | 0.00582 |
| Begin update | 0.01409 | 0.01245 | 0.01695 |
| NCCL weight broadcast | 0.10786 | 0.03501 | 2.35066 |
| End update | 0.00441 | 0.00398 | 0.00627 |
| Version publish | 0.00149 | 0.00134 | 0.00239 |
| Resume generation | 0.00124 | 0.00113 | 0.00173 |
| Total update window | 0.13224 | 0.05690 | 2.38013 |

## Tested environment

- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python` (3.10.12)
- Miles commit: `e5125a97e1fd383f005f4de258a5985026e09425`
- Miles source: `/job/miles/miles`
- SGLang commit: `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`
- SGLang source: `/sgl-workspace/sglang/python/sglang`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch`
- HIP: `7.2.26015-fc0010cf6a`
- RCCL: `2.27.7`
- Transformers: `5.12.1`
- Transformers source: `/opt/venv/lib/python3.10/site-packages/transformers`
- GPUs: two AMD Instinct MI350X, capability `(9, 5)`, `gfx950`
- Model: local Qwen3-0.6B, 1,503,300,328 downloaded bytes

## Reproduction

```bash
mkdir -p /job/cache/models /job/cache/hf
HF_HOME=/job/cache/hf /opt/venv/bin/hf download Qwen/Qwen3-0.6B \
  --local-dir /job/cache/models/Qwen3-0.6B

/opt/venv/bin/python reports/j-06a291616028/launch_investigation.py \
  --model-path /job/cache/models/Qwen3-0.6B \
  --output-dir /job/artifacts/j-06a291616028 \
  --cycles 32 --concurrency 8 --max-new-tokens 64 --fail-cycle 16

/opt/venv/bin/python reports/j-06a291616028/summarize_results.py \
  --summary /job/artifacts/j-06a291616028/summary.json \
  --environment reports/j-06a291616028/environment.json \
  --output reports/j-06a291616028/results.json
```

The launcher allocates unique HTTP, torchrun, SGLang NCCL, and weight-update
NCCL ports. It starts only its own SGLang and torchrun process groups and
terminates those groups on exit. Its only wait is bounded server-health polling;
no sleep is used as GPU work, update synchronization, or a fake result. It does
not use blind timeout increases, fake inference responses, or unrelated process
kills.

## Limitations

- This is a reduced two-GPU fixture, not a claim that a larger actor, rollout,
  tensor-parallel, or pipeline-parallel topology passed.
- The SGLang engine is TP1 on GPU 1 while the two actor ranks occupy GPUs 0 and
  1. This uses all assigned devices but is not a MI355X performance-equivalence
  claim.
- No request timeout occurred in this run. The empty timeout-cause list is the
  measured result for this topology and workload, not evidence that other
  configurations cannot time out.
- The model download is 1.5 GB, below the 8 GB limit. Source/model constraints
  are recorded as evidence only.
