# Two-GPU Miles rollout lifecycle investigation

## Conclusion

The reduced fixture exercised the supported Miles router and SGLang lifecycle/update hooks on two assigned AMD Instinct MI350X (`gfx950`) GPUs for 32 repeated generate/drain/release/train/update/wake cycles. Selective release and wake operations are supported in this fixture. This is **not** evidence for an elastic scheduler, node-wide scheduling, or MI355X performance equivalence.

## Topology

- Engine A uses GPU 1 with SGLang memory saver enabled and is removed from the router, released, trained beside, woken, updated, checked, and then re-admitted.
- Engine B uses GPU 0 and remains router-admitted during each release phase.
- Two-rank DDP training uses both assigned GPUs. Both GPUs are therefore colocated with a training rank; only engine A is selectively drained/released.
- The router, both SGLang engines, and both DDP ranks use fresh loopback rendezvous ports. Process-group and recovery operations are bounded at 300 seconds.

## Model

`make_local_model.py` creates a local two-layer Qwen2 model with 13,173,888 parameters and a small GPT-2 tokenizer. The model directory is 29,909,617 bytes, well below the 8 GB download bound. No large pretrained model is downloaded.

## Lifecycle sequence

Each cycle performs:

1. Direct generation on both engines and router generation before release.
2. Router removal of engine A and router generation through engine B only.
3. SGLang `release_memory_occupation(tags=["weights", "kv_cache"])` for engine A.
4. Real two-rank DDP forward, backward, gradient all-reduce, and AdamW optimizer step.
5. A local control forward and log-probability check.
6. NCCL weight update into engine B.
7. SGLang wake of engine A (or the injected failed wake at cycle 16).
8. NCCL weight update into engine A.
9. Weight-version, checksum, direct-output, and router re-admission checks.

The fixture uses the supported Miles/SGLang APIs for router admission, memory release/resume, generation pause/resume, distributed weight-update group creation, begin/update/end weight update, and weight-version advancement. It uses `flush_cache=True` after distributed updates so output comparison is not contaminated by pre-update KV cache reuse.

## Injected failures and bounded recovery

- **Cycle 16 wake failure:** `resume_memory_occupation(tags=["bogus"])` reaches the real SGLang scheduler path. The unsupported tag raises `KeyError: 'bogus'` and terminates engine A. The launcher performs at most one engine restart, polls health for at most 300 seconds, and the worker reconnects the NCCL update group. Recovery took 40.22 seconds and succeeded; cycle 17 also passed.
- **Cycle 24 update failure:** an intentionally invalid expected LoRA checksum fails `end_weight_update` without terminating engine B. Cycle 25 performs a successful update and passes all checks.

## Results

The final 32-cycle run completed 224 requests with zero request failures:

- 32/32 router admission cycles passed.
- 32/32 releases reported positive freed memory; mean release was 53.63 GB.
- 32/32 cycles had matching engine weight versions.
- 64/64 engine checksum checks covered all expected engine tensors with zero mismatches.
- 64/64 direct output token comparisons matched the local control forward.
- 64 real distributed weight-update events occurred: 63 succeeded and 1 was the intentional cycle-24 failure.
- 32 unique training losses and 32 unique actor/engine checksums demonstrate state drift rather than repeated idle work.
- Maximum output log-probability difference was 0.03035.

The one-cycle baseline used the same real two-GPU path. It completed 7 requests, 2 real updates, and 2/2 checksum checks. Its engine-B output differed because the original update path used `flush_cache=False`; the final run uses the supported post-update cache flush and passes 64/64 output comparisons.

## Reproduction

From the repository root:

```bash
/opt/venv/bin/python reports/j-50c9a529dc1c/make_local_model.py

/opt/venv/bin/python reports/j-50c9a529dc1c/launch_investigation.py \
  --output-dir /tmp/miles-baseline \
  --cycles 1 \
  --baseline

/opt/venv/bin/python reports/j-50c9a529dc1c/launch_investigation.py \
  --output-dir /tmp/miles-investigation \
  --cycles 32 \
  --fail-wake-cycle 16 \
  --fail-update-cycle 24

/opt/venv/bin/python reports/j-50c9a529dc1c/summarize.py \
  --baseline /tmp/miles-baseline/summary.json \
  --investigation /tmp/miles-investigation/summary.json \
  --environment /tmp/miles-investigation/environment.json \
  --output /tmp/miles-investigation/evidence_summary.json
```

The launcher starts only its own router, engines, and worker processes. It does not change node-wide scheduling or kill unrelated processes.

## Tested environment

- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python`
- Miles commit: `df0e677f6dd51fa551d48d37860812ece904cd8f`
- Miles source: `/job/miles/miles`
- SGLang commit: `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`
- SGLang source: `/sgl-workspace/sglang/python/sglang`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch`
- HIP: `7.2.26015-fc0010cf6a`
- Transformers: `5.12.1`
- Transformers source: `/opt/venv/lib/python3.10/site-packages/transformers`
- Torch memory saver: `0.0.10b1`
- Native preload: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so`

`NCCL_P2P_DISABLE=1` is set for this fixture because the ROCm P2P path raised `HIP invalid argument` in the qualified environment. This is a fixture environment setting, not a node-wide scheduler change.

## Limitations

This is a reduced two-GPU operator investigation. It does not establish behavior for larger topologies, multi-node scheduling, or MI355X. The failed wake is deliberately unsupported and requires an engine process restart; the fixture bounds that restart to one attempt and 300 seconds. No elastic scheduler is claimed.
