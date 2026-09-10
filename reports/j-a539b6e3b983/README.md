# Two-GPU Miles recovery/update fixture

This is a bounded hardware investigation for radixark/miles#1724, recorded under
amdpilot-org/amdpilotv2#402. It uses the mounted `amdpilot-org/miles` clone and
does not modify Miles core behavior.

## Result

The fixture passed on two assigned AMD Instinct MI350X `gfx950` GPUs. The ordered
evidence is in `evidence.json`.

The run exercised:

- Actual `MilesRouter` admission and proxying.
- Actual `RolloutServer` cell add/remove/dispose lifecycle.
- Actual `ServerCell.init()`, `ServerCell.tick()`, and
  `ServerCell.mark_weights_ready()` state transitions.
- Actual `InferenceController.start_update_weights()` and
  `InferenceController.end_update_weights()` hooks.
- Actual `TrainerController.update_weights()` orchestration.
- Actual `SGLangApiClient.update_weights_from_tensor()` weight-update calls.

The synthetic rollout workers use one-element Torch tensors on each assigned GPU.
The old weight produces `old` / `old-v0`; the new weight produces `new` /
`new-v1`. No model download is required.

The observed order was:

1. Both recovered engines entered `StatePendingWeights` while serving old output.
2. The router had no workers and rejected generation traffic.
3. The first actor update completed, registered both engines, and routed only
   `new-v1` output across both GPUs.
4. The GPU 1 engine was replaced. The replacement served old output directly but
   stayed in `StatePendingWeights` and was absent from the router.
5. During that recovery window, routed traffic used only the synchronized GPU 0
   worker and returned `new-v1`.
6. An injected non-retryable actor update failure left the replacement pending
   and unavailable to the router.
7. The successful recovery update registered the replacement. Final routed
   traffic returned `new-v1` from both GPUs.

## Reproduction

Validation on the supplied image:

- The two-GPU fixture completed successfully and wrote all 19 ordered events.
- `pytest -q tests/fast/ray/test_update_weights_ordering.py
  tests/fast/ray/rollout/test_server_cell_state_machine.py` passed 58 tests.

From the repository root on the supplied image:

```bash
mkdir -p /job/.cache/tmp
export TMPDIR=/job/.cache/tmp
export AITER_CONFIG_GEMM_A4W4=/sgl-workspace/aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8=/sgl-workspace/aiter/aiter/configs/a8w8_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=/sgl-workspace/aiter/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv
export AITER_CONFIG_FMOE=/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv
export AITER_CONFIG_GROUPED_FMOE=/sgl-workspace/aiter/aiter/configs/tuned_grouped_fmoe.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
export AITER_CONFIG_A8W8_BATCHED_GEMM=/sgl-workspace/aiter/aiter/configs/a8w8_tuned_batched_gemm.csv
export AITER_CONFIG_BF16_BATCHED_GEMM=/sgl-workspace/aiter/aiter/configs/bf16_tuned_batched_gemm.csv
export AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv

/opt/venv/bin/python reports/j-a539b6e3b983/recovery_fixture.py \
  --output reports/j-a539b6e3b983/evidence.json
```

The explicit AITER variables select single existing config files. Without them,
the installed AITER import path attempts a shared `/tmp/aiter_configs` merge,
which failed in this container because another user owned the lock. The rerun
used only job-local temporary/cache state.

## Scope and limitations

This reduced case proves the Miles admission ordering invariant when recovery
produces an updatable engine and the actor update hook is allowed to complete:
an engine serving old checkpoint weights is not admitted to routed generation,
and admission occurs only after the synchronized new version is installed.

It does not prove:

- Correctness of a full SGLang engine, tokenizer, KV cache, or generation path.
- Correctness of Ray worker recovery, process replacement, or health monitoring.
- Correctness of a real Megatron/FSDP actor or a distributed weight broadcast.
- Behavior under concurrent full-scale training traffic.
- Performance or parity with MI355X; this run used MI350X only.
- Any claim about a different Miles commit. The tested commit is recorded in
  `evidence.json`.

No Torch distributed process group is created. The fixture uses only its own
child processes, bounded HTTP timeouts, bounded readiness waits, and bounded
process joins. It contains no blanket sleep or distributed barrier.
