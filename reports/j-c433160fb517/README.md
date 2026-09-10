# Miles issue 1724 recovery-order fixture

## Result

**PASS** on mirror `main` commit `e5125a97e1fd383f005f4de258a5985026e09425` with two assigned AMD Instinct MI355X GPUs.

The current mirror already contains the likely closing change, `7ecf7cc6f32c5fb230231f9824a0f40d229b3f3c` (“Mark cell weights ready only once the update weights window closes”). This change adds an executable, reduced-scale check of that behavior; it does not change production Miles code.

## Fixture

`recovery_fixture.py` launches two Torch ranks. Each rank places a tiny actor model and a tiny rollout model on its assigned GPU. The old weight produces `7.0` and has version `old-v1`; the new weight produces `13.0` and has version `new-v2`.

Rank 0 drives the real Miles objects and hooks:

- `RolloutServer`
- `ServerCell.init()`, `ServerCell.tick()`, and `ServerCell.mark_weights_ready()`
- `InferenceController.start_update_weights()`
- `InferenceController.end_update_weights()`
- `SGLangApiClient.update_weights_from_tensor()`
- `SGLangRouterApiClient.add_worker()` and `remove_worker()`

Only worker-address discovery is replaced with fixed local URLs. Local FastAPI servers implement the small HTTP subset expected by the real Miles clients. The fixture removes and re-adds engine 1 to exercise recovery, then performs both a successful and an intentionally failed weight synchronization.

The process group uses a 90-second timeout. HTTP operations use a 15-second timeout. The fixture contains no blanket sleeps and no barriers; its only synchronization collectives are the required readiness, command, and weight broadcasts.

## Recorded ordering

`evidence.jsonl` is the complete ordered output from the passing run:

1. Environment and imported native paths.
2. Rank 1 engine readiness.
3. Both cells are `StatePendingWeights` before the first sync.
4. The router has no workers before the first sync.
5. The first completed sync admits output `13.0` / `new-v2`.
6. After recovery, engine 1 directly reports `old-v1` and remains `StatePendingWeights`; router traffic still returns `13.0` / `new-v2` from engine 0.
7. After successful sync, both engines report `new-v2` and router traffic returns `13.0`.
8. After an injected engine-1 sync failure, engine 1 remains `StatePendingWeights` and directly reports `old-v1`; router traffic still returns `13.0` / `new-v2` from engine 0.
9. Fixture completion.

## Reproduction

From the repository root, choose a free private rendezvous port and run:

```bash
PYTHONPATH="$PWD" /opt/venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=<free-port> \
  reports/j-c433160fb517/recovery_fixture.py
```

The adjacent fast suites were also run:

```bash
/opt/venv/bin/python -m pytest -q \
  tests/fast/ray/test_update_weights_ordering.py \
  tests/fast/ray/rollout/test_server_cell_state_machine.py
```

Result: **58 passed**.

## Scope and limits

This reduced case proves that the current mirror's real recovery state machine and update-window hooks keep a recovered, updatable engine unroutable until a successful weight update, under both success and injected-failure paths. It also proves that old and new weights are distinguishable at the router and directly at each engine.

It does not prove:

- full SGLang engine startup, model loading, or generation internals;
- the Ray worker provider or real worker-address discovery;
- the complete Megatron/FSDP actor update implementation;
- behavior under concurrent traffic or a crash during an in-flight request;
- router load-balancing behavior beyond admission/registration ordering.

No model weights were downloaded. The environment's shared `/tmp/aiter_configs` directory is not writable by this job, so the fixture reads Aiter's installed config files directly through a process-local compatibility shim. This avoids changing node-wide state and does not affect the Miles admission behavior under test.
