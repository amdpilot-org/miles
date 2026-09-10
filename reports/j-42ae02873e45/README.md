# Reduced P2P MoE weight-sync investigation

Investigates upstream `radixark/miles#2856`; coordination is `amdpilot-org/amdpilotv2#402`.

## Finding

The reported failure is reproducible without a checkpoint. A synthetic one-layer Qwen3-MoE
config with TP=2 and EP=2, published as `nnodes=2`, reaches SGLang's
`MultimemAllGatherer` topology probe through Miles' CPU replica path. The probe calls
`torch.distributed.get_backend()` on `ParallelismContext`'s unregistered `MagicMock`
CPU group and raises the same `ValueError` as the 235B production report.

The CPU replica is structural: it derives parameter layout and does not join the rollout
engine's real process group. Miles now scopes its construction to `nnodes=1`, preserves the
remote TP/EP geometry in `RankParallelismConfig`, and restores the published `nnodes` value
after construction, including on a loader error.

## Reproduction

```bash
/opt/venv/bin/python -m pytest -q --noconftest \
  tests/fast/backends/training_utils/weight_update/test_p2p_cpu_replica.py

FIXTURE_OUTPUT=$PWD/reports/j-42ae02873e45/gpu_result.json \
/opt/venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  reports/j-42ae02873e45/gpu_p2p_weight_sync_fixture.py
```

The fixture uses only its own torchrun subprocesses, both assigned GPUs, and 30-second
process-group timeouts. It downloads no model or checkpoint.

## GPU evidence

`gpu_result.json` records the successful run on two AMD Instinct MI355X devices
(gfx950, capability `(9, 5)`):

- The real Miles CPU-replica method constructed `Qwen3MoeForCausalLM` with 3,472 parameters.
- The published topology was `nnodes=2`, TP=2, EP=2; construction restored `nnodes=2`.
- Explicit NCCL and Gloo groups were constructed with 30-second timeouts.
- An NCCL all-reduce probe produced `3.0` on both ranks.
- All 12 state-dict tensors, totaling 6,944 bytes, were sent rank 0 to rank 1 with NCCL
  point-to-point operations and echoed back; every tensor compared exactly equal.
- Mooncake HIP initialized successfully, registered both GPU buffers, and `send_probe`
  returned `0`, but cross-process `transfer_sync_write` returned `-1`.

## What this proves

- The exact mocked-process-group construction failure is model-size independent and is
  reached by the current Miles P2P CPU-replica path.
- The scoped `nnodes=1` construction workaround preserves the reduced TP/EP shard layout
  and restores the original runtime topology.
- Real two-GPU process-group construction, point-to-point transfer, and reconstructed
  weight equality pass on the assigned gfx950 hardware.

## What this does not prove

- The passing transfer is NCCL point-to-point, not a passing Mooncake production transfer.
  The node has no RDMA HCAs, and this Mooncake build's HIP transport returned `-1` for a
  cross-process GPU write despite successful engine initialization, registration, and probe.
  Therefore the reduced topology is an unsupported Mooncake reproduction, not proof that
  multi-node Mooncake RDMA works.
- The tiny model has one layer, four experts, and 3,472 parameters. It does not cover the
  235B model's full parameter count, quantization layout, multi-node placement, engine
  lifecycle, or end-to-end rollout behavior.
- The fixture verifies state-dict bytes and shapes, not a forward pass through the
  reconstructed MoE model.

## Failed attempt retained

The first GPU launch accidentally imported `miles` from `/root/miles`, which predates this
branch, and therefore reproduced the `MagicMock` failure rather than testing the working
clone. `gpu_run_first_attempt.txt` preserves that log. The fixture now puts `/job/miles`
first on `sys.path`, and `gpu_result.json` records `/job/miles/miles/__init__.py` as the
imported Miles path.

## Runtime provenance

The tested process used `/opt/venv/bin/python`, Torch `2.9.1+rocm7.2.0.git7e1940d4`, and HIP
`7.2.26015-fc0010cf6a`. Imported paths were:

- Miles: `/job/miles/miles/__init__.py`
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron core: `/root/Megatron-LM/megatron/core/__init__.py`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Mooncake native module: `/opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so`
- Transformer Engine: `/opt/venv/lib/python3.10/site-packages/transformer_engine/__init__.py`
- Aiter: `/sgl-workspace/aiter/aiter/__init__.py`

Preinstalled source paths are environment context, not proof of the tested Miles revision.
The fixture records them at runtime and the working clone is under `/job/miles`.
