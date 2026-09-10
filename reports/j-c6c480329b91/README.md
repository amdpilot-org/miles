# Two-rank Miles weight-transfer investigation

This investigation exercises the real Miles Llama canonical converter, the
existing `broadcast` weight-transfer hook, and an explicit two-rank
send/receive leg on two assigned AMD Instinct MI350X GPUs. It does **not**
claim coverage for a multi-relay rollout topology, and it does not add the
proposed `sendrecv_broadcast` transport mode.

Reference issue: `radixark/miles#1120`.

## Scope

- Two assigned GPUs: `AMD Instinct MI350X` (`gfx950`).
- Real optimizer/update cycles: 64.
- Transfer dtypes: alternating `bfloat16` and `float32`.
- Bucket boundaries: non-divisible (`3 * hidden * hidden * itemsize + 7`).
- Checksums: SHA-256 per bucket.
- Version commit: explicit per-cycle version tensor.
- Failure case: bounded truncated bucket at cycle 8, followed by recovery.
- Consumer forward: compared against the committed canonical weights.
- Rendezvous: unique port selected by `torchrun --standalone`.
- Process-group timeout: 120 seconds.
- Model downloads: 0 GB; synthetic/local random initialization only.

## Reproduction

Baseline:

```bash
PYTHONPATH=/job/miles TORCH_NCCL_BLOCKING_WAIT=1 \
/opt/venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 \
reports/j-c6c480329b91/investigate_weight_transfer.py \
--cycles 2 --failure-cycle 0 \
--output reports/j-c6c480329b91/baseline.json
```

Full run:

```bash
PYTHONPATH=/job/miles TORCH_NCCL_BLOCKING_WAIT=1 \
/opt/venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 \
reports/j-c6c480329b91/investigate_weight_transfer.py \
--cycles 64 --failure-cycle 8 \
--output reports/j-c6c480329b91/results.json
```

## Results

- Cycles completed: 64.
- Broadcast forward match: passed.
- Broadcast checksum match: passed.
- Canonical conversion match: passed.
- Send/receive forward match: passed.
- Send/receive checksum match: passed.
- Truncated-bucket failure detected: cycle 8.
- Truncated-bucket recovery: passed.
- Maximum broadcast forward error: 0.
- Maximum send/receive forward error: 0.
- Maximum canonical conversion error: 0.0369.
- Average broadcast transfer time: 4.825 ms.
- Median send/receive transfer time: 7.596 ms.
- Average send/receive overlap fraction: 0.941.
- Average source compute time: 0.324 ms.
- Average reference consumer forward time: 0.306 ms.

## Environment

- Miles commit under test: `02fe59624c335d5490bf08e6685c43415df3859f`.
- Miles base commit: `df0e677f6dd51fa551d48d37860812ece904cd8f`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm/HIP: `7.2.26015-fc0010cf6a`.
- SGLang: `0.5.17.dev2157+ga8e5c632f`.
- Miles import path: `/job/miles/miles/__init__.py`.
- Torch import path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- SGLang import path: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Aiter native path: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

## Implementation note

The qualified environment initially failed when Miles’ unquantized canonical
conversion eagerly imported SGLang/Aiter quantizers and hit a read-only
`/tmp/aiter_configs` lock. This change makes those quantizer imports lazy so
the real conversion path works without altering quantization behavior.

## Tests

- `tests/fast/backends/megatron_utils/test_qwen2_true_on_policy_conversion.py`
- `tests/fast/backends/training_utils/weight_update/test_hf_weight_iterator.py`

Both pass with `pytest --noconftest`. The broader test conftest still imports
SGLang/Aiter and can hit the same environment lock; it was not run in full.
