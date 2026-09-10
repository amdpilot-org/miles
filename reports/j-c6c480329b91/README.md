# Two-rank Miles weight-transfer investigation

This fixture exercises the real Miles Llama canonical converter, the existing
`broadcast` weight-transfer hook, and an explicit two-rank send/receive leg on
two assigned AMD Instinct MI350X GPUs. It does not claim coverage for a
multi-relay rollout topology.

## Baseline

The smallest real GPU baseline is two optimizer/update cycles:

```bash
PYTHONPATH=/job/miles TORCH_NCCL_BLOCKING_WAIT=1 \
/opt/venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 \
reports/j-c6c480329b91/investigate_weight_transfer.py \
--cycles 2 --failure-cycle 0 \
--output reports/j-c6c480329b91/baseline.json
```

Baseline results:

- Broadcast forward match: passed.
- Broadcast checksum match: passed.
- Canonical conversion match: passed.
- Send/receive forward match: passed.
- Send/receive checksum match: passed.
- Send/receive overlap fraction: approximately 0.94–0.97.

The full 64-cycle run and bounded truncated-bucket recovery are still pending.
