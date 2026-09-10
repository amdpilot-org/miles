# Miles DP balancing investigation

This branch fixes two related semantics:

- Dynamic global batch size now consumes every valid sample instead of rounding down to a DP multiple.
- The legacy balanced DP split uses token-balanced partitions with variable sample counts.

The GPU fixture in `gpu_balance_fixture.py` runs 72 batches on two MI350X ranks. Each batch starts with 12 synthetic samples, drops invalid samples before conversion, alternates 10/11 valid samples, uses changing token lengths, performs the real Miles conversion and rollout-side schedule, trains with DDP/NCCL, updates an optimizer, and compares gradients to a same-state single-rank reference.

Run:

```bash
torchrun --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29631 --rdzv-id=j-14c79d0ec76f reports/j-14c79d0ec76f/gpu_balance_fixture.py
```

GPU evidence is written to `gpu_validation.json`.
