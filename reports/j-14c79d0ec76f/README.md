# Miles DP balancing investigation

## Result

This branch fixes two related semantics:

- Dynamic global batch size now consumes every valid sample instead of rounding down to a DP multiple.
- The legacy balanced DP split uses token-balanced partitions with variable sample counts.

The pre-fix baseline demonstrated both mismatches with 13 valid samples and DP size 4:

- dynamic GBS returned 12, silently discarding one valid sample;
- the legacy balanced split raised `AssertionError: 13 % 4 != 0`.

## GPU evidence

The fixture in `gpu_balance_fixture.py` ran 72 batches on two assigned AMD Instinct MI350X ranks. Each batch started with 12 synthetic samples, dropped invalid samples before conversion, alternated 10/11 valid samples, used token lengths from 3 through 14 tokens, performed the real Miles conversion and rollout-side schedule, trained with DDP/NCCL, updated an optimizer, and compared gradients to a same-state single-rank reference.

Actual passing run:

- Tested candidate commit: `6e394cab3b869f8db164eecf8a5f92c9d05e057a`
- Base commit: `e5125a97`
- Devices: 2 × AMD Instinct MI350X (`gfx950`)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Collective backend: NCCL/RCCL
- Batches: 72 total; 36 divisible by DP size and 36 non-divisible
- Valid counts: alternating 10 and 11 samples
- Valid token lengths: 3 through 14 tokens
- Consumed samples: every batch's shard partitions covered exactly the valid sample range, with no duplicate or missing sample
- Effective loss weights: every batch's DP-wide effective sample weight summed to 1.0
- Gradient reference: maximum absolute difference was `1.1920928955078125e-07`; all 72 batches were below `2e-5`
- Optimizer: SGD updated DDP parameters after every gradient comparison, so later batches also checked state drift

Phase timings are recorded in `gpu_validation.json`. The passing run's means were:

- Forward: 13.09 ms
- Backward/collective: 2.68 ms
- Reference forward/backward: 1.48 ms
- Optimizer: 0.59 ms

The first maxima include one-time GPU initialization and kernel warmup; no idle loops, spin waits, or sleeps are used.

## Imported paths

Miles source paths exercised:

- `miles/ray/rollout/rollout_data_conversion.py`
- `miles/ray/rollout/train_data_conversion.py`
- `miles/utils/dp_schedule.py`
- `miles/utils/seqlen_balancing.py`
- `miles/backends/training_utils/data.py`
- `miles/backends/training_utils/loss.py`
- `miles/backends/training_utils/cp_utils.py`

Native paths exercised:

- `torch`
- `torch.distributed`
- `nccl`

## Reproduction

Run:

```bash
torchrun --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29631 --rdzv-id=j-14c79d0ec76f reports/j-14c79d0ec76f/gpu_balance_fixture.py
```

GPU evidence is written to `gpu_validation.json`.

## Scope and limitations

- This is a synthetic, local-random-initialization fixture and performs no model downloads.
- The passing topology is exactly two MI350X training ranks; it is not a claim about a larger topology.
- Two fixture-only corrections were needed before the passing run: tensor loss denominators and unpacking Miles' three-element loss result. Neither changed framework behavior.
- The full GPU run passed at the candidate commit listed above; report-only commits after that run do not alter the tested framework code.
