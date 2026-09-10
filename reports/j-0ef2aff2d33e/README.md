# Two-rank MI350X investigation of fully async ownership boundaries

## Scope

This investigation exercises the supported Miles fully-async source, buffer, and trainer-acknowledgement path on two assigned AMD Instinct MI350X (`gfx950`) GPUs. It does not claim that the full reservation/admission/settlement RFC from [radixark/miles#2254](https://github.com/radixark/miles/issues/2254) is implemented.

Tested Miles commit: `df0e677f6dd51fa551d48d37860812ece904cd8f`.

## Environment

- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native libraries:
  - `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so`
  - `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Ray: `2.58.0`, `/opt/venv/lib/python3.10/site-packages/ray/__init__.py`
- SGLang: `0.5.17.dev2157+ga8e5c632f`, `/sgl-workspace/sglang/python/sglang/__init__.py`
- Aiter: `/sgl-workspace/aiter/aiter/__init__.py`
- Miles source: `/job/miles/miles/__init__.py`
- GPUs: 2 × AMD Instinct MI350X, `gfx950`, 270566162432 B VRAM each
- Model downloads: 0 B

## Fixture

`async_recovery_gpu_probe.py` uses the installed Miles classes:

- `miles.rollout.data_source.RolloutDataSourceWithBuffer`
- `miles.rollout.fully_async_data_buffer.DefaultDataBuffer`
- `miles.rollout.fully_async_data_buffer.DataBufferInput`
- `miles.utils.types.Sample`

The fixture creates a local tokenizer and a 64-row JSONL prompt dataset, then runs a two-rank DDP model on both MI350X devices. Each batch performs real GPU forward, backward, optimizer, and `dist.all_reduce` acknowledgement work. The source cursor, model, and optimizer are checkpointed at bounded boundaries. No idle loops, spin waits, or sleeps are used.

## Results

### Uninterrupted control

- Committed source groups: `0..63`
- Optimizer steps: `64`
- Final checkpoint SHA-256: `0b72323efcd90da18bed8f1b10a6a8837f88f09b85623c7a9007da377b2fa6cf`

### Pre-admission interruption and restore

After 32 admitted batches, source group `32` is reserved and placed in the in-memory buffer, but not admitted. The source cursor is checkpointed at `33`.

After restore:

- Committed source groups: `0..31`, `33..63`, `64`
- Missing source group: `32`
- Extra wrapped source group: `64` (prompt `0`)
- Optimizer steps: `64`
- Final checkpoint SHA-256: `57a90cbc844b3a2bdd62ce4fb66386527a07436e2e1562a95a6141f100fc9add`
- Maximum absolute final-weight difference from control: `0.0030214302241802216`
- Mean absolute final-weight difference from control: `0.0019738031551241875`

This is the expected unsupported-path failure: the source checkpoint advances past a group whose only durable record is an in-memory buffer entry.

### Post-admission interruption and restore

After 33 admitted batches, source group `32` has been acknowledged by both trainer ranks. The source cursor is checkpointed at `33`.

After restore:

- Committed source groups: `0..63`
- Optimizer steps: `64`
- Final checkpoint SHA-256: `0b72323efcd90da18bed8f1b10a6a8837f88f09b85623c7a9007da377b2fa6cf`
- Maximum absolute final-weight difference from control: `0.0`
- Mean absolute final-weight difference from control: `0.0`
- Final weights are bit-exact with control.

This positive result is limited to the supported cursor/model/optimizer checkpoint alignment after a successful trainer acknowledgement. It does not prove recovery from an uncertain remote response.

### Duplicate/late completion

After the post-admission checkpoint, a duplicate completion for already-admitted source group `32` is inserted into the buffer.

Result:

- Committed source groups: `0..32`, `32`, `33..62`
- Duplicate source group: `32`
- Missing source group: `63`
- Optimizer steps: `64`
- Final checkpoint SHA-256: `c3cd547d4aa71df5c50b6e7f0c0308e7753f109276812b7248396f186b93eb97`
- Maximum absolute final-weight difference from control: `0.0030624978244304657`
- Mean absolute final-weight difference from control: `0.0020041093230247498`

The current `DefaultDataBuffer` accepts and trains the duplicate completion; it has no reservation identity, attempt receipt, or deduplication state.

## GPU phase timings

Median and mean per-batch timings are recorded in `results.json`. The control run measured:

- Forward median: `0.0002164323814213276 s`
- Backward median: `0.000301227904856205 s`
- Optimizer median: `0.00011370144784450531 s`
- Acknowledgement median: `0.00009069126099348068 s`

The first batch in each process includes kernel-compilation warmup, so its forward and backward timings are higher than the steady-state median.

## Honest limitations

- Current `main` has no reservation, lease, admission-receipt, settlement, or deduplication API for this path.
- `DefaultDataBuffer` is in-memory only; its contents are not checkpointed.
- `RolloutDataSourceWithBuffer.save()` persists only the source cursor and counters, not active reservations or buffered groups.
- `train_async.py` has no lifecycle method to stop new work, drain accepted work, or report an open train-batch lease.
- This fixture uses synthetic local completions and direct DDP trainer acknowledgements. It does not exercise SGLang generation, Ray object-store publication, or uncertain remote-response settlement.
- The post-admission result is therefore a supported checkpoint-alignment success, not evidence that the full RFC lifecycle is implemented.

## Reproduction

Run from the Miles repository root:

```bash
/opt/venv/bin/torchrun \
  --nnodes=1 \
  --nproc-per-node=2 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:<free-port> \
  --rdzv-id=<unique-id> \
  reports/j-0ef2aff2d33e/async_recovery_gpu_probe.py \
  --mode control \
  --output-dir /tmp/miles-control \
  --total-batches 64
```

Use `--mode pre_admission`, `pre_admission_restore`, `post_admission`, `post_admission_restore`, and `duplicate_late` for the bounded interruption cases. The restore modes require `--load-checkpoint-dir` to point at the corresponding boundary checkpoint directory.
