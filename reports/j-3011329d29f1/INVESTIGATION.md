# Vision-tower FSDP desynchronization investigation

## Result

- Issue: `radixark/miles#2406`
- Branch: `amdpilot/j-3011329d29f1`
- Tested commit: `e5125a97e1fd383f005f4de258a5985026e09425`
- Runtime: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- NCCL: `2.27.7`
- GPUs: 2× `AMD Instinct MI350X`, capability `(9, 5)`

Imported source and native paths:

- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- FSDP2 parameter group: `/opt/venv/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py`
- FSDP2 collectives: `/opt/venv/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`
- Native Torch/HIP library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`

## What was tested

I built a reduced synthetic multimodal model and ran it under FSDP2 on both assigned MI350X GPUs. The model separately wraps:

- the text embedding
- the vision tower
- each language block
- the output projection

Each rank uses a different local microbatch. One sample in every batch is a write-off with loss weight `0.0`, matching the existing `remove_sample` behavior in `miles/ray/rollout/train_data_conversion.py:87`.

The fixture compares:

- FSDP2 gradients against an explicit unsharded reference model with all-reduced local gradients
- FSDP2 weights against the reference weights after optimizer steps
- per-rank FSDP2 collective order for `unshard`, `wait_for_unshard`, `all_gather_into_tensor`, and `reduce_scatter_tensor`
- forward, backward, reference, and optimizer phase timings

I used bounded NCCL process-group timeouts of 30 seconds and did not raise them or substitute idle loops, spin waits, or sleeps.

## Desynchronized reproduction

Command:

```bash
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
/opt/venv/bin/torchrun \
  --nproc-per-node=2 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29633 \
  reports/j-3011329d29f1/reproduce_fsdp_vision_desync.py \
  --mode desync \
  --cycles 1 \
  --timeout-seconds 30 \
  --output-dir /tmp/miles-j-3011329d29f1-desync-2 \
  --check-every 1
```

Observed all-gather order:

| Rank | Order |
|---|---|
| 0 | `embedding`, `vision`, `language_block_0`, `language_block_1`, `output` |
| 1 | `embedding`, `language_block_0`, `language_block_1`, `output` |

Rank 0 executes the vision tower because its microbatch contains an image. Rank 1 skips it because its microbatch is text-only. This changes the FSDP2 all-gather sequence, and NCCL times out on the mismatched collective:

```text
Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=2, OpType=_ALLGATHER_BASE, NumelIn=272, NumelOut=544,
Timeout(ms)=30000) ran for 30093 milliseconds before timing out.
```

Both ranks eventually abort from the NCCL watchdog. This reproduces the reported desynchronization without increasing any timeout.

## Synchronized validation

Command:

```bash
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
/opt/venv/bin/torchrun \
  --nproc-per-node=2 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29634 \
  reports/j-3011329d29f1/reproduce_fsdp_vision_desync.py \
  --mode sync \
  --cycles 64 \
  --timeout-seconds 30 \
  --output-dir /tmp/miles-j-3011329d29f1-sync-64 \
  --check-every 8
```

The synchronized mode keeps every rank on the same vision-tower collective schedule. The text-only rank uses a dummy image with image weight `0.0`, and a small vision regularization term keeps the vision tower’s backward reduce-scatter active on both ranks.

Results:

- 64 alternating image/text cycles completed on both ranks
- 2,560 FSDP2 events per rank
- Collective-order mismatch count: `0`
- Maximum gradient absolute error: `0`
- Maximum gradient relative error: `0`
- Maximum weight absolute error: `0`
- Maximum weight relative error: `0`
- Weight comparisons: 8 per rank

### Phase timings

| Rank | Forward mean | Backward mean | Reference mean | Optimizer mean |
|---|---:|---:|---:|---:|
| 0 | 19.282 ms | 21.650 ms | 2.132 ms | 0.007 ms |
| 1 | 19.647 ms | 21.468 ms | 2.132 ms | 0.007 ms |

The first cycle includes FSDP2 and ROCm warm-up, so its forward and backward times are higher than the mean.

## Existing handling

The generic actor path conditionally adds multimodal inputs in `miles/backends/fsdp_utils/actor.py:687`, so a text-only rank can skip the vision tower while another rank executes it.

The existing HUD workaround in `examples/experimental/hud/rollout.py:340` inserts a dummy screenshot for write-offs and documents the collective-schedule requirement. That workaround is local to the HUD example and does not provide a generic actor-level guard.

My synchronized fixture validates a possible contract: keep every rank on the same vision-tower schedule using a dummy image with zero image weight, and preserve a gradient-producing vision path so backward reduce-scatters also match. I did not modify production behavior, because the correct production integration depends on the real multimodal model’s input contract and loss masking.

## Limitations

- This is a synthetic reduced model, not a full production multimodal checkpoint.
- The synchronized fixture uses a small vision regularization term to keep vision gradients active on text-only ranks; a production fix must choose an equivalent model-specific mechanism.
- I did not validate a production actor-level correction.
- I did not test topologies larger than two GPUs.
- No model weights were downloaded; total downloads were `0 GB`.

## Artifacts

- Fixture: `reports/j-3011329d29f1/reproduce_fsdp_vision_desync.py`
- Result summarizer: `reports/j-3011329d29f1/summarize_results.py`
- Numerical and timing summary: `reports/j-3011329d29f1/results/summary.json`
