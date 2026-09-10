# Miles AMD LoRA checkpoint/resume investigation

## Result

The existing native LoRA checkpoint path was close to sufficient but did not meet the issue-2705 resume requirements as found. It saved per-rank native adapter shards and optimizer/scheduler state, but it did not publish a standard latest-iteration marker, did not infer the rollout from a standard `iter_*/adapter` path, and accepted malformed native shards. This change adds those narrow behaviors and validates them with real two-GPU training.

The final fixture compared an uninterrupted 256-step run with direct resumes from steps 32, 64, and 128, plus a repeated chain `32 -> 64 -> 128 -> 256`. Every direct and repeated-chain comparison passed on both ranks for:

- LoRA adapter weights
- FP32 master weights
- Adam first and second moments
- Adam step counts
- LR progression
- rollout data-source cursor
- fixed-token outputs

The observed maximum absolute difference was `0.0` for all tensor comparisons in the passing direct and repeated-chain cases. The warm-start case loaded adapter weights only, started a fresh optimizer/scheduler/data cursor at rollout 0, and was explicitly different from full resume. A native shard with a missing tensor was rejected before weight copy.

## Tested topology and stack

- GPUs: 2 assigned AMD Instinct MI350X, `gfx950`, capability `9.5`
- Process topology: DP=2, TP=1, PP=1, EP=1, ETP=1
- Runtime: `/opt/venv/bin/python`, Torch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Megatron-Core: `0.19.0+8c1e05747`, imported from `/root/Megatron-LM/megatron/core/parallel_state.py`
- Megatron-Bridge: `0.5.0+582783a0`, imported from `/opt/venv/lib/python3.10/site-packages/megatron`
- Transformers: `5.12.1`; PEFT: `0.18.1`
- Process group: NCCL/HIP backend, 120-second timeout, unique c10d rendezvous port for each launch
- Downloads: 0 bytes; the tokenizer and prompt data are generated locally

Sharing `gfx950` is not a MI355X performance-equivalence claim. This is a tiny supported fixture, not the 30B recipe, and no larger topology is claimed.

## Model and layouts

The fixture uses local random initialization with vocabulary 8, hidden size 16, LoRA rank 8, 8 tokens per step, and 2 local experts. It exercises:

- dense Megatron-Bridge `LoRA` through `LinearAdapter`
- per-expert `GroupedExpertLinearAdapter`
- available shared-outer `SharedOuterGroupedExpertAdapter`, including `torch._grouped_mm`

Every training step performs real GPU forward, DDP gradient synchronization, explicit collective reduction, FP32-master Adam update, and BF16 model synchronization. CUDA events record forward/backward/collective/update/eval timings; save/load also record wall time. There are no idle loops, spin waits, or sleeps.

## Hooks exercised

- `miles.backends.megatron_utils.checkpoint.save_checkpoint_with_lora`
- `miles.backends.megatron_utils.lora_utils.save_lora_checkpoint`
- `miles.backends.megatron_utils.lora_utils.load_lora_adapter`
- `miles.rollout.data_source.RolloutDataSource.save`
- `miles.rollout.data_source.RolloutDataSource.load`
- `torch.nn.parallel.DistributedDataParallel`
- `torch.optim.Adam` with explicit FP32 masters
- `torch.optim.lr_scheduler.LambdaLR`

The production implementation under test is commit `e73bca8c7438cebcfbb0c488b10ca27053f7b338`. The complete machine-readable evidence, including both ranks, all comparisons, LR histories, cursor values, timings, and native paths, is in `gpu_resume_results.json`.

## Reproduction

Run from the repository root with the qualified environment:

```bash
PYTHONPATH=/job/miles TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
/opt/venv/bin/torchrun --nnodes=1 --nproc-per-node=2 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:<unique-port> \
  --rdzv-id=j-4b92ddb9985d \
  reports/j-4b92ddb9985d/gpu_resume_fixture.py \
  --root /job/cache/j-4b92ddb9985d \
  --output reports/j-4b92ddb9985d/gpu_resume_results.json \
  --steps 256 --checkpoints 32 64 128 \
  --layouts dense expert_per_expert expert_shared_outer
```

Focused CPU tests pass with:

```bash
TMPDIR=/job/tmp /opt/venv/bin/python -m pytest --noconftest -q \
  tests/fast/backends/megatron_utils/test_lora_utils.py::TestNativeLoRALoad \
  tests/fast/backends/megatron_utils/test_lora_checkpoint_helpers.py
```

## Limitations and honest notes

- This does not run or claim support for the full 30B recipe.
- The tiny model is synthetic/local and is not a production performance benchmark.
- The fixture's HF PEFT export is intentionally skipped because the local tokenizer directory has no model config; native per-rank shards and training state are the tested resume path.
- Normal pytest conftest discovery is blocked by a pre-existing `/tmp/aiter_configs/bf16_tuned_gemm.csv.lock` permission error. The focused tests above pass with `--noconftest`; no node-wide state was changed to work around it.
- The full 256-step matrix passed after fixing a fixture-only stale `master_param` alias issue discovered by the repeated-chain case. The final `gpu_resume_results.json` is from that passing run.
