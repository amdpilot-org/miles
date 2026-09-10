# Two-rank LoRA checkpoint investigation

## Scope

This is a bounded GPU investigation for [radixark/miles#2668](https://github.com/radixark/miles/issues/2668),
not a fix PR. It uses the real Miles `save_lora_checkpoint` and
`load_lora_adapter` implementations from mirror `main` commit
`e5125a97e1fd383f005f4de258a5985026e09425`.

The fixture runs two actual ranks on two assigned AMD Instinct MI355X (`gfx950`)
GPUs with DP=2, TP=PP=CP=EP=1. It builds a synthetic one-layer Qwen2 model
(3,504 parameters), applies a rank-2 Miles/Megatron-Bridge LoRA to
`linear_qkv`, and saves through the Miles checkpoint path. No public weights or
large model download are required.

## Reproduction

Run from the mounted clone with the private caches outside the worktree:

```bash
export PYTHONPATH=/job/miles
export CUDA_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export HF_HOME=/job/cache/hf
export TORCH_HOME=/job/cache/torch
export TRITON_CACHE_DIR=/job/cache/triton

/opt/venv/bin/torchrun --standalone --nproc-per-node=2 \
  reports/j-605503eb1799/two_rank_lora_checkpoint_fixture.py \
  --mode baseline \
  --model-dir /job/models/tiny-qwen2 \
  --save-dir /job/cache/j-605503eb1799/baseline \
  --evidence-dir /job/cache/j-605503eb1799/evidence-baseline \
  --timeout-seconds 120
```

Repeat with `--mode hf-error` for the contained export-failure control. Repeat
with `--mode native-error` and `--timeout-seconds 25` for the interrupted
native-save control. Each mode uses a separate save directory; the fixture does
not delete rank artifacts or turn duplication into a pass by removing unverified
files.

## Observed environment

- Runtime image requested: `amdpilotv2/miles-job:central-d957-20260909`.
- Python: `/opt/venv/bin/python3`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- HIP: `7.2.26015-fc0010cf6a`.
- Miles under test: `/job/miles/miles/__init__.py` at commit `e5125a9`.
- Megatron core: `/root/Megatron-LM/megatron/core/__init__.py`.
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- AIter Python package: `/sgl-workspace/aiter/aiter/__init__.py`.
- AIter native module observed at import: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

The preinstalled source trees are environment context. The subprocesses were
launched with `PYTHONPATH=/job/miles`, and the evidence records
`/job/miles/miles/__init__.py`; the tested Miles source is therefore the mounted
clone, not the preinstalled `/root/miles` tree.

## Baseline result

The baseline exited 0 and produced:

- `adapter_megatron_rank0.pt`
- `adapter_megatron_rank1.pt`
- `training_state_rank0.pt`
- `training_state_rank1.pt`
- `adapter_model.bin`
- `adapter_config.json`

Both DP ranks wrote a native adapter shard. Their keys, shapes, and tensor
values were exactly equal, confirming the issue's duplicate-shard behavior for
DP=2. Their SHA-256 hashes differ because `torch.save` serialization metadata
differs; tensor equality, not byte equality, is the correct comparison here.

Writer ownership was:

- Native adapter shard: each global rank (`rank0` and `rank1`).
- Training state: each global rank (`rank0` and `rank1`).
- HF PEFT `adapter_model.bin` and `adapter_config.json`: only effective DP rank
  0 with TP=PP=CP=0.

After perturbing adapter, optimizer, and scheduler state on both ranks,
`load_lora_adapter` restored each rank from its own native shard. Both ranks
reported:

- adapter restore equal: `true`
- optimizer restore equal: `true`
- scheduler restore equal: `true`
- loaded: `true`
- iteration: `7`
- all-rank restore equal: `true`

This verifies exact current-format restore equivalence for the reduced DP=2
case. It does not establish that a deduplicated shard layout would be
backward compatible.

## Error controls

### Contained HF export failure

With `--mode hf-error`, `AutoBridge.from_hf_pretrained` raises on both ranks.
The save exited 0. Miles caught the export error, logged that native shards plus
training state remain sufficient, and left:

- both native adapter shards
- both training-state files
- no HF PEFT files

This is a useful negative control: the native resume path remains complete when
the optional collective HF export fails before writing.

### Interrupted native save

With `--mode native-error`, rank 1 raises from `torch.save` immediately before
writing `adapter_megatron_rank1.pt`. The launcher exited nonzero as expected.
The preserved directory contains:

- `adapter_megatron_rank0.pt`
- `training_state_rank0.pt`

Rank 1's native shard and training state are absent. The rank-0 native shard
loads with two expected adapter tensors, and the rank-0 training state loads
with iteration 7. No files were deleted or repaired.

The 25-second process-group timeout was bounded, but it was not reached:
`torchrun` observed the rank-1 failure and terminated rank 0 first. Rank 0
therefore did not emit its evidence JSON, although its preserved files and the
run log show its partial writes. This is an honest limitation of the fault
control, not evidence of atomic checkpoint completion.

## What this does and does not prove

Proved on two real MI355X ranks:

- Current Miles duplicates identical native adapter shards across DP replicas.
- Native and training-state ownership is per global rank.
- HF PEFT output ownership is already restricted to the first DP/TP/PP replica.
- Current per-rank native restore is exact for adapter, optimizer, and scheduler
  state in this reduced case.
- HF export failure is contained without destroying native resume artifacts.
- A native rank failure leaves a partial, non-atomic checkpoint that must not be
  treated as complete.

Not proved:

- TP>1, PP>1, CP>1, EP>1, expert-adapter, or larger DP layouts.
- A safe deduplicated filename/ownership scheme or its backward compatibility.
- Full training, rollout, or SGLang resume behavior.
- Real disk-full, SIGKILL, or multi-node failure timing; the native fault is an
  explicit in-process `torch.save` injection.
- Byte-identical shard files across ranks; only tensor equality was asserted.

No workload fix or checkpoint migration is included. The duplication result
and the preserved partial-save failure are the deliverable.
