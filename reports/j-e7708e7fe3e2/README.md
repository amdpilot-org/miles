# LoRA checkpoint DP duplication investigation

## Scope

This investigates radixark/miles issue 2668 against the single-LoRA
`save_lora_checkpoint` / `load_lora_adapter` path. The reduced case uses a real
two-process distributed group and real GPU tensors, but replaces the large base
model with a tiny `torch.nn.Linear` fixture carrying two LoRA parameters. No
model weights were downloaded.

## Hardware and runtime

- GPUs: 2 assigned AMD Instinct MI350X, `gfx950`, 270,566,162,432 bytes VRAM each.
- Torch: 2.9.1+rocm7.2.0.git7e1940d4, HIP 7.2.26015-fc0010cf6a.
- Miles import: `/job/miles/miles/__init__.py`.
- SGLang import: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Megatron Bridge import: `/opt/venv/lib/python3.10/site-packages/megatron/bridge/__init__.py`.
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Aiter import/native module: `/sgl-workspace/aiter/aiter/__init__.py` and `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Mirror base commit: `e5125a97e1fd383f005f4de258a5985026e09425`.

Preinstalled source paths are environment context, not proof of the tested
revision. The tested Miles revision is this branch's checkout under `/job/miles`.

## Baseline result

The baseline run used two gloo ranks, one GPU per rank, DP=2 and
TP=PP=CP=EP=1:

```text
PYTHONPATH=/job/miles torchrun --standalone --nproc_per_node=2 \
  /job/baseline_lora_checkpoint_fixture.py baseline /job/lora-baseline-artifacts
```

It produced:

```text
adapter_megatron_rank0.pt
adapter_megatron_rank1.pt
training_state_rank0.pt
training_state_rank1.pt
```

Both adapter files contained the same `lora_A.weight` and `lora_B.weight`
tensor values. Their SHA-256 digests differed because Torch serialization is not
byte-deterministic; tensor equality, not file-hash equality, established the
DP duplication. Both per-rank training-state files were required and were
preserved.

## Fixed result

The native shard is now named by its realized `(tp, pp, ep)` coordinate and is
written only by the DP=0, CP=0 owner of that coordinate. DP replicas share the
same shard on load. The old global-rank and tp/pp layouts remain load fallbacks;
no existing checkpoint files are deleted to hide duplication.

The fixed full-path run produced:

```text
adapter_megatron_tp0_pp0.pt
training_state_rank0.pt
training_state_rank1.pt
```

Both ranks perturbed their local adapter differently, loaded the shared shard,
and restored exactly the saved values. The committed two-GPU test also checks:

- rank 1 does not write a native shard;
- both ranks retain their own training state;
- old `adapter_megatron_rank{rank}.pt` checkpoints still load;
- a failed HF export is non-fatal and leaves native/training state usable;
- a failed atomic save leaves no final file or temporary file.

Run it with:

```text
PYTHONPATH=/job/miles torchrun --standalone --nproc_per_node=2 \
  tests/fast-gpu/test_lora_checkpoint_dp_shards.py
```

## Interrupted-save and error controls

Baseline HF-export failure was already caught and skipped, leaving native and
training state available. A simulated native-write interruption on rank 0
left a 7-byte partial `adapter_megatron_rank0.pt` in the final directory while
rank 1 wrote a full duplicate.

After the change, the same full-path interruption left no final adapter file
and no temporary file. The collective still failed when the writer exited, so
the caller remains responsible for retrying the save; this is intentional and
prevents a checkpoint from being reported as complete after a failed write.

## Validation

- Two-GPU fixture: passed on 2 AMD Instinct MI350X GPUs.
- Adjacent fast tests: 18 passed with `pytest --noconftest`.
- Normal pytest collection was blocked by a pre-existing
  `/tmp/aiter_configs/bf16_tuned_gemm.csv.lock` permission error owned outside
  this job; no shared lock or node-wide state was modified.

## Limits

This reduced case proves the Miles checkpoint ownership, naming, restore, and
interrupted-write behavior for DP=2 with trivial model parallelism. It does not
benchmark MI350X performance, exercise a full Megatron model or real HF bridge
conversion, prove TP/PP/EP>1 end-to-end training, cover multi-node storage, or
replace the existing multi-LoRA checkpoint tests. The shared naming helper and
adjacent tests cover expert-suffixed naming, but only the DP=2 topology was run
on hardware.
