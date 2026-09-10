# Bounded FSDP2 vs Megatron-Core DDP parity probe

## Result

The reduced two-GPU fixture matches on loss, gradients, fp32 master-parameter deltas, consumed samples, and native checkpoint reload after three fixture-level corrections:

1. Megatron's optimizer default is decoupled AdamW; the probe now explicitly sets `decoupled_weight_decay=False` to match Torch Adam/L2 on the FSDP side.
2. FSDP retains an fp32 master while Megatron derives its master from bf16 model parameters; the synthetic initialization is now quantized once through bf16 before sharding so both masters start identical.
3. DCP's fresh Megatron load placeholder omitted the optimizer `step` key; the probe now supplies that key before `dcp.load`, restoring the saved step count.

No product-code fix is proposed from this reduced case. The mismatches were fixture/configuration issues, not evidence of a Miles backend defect.

## Fixture

- Model: `nn.Linear(16, 8, bias=True)`, 144 parameters.
- Data: deterministic synthetic batches, no downloads.
- Distributed setup: 2 ranks on 2 assigned AMD Instinct MI355X devices.
- Batch: global 16, local 8.
- Precision: bf16 compute, fp32 loss, fp32 gradient reduction, fp32 master parameters.
- Optimizer: Adam/L2, learning rate 0.01, weight decay 0.01, betas `(0.9, 0.999)`, epsilon `1e-8`.
- Compared steps: 2 training steps plus a checkpoint-reload step.
- Process-group timeout: 180 seconds.

## Reproduction

```bash
cd /job/miles
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
timeout 300 /opt/venv/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  reports/j-83908b439bbf/parity_probe.py \
  --checkpoint-root /job/.cache/j-83908b439bbf/checkpoints-final \
  --output reports/j-83908b439bbf/results.json
```

The command uses only its own two `torchrun` subprocesses and does not modify node-wide state.

## Evidence

`results.json` records:

- Initial fp32 master and visible bf16 parameters match exactly.
- Step 1 and step 2 loss, gradients, and fp32 master-parameter deltas match.
- Consumed samples match at 16 and 32 samples.
- FSDP and Megatron checkpoint reload both restore model parameters, master parameters, optimizer state, and step count exactly.
- Reloaded step-3 loss, gradients, and parameters match the uninterrupted path for both backends.

The visible-bf16 parameter comparison can differ by one bf16 ULP when nearly equal fp32 masters round to adjacent bf16 values; this is expected and does not indicate a distributed-training mismatch.

## Scope

This proves only the reduced linear model under:

- FSDP2 full sharding with bf16 compute and fp32 reduction/master state.
- Megatron-Core DDP with `Float16Module`, fp32 gradient reduction, and non-distributed Adam/L2.
- Native DCP save/load for both backends.

It does not prove:

- Tensor, pipeline, context, expert, or sequence parallelism.
- Megatron distributed optimizer or Megatron FSDP.
- Transformer Engine fused kernels beyond the installed `FusedAdam` path.
- Real language-model numerics, tokenizer behavior, rollout integration, or multi-LoRA behavior.
- Performance, scaling, or production checkpoint compatibility.

Unsupported combinations are explicit in `results.json` under `scope.unsupported_combinations`.

## Environment

- Miles commit: `e5125a97e1fd383f005f4de258a5985026e09425`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Megatron Core source: `/root/Megatron-LM/megatron/core`.
- SGLang source: `/sgl-workspace/sglang/python/sglang`.
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.

The preinstalled Megatron and SGLang source paths are environment context, not proof of the tested revision. The tested Miles revision is the commit above.
