# Reduced two-GPU Miles backend investigation

## Result

This investigation establishes a reproducible, synthetic reduced-model control for the installed Miles FSDP and Megatron training backends on two AMD Instinct MI350X (`gfx950`) GPUs. It does **not** establish a Miles-versus-VERL result: VERL is not installed, and no additional framework stack was installed to manufacture a comparison.

Both supported Miles backends completed the full bounded matrix:

- Four cases: micro-batch/sequence `{1,128}`, `{2,128}`, `{1,512}`, `{2,512}`.
- Global batch size: 8 samples per optimizer step (4 samples per GPU).
- Per case: 5 warmup and 25 measured optimizer steps.
- Per backend total: 20 warmup and 100 measured optimizer steps.
- Model: 3,814,912-parameter shape-matched reduced GPT, local random initialization, 0 bytes downloaded.
- Precision/optimizer: bf16 parameters, fp32 gradient reduction/master state, Adam with lr `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, and no effective gradient clipping.

Raw evidence is in [`fsdp.json`](./fsdp.json) and [`megatron.json`](./megatron.json).

## Environment

- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- GPUs: 2× AMD Instinct MI350X, capability `(9, 5)`
- Torch/ROCm: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Miles harness commit: `8983c7f744829524e2525d827c639963810bd2ae`
- Megatron source commit: `8c1e05747eb612b382df2632783df5c83a853646`
- Megatron-core distribution: `0.19.0+8c1e05747`
- Transformers: `5.12.1`
- FlashAttention: `2.8.3`
- TransformerEngine: `2.17.0`
- VERL: not installed (explicit gap; not installed by this investigation)

The JSON evidence records the imported source and native paths, including:

- Miles: `/job/miles/miles/__init__.py`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Megatron: `/root/Megatron-LM/megatron/core/__init__.py`
- Transformers: `/opt/venv/lib/python3.10/site-packages/transformers/__init__.py`
- Aiter source: `/sgl-workspace/aiter/aiter/__init__.py`
- Aiter native: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- TransformerEngine: `/opt/venv/lib/python3.10/site-packages/transformer_engine/__init__.py`

## Benchmark paths

### FSDP

The FSDP path uses the installed Miles backend helper `miles.backends.fsdp_utils.actor.apply_fsdp2`, Miles `DataIterator`/`get_batch`, Miles `loss_function`, and Torch `AdamW`. Forward, backward (including FSDP gradient reduction), and optimizer phases are timed with CUDA events. The maximum phase time across the two ranks is reported.

### Megatron

The Megatron path uses Miles `miles.backends.megatron_utils.model.train_one_step`, Megatron DDP, the Megatron optimizer/scheduler, and the same Miles data and loss utilities. A timing wrapper records forward and total forward/backward/collective time; backward is the synchronized total minus forward. `finish_grad_sync()` is included in the collective/backward phase. Optimizer time is recorded around the Megatron optimizer step.

The installed Megatron GPT forward does not accept the `fp32_output` argument used by the current Miles `train_one_step`, and it requires explicit position IDs. The benchmark installs a per-model compatibility wrapper that accepts/ignores `fp32_output` and constructs position IDs from `input_ids`. This is a source/API limitation recorded here; it does not replace training, collective, or optimizer work.

## Controls and checks

- Input wait is measured separately as CPU wall time around local synthetic CPU-to-GPU transfer plus synchronization. It is excluded from training tokens/s.
- Training tokens/s uses synchronized forward + backward/collective + optimizer phase time and the fixed global token count.
- Memory is the maximum `torch.cuda.max_memory_allocated()` across ranks after each case.
- Loss and gradient norm are finite for every measured step.
- Full all-gathered SHA-256 weight digests match across both ranks before and after each case.
- Full all-gathered gradient digests match across both ranks on the first measured step.
- Initial and final weight digests differ, showing actual optimizer state drift/update behavior.
- No idle loops, spin waits, or sleeps are used as GPU work.

## Results

Mean training throughput excludes input wait. The baseline is each backend's smallest case (`mb1_seq128`).

| Backend | Case | Mean training tok/s | Baseline ratio | Mean input wait (s) | Peak memory (MiB) |
|---|---|---:|---:|---:|---:|
| FSDP | mb1_seq128 | 13,656.15 | 1.00× | 0.0000856 | 184.39 |
| FSDP | mb2_seq128 | 26,222.86 | 1.92× | 0.0000836 | 204.56 |
| FSDP | mb1_seq512 | 58,297.56 | 4.27× | 0.0000778 | 241.78 |
| FSDP | mb2_seq512 | 105,880.34 | 7.75× | 0.0000908 | 306.64 |
| Megatron | mb1_seq128 | 23,265.99 | 1.00× | 0.0000583 | 225.84 |
| Megatron | mb2_seq128 | 39,745.75 | 1.71× | 0.0000593 | 298.86 |
| Megatron | mb1_seq512 | 89,580.34 | 3.85× | 0.0000635 | 350.04 |
| Megatron | mb2_seq512 | 165,152.24 | 7.10× | 0.0000556 | 483.73 |

For this reduced fixture only, Megatron mean training throughput was 1.52×–1.70× FSDP across the four cases. This is not a production-model or framework-level conclusion.

All 16 backend/case numerical checks passed:

- Loss finite: 8/8 cases.
- Gradient norm finite: 8/8 cases.
- Cross-rank gradient digest match: 8/8 cases.
- Initial cross-rank weight digest match: 8/8 cases.
- Final cross-rank weight digest match: 8/8 cases.
- Weight changed after measured updates: 8/8 cases.

## Reproduction

Run from the Miles checkout with the qualified `/opt/venv/bin/python`. The Aiter environment variable avoids an unwritable shared `/tmp/aiter_configs` merge lock by selecting the installed default BF16 GEMM config directly.

```bash
export PYTHONPATH="$PWD"
export AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv

/opt/venv/bin/python -m torch.distributed.run \
  --nproc-per-node=2 --master-port=29748 \
  tools/benchmark_reduced_training.py \
  --backend fsdp \
  --output /tmp/fsdp.json

/opt/venv/bin/python -m torch.distributed.run \
  --nproc-per-node=2 --master-port=29749 \
  tools/benchmark_reduced_training.py \
  --backend megatron \
  --output /tmp/megatron.json
```

The process group timeout is bounded at two minutes. The recorded final runs used unique ports 29748 and 29749 and all assigned devices.

## Limitations and unfinished work

- This is a reduced synthetic control, not a rollout or end-to-end training benchmark.
- FSDP uses HuggingFace GPT-2 and Megatron uses Megatron-core GPT with the same shape controls and parameter count. They are not bit-identical implementations, so cross-backend loss values are not expected to match; numerical checks establish finiteness and cross-rank consistency, not cross-backend equivalence.
- The Megatron compatibility wrapper described above is necessary for the installed source/API combination and should be removed when the backend and Megatron APIs realign.
- VERL remains an explicit unavailable backend. No VERL comparison or performance-equivalence claim is made.
- Results apply only to two MI350X (`gfx950`) GPUs. Sharing `gfx950` is not a MI355X performance-equivalence claim, and no larger topology is claimed.
