# gfx950 train/rollout log-prob parity investigation

Status: draft investigation in progress.

## Scope

Measure the numerical mismatch between training and rollout log probabilities for a small representative model on one assigned AMD Instinct MI350X (`gfx950`). Compare identical weights, tokens, and settings across packed versus unpacked sequences and controlled batch sizes through the available SGLang rollout and Miles FSDP training paths.

## Planned evidence

- Record exact Torch, ROCm, Miles, SGLang, Megatron, and native module paths.
- Use a small public model or reduced fixture, bounded to 8 GB of downloads.
- Run controlled batch-size sweeps and packed/unpacked training forwards.
- Report absolute and signed log-prob differences, including negative results.
- Apply a focused correction only if the measurements justify one.

Reproduction commands and final numerical results will be added before this draft is marked ready.
