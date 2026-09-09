# CanonicalLoRA gated-QKV investigation

## Result

The installed Megatron-Bridge `CanonicalLoRA` path omits attention-output gate rows.
On a synthetic fused `ColumnParallelLinear` with 4 query heads, 2 query groups, and
head size 8, the wrapped projection has 96 output rows:

- Q: 32 rows
- attention-output gate: 32 rows
- K: 16 rows
- V: 16 rows

The unmodified bridge adapter allocates only 32 Q rows plus 16 K and 16 V rows. Its
interleaved output is therefore 64 rows, and forward fails with:

```text
The size of tensor a (96) must match the size of tensor b (64) at non-singleton dimension 1
```

This reproduces radixark/miles issue 2008 without a public-model download.

## Fix

Miles now uses a compatibility subclass at the `create_lora_instance` boundary. For
gated `linear_qkv` targets, it allocates the query adapter with the HuggingFace
`q_proj` width (Q plus gate), restores the original Megatron config, and interleaves
local `[Q, gate, K, V]` groups. Non-gated modules still use the upstream wrapper.

The reduced GPU test compares:

- converted adapter tensor shapes and values,
- fused forward output against an independent unfused Q/gate/K/V reference,
- input and adapter gradients,
- frozen base weights before and after backward,
- native adapter checkpoint save and load.

## Reproduction

Run from the Miles checkout:

```bash
/opt/venv/bin/python -m pytest --noconftest \
  tests/fast/backends/megatron_utils/test_canonical_lora_gated_qkv.py -q
/opt/venv/bin/python -m pytest --noconftest \
  tests/fast/backends/megatron_utils/test_lora_utils.py -q
```

Observed results:

```text
1 passed
63 passed
```

The focused test uses one NCCL process and one assigned AMD Instinct MI355X GPU with
a 30-second process-group timeout. It creates its own `FileStore`; it does not use a
fixed rendezvous port or broad process management.

## Environment

The test used the current `/job/miles` checkout plus these preinstalled packages:

- Torch 2.9.1+rocm7.2.0.lw.git7e1940d4: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Megatron-Core 0.19.0+8c1e05747: `/root/Megatron-LM/megatron/core/__init__.py`
- Megatron-Bridge 0.5.0+582783a0: `/opt/venv/lib/python3.10/site-packages/megatron/bridge/__init__.py`
- SGLang 0.5.17.dev2157+ga8e5c632f: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Miles 0.1.0: `/job/miles/miles/__init__.py`
- Aiter native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

The GPU was `AMD Instinct MI355X` with capability `(9, 5)`.

## Scope and limitations

This reduced case proves the missing-row diagnosis and the narrow compatibility fix
for a real Megatron `ColumnParallelLinear` fused QKV projection. It does not prove:

- full Qwen3.5/Qwen3.6 model construction or training,
- SGLang adapter publication,
- tensor/pipeline/expert parallelism above one rank,
- HF PEFT export (the test intentionally has no HF checkpoint; that export is skipped),
- a Megatron-Bridge upstream fix or dependency-pin update.

Standard pytest collection without `--noconftest` fails in this container before this
test runs because the repository conftest imports SGLang/Aiter and cannot acquire
`/tmp/aiter_configs/bf16_tuned_gemm.csv.lock`. The focused commands above bypass that
unrelated environment issue. `black` and `isort` are not installed in this runtime, so
only `py_compile`, `git diff --check`, and the focused tests were run.
