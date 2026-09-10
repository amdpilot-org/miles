# CanonicalLoRA gated fused-QKV investigation

## Result

The installed Megatron-Bridge revision (`582783a05442245647239e4c5e7d733d7f0e00ea`)
and the live `bridge` branch omit attention-output gate rows from
`CanonicalLoRA`. A reduced fused `ColumnParallelLinear` QKV module reproduces the
contract mismatch: the base output is 64 wide while the upstream adapter packs only
40 Q/K/V rows. The Miles compatibility subclass widens the query-side adapter to
48 rows and packs Q, Gate, K, and V in Megatron's grouped order.

The focused GPU fixture passed all four tests on one AMD Instinct MI350X
(`gfx950`, compute capability 9.5). It exercised the actual
`create_lora_instance()` path, compared forward and gradient outputs with an
independent unfused reference, checked converted adapter tensors and gradients,
checked base/adapter state-dict boundaries, preserved HF-to-Megatron canonical
target conversion, and confirmed non-gated Q/K/V behavior remains unchanged.

## Reproduction

```bash
PYTHONPATH="$PWD" /opt/venv/bin/python reports/j-bc5283635bb4/run_gpu_fixture.py
```

The runner writes `results.json` and `run.log` beside this report. The direct test
command is:

```bash
PYTHONPATH="$PWD" /opt/venv/bin/python -m pytest --noconftest -q \
  tests/fast/backends/megatron_utils/test_canonical_lora_gate.py
```

`--noconftest` avoids this host's unrelated `/tmp/aiter_configs` permission failure
while importing the repository-wide test conftest. The focused test itself does not
depend on that conftest.

## Evidence

- GPU: one assigned AMD Instinct MI350X, UUID
  `63646365-3134-6334-6564-393765333530`, `gfx950:sramecc+:xnack-`.
- Torch: `2.9.1+rocm7.2.0.lw.git7e1940d4`.
- Megatron-Bridge: `0.5.0+582783a0`, module
  `/opt/venv/lib/python3.10/site-packages/megatron/bridge`.
- SGLang: `0.5.17.dev2157+ga8e5c632f`, module `/sgl-workspace/sglang/python/sglang`.
- Miles under test: `/job/miles/miles`.
- Native modules: Torch `_C`, `sgl_kernel.common_ops`, and AITER
  `module_aiter_core` paths are recorded in `results.json`.

The fixture uses 6 attention heads, 2 query groups, head size 4, hidden size 8,
LoRA rank 4, and alpha 4. It initializes a one-rank NCCL process group with a
60-second timeout and uses GPU 0.

## Scope

This reduced case proves the row allocation, grouped Q/Gate/K/V packing, forward
output, input/adapter gradients, state-dict boundary preservation, and unchanged
non-gated behavior. It does not prove full Qwen model loading, TP greater than one,
distributed checkpoint resharding, SGLang publication end to end, or performance
equivalence with MI355X. Those remain follow-up validation work.

The Miles subclass is a compatibility shim until Megatron-Bridge carries the fix;
the Docker image still installs the live `bridge` branch.
