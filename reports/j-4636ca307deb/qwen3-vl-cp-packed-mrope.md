# Qwen3-VL packed MRoPE and context-parallel investigation

## Result

The existing packed MRoPE position reconstruction passes the bounded two-GPU fixture, and the non-CP packed path is independent per sample. The CP2 forward and backward path does **not** match the full-sequence independent reference. This is a negative equivalence result; it is not a performance claim and does not establish that any larger topology works.

| Check | Result | Evidence |
| --- | --- | --- |
| CP2 reconstructed MRoPE positions | **Pass** | Exact tensor equality for all 32 iterations. |
| Non-CP packed cross-sample independence | **Pass** | Independent and packed outputs are exactly equal for every sample and iteration. |
| Per-sample padding and zigzag boundaries | **Pass** | Padded lengths are CP2-zigzag aligned; recorded real/padding position boundaries are consistent. |
| CP2 forward output equivalence | **Fail** | Worst max absolute difference is `0.943359375`. |
| CP2 local zigzag output equivalence | **Fail** | Worst local-slice max absolute difference is `0.943359375`. |
| CP2 gradient equivalence | **Fail** | Worst max absolute difference is `37.90625`. |
| Repeated SGD update drift | **Fail** | Worst post-update weight max absolute difference is `0.0006978511810302734`. |

All 32 iterations and all compared tensors remained finite. The comparison intentionally exits nonzero when any non-position difference exceeds `2e-4`.

## Fixture

The fixture is `tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py`. It uses a tiny, locally initialized Qwen3-VL bridge model, so no checkpoint is downloaded:

- `bf16`, one text layer, hidden size `64`, two attention heads, head dimension `32`, vocabulary size `128`.
- Real bridge vision inputs, MRoPE, THD packed sequence parameters, and `AttnBackend.flash`.
- CP size `2` with zigzag layout on both assigned GPUs.
- Three samples per iteration with varied real lengths and image grids.
- Per-sample padding to a multiple of `2 * CP_SIZE`; padding remains inside each sample and no final padding-only segment is added.
- Real forward, backward, gradient all-reduce, and SGD update on every iteration.
- A 120-second process-group timeout and unique localhost rendezvous ports.

The six repeated length layouts are:

1. `[7, 11, 18]`
2. `[11, 15, 14]`
3. `[15, 7, 18]`
4. `[19, 9, 12]`
5. `[7, 17, 16]`
6. `[11, 7, 22]`

The corresponding image-grid layouts are:

1. `[(1,4,4), (1,6,4), (1,4,6)]`
2. `[(1,6,4), (1,4,6), (1,4,4)]`
3. `[(1,4,6), (1,4,4), (1,6,4)]`

The learning rate is `1e-6`. An earlier `1e-3` probe reached non-finite bf16 tensors by iteration 14 because of the large CP gradient mismatch; the committed fixture uses the lower rate so all 32 repeated updates remain finite and interpretable.

## Numerical evidence

The committed summary is `reports/j-4636ca307deb/qwen3-vl-cp-32-stable-summary.json`.

- `position_equal`: `true` for all 32 iterations.
- `independent_vs_packed_output_equal`: `true` for all 32 iterations.
- `cross_sample_contamination`: `false` for all 32 iterations.
- Worst CP output max absolute difference: `0.943359375`.
- Worst CP local-output max absolute difference: `0.943359375`.
- Worst CP gradient max absolute difference: `37.90625`.
- Worst post-update weight max absolute difference: `0.0006978511810302734`.
- Iteration 0 output and gradient maxima: `0.8125` and `27.3125`.
- Iteration 31 output and gradient maxima: `0.7509765625` and `26.5625`.

The per-sample checks record the first position, last real-token position, first padding position, last padding position, padding count, and alignment. For example, iteration 0 sample 0 has real length `7`, padded length `8`, last real position `[4, 4, 4]`, and first padding position `[5, 5, 5]`.

The CP local output also differs from the expected zigzag slices of the non-CP packed reference. Therefore the failure is not explained by incorrect full-row position reassembly alone.

## GPU timings

Wall-clock phase timings from the stable 32-iteration run:

- CP2 phase: `31.432 s` on two MI350X GPUs.
- Independent reference phase: `25.771 s` on one MI350X GPU.
- CPU comparison phase: `41.337 s`.

CUDA-event totals across the 32 CP iterations:

| Phase | Total | Mean per iteration |
| --- | ---: | ---: |
| Positions | `2949.694 ms` | `92.178 ms` |
| Forward | `3179.056 ms` | `99.345 ms` |
| Backward | `2914.849 ms` | `91.089 ms` |
| Gradient all-reduce | `35.350 ms` | `1.105 ms` |
| SGD optimizer | `28.571 ms` | `0.893 ms` |

The first iteration includes native kernel compilation/warmup. Later iterations are much shorter; for example, iteration 31 records approximately `2.404 ms` positions, `12.007 ms` forward, `4.510 ms` backward, `1.041 ms` all-reduce, and `0.189 ms` optimizer time.

## Environment and tested code

- Repository base: `e5125a97e1fd383f005f4de258a5985026e09425`
- Existing packed MRoPE implementation commit: `803016a4622a7f7f45c26140cb9bd8e016aad217` (`feat: Fix Qwen3-VL THD packed mRoPE positions (#1272)`)
- Implementation under test: `miles_plugins/models/qwen3_vl.py`
- Source issue: `https://github.com/radixark/miles/issues/1296`
- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python` (`3.10.12`)
- GPUs: 2 × AMD Instinct MI350X, `gfx950` / CUDA capability `(9, 5)`
- Torch: `2.9.1+rocm7.2.0.lw.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Megatron Core: `0.19.0+8c1e05747`
- Megatron Bridge: `0.5.0+582783a0`
- Transformer Engine: `2.17.0`
- Transformers: `5.12.1`
- Flash Attention: `2.8.3`
- Flash-linear attention: `0.5.2`
- Apex: `1.9.0+rocm7.2.0.gite37ed124`

Imported source/native paths exercised by the fixture include:

- `/root/Megatron-LM/megatron/core/`
- `/opt/venv/lib/python3.10/site-packages/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py`
- `/opt/venv/lib/python3.10/site-packages/transformer_engine/`
- `/opt/venv/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- `/opt/venv/lib/python3.10/site-packages/flash_attn/`
- `/opt/venv/lib/python3.10/site-packages/fla/`
- `/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py`
- `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`

## Reproduction

Run from the repository root:

```bash
/opt/venv/bin/python -m torch.distributed.run \
  --standalone --rdzv-endpoint=localhost:29335 --nproc_per_node=2 \
  tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py \
  --mode cp --iterations 32 --output /tmp/qwen3-vl-cp.pt

/opt/venv/bin/python -m torch.distributed.run \
  --standalone --rdzv-endpoint=localhost:29337 --nproc_per_node=1 \
  tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py \
  --mode reference --iterations 32 \
  --cp /tmp/qwen3-vl-cp.pt --output /tmp/qwen3-vl-reference.pt

/opt/venv/bin/python tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py \
  --mode compare \
  --cp /tmp/qwen3-vl-cp.pt \
  --reference /tmp/qwen3-vl-reference.pt \
  --report /tmp/qwen3-vl-report.json
```

The compare phase prints the maxima and exits with status `1` because the CP output and gradient checks fail. Positions, packed independence, and boundary checks still pass.

## Scope and uncertainty

- This tests CP2 only on the two assigned MI350X GPUs. It does not test or validate CP4, CP8, tensor parallelism, pipeline parallelism, expert parallelism, or any larger topology.
- This uses a tiny synthetic/local model and `bf16` flash attention. It does not validate a full large VLM checkpoint or production numerics.
- The root cause of the CP forward/backward mismatch is not isolated. The exact position equality and local-output mismatch narrow the issue to the CP execution path, but do not identify whether the defect is in attention, backward communication, loss partitioning, or another CP component.
- No performance equivalence is claimed. Sharing `gfx950` is not a claim that MI350X and MI355X are performance-equivalent.
