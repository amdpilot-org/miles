# Investigation: bridge-built GDN distributed weight updates

## Result

The existing implementation already satisfies the operator-scoped requirement
for the tested dense bridge-built Qwen3.5 GDN topology. No core change is
proposed. `get_hf_weight_iterator` routes bridge mode to
`HfWeightIteratorBridge`, which uses Megatron Bridge's real `Qwen35Bridge`
conversion table rather than a fake bridge or hand-written substitute.

The validation fixture in `validate_bridge_gdn_updates.py` creates a tiny local
Qwen3.5 text model with three GDN layers and one full-attention layer. It runs
32 real optimizer/update cycles, converts through Miles's bridge iterator,
uses `UpdateWeightFromDistributed.send_bucket` and
`update_weights_from_distributed`, broadcasts each converted bucket over NCCL,
applies the tensors to a consumer HF model on the second GPU, and compares
consumer forward logits with a reference HF model on the first GPU.

## Evidence

- Cycles: 32/32 valid.
- Megatron parameter coverage: 41/41 bridge tasks; no missing or unexpected names.
- Initial converted tensor equality: 56/56 HF tensors exact.
- Per-cycle converted tensors: 56/56 exact on both trainer/reference and consumer.
- Missing-name diagnostics: 18 bridge-built GDN names rejected by the legacy direct converter.
- Later valid update: cycle 1 succeeded after recording those diagnostics.
- Consumer forward outputs: `allclose` true for all cycles; maximum absolute difference was 0.
- Optimizer loss: 4.1437 on cycle 1 and 2.5451 on cycle 32.
- GPU phase timing ranges: optimizer 17.24–5530.17 ms, conversion 7.73–10.42 ms,
  broadcast 5.98–41.91 ms, reference forward 6.08–617.77 ms,
  consumer forward 5.84–3006.10 ms, full cycle 38.45–8644.11 ms.
- First-cycle timings include warmup/compilation and are not performance claims.

The complete numerical record is in `gpu_validation.json`.

## Reproduction

The model is generated locally under `/job/cache`; it performs no network
downloads. Run from the repository root:

```bash
export PYTHONPATH=/job/miles
export MILES_DISABLE_IMPORT_WARNINGS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
/opt/venv/bin/torchrun \
  --nnodes=1 \
  --nproc-per-node=2 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --rdzv-id="j-3077708ab2c8-$(date +%s)-$$" \
  reports/j-3077708ab2c8/validate_bridge_gdn_updates.py
```

The run uses both assigned AMD Instinct MI350X (gfx950) GPUs: rank 0 is the
trainer/reference process on GPU 0 and rank 1 is the consumer on GPU 1. The
recorded rendezvous port was 42371 and the process-group timeout was 120 seconds.

## Environment and tested commits

- Miles: `d37ac22961743ae0ae78a1bef5e32b354ee04497`
- PR base: `e5125a97e1fd383f005f4de258a5985026e09425`
- Megatron-LM: `8c1e05747eb612b382df2632783df5c83a853646`
- Torch/ROCm: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Transformers: `5.12.1`
- Imported paths: Miles `/job/miles/miles`, Megatron Bridge
  `/opt/venv/lib/python3.10/site-packages/megatron/bridge/__init__.py`,
  Transformers `/opt/venv/lib/python3.10/site-packages/transformers/__init__.py`,
  Torch `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, and AITER
  `/sgl-workspace/aiter/aiter/jit/core.py`.

## Limitations

- This validates dense Qwen3.5 text only. It does not claim support for VL,
  MoE, MTP, quantization, pipeline parallelism greater than one, tensor
  parallelism greater than one, or the full 27B checkpoint.
- The consumer is a local Hugging Face process, not a production SGLang rollout
  engine. The Miles distributed tensor hook and NCCL transfer are real, but
  HTTP engine lifecycle and SGLang weight application are not covered.
- The legacy direct converter still rejects the bridge-built GDN names. That is
  recorded as diagnostic evidence; the successful path is the existing bridge
  iterator.
- These are correctness results, not MI355X or MI350X performance-equivalence
  claims.
