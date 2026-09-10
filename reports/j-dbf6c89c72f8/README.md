# SWA KV memory-saver validation

## Result

- **Passed:** 32/32 allocate → generate → release → train → resume cycles on one assigned `AMD Instinct MI350X` (`gfx950`, capability `(9, 5)`).
- **Correction:** current SGLang’s generic `SWAKVPool` constructor already honors a caller-provided `enable_memory_saver` value, but the Ascend, hybrid-MLA, and hybrid-MHA construction sites still forced `False`. The packaged patch propagates `get_exec().features.enable_memory_saver` at those three sites.
- **Allocator:** after each 4,096-token allocation, full/SWA/combined availability was `126,976`; after release it returned to `131,072` with no drift across all cycles.
- **Physical release:** every pause released `1.0078125 GiB` measured with `torch.cuda.mem_get_info`, not just allocator accounting.
- **Training reuse:** each cycle allocated a 64.015625 MiB synthetic model after the KV release, ran forward/backward, NCCL gradient all-reduce, and SGD update, then deleted it before resume.
- **Continuity:** token IDs and logits were identical before release and after resume in every cycle.

## Numerical checks

- Loss range: `8.38023853302002`–`8.395315170288086`; all values finite.
- Local gradient norm range: `0.4165954887866974`–`0.4173724353313446`; all values finite.
- NCCL gradient norm matched the local norm exactly in every cycle.
- Optimizer update norm range: `5.5082251492422074e-05`–`5.520970080397092e-05`; every update was nonzero and finite.

## Timings

Mean / maximum seconds per cycle:

- Allocate: `0.0011598510318435729` / `0.030067439191043377`
- Release allocator: `0.00450361316325143` / `0.13631709199398756`
- Pause: `0.0008363317465409636` / `0.000910148024559021`
- Forward/backward: `0.03860936645651236` / `1.137212673202157`
- NCCL all-reduce: `0.05020259725279175` / `1.5915120150893927`
- Optimizer update: `0.0009105403441935778` / `0.021191509440541267`
- Resume: `0.000268367410171777` / `0.00030416250228881836`

## Reproduction

Run from the qualified image:

```bash
LD_PRELOAD=/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so \
SGLANG_USE_AITER=0 \
/opt/venv/bin/python reports/j-dbf6c89c72f8/swa_memory_saver_gpu_validation.py
```

The fixture uses the real `SWAKVPool` and `SWATokenToKVPoolAllocator`, a synthetic local model, a unique process-group rendezvous file, and a 60-second process-group timeout. It performs no model downloads and does not modify allocator behavior.

The SGLang correction is packaged in `docker/amd_patch/latest/sglang_swa_memory_saver.patch` and applied by `docker/Dockerfile.rocm`.

## Tested versions and paths

- Miles commit: `f21cd30c8d02e8c38c47bde1d3d7f855f47b8a04`
- SGLang commit: `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6`
- SGLang pool: `/sgl-workspace/sglang/python/sglang/srt/mem_cache/swa_memory_pool.py`
- SGLang configurator: `/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py`
- Torch memory saver Python: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver/__init__.py`
- Torch memory saver native preload: `/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so`

## Limitations

- This is a one-GPU validation; no multi-GPU topology is claimed.
- The fixture uses synthetic local models, not a downloaded production model.
- It validates the pool/allocator lifecycle directly rather than launching a full SGLang HTTP server.
- No performance-equivalence claim is made for MI355X or any other GPU.
- The process-group rendezvous file is unique per run and removed after teardown.
