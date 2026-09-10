# GDN packed-sequence validation on MI350X

## Result

The installed Megatron GDN and the Miles Qwen3.5 attention boundary already support the tested reduced BSHD dense-metadata case and genuinely packed THD inputs. No framework guard was weakened or removed.

- **Overall pass:** `true`
- **Guard probe:** deterministic THD is still rejected by Megatron GDN.
- **Training cycles:** 64
- **Forward/backward calls:** 640
- **GPU:** 1 × AMD Instinct MI350X (`gfx950`, capability `9.5`)
- **Wall time:** 11.31 seconds

## Numerical checks

The fixture compares:

1. Megatron GDN BSHD with no metadata versus a supplied `PackedSeqParams(qkv_format="bshd")` object.
2. Megatron GDN genuinely packed THD `[0, 13, 32]` versus independent per-sequence BSHD controls.
3. Miles Qwen3.5 attention BSHD with no metadata versus a supplied dense `PackedSeqParams(qkv_format="bshd")` object.
4. Miles Qwen3.5 attention genuinely packed THD `[0, 13, 32]` versus independent per-sequence BSHD controls.

The gating check is `torch.allclose(..., rtol=2e-2, atol=2e-2)` on outputs and input gradients.

### Megatron GDN

- BSHD dense metadata:
  - Output max absolute difference: `0.0`
  - Input-gradient max absolute difference: `0.0`
  - Parameter-gradient max absolute difference: `0.0`
- THD packed:
  - Output max absolute difference: `0.0`
  - Input-gradient max absolute difference: `0.00244903564453125`
  - Parameter-gradient max absolute difference: `0.059326171875`
- First loss: `0.9795478582382202`
- Last loss: `0.2360016405582428`
- Initial parameter norm: `193.07184970378876`
- Final parameter norm: `228.9421055316925`
- Forward time: `6570.28 ms`
- Backward time: `2661.22 ms`

### Miles Qwen3.5 attention

- BSHD dense metadata:
  - Output max absolute difference: `0.0`
  - Input-gradient max absolute difference: `0.0`
  - Parameter-gradient max absolute difference: `0.0`
- THD packed:
  - Output max absolute difference: `0.0`
  - Input-gradient max absolute difference: `0.010711669921875`
  - Parameter-gradient max absolute difference: `0.10986328125`
- First loss: `1.0321874618530273`
- Last loss: `0.17531102895736694`
- Initial parameter norm: `174.23547959327698`
- Final parameter norm: `197.99971184190483`
- Forward time: `612.52 ms`
- Backward time: `711.54 ms`

Parameter-gradient absolute differences are recorded but are not used as the pass criterion because bf16 reductions produce large relative errors near zero. Output and input-gradient checks are the gating numerical checks.

## Limitation

The Miles Qwen3.5 wrapper requires batch size 1 when `cu_seqlens` is supplied. The fixture therefore uses:

- Megatron GDN BSHD batch: `2`
- Miles Qwen3.5 BSHD batch: `1`
- THD boundaries: `[0, 13, 32]`

This is a source-level limitation, not a claim that a larger topology passed.

## Reproduction

From `/job/miles`:

```bash
/opt/venv/bin/python reports/j-20915bb3a656/gdn_packing_validation.py
```

The script:

- Uses one assigned MI350X.
- Initializes a unique file-based NCCL rendezvous with a 2-minute timeout.
- Uses synthetic random initialization; no model weights are downloaded.
- Runs 64 training cycles with 640 total forward/backward calls.
- Records GPU forward and backward timings with CUDA events.
- Proves the deterministic THD unsupported-format guard still rejects the case.

## Provenance

### Versions

- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Megatron Core: `0.19.0+8c1e05747`
- Transformer Engine: `2.17.0`
- Flash-linear-attention: `0.5.2`
- Triton: `3.6.0+git42270451`

### Commits

- Miles workspace: `e5125a97e1fd383f005f4de258a5985026e09425`
- Miles installed editable source: `8d9826eacc8b5c279546f96711bb401b7f62c54c`
- Megatron: `8c1e05747eb612b382df2632783df5c83a853646`
- Transformer Engine source commit: unavailable in this environment

### Source and native paths

- Megatron GDN: `/root/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py`
- Megatron packed-sequence parameters: `/root/Megatron-LM/megatron/core/packed_seq_params.py`
- Miles Qwen3.5: `/job/miles/miles_plugins/models/qwen3_5.py`
- Miles Hugging Face attention boundary: `/job/miles/miles_plugins/models/hf_attention.py`
- Flash-linear-attention chunk operation: `/opt/venv/lib/python3.10/site-packages/fla/ops/gated_delta_rule/chunk.py`
- Transformer Engine: `/opt/venv/lib/python3.10/site-packages/transformer_engine`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch`
- Triton: `/opt/venv/lib/python3.10/site-packages/triton`

## Uncertainty

- The Miles wrapper’s BSHD batch-size-1 limitation is source-level evidence, not a broader topology claim.
- Parameter-gradient differences are recorded but not gated because bf16 reductions produce large relative errors near zero.
- Transformer Engine’s source commit is unavailable in this environment; only the installed package version is recorded.
