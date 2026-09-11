# OPD top-k investigation: issue 2362

## Result

The reported core defect is real, but the first obvious fix is also insufficient.

1. **Old path:** `opd_reverse_kl` is a function of the prefix/state only. Miles detaches it and subtracts it from every action's advantage. The expected score-function gradient is therefore zero on-policy.
2. **False fix:** moving the existing frozen `student_p` weights into a differentiable loss does not distill the teacher. At the on-policy snapshot, `∇Σ p_rollout log p_current = 0`; off-policy it pulls the current policy toward its own rollout snapshot.
3. **Validated correction:** carry the selected token IDs and teacher log-probs into training, gather current student log-probs at those IDs, renormalize both current student and teacher distributions over the selected set, and add the differentiable subset reverse KL to the policy loss. Mass outside the selected set is discarded, matching the paper's top-k approximation.

The old action-independent reward has measured expected gradient max-norm `1.18e-8`. The corrected Miles gradient has max-norm `0.5801` and matches a direct PyTorch reference to `3.73e-9` and an analytic gradient to `1.19e-7`.

## GPU fixture

The executable fixture is `reports/j-f1eb9401aa1f/opd_gpu_fixture.py`. It uses the real Miles rollout target extraction (`_compute_topk_distillation`) and real policy loss (`policy_loss_function`), with synthetic teacher/student logits and no model weights.

It ran on the sole assigned GPU:

- AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, capability `(9, 5)`
- Torch `2.9.1+rocm7.2.0.git7e1940d4`
- 30 Adam steps at learning rate `0.05`

The union strategy selected tokens `[0, 1, 2]`. The initial student distribution was `[0.20, 0.79, 0.009999, 0.000001]`; the teacher was `[0.80, 0.10, 0.099999, 0.000001]`.

After 30 steps:

- OPD subset-KL trajectory: `[0.7598, 0.1957, 0.0445, 0.000001]`, KL to teacher `0.0550`
- entropy-only control: `[0.5048, 0.4227, 0.0725, 0.0000075]`, KL to teacher `0.2563`
- initial KL to teacher: `1.1326`

Thus the corrected loss performs teacher matching rather than merely maximizing entropy. Full machine-readable output is in `reports/j-f1eb9401aa1f/gpu_fixture_results.json`.

## What this proves

- The action-independent top-k reward has zero expected on-policy score-function gradient.
- The corrected training loss has a nonzero, independently verified gradient.
- Top-k/tail-mass semantics are preserved by renormalizing over the selected set and discarding outside mass.
- Several optimizer steps move a controlled student toward the controlled teacher, unlike an entropy-only control.
- Rollout-to-training plumbing carries candidate IDs, teacher log-probs, and validity weights through the existing data conversion and CP slicing paths.

## What this does not prove

- It does not run an external SGLang teacher server or a public model; teacher/student top-logprobs are synthetic.
- It does not establish end-to-end model quality, multi-teacher routing behavior, or long-horizon OPD stability.
- The GPU fixture uses one assigned GPU and a single-process trivial parallel state. It does not exercise a multi-GPU process group; unit tests cover the data plumbing and loss math only.
- Tensor-parallel and context-parallel paths reuse existing Miles helpers, but only their non-distributed behavior was executed on this one-GPU allocation.
- Megatron top-k OPD is not claimed: existing argument validation restricts top-k to SGLang, and sampled-token Megatron OPD remains unchanged.

## Reproduction

From the repository root:

```bash
PYTORCH_HIP_VISIBLE_DEVICES=0 /opt/venv/bin/python reports/j-f1eb9401aa1f/opd_gpu_fixture.py

PYTORCH_HIP_VISIBLE_DEVICES=0 /opt/venv/bin/python -m pytest --noconftest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/backends/training_utils/loss/test_opd.py
```

`--noconftest` avoids an unrelated, node-wide AITER cache lock in this container's shared `/tmp/aiter_configs`; it does not alter the Miles code under test.

## Provenance

- PR base: `e5125a97e1fd383f005f4de258a5985026e09425`
- Tested Miles import: `/job/miles/miles`
- SGLang import: `/sgl-workspace/sglang/python/sglang`
- Megatron namespace: `/opt/venv/lib/python3.10/site-packages/megatron` (editable finder)
- Torch: `/opt/venv/lib/python3.10/site-packages/torch`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Triton: `/sgl-workspace/triton-custom/python/triton`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

Preinstalled source paths are environment context; the tested Miles revision is the working clone under `/job/miles`.

## References

- Upstream issue: https://github.com/radixark/miles/issues/2362
- H2 roadmap: https://github.com/radixark/miles/issues/2853
- AMD Q3 roadmap: https://github.com/radixark/miles/issues/2025
- Paper: https://arxiv.org/abs/2604.13016
