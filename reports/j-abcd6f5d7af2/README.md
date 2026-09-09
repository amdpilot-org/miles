# OPD top-k gradient investigation

## Result

This is an explicit negative result for the current Miles top-k OPD implementation. The detached
top-k reverse-KL estimate is applied as an action-independent advantage penalty, so its expected
on-policy policy-gradient contribution is zero. The fixture confirms that five optimizer steps do
not change analytically controlled student logits when the full action expectation is evaluated.

The investigation does **not** provide a runtime correction. A correct correction requires a design
decision and additional data plumbing: Miles currently persists only the scalar
`opd_reverse_kl`, not the selected token IDs and teacher log-probabilities needed by a
differentiable training-forward loss. The paper reference also uses teacher-weighted forward KL
over teacher top-k, while Miles documents and implements student-weighted reverse KL. Changing the
loss direction, weighting, and normalization would alter intended semantics and should be reviewed
as part of a runtime fix.

## Reproduction

Run from the repository root:

```bash
/opt/venv/bin/python reports/j-abcd6f5d7af2/opd_topk_gpu_fixture.py
```

The fixture uses the single assigned MI355X GPU (`cuda:0`), initializes a one-rank NCCL process
group with a 60-second timeout, and destroys only that process group. It does not launch external
teacher or student servers, download model weights, or create other subprocesses.

## What the fixture exercises

- The real Miles rollout path: `_compute_topk_reverse_kl` with `only-student`, `student_p`, and
  top-k/tail mass retained (`VOCAB_SIZE=8`, `TOP_K=3`).
- The real Miles consumer and policy paths: `apply_opd_kl_to_advantages` and `compute_policy_loss`.
- An independent hand-coded top-k reverse-KL reference.
- An independent finite-difference gradient for the differentiable reverse-KL reference.
- A teacher-weighted top-k forward-KL reference with unnormalized top-k weights and its
  tail-mass-preserving analytic gradient
  `teacher_top_mass * softmax(student) - teacher_top_probability`.
- A separate entropy-maximization reference.
- Five optimizer steps for the Miles expected-gradient path, differentiable reverse KL,
  forward KL, and entropy.

## Controlled observations

The PR base revision was `8d9826ea`. The environment reported:

- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- GPU: one AMD Instinct MI355X
- Miles module: `/job/miles/miles/__init__.py`
- SGLang module: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron module: unavailable from this interpreter
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`

The latest MI355X run reported:

| Check | Result |
|-------|--------|
| Miles expected-gradient maximum | `2.619345e-09` |
| Miles logits maximum change after five steps | `1.797161e-09` |
| Reverse KL teacher match | `0.252776 -> 0.187413` |
| Reverse KL entropy | `1.394861 -> 1.283343` |
| Forward KL teacher match | `0.252776 -> 0.202569` |
| Forward KL entropy | `1.394861 -> 1.262033` |
| Entropy reference teacher match | `0.252776 -> 0.344078` |
| Entropy reference entropy | `1.394861 -> 1.546048` |
| Reverse-KL/entropy gradient cosine | `-0.646962` |

The fixture uses a student whose top set is `[0, 1, 2]` and a sharply peaked teacher on token 0.
Both distributions leave more than 10% probability outside the selected top-k set. The Miles
top-k estimate matches the independent reference. Its expected policy gradient is zero to floating
point, and five optimizer steps leave the logits unchanged.

The differentiable reverse-KL and paper-style forward-KL references both reduce full teacher KL and
reduce entropy over five steps. The entropy reference increases entropy and increases teacher KL.
Their gradients are not parallel, and changing the teacher changes the reverse-KL gradient. This
separates teacher matching from entropy maximization and rules out the explanation that the observed
zero Miles gradient is merely an entropy-gradient cancellation.

## Diagnosis

For a fixed state, Miles computes `r_t = KL_topk(student || teacher)` without reading the sampled
action, detaches it, and forms `A_t = A_t - lambda * r_t`. The on-policy policy gradient contains
`E[a~student][A_t * grad log student(a)]`. Since `A_t` is constant in `a`,

```text
E[a~student][grad log student(a)] = 0
```

so the top-k OPD contribution is zero in expectation. A finite one-action batch can move logits,
but its direction is determined by the accidentally sampled action rather than by teacher matching.
The issue's core zero-gradient diagnosis is therefore confirmed.

## What this does not prove

- It does not run a full SGLang rollout server or Megatron training job; the fixture intentionally
  isolates the exact Miles reward, advantage, and policy-loss functions with controlled logits.
- It does not prove that making the current reverse-KL estimate differentiable is the correct fix.
  That reference is teacher-dependent and distinct from entropy, but it is not the paper's
  teacher-weighted forward-KL implementation.
- It does not measure numerical behavior of a large vocabulary, tensor-parallel vocabulary gather,
  context parallelism, or a real model checkpoint.
- It does not establish whether the existing sampled-token (`top_k=0`) OPD path is sufficient for
  all intended use cases; only the top-k path is under test.

## Roadmap context

This reduced case addresses the multi-step correctness intent in `radixark/miles#2853` and the
AMD OPD roadmap item in `radixark/miles#2025`: teacher/student log-probability alignment, reverse
KL, reward, and multi-step training behavior on MI355X. It does not claim the full Qwen3-8B,
multi-teacher, Megatron-loaded-teacher, or colocated memory-saving recipes from that roadmap.

## Reference

- Upstream issue: `radixark/miles#2362`
- Paper-style independent implementation inspected from `thunlp/OPD`,
  `verl/recipe/gkd/megatron_kl_loss.py`, which computes teacher-weighted forward KL over teacher
  top-k. Its custom backward returns `softmax(student) - teacher_top_probability`. With the raw,
  unnormalized top-k probabilities passed in that file, autograd instead requires
  `teacher_top_mass * softmax(student) - teacher_top_probability`; the fixture preserves and tests
  the tail-mass form. This discrepancy is an upstream design question and is not silently
  normalized away.
