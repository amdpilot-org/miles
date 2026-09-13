# Reduced two-GPU train/rollout parity investigation

This branch records the operator-scoped investigation requested in upstream issue 2722.

## Planned evidence

- Run a reduced Qwen3-0.6B path on two assigned AMD Instinct MI350X GPUs.
- Compare Miles/Megatron training logprobs with SGLang rollout logprobs across at least 32 real optimizer updates.
- Control tokens, routing, and precision while separating first-token, EOS, and sampled-token behavior.
- Exercise BF16 and the available FP8 path, record distributional errors and their effect on gradient/update decisions, and report GPU phase timings.
- Preserve the qualified Torch/ROCm and framework stack, use unique rendezvous endpoints, and bound distributed initialization.

The final report will explicitly avoid claiming reproduction of the original TP8/EP8 model or MI350X/MI355X performance equivalence.
