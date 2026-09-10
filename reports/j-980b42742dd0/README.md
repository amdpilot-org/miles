# Reduced PPO two-rank GPU investigation

## Scope and result

This is a bounded correctness and stability investigation for the H2 actor-critic roadmap item at https://github.com/radixark/miles/issues/2853. It does not claim task-level RL gains, MI355X equivalence, or validation of a topology larger than the two assigned MI350X devices.

The fixture in `tests/fast-gpu/reduced_ppo_two_rank.py` trains synthetic local actor and critic models with DDP on two ranks, using Miles PPO GAE, clipping, distributed advantage normalization, and distributed checkpoint state helpers. The required uninterrupted segment completed 256 optimizer steps. The checkpoint comparison separately completed 128 steps, saved model and optimizer state, resumed for 128 steps, and matched the uninterrupted segment exactly.

## Environment and commits

- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Interpreter: `/opt/venv/bin/python`
- Miles base commit: `e5125a97e1fd383f005f4de258a5985026e09425`
- Tested fixture commit: `86f8ed953c20823f247c42e5f0712a591f240c8f`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, git commit `7e1940d4b11fd6128be4c42ba41567ec5ab87102`
- HIP: `7.2.26015-fc0010cf6a`
- GPUs: two assigned AMD Instinct MI350X, `gfx950`, serials `692517020513` and `692517020502`
- Process group: NCCL, world size 2, 180-second bounded timeout
- Rendezvous: unique `c10d` endpoint `127.0.0.1:29740`
- Downloads: none; all model data and initialization are synthetic/local

Imported Miles source paths:

- `miles/backends/training_utils/loss_hub/math_utils.py`
- `miles/backends/training_utils/loss_hub/advantages.py`
- `miles/backends/training_utils/parallel.py`
- `miles/backends/fsdp_utils/checkpoint.py`

Imported native/Torch paths:

- `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- `/opt/venv/lib/python3.10/site-packages/torch/distributed/__init__.py`

## Numerical checks

The fixture compares Miles outputs against explicit references on every step:

- PPO clipping compares `compute_policy_loss` with `max(-ratio * advantage, -clamp(ratio) * advantage)`, using `eps_clip=eps_clip_high=0.2`.
- GAE compares Miles advantages and returns with a masked reverse recursion over active tokens, `gamma=0.99`, and `lambd=0.95`, using relative and absolute tolerance `2e-5`.
- Terminal rewards are placed on the last active token, not the last padded response token. The terminal mask case has a zero final response mask and a zero return at that inactive terminal position.
- Value bootstrapping uses zero after the last active token for the bounded terminated/truncated semantics exercised here.
- Sample accounting all-reduces local counts each step and requires 8 global samples and 74 global active tokens.
- Actor and critic parameter deltas are finite and nonzero after 256 steps.

Final measured values are recorded in `reduced_ppo_two_rank_results.json`:

- Actor parameter delta: max absolute `0.13822674751281738`, L2 `3.5080652236938477`, nonzero count `10432`
- Critic parameter delta: max absolute `0.0950658917427063`, L2 `0.8332765698438752`, nonzero count `2177`
- Final actor loss: `-0.1749095767736435`
- Final critic loss: `2.559025287628174`
- Final clipping fraction: `0.42500001192092896`
- Checkpoint model load difference: exact equality for actor and critic
- Checkpoint optimizer state: exact equality for Adam state
- Resumed final parameter difference: exact equality for actor and critic

## GPU timings

The aggregate timings cover the 256-step uninterrupted segment plus the separate 128-step pre-checkpoint and 128-step resumed segments:

- Total: `11.562702177092433` seconds
- Setup: `2.6389261884614825` seconds
- GAE and explicit reference checks: `4.158927697688341` seconds
- Forward: `0.23532244376838207` seconds
- Backward and DDP synchronization: `3.7458116356283426` seconds
- Optimizer steps: `0.25829692371189594` seconds
- Checkpoint save: `0.12143276166170835` seconds
- Checkpoint load: `0.014838095754384995` seconds

## Reproduction

From the Miles checkout, with both assigned MI350X devices visible:

```bash
/opt/venv/bin/torchrun \
  --nnodes=1 \
  --nproc-per-node=2 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29740 \
  --rdzv-id=miles-j-980b42742dd0-final-formatted \
  tests/fast-gpu/reduced_ppo_two_rank.py \
  --steps 256 \
  --resume-split 128 \
  --output /job/reduced_ppo_two_rank_results.json \
  --checkpoint-dir /job/reduced-ppo-checkpoint
```

## Limitations and unfinished scope

This validates the supported reduced PPO actor/critic path on two MI350X ranks only. It does not run a pretrained model, task-level evaluation, context parallelism, tensor/pipeline parallelism, multi-node training, or a larger topology. Those limitations are evidence about this bounded test and are not permission to claim broader coverage or performance equivalence.
