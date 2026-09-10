# Investigation: reduced MoE R3 replay fixture

## Scope

- Upstream issue: https://github.com/radixark/miles/issues/1002
- Target topology: two AMD Instinct MI350X GPUs (`gfx950`).
- Goal: exercise the actual Miles R3 record/replay manager, reloadable process-group path, Megatron `_AllToAll`, and real `torch.distributed.all_to_all_single` backward operations across at least 64 changing token/expert distributions.

## Initial source trace

- Miles R3 state and cursors: `miles/utils/replay_base.py`
- Megatron replay-stream registration: `miles/backends/megatron_utils/replay_utils.py`
- Miles reloadable process groups: `miles/utils/reloadable_process_group.py`
- Megatron MoE split derivation: `/root/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`
- Megatron all-to-all autograd wrapper: `/root/Megatron-LM/megatron/core/tensor_parallel/mappings.py`

The dispatcher computes `input_splits` and `output_splits` from the routing map and invokes `_AllToAll`. `_AllToAll` allocates an output using `sum(output_split_sizes)` and calls `torch.distributed.all_to_all_single`; it does not first verify that the local input length equals `sum(input_split_sizes)` or that all ranks agree on complementary split metadata. A malformed replay can therefore surface as Torch's generic `Split sizes doesn't match total dim 0 size` error rather than a routing-specific diagnostic.

## Result

The reduced two-GPU fixture passes on the assigned AMD Instinct MI350X (`gfx950`) devices.

- 64 unique, changing token/expert distributions.
- 128 real `all_to_all_single` forward calls and 128 real backward calls per rank.
- 128 split checks, 64 token identity checks, and 128 gradient checks per rank.
- Exact token identity and input-gradient correspondence.
- Maximum output error: `0.0`.
- Maximum input-gradient error: `0.0`.
- Maximum weight-gradient error: `8.94e-08`.
- Both malformed-split controls fail promptly with routing-specific diagnostics.

This validates the tested Miles R3 replay, reloadable process-group, and Megatron all-to-all paths under the reduced fixture. It does not claim equivalence for other topologies, production models, or the original issue’s full workload.

## Environment

- Miles branch: `amdpilot/j-1649a2fa5ea9`
- Miles base commit: `e5125a97e1fd383f005f4de258a5985026e09425`
- Miles fixture commit: `aa16f95934348aeafce2726397def6aa55b022c8`
- Megatron commit: `8c1e05747eb612b382df2632783df5c83a853646`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Devices: 2 × AMD Instinct MI350X (`gfx950`)

## Imported source and native paths

- Miles R3 replay manager: `miles/utils/replay_base.py`
- Miles reloadable process group: `miles/utils/reloadable_process_group.py`
- Megatron replay-stream mapping: `miles/backends/megatron_utils/replay_utils.py`
- Megatron MoE split derivation: `/root/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`
- Megatron all-to-all autograd wrapper: `/root/Megatron-LM/megatron/core/tensor_parallel/mappings.py`
- Native collective: `torch.distributed.all_to_all_single`
- Native autograd: `torch.autograd.Function`

## Reproduction

Run the worker directly:

```bash
PYTHONPATH=/job/miles /opt/venv/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  tests/fast-gpu/_moe_r3_replay_worker.py --output /tmp/moe-r3-results
```

Run the pytest wrapper:

```bash
/opt/venv/bin/python -m pytest --noconftest \
  tests/fast-gpu/test_moe_r3_replay.py -q
```

`--standalone` uses a free local rendezvous port. The process-group timeout is bounded to 60 seconds.

## Timings

Per-rank CUDA event timings over 64 cycles:

| Phase | Rank 0 mean / max (ms) | Rank 1 mean / max (ms) |
|---|---:|---:|
| Record | 102.43 / 102.43 | 103.95 / 103.95 |
| Dispatch | 1.71 / 101.16 | 1.70 / 100.95 |
| Expert | 0.65 / 37.29 | 0.64 / 36.97 |
| Combine | 0.12 / 0.26 | 0.12 / 0.26 |
| Forward | 48.19 / 2982.45 | 53.27 / 3307.45 |
| Backward | 4.12 / 230.02 | 3.97 / 221.24 |

## Malformed-split controls

1. **Local input mismatch**
   - Injected `input_splits=[65,0]` with 64 local rows.
   - Both ranks fail before all-to-all with:
     `local input split sum 65 != input rows 64`.

2. **Cross-rank split mismatch**
   - Injected inconsistent `output_splits` across ranks.
   - Both ranks fail before all-to-all with:
     `cross-rank split matrices disagree`.

Both controls include rank, world size, local row count, input/output splits, gathered split matrices, and expected output rows.

## Conclusion

The existing implementation passes this reduced, real-GPU R3 replay fixture. The malformed-split controls also show that routing-specific preflight diagnostics can catch invalid split metadata before reaching Torch’s generic split-size error.

This report is an early draft. GPU fixture implementation, numerical results, timings, tested commits, and reproduction commands will be appended after execution.
