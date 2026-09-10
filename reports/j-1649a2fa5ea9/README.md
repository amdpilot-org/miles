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

## Status

This report is an early draft. GPU fixture implementation, numerical results, timings, tested commits, and reproduction commands will be appended after execution.
