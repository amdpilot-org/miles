"""Shared naming helpers for Megatron-native adapter checkpoints."""

def megatron_shard_name(tp_rank: int, pp_rank: int, ep_rank: int, ep_size: int) -> str:
    """Return the stable shard name for one realized (tp, pp, ep) coordinate."""
    name = f"adapter_megatron_tp{tp_rank}_pp{pp_rank}"
    if ep_size > 1:
        name += f"_ep{ep_rank}"
    return name + ".pt"
