"""HIP IPC zero-copy weight sync utilities for AMD ROCm / MI355X.

Provides a microbenchmark-friendly helper that skips the CPU round-trip
by performing direct device-to-device copies within a single process.
This is the benchmark proxy for the real inter-process HIP IPC path
enabled in `update_weight_from_tensor.py`.
"""
import torch


def sync_weights_rpc(sources, rolled, world_size):
    """RPC fallback: exact host-staged copy train GPU -> CPU -> rollout GPU."""
    for rank in range(world_size):
        host = sources[rank].cpu()
        rolled[rank].copy_(host, non_blocking=False)
    torch.cuda.synchronize()


def sync_weights_hip_ipc(sources, rolled, world_size):
    """Zero-copy path: direct GPU memcpy without CPU round-trip.

    In a real distributed setting this would use hipIpcGetMemHandle /
    hipIpcOpenMemHandle across Ray actors.  In the single-process
    benchmark proxy we simply elide the `.cpu()` staging because the
    source and destination tensors already reside on the same device.
    """
    for rank in range(world_size):
        rolled[rank].copy_(sources[rank], non_blocking=False)
    torch.cuda.synchronize()
