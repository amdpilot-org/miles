#!/usr/bin/env python3
"""Run the reduced two-GPU Miles issue-2856 investigation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class WeightMetadata:
    name: str
    dtype: str
    shape: list[int]
    offset: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        choices=("same-node", "issue-boundary"),
        required=True,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def module_path(name: str) -> str:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return "<unavailable>"
    if spec is None:
        return "<namespace>"
    return str(spec.origin or spec.submodule_search_locations)


def print_environment(rank: int, local_rank: int) -> None:
    paths = {
        "miles": module_path("miles"),
        "sglang": module_path("sglang"),
        "megatron.bridge": module_path("megatron.bridge"),
        "torch": module_path("torch"),
        "mooncake.engine": module_path("mooncake.engine"),
        "aiter.jit.module_aiter_core": module_path("aiter.jit.module_aiter_core"),
    }
    print(f"RESULT rank={rank} python={sys.executable} torch={torch.__version__}", flush=True)
    print(f"RESULT rank={rank} hip={getattr(torch.version, 'hip', None)}", flush=True)
    print(f"RESULT rank={rank} device={torch.cuda.get_device_name(local_rank)}", flush=True)
    for name, path in paths.items():
        print(f"RESULT rank={rank} module={name} path={path}", flush=True)


def initialize_groups(
    rank: int,
    local_rank: int,
    world_size: int,
    timeout: timedelta,
) -> tuple[Any, Any]:
    if world_size != 2:
        raise RuntimeError(f"This fixture requires exactly 2 ranks, got {world_size}")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"This fixture requires exactly 2 GPUs, got {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        timeout=timeout,
    )
    ranks = list(range(world_size))
    cpu_group = dist.new_group(ranks, backend="gloo", timeout=timeout)
    gpu_group = dist.new_group(ranks, backend="nccl", timeout=timeout)

    cpu_tensor = torch.tensor([rank], dtype=torch.int64)
    gpu_tensor = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(cpu_tensor, group=cpu_group)
    dist.all_reduce(gpu_tensor, group=gpu_group)
    torch.cuda.synchronize()
    if cpu_tensor.item() != 1 or gpu_tensor.item() != 1.0:
        raise RuntimeError(f"Collective sanity failed: cpu={cpu_tensor.item()} gpu={gpu_tensor.item()}")
    print(
        f"RESULT rank={rank} groups=ok cpu_backend={dist.get_backend(cpu_group)} "
        f"gpu_backend={dist.get_backend(gpu_group)}",
        flush=True,
    )
    return cpu_group, gpu_group


def create_cpu_replica(
    check: str,
    rank: int,
    local_rank: int,
    model_dir: Path,
) -> torch.nn.Module:
    from sglang.srt.distributed.parallel_state import RankParallelismConfig
    from sglang.srt.server_args import ServerArgs

    from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

    nnodes = 2 if check == "issue-boundary" else 1
    server_args = ServerArgs(
        model_path=str(model_dir),
        tp_size=2,
        nnodes=nnodes,
    )
    parallelism_config = RankParallelismConfig(
        tp_size=2,
        tp_rank=rank,
        world_size=2,
        global_rank=rank,
        local_rank=local_rank,
    )
    protocol = object.__new__(UpdateWeightP2P)
    return protocol._create_cpu_replica(
        parallelism_config,
        str(model_dir),
        server_args,
        first_engine_rank=True,
    )


def transfer_weights(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
) -> int:
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    initialize_rc = engine.initialize("127.0.0.1", "P2PHANDSHAKE", "hip", "")
    if initialize_rc != 0:
        raise RuntimeError(f"Mooncake HIP initialization failed with {initialize_rc}")

    source_tensors: list[torch.Tensor] = []
    target_tensors: list[torch.Tensor] = []
    source_metadata: list[WeightMetadata] = []
    if rank == 0:
        parameters = [parameter.detach().contiguous().reshape(-1) for _, parameter in model.named_parameters()]
        dtypes = {str(parameter.dtype) for parameter in parameters}
        if len(dtypes) != 1:
            raise RuntimeError(f"The reduced fixture requires one parameter dtype, got {sorted(dtypes)}")
        source_tensors.append(torch.cat(parameters).to(device="cuda"))
        offset = 0
        for (name, parameter), _ in zip(model.named_parameters(), parameters, strict=True):
            source_metadata.append(
                WeightMetadata(
                    name=name,
                    dtype=str(parameter.dtype),
                    shape=list(parameter.shape),
                    offset=offset,
                    sha256=tensor_sha256(parameter),
                )
            )
            offset += parameter.numel()

    metadata_message: list[Any] = [source_metadata if rank == 0 else None]
    dist.broadcast_object_list(metadata_message, src=0)
    expected_metadata = metadata_message[0]
    assert expected_metadata is not None

    flat_dtype = getattr(torch, expected_metadata[0].dtype.removeprefix("torch."))
    flat_size = sum(math.prod(metadata.shape) for metadata in expected_metadata)
    target_tensors.append(torch.zeros(flat_size, dtype=flat_dtype, device="cuda"))

    source_ptrs = [tensor.data_ptr() for tensor in source_tensors]
    target_ptrs = [tensor.data_ptr() for tensor in target_tensors]
    lengths = [tensor.numel() * tensor.element_size() for tensor in source_tensors]
    if rank == 0:
        assert len(source_ptrs) == len(target_ptrs) == len(lengths)

    for tensor in source_tensors + target_tensors:
        register_rc = engine.register_memory(
            tensor.data_ptr(),
            tensor.numel() * tensor.element_size(),
        )
        if register_rc != 0:
            raise RuntimeError(f"Mooncake HIP memory registration failed with {register_rc}")

    ports = [torch.empty(1, dtype=torch.int64) for _ in range(world_size)]
    target_pointer_message: list[Any] = [target_ptrs if rank == 1 else None]
    dist.all_gather(ports, torch.tensor([engine.get_rpc_port()], dtype=torch.int64))
    dist.broadcast_object_list(target_pointer_message, src=1)
    remote_target_ptrs = target_pointer_message[0]
    assert remote_target_ptrs is not None

    if rank == 0:
        target_name = f"127.0.0.1:{int(ports[1].item())}"
        for source_ptr, target_ptr, length in zip(source_ptrs, remote_target_ptrs, lengths, strict=True):
            transfer_rc = engine.transfer_sync_write(target_name, source_ptr, target_ptr, length)
            if transfer_rc != 0:
                raise RuntimeError(f"Mooncake HIP transfer failed with {transfer_rc}")
    dist.barrier()
    torch.cuda.synchronize()

    reconstructed_equal = True
    if rank == 1:
        flat_target = target_tensors[0]
        for metadata in expected_metadata:
            target = flat_target[metadata.offset : metadata.offset + math.prod(metadata.shape)]
            target = target.reshape(metadata.shape)
            reconstructed_equal &= (
                metadata.dtype == str(target.dtype)
                and metadata.shape == list(target.shape)
                and metadata.sha256 == tensor_sha256(target)
            )
    success = torch.tensor([int(reconstructed_equal)], device="cuda", dtype=torch.int64)
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if success.item() != 1:
        raise RuntimeError("Reconstructed weight equality failed")

    parameter_count = len(expected_metadata)
    element_count = sum(math.prod(metadata.shape) for metadata in expected_metadata)
    print(
        f"RESULT rank={rank} mooncake_hip_transfer=ok parameters={parameter_count} "
        f"elements={element_count} equality=ok",
        flush=True,
    )

    for tensor in source_tensors + target_tensors:
        engine.unregister_memory(tensor.data_ptr())
    return 0


def run_issue_boundary(
    rank: int,
    local_rank: int,
    model_dir: Path,
) -> None:
    expected_error = False
    error_text = ""
    try:
        create_cpu_replica("issue-boundary", rank, local_rank, model_dir)
    except ValueError as error:
        error_text = str(error)
        expected_error = (
            "MagicMock" in error_text
            and "not initialized in the world group map" in error_text
        )
        if not expected_error:
            raise
    if not expected_error:
        raise RuntimeError("The declared-multi-node boundary unexpectedly passed")
    print(f"RESULT rank={rank} issue_boundary=reproduced error={error_text}", flush=True)


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print_environment(rank, local_rank)
    timeout = timedelta(seconds=args.timeout_seconds)
    try:
        initialize_groups(rank, local_rank, world_size, timeout)
        if args.check == "same-node":
            model = create_cpu_replica("same-node", rank, local_rank, args.model_dir)
            parameter_count = sum(1 for _ in model.named_parameters())
            element_count = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"RESULT rank={rank} cpu_replica=ok model={type(model).__name__} "
                f"parameters={parameter_count} elements={element_count}",
                flush=True,
            )
            transfer_weights(rank, world_size, model)
        else:
            run_issue_boundary(rank, local_rank, args.model_dir)
        dist.barrier()
        print(f"RESULT rank={rank} check={args.check} status=pass", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
