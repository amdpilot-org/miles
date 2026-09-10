from datetime import timedelta
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["mtp", "weight-sync"])
register_rocm_ci(est_time=60, suite="nightly-stage-c-2-gpu-mi350", labels=["mtp", "weight-sync"])

_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT = timedelta(seconds=60)


class TinyMTPModel(torch.nn.Module):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        self.target = torch.nn.Linear(dimension, dimension)
        self.draft = torch.nn.Linear(dimension, dimension)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target(inputs), self.draft(inputs)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, tensor.detach().contiguous())
    return peers


def _assert_tensor_synced(tensor: torch.Tensor, label: str) -> None:
    peers = _gather_tensor(tensor.detach().reshape(-1))
    assert all(torch.equal(peers[0], peer) for peer in peers[1:]), f"{label} is not synchronized across ranks"


def _assert_tensor_not_synced(tensor: torch.Tensor, label: str) -> None:
    peers = _gather_tensor(tensor.detach().reshape(-1))
    assert any(not torch.equal(peers[0], peer) for peer in peers[1:]), f"{label} unexpectedly synchronized across ranks"


def _assert_module_synced(module: torch.nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        _assert_tensor_synced(parameter, f"{label}.{name}")


def _assert_module_not_synced(module: torch.nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        peers = _gather_tensor(parameter)
        if any(not torch.equal(peers[0], peer) for peer in peers[1:]):
            return
    raise AssertionError(f"{label} unexpectedly synchronized across ranks")


def _assert_versions_synced(target_version: torch.Tensor, draft_version: torch.Tensor) -> None:
    target_peers = _gather_tensor(target_version)
    draft_peers = _gather_tensor(draft_version)
    assert all(torch.equal(target_peers[0], peer) for peer in target_peers[1:]), "target update version is not synchronized"
    assert all(torch.equal(draft_peers[0], peer) for peer in draft_peers[1:]), "draft update version is not synchronized"
    assert torch.equal(target_version, draft_version), "target and draft update versions do not correspond to the same optimizer step"


def _assert_versions_not_synced(target_version: torch.Tensor, draft_version: torch.Tensor) -> None:
    target_peers = _gather_tensor(target_version)
    draft_peers = _gather_tensor(draft_version)
    assert any(not torch.equal(target_peers[0], peer) for peer in target_peers[1:]), "target update version unexpectedly synchronized"
    assert any(not torch.equal(draft_peers[0], peer) for peer in draft_peers[1:]), "draft update version unexpectedly synchronized"
    assert torch.equal(target_version, draft_version), "target and draft update versions diverged within a rank"


def _worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE, timeout=_PROCESS_GROUP_TIMEOUT)
    device = torch.device("cuda", rank)

    torch.manual_seed(1234)
    model = TinyMTPModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    target_version = torch.zeros(1, device=device)
    draft_version = torch.zeros(1, device=device)
    generator = torch.Generator(device=device).manual_seed(42)
    inputs = torch.randn(2, 4, device=device, generator=generator)

    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    target_output, draft_output = ddp_model(inputs)
    loss = target_output.sum() + draft_output.sum()
    loss.backward()
    optimizer.step()
    target_version += 1
    draft_version += 1
    optimizer.zero_grad(set_to_none=True)

    _assert_module_synced(ddp_model.module.target, "target after full update")
    _assert_module_synced(ddp_model.module.draft, "draft after full update")
    _assert_versions_synced(target_version, draft_version)
    with torch.no_grad():
        target_output, draft_output = ddp_model(inputs)
    _assert_tensor_synced(target_output, "target forward output after full update")
    _assert_tensor_synced(draft_output, "draft forward output after full update")

    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    target_output, draft_output = ddp_model(inputs)
    loss = target_output.sum() + draft_output.sum()
    loss.backward()
    if rank == 0:
        optimizer.step()
        target_version += 1
        draft_version += 1
    optimizer.zero_grad(set_to_none=True)

    _assert_module_not_synced(ddp_model.module.target, "target after partial update")
    _assert_module_not_synced(ddp_model.module.draft, "draft after partial update")
    _assert_versions_not_synced(target_version, draft_version)
    with torch.no_grad():
        target_output, draft_output = ddp_model(inputs)
    _assert_tensor_not_synced(target_output, "target forward output after partial update")
    _assert_tensor_not_synced(draft_output, "draft forward output after partial update")

    with torch.no_grad():
        for parameter in ddp_model.module.parameters():
            dist.broadcast(parameter.data, src=0)
        dist.broadcast(target_version, src=0)
        dist.broadcast(draft_version, src=0)

    _assert_module_synced(ddp_model.module.target, "target after recovery")
    _assert_module_synced(ddp_model.module.draft, "draft after recovery")
    _assert_versions_synced(target_version, draft_version)
    with torch.no_grad():
        target_output, draft_output = ddp_model(inputs)
    _assert_tensor_synced(target_output, "target forward output after recovery")
    _assert_tensor_synced(draft_output, "draft forward output after recovery")

    dist.barrier()
    dist.destroy_process_group()


def test_mtp_weight_sync_partial_update_recovery() -> None:
    port = _free_port()
    mp.spawn(_worker, args=(port,), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
