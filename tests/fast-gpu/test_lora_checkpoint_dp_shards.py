"""Two-GPU check for single-LoRA Megatron-native checkpoint ownership."""

import os
import sys
import tempfile
from argparse import Namespace
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn import Linear

from miles.backends.megatron_utils.lora_utils import (
    _atomic_torch_save,
    load_lora_adapter,
    save_lora_checkpoint,
)
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=30, suite="stage-c-4-gpu-h200", labels=["checkpoint", "lora"])
register_rocm_ci(est_time=30, suite="nightly-stage-c-4-gpu-mi350", labels=["checkpoint", "lora"])


class FakeBridge:
    @classmethod
    def from_hf_pretrained(cls, *_args, **_kwargs):
        return cls()

    def export_adapter_weights(self, *_args, **_kwargs):
        return iter([])


class FakeOptimizer:
    def __init__(self, rank):
        self.rank = rank

    def state_dict(self):
        return {"rank": self.rank, "step": torch.tensor([7], device="cuda")}

    def load_state_dict(self, state):
        assert state["rank"] == self.rank
        assert torch.equal(state["step"].cpu(), torch.tensor([7]))


@contextmanager
def no_model_patch(*_args, **_kwargs):
    yield


def install_parallel_state(rank, world_size):
    dp_group = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
    singleton = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=dp_group,
            intra_dp_cp=dp_group,
            cp=singleton,
            tp=singleton,
            pp=singleton,
            ep=singleton,
            etp=singleton,
            indep_dp=singleton,
        )
    )


def make_model():
    torch.manual_seed(1234)
    model = Linear(4, 4).cuda()
    model.lora_A = Linear(4, 2, bias=False).cuda()
    model.lora_B = Linear(2, 4, bias=False).cuda()
    with torch.no_grad():
        model.lora_A.weight.fill_(0.25)
        model.lora_B.weight.fill_(0.5)
    return model


def make_args():
    return Namespace(
        hf_checkpoint="/unused-small-fixture",
        target_modules=[],
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )


def assert_restore(model, root, rank, expected_iteration=7):
    with torch.no_grad():
        model.lora_A.weight.fill_(rank + 100)
        model.lora_B.weight.fill_(rank + 200)

    loaded, iteration = load_lora_adapter(
        [model],
        str(root),
        optimizer=FakeOptimizer(rank),
        opt_param_scheduler=None,
    )
    assert loaded
    assert iteration == expected_iteration
    assert torch.equal(model.lora_A.weight, torch.full((2, 4), 0.25, device="cuda"))
    assert torch.equal(model.lora_B.weight, torch.full((4, 2), 0.5, device="cuda"))


def run(rank, world_size):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=120))
    install_parallel_state(rank, world_size)

    root_buffer = [None]
    if rank == 0:
        root_buffer[0] = Path(tempfile.mkdtemp(prefix="miles-lora-dp-shards-"))
    dist.broadcast_object_list(root_buffer, src=0)
    root = root_buffer[0]

    model = [make_model()]
    args = make_args()
    with (
        patch("megatron.bridge.AutoBridge", FakeBridge),
        patch("miles.utils.megatron_bridge_utils.patch_megatron_model", no_model_patch),
    ):
        save_lora_checkpoint(
            model,
            args,
            str(root / "new"),
            optimizer=FakeOptimizer(rank),
            opt_param_scheduler=None,
            iteration=7,
        )
        assert_restore(model[0], root / "new", rank)

        with patch("megatron.bridge.AutoBridge", object):
            save_lora_checkpoint(
                model,
                args,
                str(root / "hf-error"),
                optimizer=FakeOptimizer(rank),
                opt_param_scheduler=None,
                iteration=7,
            )
        assert_restore(model[0], root / "hf-error", rank)

    native_name = "adapter_megatron_tp0_pp0.pt"
    for directory in (root / "new", root / "hf-error"):
        assert (directory / native_name).is_file()
        assert not (directory / "adapter_megatron_rank0.pt").exists()
        assert not (directory / "adapter_megatron_rank1.pt").exists()
        assert (directory / "training_state_rank0.pt").is_file()
        assert (directory / "training_state_rank1.pt").is_file()

    legacy_root = root / "legacy"
    legacy_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora_A.weight": torch.full((2, 4), 0.25),
            "lora_B.weight": torch.full((4, 2), 0.5),
        },
        legacy_root / f"adapter_megatron_rank{rank}.pt",
    )
    assert_restore(model[0], legacy_root, rank, expected_iteration=None)

    atomic_failure_path = root / "atomic-failure.pt"
    original_save = torch.save

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("simulated interrupted save")

    torch.save = fail_save
    try:
        _atomic_torch_save({"tensor": torch.empty(1, device="cuda")}, atomic_failure_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("simulated save unexpectedly succeeded")
    finally:
        torch.save = original_save

    assert not atomic_failure_path.exists()
    assert not list(root.glob(".atomic-failure.pt.*.tmp"))

    dist.barrier()
    if rank == 0:
        import shutil

        shutil.rmtree(root)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp(
            sys.executable,
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", __file__],
        )
    run(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
