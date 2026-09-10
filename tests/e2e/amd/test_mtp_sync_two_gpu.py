"""Two-GPU target/MTP synchronization and recovery fixture.

Run with:
    torchrun --nproc_per_node=2 tests/e2e/amd/test_mtp_sync_two_gpu.py

The fixture keeps a reduced target model and MTP head in one DDP module, trains
them jointly for two optimizer steps, synchronizes both parameter sets into a
separate rollout replica after the first step, and verifies that the replica's
target and MTP forward outputs match the training model. It then injects a
target-only partial update after the second step and verifies that the MTP
branch is stale before a full resync recovers both branches to the same
optimizer step.
"""

import os
from datetime import timedelta

import megatron.core
import miles
import sglang
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class ReducedTargetMTP(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.target = nn.Linear(width, width)
        self.mtp_head = nn.Linear(width, width)

    def forward(self, inputs):
        return self.target(inputs), self.mtp_head(inputs)


class RolloutReplica:
    def __init__(self, model: ReducedTargetMTP):
        self.model = model
        self.weight_version = 0
        self.target_step = 0
        self.mtp_step = 0
        self.partial_update = False

    def apply_full_update(self, optimizer_step: int):
        self.target_step = optimizer_step
        self.mtp_step = optimizer_step
        self.weight_version += 1
        self.partial_update = False

    def apply_partial_update(self, optimizer_step: int):
        self.target_step = optimizer_step
        self.partial_update = True


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 2, f"this fixture requires exactly two GPUs, got {world_size}"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    return rank, world_size, local_rank


def broadcast_metadata(device, weight_version, optimizer_step, complete):
    metadata = torch.tensor(
        [weight_version, optimizer_step, int(complete)],
        dtype=torch.int64,
        device=device,
    )
    dist.broadcast(metadata, src=0)
    return metadata


def broadcast_parameters(training_model, rollout_model, include_mtp):
    for training_parameter, rollout_parameter in zip(
        training_model.target.parameters(), rollout_model.target.parameters()
    ):
        dist.broadcast(training_parameter.data, src=0)
        rollout_parameter.data.copy_(training_parameter.data)
    if include_mtp:
        for training_parameter, rollout_parameter in zip(
            training_model.mtp_head.parameters(), rollout_model.mtp_head.parameters()
        ):
            dist.broadcast(training_parameter.data, src=0)
            rollout_parameter.data.copy_(training_parameter.data)


def optimizer_step_count(optimizer):
    steps = {
        int(optimizer.state[parameter]["step"].item())
        for parameter in optimizer.param_groups[0]["params"]
    }
    assert len(steps) == 1, f"optimizer parameters advanced to different steps: {sorted(steps)}"
    return steps.pop()


def parameters_match(left, right):
    return all(
        torch.equal(left_parameter, right_parameter)
        for left_parameter, right_parameter in zip(left.parameters(), right.parameters())
    )


def forward_outputs_match(training_model, rollout_model, inputs):
    with torch.no_grad():
        training_target, training_mtp = training_model(inputs)
        rollout_target, rollout_mtp = rollout_model(inputs)
    target_match = torch.equal(training_target, rollout_target)
    mtp_match = torch.equal(training_mtp, rollout_mtp)
    return target_match, mtp_match


def train_one_step(ddp_model, optimizer, inputs, target_labels, mtp_labels):
    optimizer.zero_grad(set_to_none=True)
    target_output, mtp_output = ddp_model(inputs)
    loss = (target_output - target_labels).square().mean()
    loss += (mtp_output - mtp_labels).square().mean()
    loss.backward()
    optimizer.step()
    return optimizer_step_count(optimizer)


def print_module_provenance(rank):
    if rank != 0:
        return
    print(f"miles={miles.__file__}")
    print(f"sglang={sglang.__file__}")
    print(f"megatron.core={megatron.core.__file__}")
    print(f"torch={torch.__file__}")
    print(f"torch._C={torch._C.__file__}")


def run_fixture(rank, world_size, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    print_module_provenance(rank)

    torch.manual_seed(1234)
    training_model = ReducedTargetMTP().to(device)
    ddp_model = DDP(training_model, device_ids=[local_rank], output_device=local_rank)
    rollout_model = ReducedTargetMTP().to(device)
    rollout = RolloutReplica(rollout_model)

    torch.manual_seed(4321)
    inputs = torch.randn(8, 16, device=device)
    target_labels = torch.randn_like(inputs)
    mtp_labels = torch.randn_like(inputs)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.05)

    optimizer_step = train_one_step(ddp_model, optimizer, inputs, target_labels, mtp_labels)

    broadcast_metadata(device, rollout.weight_version + 1, optimizer_step, True)
    broadcast_parameters(training_model, rollout_model, include_mtp=True)
    rollout.apply_full_update(optimizer_step)

    target_parameters_match = parameters_match(training_model.target, rollout_model.target)
    mtp_parameters_match = parameters_match(training_model.mtp_head, rollout_model.mtp_head)
    target_forward_match, mtp_forward_match = forward_outputs_match(training_model, rollout_model, inputs)
    full_sync_passed = (
        target_parameters_match
        and mtp_parameters_match
        and target_forward_match
        and mtp_forward_match
        and rollout.target_step == optimizer_step
        and rollout.mtp_step == optimizer_step
        and rollout.weight_version == 1
    )

    optimizer_step = train_one_step(ddp_model, optimizer, inputs, target_labels, mtp_labels)

    broadcast_metadata(device, rollout.weight_version + 1, optimizer_step, False)
    broadcast_parameters(training_model, rollout_model, include_mtp=False)
    rollout.apply_partial_update(optimizer_step)
    partial_target_match, partial_mtp_match = forward_outputs_match(training_model, rollout_model, inputs)
    partial_update_detected = partial_target_match and not partial_mtp_match

    broadcast_metadata(device, rollout.weight_version + 1, optimizer_step, True)
    broadcast_parameters(training_model, rollout_model, include_mtp=True)
    rollout.apply_full_update(optimizer_step)
    recovery_target_match, recovery_mtp_match = forward_outputs_match(training_model, rollout_model, inputs)
    recovery_passed = (
        recovery_target_match
        and recovery_mtp_match
        and rollout.target_step == optimizer_step
        and rollout.mtp_step == optimizer_step
        and rollout.weight_version == 2
        and not rollout.partial_update
    )

    passed = full_sync_passed and partial_update_detected and recovery_passed
    if rank == 0:
        print(f"optimizer_step={optimizer_step}")
        print(f"full_sync_target_parameters={target_parameters_match}")
        print(f"full_sync_mtp_parameters={mtp_parameters_match}")
        print(f"full_sync_target_forward={target_forward_match}")
        print(f"full_sync_mtp_forward={mtp_forward_match}")
        print(f"partial_update_target_forward={partial_target_match}")
        print(f"partial_update_mtp_forward={partial_mtp_match}")
        print(f"recovery_target_forward={recovery_target_match}")
        print(f"recovery_mtp_forward={recovery_mtp_match}")
        print(f"rollout_weight_version={rollout.weight_version}")
        print(f"fixture_passed={passed}")

    failure_count = torch.tensor([int(not passed)], dtype=torch.int64, device=device)
    dist.all_reduce(failure_count, op=dist.ReduceOp.SUM)
    assert failure_count.item() == 0, "target/MTP synchronization fixture failed on at least one rank"


def main():
    rank, world_size, local_rank = setup_distributed()
    try:
        run_fixture(rank, world_size, local_rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
