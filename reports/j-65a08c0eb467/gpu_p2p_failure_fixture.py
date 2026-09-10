#!/usr/bin/env python3
"""Exercise the P2P weight-update failure boundary on two gfx950 GPUs."""

from argparse import Namespace
from datetime import timedelta
from importlib import import_module
from json import dump
from os import environ
from pathlib import Path
from sys import path
from types import SimpleNamespace

import aiter.jit.core as aiter_core
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
path.insert(0, str(REPO_ROOT))

aiter_core.AITER_CONFIGS.get_config_file = lambda _env_name, default_file, _tuned_file_name: default_file

from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P
from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import (
    P2PTransferManager,
    RemoteWeightInfo,
)
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils import distributed_utils


PROCESS_GROUP_TIMEOUT_SECONDS = 20
TRANSFER_TIMEOUT_SECONDS = 2
WEIGHT_VALUE = 7.0


class TinyMapper:
    def map(self, name: str):
        return SimpleNamespace(sglang_name="weight", num_shards=1, num_local_experts=None)


class TinyReplica:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def load_weights(self, named_tensors):
        for name, tensor in named_tensors:
            if name != "weight":
                raise RuntimeError(f"unexpected tiny-model weight: {name}")
            self.weight.copy_(tensor.to(device=self.weight.device))


class FakeTransferEngine:
    def __init__(self, source: torch.Tensor, target: torch.Tensor, *, fail: bool):
        self.source = source
        self.target = target
        self.fail = fail

    def batch_transfer_sync_write(self, _session_id, source_ptrs, target_ptrs, _lengths):
        if self.fail:
            return -1
        if source_ptrs != [self.source.data_ptr()] or target_ptrs != [self.target.data_ptr()]:
            raise RuntimeError("tiny-model transfer pointers do not match their registered tensors")
        self.target.copy_(self.source)
        torch.cuda.synchronize()
        return 0


class Consumer:
    def __init__(self):
        self.paused = False
        self.session_open = False
        self.finalized = False
        self.ready_version = None
        self.resumed = False

    def claim(self, weight_version: int) -> bool:
        return self.finalized and self.ready_version == weight_version and self.resumed


class TinyWeightIterator:
    placement = WeightUpdatePlacement(gather_pp=False)
    weight_update_selector = "all"

    def __init__(self, value: float):
        self.value = value

    def iter_hf_weights(self, _weights, **_kwargs):
        yield [("weight", torch.full((4,), self.value, device=torch.cuda.current_device()))]


def make_protocol(device: torch.device, *, fail: bool) -> tuple[UpdateWeightP2P, torch.Tensor, torch.Tensor]:
    source = torch.zeros(4, device=device)
    target = torch.zeros(4, device=device)
    protocol = UpdateWeightP2P.__new__(UpdateWeightP2P)
    protocol.args = Namespace()
    protocol.rollout_engines = [object()]
    protocol.is_sender = True
    protocol.group_name = "tiny-p2p"
    protocol.update_weight_metrics = {}
    protocol.transfer_manager = P2PTransferManager(
        num_workers=1,
        transfer_timeout=TRANSFER_TIMEOUT_SECONDS,
    )
    protocol._model_registered = True
    protocol._tensor_update_pending = {}
    protocol._staged_tensors = {}
    protocol._shared_params_dict = {"weight": source}
    protocol._shared_param_mapper = TinyMapper()
    protocol._weight_memory_registry = {
        "weight": (source.data_ptr(), source.numel(), source.element_size())
    }
    protocol._transfer_engine = FakeTransferEngine(source, target, fail=fail)
    protocol._transfer_engine_meta_list = [
        (
            TinyReplica(source),
            [
                RemoteWeightInfo(
                    session_id="tiny-session",
                    weights_info={"weight": (target.data_ptr(), target.numel(), target.element_size())},
                )
            ],
        )
    ]
    return protocol, source, target


def make_updater(protocol: UpdateWeightP2P, value: float) -> WeightUpdater:
    args = Namespace(pause_generation_mode="in_place")
    return WeightUpdater(
        args,
        [SimpleNamespace()],
        weights_getter=lambda: {},
        model_name="tiny-deterministic",
        quantization_config=None,
        iterator_factory=lambda *_args, **_kwargs: TinyWeightIterator(value),
        parallel_state=SimpleNamespace(),
        is_lora=False,
    )


def install_lifecycle_hooks(updater_module, consumer: Consumer, events: list[str]):
    def pause_engines(_args, _engines):
        consumer.paused = True
        events.append("pause")

    def begin_weight_update(_engines, *_args, **_kwargs):
        consumer.session_open = True
        events.append("begin")

    def end_weight_update(_engines, **_kwargs):
        consumer.finalized = True
        events.append("end")

    def set_weight_version(_engines, weight_version):
        consumer.ready_version = weight_version
        events.append("set_version")

    def resume_engines(_engines):
        consumer.resumed = True
        events.append("resume")

    updater_module.pause_engines = pause_engines
    updater_module.begin_weight_update = begin_weight_update
    updater_module.end_weight_update = end_weight_update
    updater_module.set_weight_version = set_weight_version
    updater_module.resume_engines = resume_engines


def module_path(name: str) -> str | None:
    module = import_module(name)
    return getattr(module, "__file__", None)


def main() -> None:
    rank = int(environ["RANK"])
    local_rank = int(environ["LOCAL_RANK"])
    world_size = int(environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"this fixture requires exactly two GPU processes, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    gloo_group = dist.new_group(
        backend="gloo",
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    distributed_utils.GLOO_GROUP = gloo_group

    updater_module = import_module("miles.backends.training_utils.weight_update.updater")
    consumer = Consumer()
    lifecycle_events = []
    install_lifecycle_hooks(updater_module, consumer, lifecycle_events)

    failing_protocol, source, target = make_protocol(device, fail=rank == 1)
    updater_module.get_weight_transfer_protocol = lambda _args: failing_protocol
    failing_updater = make_updater(failing_protocol, WEIGHT_VALUE)

    failure = None
    try:
        failing_updater.update_weights()
    except RuntimeError as error:
        failure = str(error)

    if failure is None:
        raise RuntimeError("rank 1 transfer failure did not fail the distributed update")

    reuse_refusal = None
    try:
        failing_protocol.connect([], None, None, None, None, "all")
    except RuntimeError as error:
        reuse_refusal = str(error)
    if reuse_refusal is None:
        raise RuntimeError("a failed P2P updater was allowed to reconnect")

    failed_claim = consumer.claim(failing_updater.weight_version) if rank == 0 else None
    failed_consumer_state = {
        "paused": consumer.paused,
        "session_open": consumer.session_open,
        "finalized": consumer.finalized,
        "ready_version": consumer.ready_version,
        "resumed": consumer.resumed,
    }
    failed_lifecycle_events = list(lifecycle_events)
    lifecycle_events.clear()

    recovered_protocol, recovered_source, recovered_target = make_protocol(device, fail=False)
    updater_module.get_weight_transfer_protocol = lambda _args: recovered_protocol
    recovered_updater = make_updater(recovered_protocol, WEIGHT_VALUE)
    recovered_updater.update_weights()
    recovered_claim = consumer.claim(recovered_updater.weight_version) if rank == 0 else None
    if rank == 0 and recovered_claim is not True:
        raise RuntimeError("consumer could not claim weights after successful recovery")

    result = {
        "rank": rank,
        "local_rank": local_rank,
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "failure": failure,
        "failed_updater_weight_version": failing_updater.weight_version,
        "failed_consumer_claim": failed_claim,
        "failed_lifecycle_events": failed_lifecycle_events,
        "failed_consumer_state": failed_consumer_state,
        "failed_updater_reuse_refusal": reuse_refusal,
        "recovery_successful": True,
        "recovered_updater_weight_version": recovered_updater.weight_version,
        "recovered_consumer_claim": recovered_claim,
        "recovered_lifecycle_events": list(lifecycle_events),
        "recovered_consumer_state": {
            "paused": consumer.paused,
            "session_open": consumer.session_open,
            "finalized": consumer.finalized,
            "ready_version": consumer.ready_version,
            "resumed": consumer.resumed,
        },
        "source_value": recovered_source.detach().cpu().tolist(),
        "target_value": recovered_target.detach().cpu().tolist(),
        "module_paths": {
            "miles": module_path("miles"),
            "sglang": module_path("sglang"),
            "megatron.core": module_path("megatron.core"),
            "torch": module_path("torch"),
            "aiter": module_path("aiter"),
            "mooncake": module_path("mooncake"),
        },
    }

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result, group=gloo_group)
    if rank == 0:
        output = Path(environ["FIXTURE_OUTPUT"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as output_file:
            dump({"ranks": gathered}, output_file, indent=2, sort_keys=True)
            output_file.write("\n")

    dist.barrier(group=gloo_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
