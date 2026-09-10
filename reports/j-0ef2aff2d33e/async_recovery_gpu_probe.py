#!/usr/bin/env python3
"""Exercise the supported Miles async ownership path on two ROCm ranks."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.fully_async_data_buffer import (
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
)
from miles.utils.types import Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "control",
            "pre_admission",
            "pre_admission_restore",
            "post_admission",
            "post_admission_restore",
            "duplicate_late",
            "compare",
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--load-checkpoint-dir", type=Path)
    parser.add_argument("--boundary", type=int, default=32)
    parser.add_argument("--total-batches", type=int, default=64)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--case-checkpoint", type=Path)
    parser.add_argument("--comparison-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_local_assets(output_dir: Path) -> tuple[Path, Path]:
    tokenizer_dir = output_dir / "tokenizer"
    prompt_path = output_dir / "prompts.jsonl"
    if dist.get_rank() == 0:
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        raw_tokenizer = Tokenizer(
            models.WordLevel(
                vocab={"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4, "d": 5},
                unk_token="<unk>",
            )
        )
        raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
        )
        tokenizer.save_pretrained(tokenizer_dir)
        with prompt_path.open("w", encoding="utf-8") as handle:
            for prompt_index in range(64):
                handle.write(json.dumps({"prompt": "a b c d", "metadata": {"prompt_index": prompt_index}}) + "\n")
    dist.barrier()
    return tokenizer_dir, prompt_path


def build_args(tokenizer_dir: Path, prompt_path: Path, checkpoint_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint=str(tokenizer_dir),
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(prompt_path),
        rollout_max_prompt_len=32,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=0,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        buffer_filter_path=None,
        save=str(checkpoint_dir),
        load=str(checkpoint_dir),
        reward_key=None,
        async_data_buffer_capacity_factor=4,
        max_weight_staleness=None,
        dynamic_sampling_filter_path=None,
        async_unused_samples_handler="drop",
    )


def create_model(device: torch.device) -> nn.Module:
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32))
    return model.to(device)


def synthetic_completion(prompt_group: list[Sample]) -> list[Sample]:
    completed_group = []
    for prompt_sample in prompt_group:
        completed_sample = copy.deepcopy(prompt_sample)
        completed_sample.status = Sample.Status.COMPLETED
        completed_sample.response = "a b c d"
        completed_sample.response_length = 4
        completed_sample.tokens = [2, 3, 4, 5]
        completed_sample.reward = 1.0
        completed_sample.loss_mask = [1, 1, 1, 1]
        completed_sample.rollout_id = prompt_sample.group_index
        completed_group.append(completed_sample)
    return completed_group


def sample_identity(entry: DataBufferInput) -> dict[str, object]:
    return {
        "group_index": entry.prompt_group[0].group_index,
        "sample_indices": [sample.index for sample in entry.prompt_group],
        "prompt_indices": [sample.metadata.get("prompt_index") for sample in entry.prompt_group],
    }


def training_batch(group_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(10_000 + group_index)
    inputs = torch.randn((16, 32), generator=generator)
    targets = torch.randn((16, 32), generator=generator)
    return inputs.to(device), targets.to(device)


def save_checkpoint(
    checkpoint_dir: Path,
    rollout_id: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source: RolloutDataSourceWithBuffer,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source.save(rollout_id)
    torch.save(model.module.state_dict(), checkpoint_dir / "model.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")


def load_checkpoint(
    checkpoint_dir: Path,
    rollout_id: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source: RolloutDataSourceWithBuffer,
) -> None:
    source.load(rollout_id)
    model.module.load_state_dict(torch.load(checkpoint_dir / "model.pt", weights_only=True))
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", weights_only=True))


def timed_synchronize() -> None:
    torch.cuda.synchronize()


async def reserve_entry(source: RolloutDataSourceWithBuffer, buffer: DefaultDataBuffer) -> DataBufferInput:
    prompt_group = source.get_samples(1)[0]
    completed_group = synthetic_completion(prompt_group)
    entry = DataBufferInput(prompt_group=prompt_group, group=completed_group)
    await buffer.put(entry)
    return entry


async def train_entry(
    entry: DataBufferInput,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    group_index = entry.prompt_group[0].group_index
    inputs, targets = training_batch(group_index, device)
    optimizer.zero_grad(set_to_none=True)

    timed_synchronize()
    forward_start = time.perf_counter()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    timed_synchronize()
    forward_seconds = time.perf_counter() - forward_start

    backward_start = time.perf_counter()
    loss.backward()
    timed_synchronize()
    backward_seconds = time.perf_counter() - backward_start

    optimizer_start = time.perf_counter()
    optimizer.step()
    timed_synchronize()
    optimizer_seconds = time.perf_counter() - optimizer_start

    acknowledgement = torch.tensor([step, group_index], device=device, dtype=torch.int64)
    acknowledgement_start = time.perf_counter()
    dist.all_reduce(acknowledgement, op=dist.ReduceOp.MIN)
    timed_synchronize()
    acknowledgement_seconds = time.perf_counter() - acknowledgement_start
    expected = torch.tensor([step, group_index], device=device, dtype=torch.int64)
    if not torch.equal(acknowledgement, expected):
        raise RuntimeError(f"trainer acknowledgement mismatch: {acknowledgement.tolist()} != {expected.tolist()}")

    return {
        **sample_identity(entry),
        "step": step,
        "loss": float(loss.detach().cpu()),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "acknowledgement_seconds": acknowledgement_seconds,
    }


async def run_control(args: argparse.Namespace, tokenizer_dir: Path, prompt_path: Path) -> None:
    checkpoint_dir = args.output_dir / "checkpoint"
    namespace = build_args(tokenizer_dir, prompt_path, checkpoint_dir)
    source = RolloutDataSourceWithBuffer(namespace)
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=namespace, unused_handler_fn=lambda prompt_group: None)
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = create_model(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    records = []
    for step in range(args.total_batches):
        entry = await reserve_entry(source, buffer)
        entry = await buffer.get()
        records.append(await train_entry(entry, step, model, optimizer, device))
    save_checkpoint(checkpoint_dir, args.total_batches, model, optimizer, source)
    result = {
        "mode": "control",
        "records": records,
        "optimizer_steps": len(records),
        "source_sample_offset": source.sample_offset,
        "source_sample_group_index": source.sample_group_index,
        "source_sample_index": source.sample_index,
        "final_checkpoint": str(checkpoint_dir / "model.pt"),
        "final_checkpoint_sha256": sha256_file(checkpoint_dir / "model.pt"),
    }
    if dist.get_rank() == 0:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


async def run_pre_admission(args: argparse.Namespace, tokenizer_dir: Path, prompt_path: Path) -> None:
    checkpoint_dir = args.output_dir / "checkpoint"
    namespace = build_args(tokenizer_dir, prompt_path, checkpoint_dir)
    source = RolloutDataSourceWithBuffer(namespace)
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=namespace, unused_handler_fn=lambda prompt_group: None)
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = create_model(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    records = []
    for step in range(args.boundary):
        entry = await reserve_entry(source, buffer)
        entry = await buffer.get()
        records.append(await train_entry(entry, step, model, optimizer, device))
    reserved_entry = await reserve_entry(source, buffer)
    save_checkpoint(checkpoint_dir, args.boundary, model, optimizer, source)
    result = {
        "mode": "pre_admission",
        "records": records,
        "reserved_entry": sample_identity(reserved_entry),
        "buffer_length_after_checkpoint": buffer.get_metrics()["rollout/fully_async/queue_size"],
        "optimizer_steps": len(records),
        "source_sample_offset": source.sample_offset,
        "source_sample_group_index": source.sample_group_index,
        "source_sample_index": source.sample_index,
        "checkpoint": str(checkpoint_dir / "model.pt"),
        "checkpoint_sha256": sha256_file(checkpoint_dir / "model.pt"),
    }
    if dist.get_rank() == 0:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


async def run_pre_admission_restore(args: argparse.Namespace, tokenizer_dir: Path, prompt_path: Path) -> None:
    checkpoint_dir = args.output_dir / "checkpoint"
    namespace = build_args(tokenizer_dir, prompt_path, checkpoint_dir)
    namespace.load = str(args.load_checkpoint_dir)
    source = RolloutDataSourceWithBuffer(namespace)
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=namespace, unused_handler_fn=lambda prompt_group: None)
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = create_model(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    load_checkpoint(args.load_checkpoint_dir, args.boundary, model, optimizer, source)
    records = []
    for step in range(args.boundary, args.total_batches):
        entry = await reserve_entry(source, buffer)
        entry = await buffer.get()
        records.append(await train_entry(entry, step, model, optimizer, device))
    save_checkpoint(checkpoint_dir, args.total_batches, model, optimizer, source)
    result = {
        "mode": "pre_admission_restore",
        "records": records,
        "optimizer_steps": args.boundary + len(records),
        "source_sample_offset": source.sample_offset,
        "source_sample_group_index": source.sample_group_index,
        "source_sample_index": source.sample_index,
        "final_checkpoint": str(checkpoint_dir / "model.pt"),
        "final_checkpoint_sha256": sha256_file(checkpoint_dir / "model.pt"),
    }
    if dist.get_rank() == 0:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


async def run_post_admission(args: argparse.Namespace, tokenizer_dir: Path, prompt_path: Path) -> None:
    checkpoint_dir = args.output_dir / "checkpoint"
    namespace = build_args(tokenizer_dir, prompt_path, checkpoint_dir)
    source = RolloutDataSourceWithBuffer(namespace)
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=namespace, unused_handler_fn=lambda prompt_group: None)
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = create_model(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    records = []
    for step in range(args.boundary + 1):
        entry = await reserve_entry(source, buffer)
        entry = await buffer.get()
        records.append(await train_entry(entry, step, model, optimizer, device))
    save_checkpoint(checkpoint_dir, args.boundary + 1, model, optimizer, source)
    result = {
        "mode": "post_admission",
        "records": records,
        "optimizer_steps": len(records),
        "source_sample_offset": source.sample_offset,
        "source_sample_group_index": source.sample_group_index,
        "source_sample_index": source.sample_index,
        "checkpoint": str(checkpoint_dir / "model.pt"),
        "checkpoint_sha256": sha256_file(checkpoint_dir / "model.pt"),
    }
    if dist.get_rank() == 0:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


async def run_post_admission_restore(args: argparse.Namespace, tokenizer_dir: Path, prompt_path: Path) -> None:
    checkpoint_dir = args.output_dir / "checkpoint"
    namespace = build_args(tokenizer_dir, prompt_path, checkpoint_dir)
    namespace.load = str(args.load_checkpoint_dir)
    source = RolloutDataSourceWithBuffer(namespace)
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=namespace, unused_handler_fn=lambda prompt_group: None)
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = create_model(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    load_checkpoint(args.load_checkpoint_dir, args.boundary + 1, model, optimizer, source)
    records = []
    for step in range(args.boundary + 1, args.total_batches):
        entry = await reserve_entry(source, buffer)
        entry = await buffer.get()
        records.append(await train_entry(entry, step, model, optimizer, device))
    save_checkpoint(checkpoint_dir, args.total_batches, model, optimizer, source)
    result = {
        "mode": "post_admission_restore",
        "records": records,
        "optimizer_steps": args.boundary + 1 + len(records),
        "source_sample_offset": source.sample_offset,
        "source_sample_group_index": source.sample_group_index,
        "source_sample_index": source.sample_index,
        "final_checkpoint": str(checkpoint_dir / "model.pt"),
        "final_checkpoint_sha256": sha256_file(checkpoint_dir / "model.pt"),
    }
    if dist.get_rank() == 0:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


async def run_duplicate_late(args: argparse.Namespace, tokenizer_dir: Path, prompt_path: Path) -> None:
    checkpoint_dir = args.output_dir / "checkpoint"
    namespace = build_args(tokenizer_dir, prompt_path, checkpoint_dir)
    namespace.load = str(args.load_checkpoint_dir)
    source = RolloutDataSourceWithBuffer(namespace)
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=namespace, unused_handler_fn=lambda prompt_group: None)
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model = create_model(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    load_checkpoint(args.load_checkpoint_dir, args.boundary + 1, model, optimizer, source)
    records = []
    duplicate_prompt_group = []
    duplicate_completed_group = []
    for sample_index in range(2):
        prompt_sample = Sample(
            group_index=args.boundary,
            index=args.boundary * 2 + sample_index,
            prompt="a b c d",
            metadata={"prompt_index": args.boundary, "late_completion": True},
        )
        duplicate_prompt_group.append(prompt_sample)
        duplicate_completed_group.extend(synthetic_completion([prompt_sample]))
    duplicate_entry = DataBufferInput(
        prompt_group=duplicate_prompt_group,
        group=duplicate_completed_group,
    )
    await buffer.put(duplicate_entry)
    entry = await buffer.get()
    records.append(await train_entry(entry, args.boundary + 1, model, optimizer, device))
    for step in range(args.boundary + 2, args.total_batches):
        entry = await reserve_entry(source, buffer)
        entry = await buffer.get()
        records.append(await train_entry(entry, step, model, optimizer, device))
    save_checkpoint(checkpoint_dir, args.total_batches, model, optimizer, source)
    result = {
        "mode": "duplicate_late",
        "records": records,
        "optimizer_steps": args.boundary + 1 + len(records),
        "source_sample_offset": source.sample_offset,
        "source_sample_group_index": source.sample_group_index,
        "source_sample_index": source.sample_index,
        "final_checkpoint": str(checkpoint_dir / "model.pt"),
        "final_checkpoint_sha256": sha256_file(checkpoint_dir / "model.pt"),
    }
    if dist.get_rank() == 0:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def compare_checkpoints(args: argparse.Namespace) -> None:
    control_state = torch.load(args.control_checkpoint, weights_only=True)
    case_state = torch.load(args.case_checkpoint, weights_only=True)
    if control_state.keys() != case_state.keys():
        raise RuntimeError("checkpoint parameter sets differ")
    differences = []
    for name in control_state:
        difference = (control_state[name] - case_state[name]).abs()
        differences.append(difference.max().item())
    maximum_difference = max(differences)
    mean_difference = sum(differences) / len(differences)
    result = {
        "control_checkpoint": str(args.control_checkpoint),
        "case_checkpoint": str(args.case_checkpoint),
        "maximum_absolute_difference": maximum_difference,
        "mean_absolute_difference": mean_difference,
        "exact_equal": all(torch.equal(control_state[name], case_state[name]) for name in control_state),
    }
    args.comparison_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    if args.mode == "compare":
        compare_checkpoints(args)
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    tokenizer_dir, prompt_path = create_local_assets(args.output_dir)
    if args.mode == "control":
        await run_control(args, tokenizer_dir, prompt_path)
    elif args.mode == "pre_admission":
        await run_pre_admission(args, tokenizer_dir, prompt_path)
    elif args.mode == "pre_admission_restore":
        await run_pre_admission_restore(args, tokenizer_dir, prompt_path)
    elif args.mode == "post_admission":
        await run_post_admission(args, tokenizer_dir, prompt_path)
    elif args.mode == "post_admission_restore":
        await run_post_admission_restore(args, tokenizer_dir, prompt_path)
    elif args.mode == "duplicate_late":
        await run_duplicate_late(args, tokenizer_dir, prompt_path)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
