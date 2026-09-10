#!/usr/bin/env python3
"""Two-GPU LoRA checkpoint/resume investigation fixture.

The model is deliberately tiny, but every training step uses DDP, HIP/ROCm GEMMs,
FP32 master parameters, Adam, and a real Miles rollout data source.  No model or
dataset is downloaded.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import ModelParallelConfig
from megatron.core import parallel_state as mcore_parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.utils import GroupedExpertLinearAdapter, SharedOuterGroupedExpertAdapter
from megatron.training.global_vars import set_args
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from miles.backends.megatron_utils.checkpoint import save_checkpoint_with_lora
from miles.backends.megatron_utils.lora_utils import (
    _is_adapter_param_name,
    load_lora_adapter,
    standard_lora_resume_iteration,
)
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.rollout.data_source import RolloutDataSource
from miles.utils.ft_utils.process_group_utils import GroupInfo


HIDDEN = 16
RANK = 8
TOKEN_COUNT = 8
VOCAB = 8
EXPERTS = 2
LR = 0.003
ATOL = 1.5e-6
RTOL = 1.5e-5


class DenseLoRAModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HIDDEN).to(device=device, dtype=torch.bfloat16)
        container = nn.Module()
        container.linear_fc1 = nn.Linear(HIDDEN, HIDDEN).to(device=device, dtype=torch.bfloat16)
        lora = LoRA(
            target_modules=["linear_fc1"],
            dim=RANK,
            alpha=16,
            dropout=0.0,
            lora_dtype=torch.bfloat16,
        )
        lora(container)
        self.adapter = container.linear_fc1

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.embedding(tokens))


class ExpertLoRAModel(nn.Module):
    def __init__(self, device: torch.device, layout: str):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HIDDEN).to(device=device, dtype=torch.bfloat16)
        config = ModelParallelConfig(params_dtype=torch.bfloat16)
        if layout == "expert_per_expert":
            self.adapter = GroupedExpertLinearAdapter(
                HIDDEN,
                HIDDEN,
                RANK,
                num_local_experts=EXPERTS,
                base_linear_name="mlp.experts.linear_fc1",
                activation="identity",
                model_parallel_config=config,
                params_device=device,
                params_dtype=torch.bfloat16,
            )
        elif layout == "expert_shared_outer":
            self.adapter = SharedOuterGroupedExpertAdapter(
                HIDDEN,
                HIDDEN,
                RANK,
                num_local_experts=EXPERTS,
                base_linear_name="mlp.experts.linear_fc1",
                activation="identity",
                input_is_parallel=False,
                model_parallel_config=config,
                params_device=device,
                params_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"unknown expert layout: {layout}")

    def forward(self, tokens: torch.Tensor, m_splits: list[int]) -> torch.Tensor:
        return self.adapter(self.embedding(tokens), m_splits)


@dataclass
class Snapshot:
    adapter: dict[str, torch.Tensor]
    masters: list[torch.Tensor]
    exp_avg: list[torch.Tensor]
    exp_avg_sq: list[torch.Tensor]
    adam_steps: list[Any]
    lr_history: list[float]
    cursor: dict[str, Any]
    fixed_output: torch.Tensor
    final_loss: float



def install_miles_parallel_state(rank: int, world_size: int) -> None:
    group = dist.group.WORLD if world_size > 1 else None
    dp = GroupInfo(rank=rank, size=world_size, group=group)
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=dp,
            intra_dp_cp=dp,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def build_workspace(root: Path, rank: int) -> tuple[Path, Path]:
    tokenizer_dir = root / "tokenizer"
    prompt_path = root / "prompts.jsonl"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        raw = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2}, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        raw.decoder = decoders.WordPiece(prefix="##")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
        tokenizer.save_pretrained(tokenizer_dir)
        with prompt_path.open("w") as handle:
            for index in range(16):
                prompt = " ".join(("a", "b", "c")[index % 3] for _ in range(index + 1))
                handle.write(json.dumps({"text": prompt, "metadata": {"index": index}}) + "\n")
    dist.barrier()
    return tokenizer_dir, prompt_path


def source_args(
    tokenizer_dir: Path,
    prompt_path: Path,
    save: Path,
    load: Path | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint=str(tokenizer_dir),
        prompt_data=str(prompt_path),
        save=str(save),
        load=None if load is None else str(load),
        rollout_global_dataset=True,
        rollout_shuffle=True,
        rollout_seed=42,
        rollout_max_prompt_len=32,
        input_key="text",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        chat_template_path=None,
        dump_details=None,
        n_samples_per_prompt=1,
    )


def checkpoint_args(tokenizer_dir: Path, save: Path, layout: str) -> SimpleNamespace:
    return SimpleNamespace(
        save=str(save),
        hf_checkpoint=str(tokenizer_dir),
        target_modules=["linear_fc1"] if layout == "dense" else ["mlp.experts.linear_fc1"],
        lora_rank=RANK,
        lora_alpha=16,
        lora_dropout=0.0,
        experts_shared_outer_loras=layout == "expert_shared_outer",
    )


def create_model(layout: str, device: torch.device) -> nn.Module:
    torch.manual_seed(1234)
    if layout == "dense":
        model: nn.Module = DenseLoRAModel(device)
    else:
        model = ExpertLoRAModel(device, layout)
    return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)


def adapter_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if _is_adapter_param_name(name)
    }


def create_training_state(
    layout: str,
    model: nn.Module,
    tokenizer_dir: Path,
    prompt_path: Path,
    save: Path,
    load: Path | None,
    lr_history: list[float] | None = None,
) -> dict[str, Any]:
    parameters = adapter_parameters(model)
    assert parameters, f"{layout} model exposed no adapter parameters"
    masters = [parameter.detach().float().clone() for parameter in parameters.values()]
    optimizer = torch.optim.Adam(
        masters,
        lr=LR,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = LambdaLR(
        optimizer,
        lambda epoch: 0.5 * (1.0 + math.cos(math.pi * min(epoch, 255) / 255.0)),
    )
    source = RolloutDataSource(source_args(tokenizer_dir, prompt_path, save, load))
    return {
        "model": model,
        "parameters": parameters,
        "masters": masters,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "source": source,
        "lr_history": [] if lr_history is None else list(lr_history),
        "timings_ms": defaultdict(float),
    }


def sync_model_from_masters(state: dict[str, Any]) -> None:
    for master, parameter in zip(state["masters"], state["parameters"].values(), strict=True):
        parameter.data.copy_(master.to(parameter.device, dtype=parameter.dtype))


def sync_masters_from_optimizer(state: dict[str, Any]) -> None:
    for master in state["masters"]:
        saved_master = state["optimizer"].state[master].get("master_param")
        if saved_master is not None:
            master.copy_(saved_master.to(device=master.device, dtype=master.dtype))
    sync_model_from_masters(state)


def timed_call(callback: Any, timings: dict[str, float], name: str) -> Any:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callback()
    end.record()
    end.synchronize()
    timings[name] += start.elapsed_time(end)
    return result


def tokens_for_prompt(
    prompt: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.Tensor:
    token_ids = tokenizer.encode(prompt)
    tokens = torch.full((TOKEN_COUNT,), tokenizer.pad_token_id, dtype=torch.long, device=device)
    count = min(len(token_ids), TOKEN_COUNT)
    if count:
         tokens[:count] = torch.tensor(token_ids[:count], dtype=torch.long, device=device)
    return tokens


def train_step(
    state: dict[str, Any],
    layout: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    target: torch.Tensor,
    m_splits: list[int],
) -> float:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    timings = state["timings_ms"]
    sample_group = state["source"].get_samples(1)
    prompt = sample_group[0][0].prompt
    tokens = tokens_for_prompt(prompt, tokenizer, device)

    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    if layout == "dense":
        output = timed_call(lambda: model(tokens), timings, "forward_ms")
    else:
        output = timed_call(lambda: model(tokens, m_splits), timings, "forward_ms")
    loss = F.mse_loss(output, target)
    timed_call(loss.backward, timings, "backward_ms")
    reduced_loss = loss.detach().clone()
    timed_call(lambda: dist.all_reduce(reduced_loss), timings, "collective_ms")

    def update() -> None:
        for master, parameter in zip(state["masters"], state["parameters"].values(), strict=True):
            master.grad = parameter.grad.detach().float()
        optimizer.step()
        for master in state["masters"]:
            state["optimizer"].state[master]["master_param"] = master
        scheduler.step()
        sync_model_from_masters(state)

    timed_call(update, timings, "update_ms")
    state["lr_history"].append(float(scheduler.get_last_lr()[0]))
    return float(reduced_loss.detach().cpu()) / dist.get_world_size()


def save_state(
    state: dict[str, Any],
    layout: str,
    tokenizer_dir: Path,
    iteration: int,
) -> None:
    model = state["model"]
    save_root = Path(state["source"].args.save)
    save_root.mkdir(parents=True, exist_ok=True)
    args = checkpoint_args(tokenizer_dir, save_root, layout)
    set_args(args)
    wall_start = time.perf_counter()
    timed_call(
        lambda: save_checkpoint_with_lora(iteration, [model], state["optimizer"], state["scheduler"]),
        state["timings_ms"],
        "save_ms",
    )
    state["timings_ms"]["save_wall_ms"] = (time.perf_counter() - wall_start) * 1000.0
    state["source"].save(iteration)
    dist.barrier()


def capture_snapshot(
    state: dict[str, Any],
    layout: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    target: torch.Tensor,
    m_splits: list[int],
) -> Snapshot:
    model = state["model"]
    fixed_tokens = torch.arange(TOKEN_COUNT, device=device) % VOCAB
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        if layout == "dense":
            fixed_output = model(fixed_tokens)
        else:
            fixed_output = model(fixed_tokens, m_splits)
        final_loss = F.mse_loss(fixed_output, target)
    end.record()
    end.synchronize()
    state["timings_ms"]["eval_ms"] += start.elapsed_time(end)

    optimizer = state["optimizer"]
    moments = []
    exp_avg_sq = []
    adam_steps = []
    for master in state["masters"]:
        moment_state = optimizer.state[master]
        moments.append(moment_state["exp_avg"].detach().cpu().clone())
        exp_avg_sq.append(moment_state["exp_avg_sq"].detach().cpu().clone())
        adam_steps.append(moment_state["step"])

    source = state["source"]
    return Snapshot(
        adapter={
            name: parameter.detach().cpu().clone()
            for name, parameter in state["parameters"].items()
        },
        masters=[master.detach().cpu().clone() for master in state["masters"]],
        exp_avg=moments,
        exp_avg_sq=exp_avg_sq,
        adam_steps=adam_steps,
        lr_history=list(state["lr_history"]),
        cursor={
            "sample_offset": source.sample_offset,
            "epoch_id": source.epoch_id,
            "sample_group_index": source.sample_group_index,
            "sample_index": source.sample_index,
        },
        fixed_output=fixed_output.detach().cpu().clone(),
        final_loss=float(final_loss.detach().cpu()),
    )


def tensor_check(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left.float() - right.float()).abs().max().item()
    return {
        "passed": bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL)),
        "max_abs_diff": difference,
        "atol": ATOL,
        "rtol": RTOL,
    }


def compare_snapshots(
    left: Snapshot,
    right: Snapshot,
    *,
    baseline_lr_start: int = 0,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "lora_weights": tensor_check(
            torch.cat([value.flatten() for value in left.adapter.values()]),
            torch.cat([value.flatten() for value in right.adapter.values()]),
        ),
        "fp32_masters": tensor_check(
            torch.cat([value.flatten() for value in left.masters]),
            torch.cat([value.flatten() for value in right.masters]),
        ),
        "adam_first_moment": tensor_check(
            torch.cat([value.flatten() for value in left.exp_avg]),
            torch.cat([value.flatten() for value in right.exp_avg]),
        ),
        "adam_second_moment": tensor_check(
            torch.cat([value.flatten() for value in left.exp_avg_sq]),
            torch.cat([value.flatten() for value in right.exp_avg_sq]),
        ),
        "adam_steps": {
            "passed": left.adam_steps == right.adam_steps,
            "left": [int(value) for value in left.adam_steps],
            "right": [int(value) for value in right.adam_steps],
        },
        "lr_progression": {
            "passed": left.lr_history[baseline_lr_start:] == right.lr_history,
            "left": left.lr_history[baseline_lr_start:],
            "right": right.lr_history,
            "baseline_lr_start": baseline_lr_start,
        },
        "dataset_cursor": {
            "passed": left.cursor == right.cursor,
            "left": left.cursor,
            "right": right.cursor,
        },
        "fixed_token_output": tensor_check(left.fixed_output, right.fixed_output),
    }
    checks["all_passed"] = all(value["passed"] for value in checks.values() if isinstance(value, dict))
    return checks


def run_segment(
    layout: str,
    run_root: Path,
    tokenizer_dir: Path,
    prompt_path: Path,
    tokenizer: AutoTokenizer,
    device: torch.device,
    target: torch.Tensor,
    m_splits: list[int],
    start: int,
    end: int,
    resume_from: Path | None,
    *,
    warm_start: bool = False,
    save_points: tuple[int, ...] = (),
    lr_history: list[float] | None = None,
) -> tuple[dict[str, Any], Snapshot]:
    model = create_model(layout, device)
    source_load = resume_from.parent.parent if resume_from is not None and not warm_start else None
    state = create_training_state(
        layout,
        model,
        tokenizer_dir,
        prompt_path,
        run_root,
        source_load,
        lr_history,
    )
    if resume_from is not None:
        load_timings = defaultdict(float)
        wall_start = time.perf_counter()
        if warm_start:
            loaded, iteration = timed_call(
                lambda: load_lora_adapter([model], resume_from),
                load_timings,
                "load",
            )
            assert loaded is True and iteration is None
        else:
            loaded, iteration = timed_call(
                lambda: load_lora_adapter(
                    [model],
                    resume_from,
                    optimizer=state["optimizer"],
                    opt_param_scheduler=state["scheduler"],
                ),
                load_timings,
                "load",
            )
            assert loaded is True and iteration == start
            sync_masters_from_optimizer(state)
            state["source"].load(start)
        state["timings_ms"]["load_gpu_ms"] = load_timings["load"]
        state["timings_ms"]["load_wall_ms"] = (time.perf_counter() - wall_start) * 1000.0

    final_loss = 0.0
    for step in range(start + 1, end + 1):
        final_loss = train_step(state, layout, tokenizer, device, target, m_splits)
        if step in save_points:
            save_state(state, layout, tokenizer_dir, step)

    snapshot = capture_snapshot(state, layout, tokenizer, device, target, m_splits)
    summary = {
        "start": start,
        "end": end,
        "warm_start": warm_start,
        "final_loss": final_loss,
        "timings_ms": dict(state["timings_ms"]),
        "steps": end - start,
    }
    return summary, snapshot


def malformed_shard_rejected(layout: str, adapter_path: Path, device: torch.device) -> bool:
    model = create_model(layout, device)
    native_path = adapter_path / f"adapter_megatron_rank{dist.get_rank()}.pt"
    state = torch.load(native_path, map_location="cpu", weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, native_path)
    try:
        load_lora_adapter([model], adapter_path)
    except RuntimeError:
        return True
    return False


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def local_summary(
    rank: int,
    device: torch.device,
    results: dict[str, Any],
    timings: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "rank": rank,
        "device": torch.cuda.get_device_name(device),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
        "results": results,
        "timings_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["dense", "expert_per_expert", "expert_shared_outer"],
        default=["dense", "expert_per_expert", "expert_shared_outer"],
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"this fixture requires exactly 2 GPUs, got {world_size}")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"this fixture requires 2 visible CUDA/HIP devices, got {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    mcore_parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    install_miles_parallel_state(rank, world_size)

    tokenizer_dir, prompt_path = build_workspace(args.root, rank)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    target = (
        torch.sin(torch.arange(TOKEN_COUNT * HIDDEN, device=device).float() * 0.17)
        .reshape(TOKEN_COUNT, HIDDEN)
        .to(torch.bfloat16)
    )
    m_splits = [TOKEN_COUNT // EXPERTS] * EXPERTS
    checkpoints = tuple(args.checkpoints)
    if any(checkpoint <= 0 or checkpoint >= args.steps for checkpoint in checkpoints):
        raise ValueError("checkpoint iterations must be inside (0, steps)")
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoint iterations are required")

    results: dict[str, Any] = {}
    timings: dict[str, dict[str, float]] = {}
    for layout in args.layouts:
        layout_root = args.root / layout / f"rank_{rank}"
        baseline_root = layout_root / "baseline"
        baseline_summary, baseline_snapshot = run_segment(
            layout,
            baseline_root,
            tokenizer_dir,
            prompt_path,
            tokenizer,
            device,
            target,
            m_splits,
            0,
            args.steps,
            None,
            save_points=checkpoints + (args.steps,),
        )
        timings[f"{layout}.baseline"] = baseline_summary["timings_ms"]
        marker = None
        if rank == 0:
            marker = int((baseline_root / "latest_checkpointed_iteration.txt").read_text())
            assert marker == args.steps

        layout_results: dict[str, Any] = {
            "baseline": baseline_summary,
            "resume_marker": {
                "passed": all(
                    standard_lora_resume_iteration(
                        baseline_root / f"iter_{checkpoint:07d}" / "adapter"
                    )
                    == checkpoint
                    for checkpoint in checkpoints
                ),
                "latest_marker": marker,
            },
        }

        for checkpoint in checkpoints:
            adapter_path = baseline_root / f"iter_{checkpoint:07d}" / "adapter"
            resume_root = layout_root / f"resume_{checkpoint}"
            resume_summary, resume_snapshot = run_segment(
                layout,
                resume_root,
                tokenizer_dir,
                prompt_path,
                tokenizer,
                device,
                target,
                m_splits,
                checkpoint,
                args.steps,
                adapter_path,
                save_points=(args.steps,),
            )
            resume_summary["comparison"] = compare_snapshots(
                baseline_snapshot,
                resume_snapshot,
                baseline_lr_start=checkpoint,
            )
            layout_results[f"resume_{checkpoint}"] = resume_summary
            timings[f"{layout}.resume_{checkpoint}"] = resume_summary["timings_ms"]

        chain_root = layout_root / "chain"
        chain_snapshot = None
        chain_lr_history: list[float] = []
        previous_root = baseline_root
        previous_checkpoint = checkpoints[0]
        for next_checkpoint in checkpoints[1:] + (args.steps,):
            chain_run_root = chain_root / f"segment_{previous_checkpoint}_{next_checkpoint}"
            adapter_path = previous_root / f"iter_{previous_checkpoint:07d}" / "adapter"
            chain_summary, chain_snapshot = run_segment(
                layout,
                chain_run_root,
                tokenizer_dir,
                prompt_path,
                tokenizer,
                device,
                target,
                m_splits,
                previous_checkpoint,
                next_checkpoint,
                adapter_path,
                save_points=(next_checkpoint,),
                lr_history=chain_lr_history,
            )
            chain_lr_history = chain_snapshot.lr_history
            timings[f"{layout}.chain_{previous_checkpoint}_{next_checkpoint}"] = chain_summary["timings_ms"]
            previous_root = chain_run_root
            previous_checkpoint = next_checkpoint
        assert chain_snapshot is not None
        layout_results["repeated_resume_chain"] = {
            "start": checkpoints[0],
            "end": args.steps,
            "comparison": compare_snapshots(
                baseline_snapshot,
                chain_snapshot,
                baseline_lr_start=checkpoints[0],
            ),
        }

        warm_source = layout_root / "warm_start"
        warm_source.mkdir(parents=True, exist_ok=True)
        warm_adapter = warm_source / "warm-adapter"
        if warm_adapter.exists():
            shutil.rmtree(warm_adapter)
        shutil.copytree(baseline_root / f"iter_{checkpoints[1]:07d}" / "adapter", warm_adapter)
        warm_summary, warm_snapshot = run_segment(
            layout,
            layout_root / "warm_start_run",
            tokenizer_dir,
            prompt_path,
            tokenizer,
            device,
            target,
            m_splits,
            0,
            args.steps,
            warm_adapter,
            warm_start=True,
        )
        warm_comparison = compare_snapshots(baseline_snapshot, warm_snapshot)
        warm_summary["comparison"] = warm_comparison
        warm_summary["clearly_separate_from_full_resume"] = not warm_comparison["all_passed"]
        layout_results["warm_start"] = warm_summary
        timings[f"{layout}.warm_start"] = warm_summary["timings_ms"]

        negative_root = layout_root / "negative"
        negative_root.mkdir(parents=True, exist_ok=True)
        negative_adapter = negative_root / f"iter_{checkpoints[0]:07d}" / "adapter"
        if negative_adapter.exists():
            shutil.rmtree(negative_adapter)
        shutil.copytree(baseline_root / f"iter_{checkpoints[0]:07d}" / "adapter", negative_adapter)
        layout_results["negative_missing_tensor_rejected"] = malformed_shard_rejected(
            layout, negative_adapter, device
        )
        results[layout] = layout_results

    dist.barrier()
    summaries = [None] * world_size
    dist.all_gather_object(
        summaries,
        local_summary(rank, device, results, timings),
    )
    if rank == 0:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
        report = {
            "schema": "miles-amd-lora-resume-investigation-v1",
            "source_issue": "https://github.com/radixark/miles/issues/2705",
            "tested_commit": commit,
            "runtime": {
                "python": "/opt/venv/bin/python",
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "megatron_core": package_version("megatron-core"),
                "megatron_bridge": package_version("megatron-bridge"),
                "transformers": package_version("transformers"),
                "peft": package_version("peft"),
                "process_group_timeout_s": 120,
                "world_size": world_size,
                "downloads_bytes": 0,
            },
            "native_paths": {
                "torch": str(torch.__file__),
                "megatron_core": str(mcore_parallel_state.__file__),
                "megatron_bridge": str(Path(__import__("megatron.bridge").__path__[0])),
                "miles": str(Path(__file__).resolve().parents[2]),
            },
            "topology": {
                "gpus": world_size,
                "dp": world_size,
                "tp": 1,
                "pp": 1,
                "ep": 1,
                "etp": 1,
                "device": "AMD Instinct MI350X gfx950",
                "performance_equivalence_claim": None,
            },
            "steps": args.steps,
            "checkpoints": list(checkpoints),
            "layouts": args.layouts,
            "ranks": summaries,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    mcore_parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
