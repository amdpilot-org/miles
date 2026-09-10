"""Reproducible two-GPU reduced-model training benchmark for Miles backends.

The benchmark uses local random initialization and synthetic tokens.  It is a
backend control, not a rollout or end-to-end framework comparison.

Run one backend at a time with torchrun, for example::

    PYTHONPATH=. torchrun --nproc-per-node=2 --master-port=29741 \
        tools/benchmark_reduced_training.py --backend fsdp \
        --output /tmp/fsdp.json

Each invocation runs four batch/sequence cases.  With the defaults, each backend
performs 20 warmup and 100 measured optimizer steps in total.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
import aiter
import aiter.jit.module_aiter_core
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as MegatronDDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
import transformer_engine
from transformers import GPT2Config, GPT2LMHeadModel

from miles.backends.fsdp_utils.actor import apply_fsdp2
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.megatron_utils import model as megatron_model_module
from miles.backends.megatron_utils.model import get_optimizer_param_scheduler
from miles.backends.megatron_utils.model import train_one_step as miles_megatron_train_one_step
from miles.backends.megatron_utils.ft.indep_dp import create_indep_dp_group
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.utils.distributed_utils import get_gloo_group, init_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo


VOCAB_SIZE = 1024
HIDDEN_SIZE = 256
LAYERS = 4
ATTENTION_HEADS = 4
FFN_SIZE = 1024
MAX_SEQUENCE_LENGTH = 512
MODEL_PARAMETER_COUNT = 3_814_912
CASES = ((1, 128), (2, 128), (1, 512), (2, 512))
SEED = 1234
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01


@dataclass(frozen=True)
class BenchmarkCase:
    micro_batch_size: int
    sequence_length: int

    @property
    def name(self) -> str:
        return f"mb{self.micro_batch_size}_seq{self.sequence_length}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fsdp", "megatron"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=25)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--timeout-minutes", type=float, default=2.0)
    return parser.parse_args()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def module_path(module: Any) -> Path:
    if getattr(module, "__file__", None):
        return Path(module.__file__).resolve()
    return Path(next(iter(module.__path__))).resolve()


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def replica_tensor(tensor: torch.Tensor) -> torch.Tensor:
    local = local_tensor(tensor).detach().contiguous()
    if not hasattr(tensor, "to_local"):
        return local
    placement = tensor.placements[0]
    shard_dimension = getattr(placement, "dim", 0)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=shard_dimension).contiguous()


def tensor_digest(tensor: torch.Tensor) -> str:
    local = replica_tensor(tensor).cpu()
    element_size = local.element_size()
    if element_size == 1:
        data = local.view(torch.uint8).numpy().tobytes()
    elif element_size == 2:
        data = local.view(torch.uint16).numpy().tobytes()
    elif element_size == 4:
        data = local.view(torch.uint32).numpy().tobytes()
    elif element_size == 8:
        data = local.view(torch.uint64).numpy().tobytes()
    else:
        raise TypeError(f"Unsupported tensor element size for digest: {element_size}")
    return hashlib.sha256(data).hexdigest()


def module_digest(model: torch.nn.Module, *, use_gradients: bool = False) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        tensor = getattr(parameter, "main_grad", None) if use_gradients else parameter
        if use_gradients and tensor is None:
            tensor = parameter.grad
        if tensor is None:
            value = "none"
        else:
            value = tensor_digest(tensor)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def gathered_digests_match(digest: str) -> bool:
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, digest, group=get_gloo_group())
    return all(value == digest for value in gathered)


def reduce_float(value: float, operation: dist.ReduceOp) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=operation)
    return tensor.item()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def initialize_distributed(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != args.expected_world_size:
        raise RuntimeError(
            f"Expected {args.expected_world_size} visible GPUs, found {torch.cuda.device_count()}"
        )
    dist.init_process_group("nccl", timeout=timedelta(minutes=args.timeout_minutes))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_gloo_group()


def initialize_fsdp_parallel_state() -> None:
    fsdp_args = SimpleNamespace(dp_replicate_size=1, fp16=False)
    set_parallel_state(create_fsdp_parallel_state(fsdp_args))


def initialize_megatron_parallel_state(args: argparse.Namespace, case: BenchmarkCase) -> SimpleNamespace:
    megatron_args = make_megatron_args(args, case)
    set_args(megatron_args)
    mpu.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)
    indep_dp = create_indep_dp_group(
        store_addr=None,
        indep_dp_info=IndepDPInfo.create_trivial(),
        megatron_rank=dist.get_rank(),
        megatron_world_size=dist.get_world_size(),
    )
    set_parallel_state(create_megatron_parallel_state(indep_dp=indep_dp))
    return megatron_args


def make_megatron_args(args: argparse.Namespace, case: BenchmarkCase) -> SimpleNamespace:
    return SimpleNamespace(
        allgather_cp=False,
        calculate_per_token_loss=False,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        cp_comm_type=None,
        custom_megatron_before_train_step_hook_path=None,
        data_pad_size_multiplier=128,
        debug_disable_optimizer=False,
        decoder_seq_length=case.sequence_length,
        dumper_dir="/tmp/miles-reduced-benchmark-dumper",
        dumper_enable=False,
        dumper_fwd_bwd=[],
        enable_mtp_training=False,
        enable_witness=False,
        end_weight_decay=WEIGHT_DECAY,
        global_batch_size=args.global_batch_size,
        log_probs_chunk_size=-1,
        loss_type="sft_loss",
        lr=LEARNING_RATE,
        lr_decay_iters=1,
        lr_decay_style="constant",
        lr_wsd_decay_iters=None,
        lr_wsd_decay_style=None,
        lr_warmup_fraction=0.0,
        lr_warmup_init=LEARNING_RATE,
        lr_warmup_iters=0,
        micro_batch_size=case.micro_batch_size,
        min_lr=LEARNING_RATE,
        multi_lora=False,
        n_samples_per_prompt=1,
        optimizer="adam",
        override_opt_param_scheduler=False,
        qkv_format="bshd",
        recompute_loss_function=False,
        rollout_max_response_len=case.sequence_length,
        rollout_batch_size=args.global_batch_size,
        rollout_temperature=1.0,
        num_rollout=1,
        save_local_weight_checksum=False,
        seq_length=case.sequence_length,
        start_weight_decay=WEIGHT_DECAY,
        true_on_policy_mode=False,
        train_iters=1,
        use_checkpoint_opt_param_scheduler=False,
        use_dynamic_global_batch_size=False,
        weight_decay_incr_style="constant",
    )


def build_fsdp_model() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(SEED)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_SEQUENCE_LENGTH,
        n_embd=HIDDEN_SIZE,
        n_layer=LAYERS,
        n_head=ATTENTION_HEADS,
        n_inner=FFN_SIZE,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = GPT2LMHeadModel(config).cuda().to(torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == MODEL_PARAMETER_COUNT, parameter_count
    model = apply_fsdp2(
        model,
        mesh=get_parallel_state().get_mesh("fsdp"),
        args=SimpleNamespace(fp16=False),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    return model, optimizer


def build_megatron_model(
    args: argparse.Namespace, case: BenchmarkCase
) -> tuple[list[MegatronDDP], Any, Any, SimpleNamespace]:
    megatron_args = make_megatron_args(args, case)
    set_args(megatron_args)
    config = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=ATTENTION_HEADS,
        ffn_hidden_size=FFN_SIZE,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="local",
        calculate_per_token_loss=False,
        variable_seq_lengths=True,
    )
    model = GPTModel(
        config,
        get_gpt_layer_local_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pre_process=True,
        post_process=True,
        parallel_output=True,
    ).cuda()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == MODEL_PARAMETER_COUNT, parameter_count
    original_forward = model.forward

    def forward_with_benchmark_compat(*forward_args: Any, fp32_output: bool = False, **forward_kwargs: Any):
        input_ids = forward_kwargs.get("input_ids")
        if forward_kwargs.get("position_ids") is None and input_ids is not None:
            sequence_length = input_ids.shape[-1]
            forward_kwargs["position_ids"] = torch.arange(
                sequence_length, device=input_ids.device, dtype=torch.long
            ).expand(input_ids.shape[0], sequence_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return original_forward(*forward_args, **forward_kwargs)

    model.forward = forward_with_benchmark_compat
    model.train()
    ddp_model = MegatronDDP(
        config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
        ),
        model,
    )
    ddp_model.broadcast_params()
    optimizer_config = OptimizerConfig(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        fp16=False,
        use_distributed_optimizer=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        clip_grad=1e9,
    )
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[ddp_model],
        use_gloo_process_groups=False,
    )
    scheduler = get_optimizer_param_scheduler(megatron_args, optimizer)
    return [ddp_model], optimizer, scheduler, megatron_args


def make_rollout_data(
    *,
    rank: int,
    step: int,
    case: BenchmarkCase,
    local_batch_size: int,
) -> tuple[dict[str, Any], float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 100_003 * rank + step)
    tokens = torch.randint(
        1,
        VOCAB_SIZE,
        (local_batch_size, case.sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    loss_masks = torch.ones(
        local_batch_size,
        case.sequence_length - 1,
        dtype=torch.float32,
    )
    wait_start = time.perf_counter()
    device = torch.cuda.current_device()
    tokens_gpu = tokens.to(device=device, non_blocking=True)
    loss_masks_gpu = loss_masks.to(device=device, non_blocking=True)
    torch.cuda.synchronize()
    input_wait_seconds = time.perf_counter() - wait_start
    rollout_data = {
        "tokens": [row for row in tokens_gpu],
        "loss_masks": [row for row in loss_masks_gpu],
        "total_lengths": [case.sequence_length] * local_batch_size,
        "response_lengths": [case.sequence_length - 1] * local_batch_size,
        "max_seq_lens": [case.sequence_length] * local_batch_size,
    }
    return rollout_data, input_wait_seconds


def fsdp_training_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    case: BenchmarkCase,
    rank: int,
    step: int,
    local_batch_size: int,
    collect_gradient_digest: bool,
) -> dict[str, Any]:
    rollout_data, input_wait_seconds = make_rollout_data(
        rank=rank,
        step=step,
        case=case,
        local_batch_size=local_batch_size,
    )
    iterator = DataIterator(rollout_data, micro_batch_size=case.micro_batch_size)
    loss_args = SimpleNamespace(
        allgather_cp=False,
        calculate_per_token_loss=False,
        global_batch_size=args.global_batch_size,
        log_probs_chunk_size=-1,
        loss_type="sft_loss",
        qkv_format="bshd",
        recompute_loss_function=False,
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        use_dynamic_global_batch_size=False,
    )
    optimizer.zero_grad(set_to_none=True)
    forward_ms = 0.0
    backward_ms = 0.0
    loss_sum = 0.0
    microbatches = local_batch_size // case.micro_batch_size
    for _ in range(microbatches):
        batch = get_batch(
            iterator,
            [
                "tokens",
                "loss_masks",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
            "bshd",
            get_position_ids=True,
        )
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=batch["tokens"], position_ids=batch["position_ids"]).logits
        forward_end.record()
        forward_end.synchronize()
        forward_ms += forward_start.elapsed_time(forward_end)
        loss, _, log_dict = loss_function(
            args=loss_args,
            batch=batch,
            num_microbatches=microbatches,
            logits=logits,
            apply_megatron_loss_scaling=False,
        )
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        backward_start.record()
        loss.backward()
        backward_end.record()
        backward_end.synchronize()
        backward_ms += backward_start.elapsed_time(backward_end)
        loss_sum += float(log_dict["values"][1].detach())
    gradient_digest = module_digest(model, use_gradients=True)
    gradient_digest_match = gathered_digests_match(gradient_digest) if collect_gradient_digest else None
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    optimizer_end.synchronize()
    optimizer_ms = optimizer_start.elapsed_time(optimizer_end)
    return {
        "input_wait_seconds": input_wait_seconds,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "loss": loss_sum,
        "grad_norm": float(grad_norm),
        "gradient_digest_match": gradient_digest_match,
    }


def install_megatron_timing_hooks(
    *,
    model: MegatronDDP,
    optimizer: Any,
) -> tuple[Callable[[], None], Callable[[], dict[str, Any]], Callable[[], None]]:
    original_get_forward_backward_func = megatron_model_module.get_forward_backward_func
    original_forward_backward = original_get_forward_backward_func()
    original_optimizer_step = optimizer.step
    collect_gradient_digest = False
    last_phase: dict[str, Any] = {"forward_ms": 0.0, "backward_ms": 0.0, "total_ms": 0.0}
    last_optimizer_ms = 0.0
    last_gradient_digest_match = None
    phase_forward_ms = 0.0

    def timed_forward_backward(*forward_args: Any, **forward_kwargs: Any):
        nonlocal phase_forward_ms
        phase_forward_ms = 0.0
        forward_step_func = forward_kwargs.get("forward_step_func")
        if forward_step_func is None and forward_args:
            forward_step_func = forward_args[0]

        def timed_forward_step(*step_args: Any, **step_kwargs: Any):
            nonlocal phase_forward_ms
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            step_start.record()
            output = forward_step_func(*step_args, **step_kwargs)
            step_end.record()
            step_end.synchronize()
            phase_forward_ms += step_start.elapsed_time(step_end)
            return output

        if "forward_step_func" in forward_kwargs:
            forward_kwargs["forward_step_func"] = timed_forward_step
        elif forward_args:
            forward_args = (timed_forward_step, *forward_args[1:])
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        output = original_forward_backward(*forward_args, **forward_kwargs)
        model.finish_grad_sync()
        total_end.record()
        total_end.synchronize()
        total_ms = total_start.elapsed_time(total_end)
        last_phase.clear()
        last_phase.update(
            {
                "forward_ms": phase_forward_ms,
                "total_ms": total_ms,
                "backward_ms": max(0.0, total_ms - phase_forward_ms),
            }
        )
        return output

    megatron_model_module.get_forward_backward_func = lambda: timed_forward_backward

    def timed_optimizer_step(*step_args: Any, **step_kwargs: Any):
        nonlocal collect_gradient_digest, last_optimizer_ms, last_gradient_digest_match
        gradient_digest_match = None
        if collect_gradient_digest:
            gradient_digest_match = gathered_digests_match(module_digest(model, use_gradients=True))
            collect_gradient_digest = False
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        output = original_optimizer_step(*step_args, **step_kwargs)
        step_end.record()
        step_end.synchronize()
        last_optimizer_ms = step_start.elapsed_time(step_end)
        last_gradient_digest_match = gradient_digest_match
        return output

    optimizer.step = timed_optimizer_step

    def request_gradient_digest() -> None:
        nonlocal collect_gradient_digest
        collect_gradient_digest = True

    def get_last_phase() -> dict[str, Any]:
        return {
            **last_phase,
            "optimizer_ms": last_optimizer_ms,
            "gradient_digest_match": last_gradient_digest_match,
        }

    def restore_forward_backward_hook() -> None:
        megatron_model_module.get_forward_backward_func = original_get_forward_backward_func

    return request_gradient_digest, get_last_phase, restore_forward_backward_hook


def megatron_training_step(
    *,
    model: list[MegatronDDP],
    optimizer: Any,
    scheduler: Any,
    megatron_args: SimpleNamespace,
    args: argparse.Namespace,
    case: BenchmarkCase,
    rank: int,
    step: int,
    local_batch_size: int,
    request_gradient_digest: Callable[[], None],
    get_last_phase: Callable[[], dict[str, Any]],
    collect_gradient_digest: bool,
) -> dict[str, Any]:
    rollout_data, input_wait_seconds = make_rollout_data(
        rank=rank,
        step=step,
        case=case,
        local_batch_size=local_batch_size,
    )
    iterator = DataIterator(rollout_data, micro_batch_size=case.micro_batch_size)
    iterator.reset()
    if collect_gradient_digest:
        request_gradient_digest()
    loss_reduced, grad_norm, outcome = miles_megatron_train_one_step(
        args=megatron_args,
        rollout_id=step,
        step_id=0,
        data_iterator=[iterator],
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        num_microbatches=local_batch_size // case.micro_batch_size,
        num_rollouts=args.global_batch_size,
        witness_info=None,
        attempt=0,
    )
    phase = get_last_phase()
    return {
        "input_wait_seconds": input_wait_seconds,
        "forward_ms": phase["forward_ms"],
        "backward_ms": phase["backward_ms"],
        "optimizer_ms": phase["optimizer_ms"],
        "loss": float(loss_reduced.get("loss", 0.0)),
        "grad_norm": float(grad_norm),
        "gradient_digest_match": phase["gradient_digest_match"],
        "outcome": outcome.name,
    }


def summarize_steps(
    *,
    steps: list[dict[str, Any]],
    tokens_per_optimizer_step: int,
) -> dict[str, Any]:
    training_seconds = [
        (step["forward_ms"] + step["backward_ms"] + step["optimizer_ms"]) / 1000 for step in steps
    ]
    end_to_end_seconds = [step["input_wait_seconds"] + value for step, value in zip(steps, training_seconds)]
    training_tokens_per_second = [tokens_per_optimizer_step / value for value in training_seconds]
    end_to_end_tokens_per_second = [tokens_per_optimizer_step / value for value in end_to_end_seconds]
    return {
        "training_seconds_mean": statistics.mean(training_seconds),
        "training_seconds_p50": percentile(training_seconds, 0.5),
        "training_seconds_p95": percentile(training_seconds, 0.95),
        "end_to_end_seconds_mean": statistics.mean(end_to_end_seconds),
        "training_tokens_per_second_mean": statistics.mean(training_tokens_per_second),
        "training_tokens_per_second_p50": percentile(training_tokens_per_second, 0.5),
        "training_tokens_per_second_p95": percentile(training_tokens_per_second, 0.95),
        "end_to_end_tokens_per_second_mean": statistics.mean(end_to_end_tokens_per_second),
        "input_wait_seconds_mean": statistics.mean(step["input_wait_seconds"] for step in steps),
        "forward_ms_mean": statistics.mean(step["forward_ms"] for step in steps),
        "backward_ms_mean": statistics.mean(step["backward_ms"] for step in steps),
        "optimizer_ms_mean": statistics.mean(step["optimizer_ms"] for step in steps),
        "loss_mean": statistics.mean(step["loss"] for step in steps),
        "grad_norm_mean": statistics.mean(step["grad_norm"] for step in steps),
    }


def run_case(
    *,
    args: argparse.Namespace,
    case: BenchmarkCase,
) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = args.global_batch_size // world_size
    if args.global_batch_size % world_size != 0 or local_batch_size % case.micro_batch_size != 0:
        raise ValueError("Global batch must divide by world size and micro-batch size")
    torch.cuda.reset_peak_memory_stats()
    if args.backend == "fsdp":
        model, optimizer = build_fsdp_model()
        request_gradient_digest: Callable[[], None] = lambda: None
        scheduler = None
        megatron_args = None
        model_for_digest = model
        restore_hook: Callable[[], None] = lambda: None
    else:
        model, optimizer, scheduler, megatron_args = build_megatron_model(args, case)
        request_gradient_digest, get_last_phase, restore_hook = install_megatron_timing_hooks(
            model=model[0],
            optimizer=optimizer,
        )
        model_for_digest = model[0]
    initial_weight_digest = module_digest(model_for_digest)
    initial_weight_digest_match = gathered_digests_match(initial_weight_digest)
    measured_steps: list[dict[str, Any]] = []
    for step in range(args.warmup_steps + args.measured_steps):
        measured = step >= args.warmup_steps
        if args.backend == "fsdp":
            result = fsdp_training_step(
                model=model,
                optimizer=optimizer,
                args=args,
                case=case,
                rank=rank,
                step=step,
                local_batch_size=local_batch_size,
                collect_gradient_digest=measured and step == args.warmup_steps,
            )
        else:
            result = megatron_training_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                megatron_args=megatron_args,
                args=args,
                case=case,
                rank=rank,
                step=step,
                local_batch_size=local_batch_size,
                request_gradient_digest=request_gradient_digest,
                get_last_phase=get_last_phase,
                collect_gradient_digest=measured and step == args.warmup_steps,
            )
        if measured:
            result["loss"] = reduce_float(result["loss"], dist.ReduceOp.SUM) / world_size
            result["grad_norm"] = reduce_float(result["grad_norm"], dist.ReduceOp.MAX)
            result["input_wait_seconds"] = reduce_float(
                result["input_wait_seconds"], dist.ReduceOp.MAX
            )
            for key in ("forward_ms", "backward_ms", "optimizer_ms"):
                result[key] = reduce_float(result[key], dist.ReduceOp.MAX)
            measured_steps.append(result)
    final_weight_digest = module_digest(model_for_digest)
    final_weight_digest_match = gathered_digests_match(final_weight_digest)
    restore_hook()
    del model, optimizer, model_for_digest, scheduler
    torch.cuda.empty_cache()
    tokens_per_optimizer_step = args.global_batch_size * case.sequence_length
    summary = summarize_steps(steps=measured_steps, tokens_per_optimizer_step=tokens_per_optimizer_step)
    peak_memory_mb = reduce_float(
        torch.cuda.max_memory_allocated() / (1024**2), dist.ReduceOp.MAX
    )
    return {
        "backend": args.backend,
        "case": case.name,
        "micro_batch_size": case.micro_batch_size,
        "sequence_length": case.sequence_length,
        "global_batch_size": args.global_batch_size,
        "local_batch_size": local_batch_size,
        "microbatches_per_optimizer_step": local_batch_size // case.micro_batch_size,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "summary": summary,
        "steps": measured_steps,
        "peak_memory_mb": peak_memory_mb,
        "numerical_checks": {
            "loss_finite": all(math.isfinite(step["loss"]) for step in measured_steps),
            "grad_norm_finite": all(math.isfinite(step["grad_norm"]) for step in measured_steps),
            "initial_weight_digest_match": initial_weight_digest_match,
            "final_weight_digest_match": final_weight_digest_match,
            "weight_changed": initial_weight_digest != final_weight_digest,
            "gradient_digest_match": measured_steps[0].get("gradient_digest_match"),
        },
    }


def runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    miles_path = Path(__import__("miles").__file__).resolve().parents[1]
    import megatron.core

    megatron_path = module_path(megatron.core).parents[1]
    return {
        "benchmark_commit": git_commit(miles_path),
        "megatron_commit": git_commit(megatron_path),
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "transformers_version": package_version("transformers"),
        "megatron_core_version": package_version("megatron-core"),
        "flash_attn_version": package_version("flash-attn"),
        "aiter_version": package_version("aiter"),
        "transformer_engine_version": package_version("transformer-engine"),
        "verl_version": package_version("verl"),
        "world_size": dist.get_world_size(),
        "devices": [
            {
                "rank": rank,
                "name": torch.cuda.get_device_name(rank),
                "capability": torch.cuda.get_device_capability(rank),
            }
            for rank in range(torch.cuda.device_count())
        ],
        "source_paths": {
            "miles": str(module_path(__import__("miles"))),
            "torch": str(module_path(torch)),
            "megatron_core": str(module_path(megatron.core)),
            "transformers": str(module_path(__import__("transformers"))),
            "aiter": str(module_path(aiter)),
            "aiter_native": str(Path(aiter.jit.module_aiter_core.__file__).resolve()),
            "transformer_engine": str(module_path(transformer_engine)),
        },
        "rendezvous": {
            "backend": "nccl",
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
            "timeout_minutes": args.timeout_minutes,
        },
        "model": {
            "initialization": "synthetic-local-random",
            "downloads_bytes": 0,
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "layers": LAYERS,
            "attention_heads": ATTENTION_HEADS,
            "ffn_size": FFN_SIZE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "precision": "bf16 parameters, fp32 gradient reduction/master state",
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": WEIGHT_DECAY,
        },
        "backend_status": {
            "fsdp": "supported" if args.backend == "fsdp" else "not-run-in-this-invocation",
            "megatron": "supported" if args.backend == "megatron" else "not-run-in-this-invocation",
            "verl": "explicit-gap-not-installed-and-not-installed-by-this-benchmark",
        },
    }


def main() -> None:
    args = parse_args()
    initialize_distributed(args)
    args.data_pad_size_multiplier = 128
    if args.backend == "fsdp":
        initialize_fsdp_parallel_state()
    else:
        initialize_megatron_parallel_state(args, BenchmarkCase(*CASES[0]))
    results = []
    for micro_batch_size, sequence_length in CASES:
        case = BenchmarkCase(micro_batch_size=micro_batch_size, sequence_length=sequence_length)
        results.append(run_case(args=args, case=case))
        torch.cuda.synchronize()
        dist.barrier()
    metadata = runtime_metadata(args)
    output = {
        "metadata": metadata,
        "cases": results,
    }
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
