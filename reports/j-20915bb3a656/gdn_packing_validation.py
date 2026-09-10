#!/usr/bin/env python3
"""Validate Megatron GDN and Miles Qwen3.5 packing on one MI350X."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

from miles_plugins.models.qwen3_5 import Attention


@dataclass
class RunResult:
    output: torch.Tensor
    input_grad: torch.Tensor
    param_grads: dict[str, torch.Tensor]
    loss: float
    forward_ms: float
    backward_ms: float


def git_commit(path: str) -> str:
    if not Path(path, ".git").exists():
        return None
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def parameter_norm(model: torch.nn.Module) -> float:
    return float(sum(parameter.float().square().sum().item() for parameter in model.parameters()))


def snapshot_param_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def restore_param_grads(model: torch.nn.Module, grads: dict[str, torch.Tensor]) -> None:
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        if name in grads:
            parameter.grad = grads[name].to(device=parameter.device, dtype=parameter.dtype)


def timed_forward(call: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def timed_backward(loss: torch.Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_single(
    model: torch.nn.Module,
    call: Callable[[], tuple[torch.Tensor, torch.Tensor | None]],
    hidden: torch.Tensor,
    target: torch.Tensor,
) -> RunResult:
    model.zero_grad(set_to_none=True)
    hidden.grad = None
    raw_output, forward_ms = timed_forward(call)
    output, _ = raw_output
    loss = F.mse_loss(output.float(), target.float())
    backward_ms = timed_backward(loss)
    return RunResult(
        output=output.detach().float().clone(),
        input_grad=hidden.grad.detach().float().clone(),
        param_grads=snapshot_param_grads(model),
        loss=float(loss.detach().item()),
        forward_ms=forward_ms,
        backward_ms=backward_ms,
    )


def run_sequence_control(
    model: torch.nn.Module,
    call: Callable[[torch.Tensor, PackedSeqParams], tuple[torch.Tensor, torch.Tensor | None]],
    hidden: torch.Tensor,
    target: torch.Tensor,
    boundaries: list[int],
) -> RunResult:
    model.zero_grad(set_to_none=True)
    outputs = []
    input_grads = []
    forward_ms = 0.0
    backward_ms = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sequence_hidden = hidden[start:end].detach().clone().requires_grad_(True)
        sequence_target = target[start:end]
        sequence_length = end - start
        sequence_cu = torch.tensor([0, sequence_length], device=hidden.device, dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="bshd",
            cu_seqlens_q=sequence_cu,
            cu_seqlens_kv=sequence_cu,
            max_seqlen_q=sequence_length,
            max_seqlen_kv=sequence_length,
        )
        raw_output, sequence_forward_ms = timed_forward(lambda: call(sequence_hidden, packed))
        output, _ = raw_output
        outputs.append(output)
        forward_ms += sequence_forward_ms
        loss = F.mse_loss(output.float(), sequence_target.float())
        backward_ms += timed_backward(loss)
        input_grads.append(sequence_hidden.grad.detach().float().clone())
    output = torch.cat(outputs, dim=0)
    input_grad = torch.cat(input_grads, dim=0)
    loss = F.mse_loss(output.float(), target.float())
    return RunResult(
        output=output.detach().float().clone(),
        input_grad=input_grad,
        param_grads=snapshot_param_grads(model),
        loss=float(loss.detach().item()),
        forward_ms=forward_ms,
        backward_ms=backward_ms,
    )


def compare(left: RunResult, right: RunResult) -> dict[str, float | bool]:
    output_diff = (left.output - right.output).abs().max().item()
    input_diff = (left.input_grad - right.input_grad).abs().max().item()
    output_relative = (
        (left.output - right.output).abs() / (right.output.abs() + 1e-6)
    ).max().item()
    input_relative = (
        (left.input_grad - right.input_grad).abs() / (right.input_grad.abs() + 1e-6)
    ).max().item()
    param_diff = 0.0
    param_relative = 0.0
    for name in left.param_grads.keys() & right.param_grads.keys():
        difference = (left.param_grads[name] - right.param_grads[name]).abs()
        param_diff = max(param_diff, difference.max().item())
        param_relative = max(
            param_relative,
            (difference / (right.param_grads[name].abs() + 1e-6)).max().item(),
        )
    return {
        "output_max_abs_diff": output_diff,
        "input_grad_max_abs_diff": input_diff,
        "param_grad_max_abs_diff": param_diff,
        "output_max_relative_diff": output_relative,
        "input_grad_max_relative_diff": input_relative,
        "param_grad_max_relative_diff": param_relative,
        "output_close": bool(
            torch.allclose(left.output, right.output, rtol=2e-2, atol=2e-2)
        ),
        "input_grad_close": bool(
            torch.allclose(left.input_grad, right.input_grad, rtol=2e-2, atol=2e-2)
        ),
    }


def aggregate(comparisons: list[dict[str, float | bool]]) -> dict[str, float | bool]:
    return {
        key: (
            max(item[key] for item in comparisons)
            if isinstance(comparisons[0][key], float)
            else all(item[key] for item in comparisons)
        )
        for key in comparisons[0]
    }


def build_megatron_gdn() -> tuple[GatedDeltaNet, ProcessGroupCollection]:
    config = TransformerConfig(
        hidden_size=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=4,
        activation_func=F.silu,
        bf16=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    spec = get_experimental_attention_variant_module_spec(config=config).submodules
    process_groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
        tp_cp=parallel_state.get_tensor_and_context_parallel_group(),
    )
    model = GatedDeltaNet(
        config,
        submodules=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=process_groups,
    ).cuda().bfloat16()
    model.train()
    return model, process_groups


def build_miles_attention() -> tuple[Attention, str]:
    config = Qwen3NextConfig(
        hidden_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
    )
    config.dtype = torch.bfloat16
    temporary_directory = tempfile.mkdtemp(prefix="miles-gdn-config-")
    config.save_pretrained(temporary_directory)
    args = Namespace(
        hf_checkpoint=Path(temporary_directory).as_posix(),
        sequence_parallel=False,
        allgather_cp=False,
        linear_attention_backend="fla",
    )
    model = Attention(args, config, layer_number=1).cuda().bfloat16()
    model.train()
    return model, temporary_directory


def guard_probe(model: GatedDeltaNet, hidden: torch.Tensor, packed: PackedSeqParams) -> bool:
    original = model.config.deterministic_mode
    model.config.deterministic_mode = True
    try:
        model(hidden, None, packed_seq_params=packed)
        return False
    except AssertionError:
        return True
    finally:
        model.config.deterministic_mode = original


def main() -> None:
    steps = 64
    sequence_length = 32
    hidden_size = 64
    boundaries = [0, 13, 32]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    rendezvous_path = Path("/tmp") / f"miles-gdn-{uuid.uuid4().hex}-rendezvous"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        world_size=1,
        rank=0,
        timeout=timedelta(minutes=2),
    )
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    megatron_gdn, _ = build_megatron_gdn()
    miles_attention, config_directory = build_miles_attention()

    megatron_optimizer = torch.optim.AdamW(megatron_gdn.parameters(), lr=1e-3)
    miles_optimizer = torch.optim.AdamW(miles_attention.parameters(), lr=1e-3)

    megatron_bshd_hidden = torch.randn(
        sequence_length, 2, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    megatron_bshd_target = torch.randn_like(megatron_bshd_hidden, dtype=torch.bfloat16)
    megatron_thd_hidden = torch.randn(
        sequence_length, 1, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    megatron_thd_target = torch.randn_like(megatron_thd_hidden, dtype=torch.bfloat16)

    miles_bshd_hidden = torch.randn(
        sequence_length, 1, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    miles_bshd_target = torch.randn_like(miles_bshd_hidden, dtype=torch.bfloat16)
    miles_thd_hidden = torch.randn(
        sequence_length, 1, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    miles_thd_target = torch.randn_like(miles_thd_hidden, dtype=torch.bfloat16)

    dense_cu = torch.tensor([0, sequence_length], device=device, dtype=torch.int32)
    packed_cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    dense_packed = PackedSeqParams(
        qkv_format="bshd",
        cu_seqlens_q=dense_cu,
        cu_seqlens_kv=dense_cu,
        max_seqlen_q=sequence_length,
        max_seqlen_kv=sequence_length,
    )
    empty_packed = PackedSeqParams(qkv_format="bshd")
    thd_packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=packed_cu,
        cu_seqlens_kv=packed_cu,
        max_seqlen_q=13,
        max_seqlen_kv=13,
    )

    deterministic_guard_supported = guard_probe(megatron_gdn, megatron_thd_hidden, thd_packed)

    megatron_bshd_comparisons = []
    megatron_thd_comparisons = []
    miles_bshd_comparisons = []
    miles_thd_comparisons = []
    megatron_losses = []
    miles_losses = []
    megatron_forward_ms = 0.0
    megatron_backward_ms = 0.0
    miles_forward_ms = 0.0
    miles_backward_ms = 0.0
    megatron_initial_norm = parameter_norm(megatron_gdn)
    miles_initial_norm = parameter_norm(miles_attention)
    started_at = time.monotonic()

    for step in range(steps):
        megatron_no_metadata = run_single(
            megatron_gdn,
            lambda: megatron_gdn(megatron_bshd_hidden, None),
            megatron_bshd_hidden,
            megatron_bshd_target,
        )
        megatron_metadata = run_single(
            megatron_gdn,
            lambda: megatron_gdn(
                megatron_bshd_hidden, None, packed_seq_params=dense_packed
            ),
            megatron_bshd_hidden,
            megatron_bshd_target,
        )
        megatron_thd = run_single(
            megatron_gdn,
            lambda: megatron_gdn(
                megatron_thd_hidden, None, packed_seq_params=thd_packed
            ),
            megatron_thd_hidden,
            megatron_thd_target,
        )
        megatron_control = run_sequence_control(
            megatron_gdn,
            lambda hidden, packed: megatron_gdn(hidden, None),
            megatron_thd_hidden,
            megatron_thd_target,
            boundaries,
        )
        megatron_bshd_comparisons.append(compare(megatron_metadata, megatron_no_metadata))
        megatron_thd_comparisons.append(compare(megatron_thd, megatron_control))
        megatron_losses.append(megatron_no_metadata.loss)
        megatron_forward_ms += (
            megatron_no_metadata.forward_ms
            + megatron_metadata.forward_ms
            + megatron_thd.forward_ms
            + megatron_control.forward_ms
        )
        megatron_backward_ms += (
            megatron_no_metadata.backward_ms
            + megatron_metadata.backward_ms
            + megatron_thd.backward_ms
            + megatron_control.backward_ms
        )
        restore_param_grads(megatron_gdn, megatron_no_metadata.param_grads)
        megatron_optimizer.step()

        miles_no_metadata = run_single(
            miles_attention,
            lambda: miles_attention(
                miles_bshd_hidden, None, packed_seq_params=empty_packed
            ),
            miles_bshd_hidden,
            miles_bshd_target,
        )
        miles_metadata = run_single(
            miles_attention,
            lambda: miles_attention(
                miles_bshd_hidden, None, packed_seq_params=dense_packed
            ),
            miles_bshd_hidden,
            miles_bshd_target,
        )
        miles_thd = run_single(
            miles_attention,
            lambda: miles_attention(
                miles_thd_hidden, None, packed_seq_params=thd_packed
            ),
            miles_thd_hidden,
            miles_thd_target,
        )
        miles_control = run_sequence_control(
            miles_attention,
            lambda hidden, packed: miles_attention(hidden, None, packed_seq_params=packed),
            miles_thd_hidden,
            miles_thd_target,
            boundaries,
        )
        miles_bshd_comparisons.append(compare(miles_metadata, miles_no_metadata))
        miles_thd_comparisons.append(compare(miles_thd, miles_control))
        miles_losses.append(miles_no_metadata.loss)
        miles_forward_ms += (
            miles_no_metadata.forward_ms
            + miles_metadata.forward_ms
            + miles_thd.forward_ms
            + miles_control.forward_ms
        )
        miles_backward_ms += (
            miles_no_metadata.backward_ms
            + miles_metadata.backward_ms
            + miles_thd.backward_ms
            + miles_control.backward_ms
        )
        restore_param_grads(miles_attention, miles_no_metadata.param_grads)
        miles_optimizer.step()

    wall_time = time.monotonic() - started_at
    megatron_final_norm = parameter_norm(megatron_gdn)
    miles_final_norm = parameter_norm(miles_attention)

    megatron_bshd_summary = aggregate(megatron_bshd_comparisons)
    megatron_thd_summary = aggregate(megatron_thd_comparisons)
    miles_bshd_summary = aggregate(miles_bshd_comparisons)
    miles_thd_summary = aggregate(miles_thd_comparisons)
    overall_pass = (
        megatron_bshd_summary["output_close"]
        and megatron_bshd_summary["input_grad_close"]
        and megatron_thd_summary["output_close"]
        and megatron_thd_summary["input_grad_close"]
        and miles_bshd_summary["output_close"]
        and miles_bshd_summary["input_grad_close"]
        and miles_thd_summary["output_close"]
        and miles_thd_summary["input_grad_close"]
        and deterministic_guard_supported
    )

    report = {
        "overall_pass": overall_pass,
        "guard_probe": {
            "deterministic_thd_rejected": deterministic_guard_supported,
        },
        "steps": steps,
        "forward_backward_calls": steps * 10,
        "layout": {
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "megatron_bshd_batch": 2,
            "miles_bshd_batch": 1,
            "thd_boundaries": boundaries,
            "dtype": "bfloat16",
        },
        "megatron_gdn": {
            "bshd_dense_metadata": megatron_bshd_summary,
            "thd_packed": megatron_thd_summary,
            "first_loss": megatron_losses[0],
            "last_loss": megatron_losses[-1],
            "initial_parameter_norm": megatron_initial_norm,
            "final_parameter_norm": megatron_final_norm,
            "forward_ms_total": megatron_forward_ms,
            "backward_ms_total": megatron_backward_ms,
        },
        "miles_qwen35_attention": {
            "bshd_dense_metadata": miles_bshd_summary,
            "thd_packed": miles_thd_summary,
            "first_loss": miles_losses[0],
            "last_loss": miles_losses[-1],
            "initial_parameter_norm": miles_initial_norm,
            "final_parameter_norm": miles_final_norm,
            "forward_ms_total": miles_forward_ms,
            "backward_ms_total": miles_backward_ms,
            "bshd_batch_limitation": "Miles wrapper requires batch size 1 when cu_seqlens is supplied.",
        },
        "thresholds": {
            "relative_diff": 2e-2,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
            "device_count": torch.cuda.device_count(),
        },
        "versions": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "megatron_core": "0.19.0+8c1e05747",
            "transformer_engine": "2.17.0",
            "fla": "0.5.2",
            "triton": "3.6.0+git42270451",
        },
        "commits": {
            "miles_workspace": git_commit("/job/miles"),
            "miles_installed_editable": git_commit("/root/miles"),
            "megatron": git_commit("/root/Megatron-LM"),
            "transformer_engine": None,
        },
        "source_paths": {
            "megatron_gdn": "/root/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py",
            "megatron_packed_seq_params": "/root/Megatron-LM/megatron/core/packed_seq_params.py",
            "miles_qwen35": "/job/miles/miles_plugins/models/qwen3_5.py",
            "miles_hf_attention": "/job/miles/miles_plugins/models/hf_attention.py",
            "fla_chunk": "/opt/venv/lib/python3.10/site-packages/fla/ops/gated_delta_rule/chunk.py",
            "transformer_engine": "/opt/venv/lib/python3.10/site-packages/transformer_engine",
        },
        "native_paths": {
            "torch": "/opt/venv/lib/python3.10/site-packages/torch",
            "transformer_engine": "/opt/venv/lib/python3.10/site-packages/transformer_engine",
            "fla": "/opt/venv/lib/python3.10/site-packages/fla",
            "triton": "/opt/venv/lib/python3.10/site-packages/triton",
        },
        "rendezvous": {
            "path": str(rendezvous_path),
            "timeout_minutes": 2,
            "world_size": 1,
        },
        "wall_time_seconds": wall_time,
    }

    output_path = Path("/job/miles/reports/j-20915bb3a656/gdn_packing_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
