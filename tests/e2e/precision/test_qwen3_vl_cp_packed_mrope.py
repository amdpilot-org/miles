"""Two-GPU Qwen3-VL packed MRoPE context-parallel training fixture.

Run the three phases from the repository root:

    torchrun --standalone --rdzv-endpoint=localhost:29321 \
        --nproc_per_node=2 tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py \
        --mode cp --output /tmp/qwen3-vl-cp.pt
    python tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py \
        --mode reference --output /tmp/qwen3-vl-reference.pt
    python tests/e2e/precision/test_qwen3_vl_cp_packed_mrope.py \
        --mode compare --cp /tmp/qwen3-vl-cp.pt \
        --reference /tmp/qwen3-vl-reference.pt --report /tmp/report.json

The model is a tiny, locally initialized Qwen3-VL bridge model. No checkpoint is
downloaded. The CP phase uses both ranks; the independent reference processes
each padded sample separately with context parallelism disabled.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnBackend
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from miles_plugins.models.qwen3_vl import (
    _build_packed_positions,
    _parse_packed_thd,
    _reassemble_full_row,
    install_qwen3_vl_packed_mrope_patch,
)


ITERATIONS = 32
CP_SIZE = 2
PAD_MULTIPLIER = 8
SEED = 4636
LEARNING_RATE = 1e-6

VISION_START_TOKEN = 10
VISION_END_TOKEN = 11
IMAGE_TOKEN = 20
VIDEO_TOKEN = 21
PAD_TOKEN = 0

LENGTH_LAYOUTS = [
    [7, 11, 18],
    [11, 15, 14],
    [15, 7, 18],
    [19, 9, 12],
    [7, 17, 16],
    [11, 7, 22],
]
GRID_LAYOUTS = [
    [(1, 4, 4), (1, 6, 4), (1, 4, 6)],
    [(1, 6, 4), (1, 4, 6), (1, 4, 4)],
    [(1, 4, 6), (1, 4, 4), (1, 6, 4)],
]


@dataclass
class Sample:
    tokens: torch.Tensor
    pixel_values: torch.Tensor
    grid: torch.Tensor
    real_length: int
    padded_length: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cp", "reference", "compare"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cp", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    return parser.parse_args()


def _init_distributed(mode: str) -> tuple[int, int, int, torch.device]:
    if mode == "compare":
        return 0, 1, 0, torch.device("cuda", 0)
    if "RANK" not in os.environ:
        process_count = 2 if mode == "cp" else 1
        os.execvp(
            sys.executable,
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={process_count}",
                __file__,
                *sys.argv[1:],
            ],
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _init_parallel_state(cp_size: int) -> None:
    parallel_state.initialize_model_parallel(1, 1, context_parallel_size=cp_size)
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)


def _make_hf_config() -> Qwen3VLConfig:
    text = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        attention_bias=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    vision = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=(),
    )
    return Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START_TOKEN,
        vision_end_token_id=VISION_END_TOKEN,
        tie_word_embeddings=False,
    )


def _build_model(cp_size: int, device: torch.device) -> tuple[torch.nn.Module, Any]:
    config = _make_hf_config()
    hf_model = Qwen3VLForConditionalGeneration(config)
    provider = Qwen3VLBridge().provider_bridge(hf_model)
    provider.context_parallel_size = cp_size
    provider.seq_length = 256
    provider.vocab_size = 128
    provider.hidden_size = 64
    provider.num_layers = 1
    provider.num_attention_heads = 2
    provider.num_query_groups = 2
    provider.head_dim = 32
    provider.intermediate_size = 128
    provider.mrope_section = [8, 8, 8]
    provider.rotary_base = 10000.0
    provider.patch_size = 2
    provider.temporal_patch_size = 2
    provider.in_channels = 3
    provider.num_position_embeddings = 64
    provider.out_hidden_size = 64
    provider.spatial_merge_size = 2
    provider.deepstack_visual_indexes = []
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.masked_softmax_fusion = False
    provider.cross_entropy_loss_fusion = False
    provider.bias_activation_fusion = False
    provider.gradient_accumulation_fusion = False
    provider.apply_rope_fusion = False
    provider.use_te_rng_tracker = False
    provider.init_model_with_meta_device = False
    provider.cuda_graph_impl = "none"
    provider.vision_dp_when_cp = False
    provider.attention_backend = AttnBackend.flash
    provider.sequence_parallel = False
    provider.share_embeddings_and_output_weights = False
    provider._pg_collection = None
    provider.finalize()
    model = provider.provide(pre_process=True, post_process=True).to(device=device, dtype=torch.bfloat16)
    return model, provider


def _zigzag_slice(tokens: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    token_count = tokens.shape[0]
    chunk_size = (token_count + 2 * cp_size - 1) // (2 * cp_size)
    pad_shape = (0, 0) * (tokens.dim() - 1) + (0, 2 * cp_size * chunk_size - token_count)
    padded = F.pad(tokens, pad_shape, value=PAD_TOKEN)
    first = padded[cp_rank * chunk_size : (cp_rank + 1) * chunk_size]
    mirror_index = 2 * cp_size - 1 - cp_rank
    second = padded[mirror_index * chunk_size : (mirror_index + 1) * chunk_size]
    return torch.cat((first, second))


def _make_sample(iteration: int, sample_index: int, real_length: int, grid: tuple[int, int, int]) -> Sample:
    temporal, height, width = grid
    image_tokens = temporal * (height // 2) * (width // 2)
    prefix = [VISION_START_TOKEN, *([IMAGE_TOKEN] * image_tokens), VISION_END_TOKEN]
    text_count = max(real_length - len(prefix), 0)
    text = [30 + ((iteration * 13 + sample_index * 7 + index * 5) % 90) for index in range(text_count)]
    tokens = torch.tensor(prefix + text, dtype=torch.long)
    padded_length = 2 * CP_SIZE * ((real_length + 2 * CP_SIZE - 1) // (2 * CP_SIZE))
    rows = temporal * height * width
    generator = torch.Generator().manual_seed(SEED + iteration * 100 + sample_index)
    pixel_values = torch.randn(rows, 24, generator=generator)
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    return Sample(tokens, pixel_values, grid_tensor, real_length, padded_length)


def _make_batch(iteration: int, cp_rank: int) -> tuple[list[Sample], torch.Tensor, list[int], PackedSeqParams]:
    lengths = LENGTH_LAYOUTS[iteration % len(LENGTH_LAYOUTS)]
    grids = GRID_LAYOUTS[iteration % len(GRID_LAYOUTS)]
    samples = [
        _make_sample(iteration, index, length, grid)
        for index, (length, grid) in enumerate(zip(lengths, grids, strict=True))
    ]
    local_tokens = [_zigzag_slice(sample.tokens, cp_rank, CP_SIZE) for sample in samples]
    input_ids = torch.cat(local_tokens).unsqueeze(0)
    local_cu = [0]
    for local_token in local_tokens:
        local_cu.append(local_cu[-1] + local_token.numel())
    full_cu = [value * CP_SIZE for value in local_cu]
    cu_tensor = torch.tensor(full_cu, dtype=torch.int32)
    max_seqlen = max(full_cu[index + 1] - full_cu[index] for index in range(len(full_cu) - 1))
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        cu_seqlens_q_padded=cu_tensor,
        cu_seqlens_kv_padded=cu_tensor,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    return samples, input_ids, full_cu, psp


def _model_inputs(samples: list[Sample], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.cat([sample.pixel_values for sample in samples]).to(device),
        "image_grid_thw": torch.stack([sample.grid for sample in samples]).to(device),
    }


def _gather_zigzag(
    local: torch.Tensor, full_cu: list[int], cp_group: Any
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    gathered = [torch.empty_like(local) for _ in range(CP_SIZE)]
    dist.all_gather(gathered, local.contiguous(), group=cp_group)
    if local.dim() > 1:
        full = torch.zeros(
            (full_cu[-1], *local.shape[1:]),
            dtype=local.dtype,
            device=local.device,
        )
        for segment_index in range(len(full_cu) - 1):
            segment_length = full_cu[segment_index + 1] - full_cu[segment_index]
            chunk_length = segment_length // (2 * CP_SIZE)
            local_offset = full_cu[segment_index] // CP_SIZE
            for cp_rank in range(CP_SIZE):
                mirror_rank = 2 * CP_SIZE - 1 - cp_rank
                full[
                    full_cu[segment_index] + cp_rank * chunk_length : (
                        full_cu[segment_index] + (cp_rank + 1) * chunk_length
                    )
                ] = gathered[cp_rank][local_offset : local_offset + chunk_length]
                full[
                    full_cu[segment_index] + mirror_rank * chunk_length : (
                        full_cu[segment_index] + (mirror_rank + 1) * chunk_length
                    )
                ] = gathered[cp_rank][local_offset + chunk_length : local_offset + 2 * chunk_length]
        return full, gathered
    return _reassemble_full_row(gathered, full_cu, CP_SIZE), gathered


def _phase_time(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


def _record_event() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _tensor_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _run_cp(mode_args: argparse.Namespace, device: torch.device, rank: int) -> None:
    bridge_module = importlib.import_module(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"
    )
    original_get_rope_index = bridge_module.get_rope_index
    install_qwen3_vl_packed_mrope_patch()
    model, _ = _build_model(CP_SIZE, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    cp_group = parallel_state.get_context_parallel_group()
    initial_state = _tensor_state_dict(model)
    iterations = []
    for iteration in range(mode_args.iterations):
        optimizer.zero_grad(set_to_none=True)
        samples, input_ids, full_cu, psp = _make_batch(iteration, rank)
        input_ids = input_ids.to(device)
        psp.cu_seqlens_q = psp.cu_seqlens_q.to(device)
        psp.cu_seqlens_kv = psp.cu_seqlens_kv.to(device)
        psp.cu_seqlens_q_padded = psp.cu_seqlens_q_padded.to(device)
        psp.cu_seqlens_kv_padded = psp.cu_seqlens_kv_padded.to(device)
        model_inputs = _model_inputs(samples, device)
        kwargs = {"input_ids": input_ids, "packed_seq_params": psp, **model_inputs}
        parsed = _parse_packed_thd((), kwargs)
        position_start, position_end = _record_event()
        position_start.record()
        local_positions = _build_packed_positions(model, parsed, kwargs, original_get_rope_index)
        position_end.record()
        forward_start, forward_end = _record_event()
        forward_start.record()
        output = model(**kwargs)
        forward_end.record()
        if isinstance(output, tuple):
            output = output[0]
        loss = output.float().square().sum()
        backward_start, backward_end = _record_event()
        backward_start.record()
        loss.backward()
        backward_end.record()
        allreduce_start, allreduce_end = _record_event()
        allreduce_start.record()
        for parameter in model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=cp_group)
        allreduce_end.record()
        optimizer_start, optimizer_end = _record_event()
        optimizer_start.record()
        optimizer.step()
        optimizer_end.record()
        torch.cuda.synchronize()
        full_positions = torch.stack(
            [
                _gather_zigzag(local_positions[channel, 0], full_cu, cp_group)[0]
                for channel in range(3)
            ],
            dim=0,
        )
        full_output, gathered_output = _gather_zigzag(output[0].detach(), full_cu, cp_group)
        gradients = {
            name: parameter.grad.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        iterations.append(
            {
                "iteration": iteration,
                "lengths": [sample.real_length for sample in samples],
                "padded_lengths": [sample.padded_length for sample in samples],
                "grids": [sample.grid.tolist() for sample in samples],
                "positions": full_positions.cpu(),
                "output": full_output.cpu(),
                "local_outputs": [local_output.detach().cpu() for local_output in gathered_output],
                "gradients": gradients,
                "post_state": _tensor_state_dict(model),
                "timings_ms": {
                    "positions": _phase_time(position_start, position_end),
                    "forward": _phase_time(forward_start, forward_end),
                    "backward": _phase_time(backward_start, backward_end),
                    "gradient_allreduce": _phase_time(allreduce_start, allreduce_end),
                    "optimizer": _phase_time(optimizer_start, optimizer_end),
                },
            }
        )
    if rank == 0:
        torch.save({"initial_state": initial_state, "iterations": iterations}, mode_args.output)
    dist.barrier(group=cp_group)


def _run_reference(mode_args: argparse.Namespace, device: torch.device) -> None:
    bridge_module = importlib.import_module(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"
    )
    original_get_rope_index = bridge_module.get_rope_index
    install_qwen3_vl_packed_mrope_patch()
    model, _ = _build_model(1, device)
    cp_artifact = torch.load(mode_args.cp, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(cp_artifact["initial_state"], strict=False)
    assert not load_result.unexpected_keys
    missing_tensor_keys = [
        key for key in load_result.missing_keys if isinstance(model.state_dict()[key], torch.Tensor)
    ]
    assert not missing_tensor_keys, missing_tensor_keys
    for key, expected_value in cp_artifact["initial_state"].items():
        assert torch.equal(model.state_dict()[key].detach().cpu(), expected_value), key
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    iterations = []
    for iteration in range(mode_args.iterations):
        optimizer.zero_grad(set_to_none=True)
        samples, _, full_cu, _ = _make_batch(iteration, 0)
        outputs = []
        positions = []
        losses = []
        for sample in samples:
            input_ids = F.pad(
                sample.tokens,
                (0, sample.padded_length - sample.tokens.numel()),
                value=PAD_TOKEN,
            ).unsqueeze(0).to(device)
            pixel_values = sample.pixel_values.to(device)
            grid = sample.grid.unsqueeze(0).to(device)
            output = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output[0].detach())
            losses.append(output.float().square().sum())
            position, _ = original_get_rope_index(
                model.config.spatial_merge_size,
                model.image_token_id,
                model.video_token_id,
                model.vision_start_token_id,
                input_ids,
                image_grid_thw=grid,
                attention_mask=None,
                packed_seq_params=None,
            )
            positions.append(position[:, 0])
        loss = torch.stack(losses).sum()
        with torch.no_grad():
            packed_tokens = torch.cat(
                [
                    F.pad(
                        sample.tokens,
                        (0, sample.padded_length - sample.tokens.numel()),
                        value=PAD_TOKEN,
                    )
                    for sample in samples
                ]
            ).unsqueeze(0).to(device)
            packed_cu = torch.tensor(full_cu, dtype=torch.int32, device=device)
            packed_psp = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=packed_cu,
                cu_seqlens_kv=packed_cu,
                cu_seqlens_q_padded=packed_cu,
                cu_seqlens_kv_padded=packed_cu,
                max_seqlen_q=max(full_cu[index + 1] - full_cu[index] for index in range(len(full_cu) - 1)),
                max_seqlen_kv=max(full_cu[index + 1] - full_cu[index] for index in range(len(full_cu) - 1)),
            )
            packed_output = model(
                input_ids=packed_tokens,
                packed_seq_params=packed_psp,
                **_model_inputs(samples, device),
            )
            if isinstance(packed_output, tuple):
                packed_output = packed_output[0]
            packed_output = packed_output[0].detach().cpu()
        loss.backward()
        optimizer.step()
        gradients = {
            name: parameter.grad.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        iterations.append(
            {
                "iteration": iteration,
                "positions": torch.cat(positions, dim=1).cpu(),
                "output": torch.cat(outputs, dim=0).cpu(),
                "packed_output": packed_output,
                "gradients": gradients,
                "post_state": _tensor_state_dict(model),
            }
        )
    torch.save({"iterations": iterations}, mode_args.output)


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0, 0.0
    difference = (left.float() - right.float()).abs()
    maximum = difference.max().item()
    relative = (difference / (right.float().abs() + 1e-6)).max().item()
    return maximum, relative


def _run_compare(mode_args: argparse.Namespace) -> None:
    cp_artifact = torch.load(mode_args.cp, map_location="cpu", weights_only=False)
    reference_artifact = torch.load(mode_args.reference, map_location="cpu", weights_only=False)
    rows = []
    for cp_row, reference_row in zip(cp_artifact["iterations"], reference_artifact["iterations"], strict=True):
        position_equal = torch.equal(cp_row["positions"], reference_row["positions"])
        output_max, output_relative = _max_difference(cp_row["output"], reference_row["output"])
        packed_output_max, _ = _max_difference(cp_row["output"], reference_row["packed_output"])
        local_output_max = 0.0
        packed_reference = reference_row["packed_output"]
        independent_reference = reference_row["output"]
        reference_positions = reference_row["positions"]
        sample_checks = []
        segment_start = 0
        for sample_index, (real_length, padded_length) in enumerate(
            zip(cp_row["lengths"], cp_row["padded_lengths"], strict=True)
        ):
            independent_slice = independent_reference[segment_start : segment_start + padded_length]
            packed_slice = packed_reference[segment_start : segment_start + padded_length]
            independent_vs_packed_max, _ = _max_difference(independent_slice, packed_slice)
            first_position = reference_positions[:, segment_start].tolist()
            last_real_position = reference_positions[:, segment_start + real_length - 1].tolist()
            first_padding_position = (
                reference_positions[:, segment_start + real_length].tolist()
                if real_length < padded_length
                else None
            )
            last_padding_position = reference_positions[:, segment_start + padded_length - 1].tolist()
            sample_checks.append(
                {
                    "sample_index": sample_index,
                    "real_length": real_length,
                    "padded_length": padded_length,
                    "padding_token_count": padded_length - real_length,
                    "padded_length_aligned": padded_length % (2 * CP_SIZE) == 0,
                    "first_position": first_position,
                    "last_real_position": last_real_position,
                    "first_padding_position": first_padding_position,
                    "last_padding_position": last_padding_position,
                    "independent_vs_packed_output_equal": torch.equal(independent_slice, packed_slice),
                    "independent_vs_packed_output_max_abs": independent_vs_packed_max,
                }
            )
            segment_start += padded_length
        for local_rank, local_output in enumerate(cp_row["local_outputs"]):
            expected_chunks = []
            segment_start = 0
            for length in cp_row["padded_lengths"]:
                expected_chunks.append(
                    _zigzag_slice(
                        packed_reference[segment_start : segment_start + length],
                        local_rank,
                        CP_SIZE,
                    )
                )
                segment_start += length
            expected_local = torch.cat(expected_chunks)
            local_max, _ = _max_difference(local_output, expected_local)
            local_output_max = max(local_output_max, local_max)
        gradient_rows = []
        for name, reference_gradient in reference_row["gradients"].items():
            cp_gradient = cp_row["gradients"].get(name)
            if cp_gradient is None:
                gradient_rows.append((name, float("inf"), float("inf")))
                continue
            gradient_max, gradient_relative = _max_difference(cp_gradient, reference_gradient)
            gradient_rows.append((name, gradient_max, gradient_relative))
        weight_rows = []
        for name, reference_weight in reference_row["post_state"].items():
            cp_weight = cp_row["post_state"][name]
            weight_max, _ = _max_difference(cp_weight, reference_weight)
            weight_rows.append((name, weight_max))
        rows.append(
            {
                "iteration": cp_row["iteration"],
                "lengths": cp_row["lengths"],
                "padded_lengths": cp_row["padded_lengths"],
                "grids": cp_row["grids"],
                "position_equal": position_equal,
                "output_max_abs": output_max,
                "output_max_relative": output_relative,
                "packed_output_max_abs": packed_output_max,
                "local_output_max_abs": local_output_max,
                "independent_vs_packed_output_equal": all(
                    check["independent_vs_packed_output_equal"] for check in sample_checks
                ),
                "cross_sample_contamination": any(
                    not check["independent_vs_packed_output_equal"] for check in sample_checks
                ),
                "sample_checks": sample_checks,
                "gradient_max_abs": max(row[1] for row in gradient_rows),
                "gradient_max_relative": max(row[2] for row in gradient_rows),
                "weight_max_abs": max(row[1] for row in weight_rows),
                "timings_ms": cp_row["timings_ms"],
            }
        )
    report = {
        "schema": "qwen3-vl-cp-packed-mrope-v1",
        "iterations": len(rows),
        "cp_size": CP_SIZE,
        "rows": rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
    }
    mode_args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    worst_output = max(row["output_max_abs"] for row in rows)
    worst_gradient = max(row["gradient_max_abs"] for row in rows)
    worst_weight = max(row["weight_max_abs"] for row in rows)
    positions_ok = all(row["position_equal"] for row in rows)
    print(f"iterations={len(rows)} positions_equal={positions_ok}")
    print(f"worst_output_max_abs={worst_output:.8g}")
    print(f"worst_gradient_max_abs={worst_gradient:.8g}")
    print(f"worst_weight_max_abs={worst_weight:.8g}")
    if not positions_ok or worst_output > 2e-4 or worst_gradient > 2e-4 or worst_weight > 2e-4:
        raise SystemExit("numerical comparison failed")


def main() -> None:
    args = _parse_args()
    rank, world_size, _, device = _init_distributed(args.mode)
    if args.mode == "compare":
        _run_compare(args)
        return
    if args.mode == "cp" and world_size != CP_SIZE:
        raise SystemExit(f"CP mode requires world_size={CP_SIZE}, got {world_size}")
    cp_size = CP_SIZE if args.mode == "cp" else 1
    _init_parallel_state(cp_size)
    if args.mode == "cp":
        _run_cp(args, device, rank)
    else:
        _run_reference(args, device)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
