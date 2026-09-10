#!/usr/bin/env python3
"""Run repeated two-GPU rollout lifecycle cycles and emit JSON evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import httpx
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils
from miles.utils.distributed_utils import init_process_group
from sglang.srt.utils.weight_checker import (
    _hash_tensor,
    _is_non_persistent_buffer_name,
    overall_checksum,
)


FIXED_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
FIXED_OUTPUT_TOKENS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-a-url", required=True)
    parser.add_argument("--server-b-url", required=True)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--update-a-port", type=int, required=True)
    parser.add_argument("--update-b-port", type=int, required=True)
    parser.add_argument("--default-port", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--pg-timeout", type=float, default=300.0)
    parser.add_argument("--fail-wake-cycle", type=int, default=16)
    parser.add_argument("--fail-update-cycle", type=int, default=24)
    parser.add_argument("--baseline", action="store_true")
    return parser.parse_args()


def run_api(coroutine, timeout: float):
    future = async_utils.submit(coroutine)
    return future.result(timeout=timeout)


def local_checksums(model: torch.nn.Module) -> dict[str, str]:
    checksums = {}
    for name, tensor in model.state_dict().items():
        if _is_non_persistent_buffer_name(name):
            continue
        if getattr(tensor, "_skip_weight_check", False):
            continue
        checksums[name] = _hash_tensor(tensor.detach())
    return checksums


def engine_compatible_checksums(model: torch.nn.Module) -> dict[str, str]:
    actor_state = model.state_dict()
    expected = {}
    for name, tensor in actor_state.items():
        if _is_non_persistent_buffer_name(name):
            continue
        if getattr(tensor, "_skip_weight_check", False):
            continue
        expected[name] = _hash_tensor(tensor.detach())

    fused_sources = {
        "qkv_proj.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
        "qkv_proj.bias": ("q_proj.bias", "k_proj.bias", "v_proj.bias"),
        "gate_up_proj.weight": ("gate_proj.weight", "up_proj.weight"),
    }
    for actor_name in actor_state:
        for fused_name, source_names in fused_sources.items():
            if not actor_name.endswith(source_names[0]):
                continue
            prefix = actor_name[: -len(source_names[0])]
            engine_name = prefix + fused_name
            source_tensors = []
            for source_name in source_names:
                source_path = prefix + source_name
                if source_path not in actor_state:
                    break
                source_tensors.append(actor_state[source_path].detach().contiguous())
            if len(source_tensors) == len(source_names):
                expected[engine_name] = _hash_tensor(torch.cat(source_tensors, dim=0))
    return expected


def compare_checksums(model: torch.nn.Module, engine_result: dict) -> dict:
    actor_checksums = local_checksums(model)
    expected_checksums = engine_compatible_checksums(model)
    ranks = engine_result.get("ranks") or []
    engine_checksums = ranks[0].get("checksums", {}) if ranks else {}
    common = sorted(set(expected_checksums) & set(engine_checksums))
    mismatches = [name for name in common if expected_checksums[name] != engine_checksums[name]]
    unexpected_engine = sorted(set(engine_checksums) - set(expected_checksums))
    actor_only = sorted(set(actor_checksums) - set(engine_checksums))
    fused_expected = sorted(set(expected_checksums) - set(actor_checksums))
    return {
        "actor_tensor_count": len(actor_checksums),
        "engine_tensor_count": len(engine_checksums),
        "common_tensor_count": len(common),
        "mismatch_count": len(mismatches),
        "unexpected_engine_tensor_count": len(unexpected_engine),
        "actor_only_tensor_count": len(actor_only),
        "fused_expected_tensor_count": len(fused_expected),
        "all_engine_tensors_checked": len(common) == len(engine_checksums),
        "mismatched_tensors": mismatches[:20],
        "unexpected_engine_tensors": unexpected_engine[:20],
        "actor_only_tensors": actor_only[:20],
        "actor_overall_checksum": overall_checksum(actor_checksums),
        "expected_engine_overall_checksum": overall_checksum(
            {name: expected_checksums[name] for name in engine_checksums}
        ),
        "engine_overall_checksum": ranks[0].get("per_gpu_checksum") if ranks else None,
    }


def control_forward(model: torch.nn.Module, device: torch.device) -> tuple[list[int], list[float], float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    sequence = list(FIXED_TOKENS)
    tokens: list[int] = []
    log_probs: list[float] = []

    start_event.record()
    with torch.no_grad():
        for _ in range(FIXED_OUTPUT_TOKENS):
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[:, -1].float()
            log_prob = torch.log_softmax(logits, dim=-1).max()
            token = int(logits.argmax(dim=-1).item())
            tokens.append(token)
            log_probs.append(float(log_prob.item()))
            sequence.append(token)
    end_event.record()
    end_event.synchronize()
    return tokens, log_probs, start_event.elapsed_time(end_event) / 1000.0


def control_log_probs_for_tokens(
    model: torch.nn.Module, output_tokens: list[int], device: torch.device
) -> list[float]:
    sequence = list(FIXED_TOKENS)
    values: list[float] = []
    with torch.no_grad():
        for token in output_tokens:
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[:, -1].float()
            values.append(float(torch.log_softmax(logits, dim=-1)[0, token].item()))
            sequence.append(token)
    return values


def memory_snapshot(device_index: int) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    stats = torch.cuda.memory_stats(device_index)
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "torch_allocated_bytes": stats["allocated_bytes.all.current"],
        "torch_reserved_bytes": stats["reserved_bytes.all.current"],
    }


def train_step(
    model: torch.nn.Module,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    torch.cuda.synchronize(device)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    input_ids = torch.arange(model.config.vocab_size, device=device)[:32].unsqueeze(0)
    optimizer.zero_grad(set_to_none=True)

    forward_start.record()
    output = ddp_model(input_ids=input_ids, labels=input_ids)
    loss = output.loss
    forward_end.record()

    backward_start.record()
    loss.backward()
    backward_end.record()

    local_norm_sq = torch.tensor(0.0, device=device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            local_norm_sq += parameter.grad.detach().float().pow(2).sum()
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    global_grad_norm = float(local_norm_sq.sqrt().item())

    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    optimizer_end.synchronize()

    return {
        "loss": float(loss.detach().cpu()),
        "local_grad_norm": float(
            sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            ).sqrt().item()
        ),
        "global_grad_norm": global_grad_norm,
        "forward_ms": forward_start.elapsed_time(forward_end),
        "backward_ms": backward_start.elapsed_time(backward_end),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
    }


def connect_update_group(
    client: SGLangApiClient,
    port: int,
    args: argparse.Namespace,
    suffix: str,
    device_index: int,
):
    torch.cuda.set_device(device_index)
    group_name = f"j50c9a529dc1c-{suffix}-{os.getpid()}"
    start = time.perf_counter()
    future = async_utils.submit(
        client.init_weights_update_group(
            "127.0.0.1",
            port,
            1,
            2,
            group_name,
            backend="nccl",
        )
    )
    group = init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=2,
        rank=0,
        group_name=group_name,
        timeout=timedelta(seconds=args.pg_timeout),
    )
    future.result(timeout=args.pg_timeout)
    return group, group_name, time.perf_counter() - start


def update_weights(
    model: torch.nn.Module,
    client: SGLangApiClient,
    group,
    group_name: str,
    version: int,
    args: argparse.Namespace,
    *,
    inject_failure: bool,
    device_index: int,
) -> dict:
    torch.cuda.set_device(device_index)
    named_tensors = list(model.state_dict().items())
    names = [name for name, _ in named_tensors]
    dtypes = [str(tensor.dtype).replace("torch.", "") for _, tensor in named_tensors]
    shapes = [list(tensor.shape) for _, tensor in named_tensors]
    transfer_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)

    timings = {"transfer_bytes": transfer_bytes}
    update_start = time.perf_counter()
    run_api(client.pause_generation(mode="in_place"), args.pg_timeout)
    timings["pause_seconds"] = time.perf_counter() - update_start

    begin_start = time.perf_counter()
    run_api(client.begin_weight_update(selector="all", sync_base=True), args.pg_timeout)
    timings["begin_seconds"] = time.perf_counter() - begin_start

    broadcast_start = time.perf_counter()
    future = async_utils.submit(
        client.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=True,
            weight_version=str(version),
            selector="all",
        )
    )
    handles = []
    for _, tensor in named_tensors:
        tensor = tensor.detach()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()
    torch.cuda.synchronize()
    future.result(timeout=args.pg_timeout)
    timings["broadcast_seconds"] = time.perf_counter() - broadcast_start

    end_start = time.perf_counter()
    failed = False
    failure_type = None
    failure_message = None
    try:
        expected = {"bogus-adapter": {}} if inject_failure else None
        run_api(client.end_weight_update(expected_lora_checksums=expected), args.pg_timeout)
    except Exception as exc:
        failed = True
        failure_type = type(exc).__name__
        failure_message = str(exc)[:1000]
    timings["end_seconds"] = time.perf_counter() - end_start

    version_start = time.perf_counter()
    run_api(
        client.update_weight_version(weight_version=str(version), abort_all_requests=False),
        args.pg_timeout,
    )
    timings["version_seconds"] = time.perf_counter() - version_start

    resume_start = time.perf_counter()
    run_api(client.continue_generation(), args.pg_timeout)
    timings["resume_seconds"] = time.perf_counter() - resume_start
    timings["total_seconds"] = time.perf_counter() - update_start
    return {
        "version": version,
        "failed": failed,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "timings": timings,
        "start_monotonic": update_start,
        "end_monotonic": time.perf_counter(),
    }


def release_engine(client: SGLangApiClient, args: argparse.Namespace, device_index: int) -> dict:
    before = memory_snapshot(device_index)
    start = time.perf_counter()
    run_api(client.release_memory_occupation(tags=["weights", "kv_cache"]), args.pg_timeout)
    seconds = time.perf_counter() - start
    after = memory_snapshot(device_index)
    return {
        "seconds": seconds,
        "before": before,
        "after": after,
        "released_bytes": after["free_bytes"] - before["free_bytes"],
    }


def wake_engine(
    client: SGLangApiClient,
    args: argparse.Namespace,
    server_url: str,
    device_index: int,
    *,
    inject_failure: bool,
) -> dict:
    before = memory_snapshot(device_index)
    start = time.perf_counter()
    failed = False
    failure_type = None
    failure_message = None
    if inject_failure:
        try:
            run_api(client.resume_memory_occupation(tags=["bogus"]), args.pg_timeout)
        except Exception as exc:
            failed = True
            failure_type = type(exc).__name__
            failure_message = str(exc)[:1000]

    recovery_seconds = None
    recovery_attempts = 0
    recovery_successful = False
    if failed:
        recovery_start = time.perf_counter()
        deadline = time.monotonic() + args.pg_timeout
        while time.monotonic() < deadline:
            recovery_attempts += 1
            try:
                response = httpx.get(f"{server_url}/health_generate", timeout=1.0)
                if response.status_code == 200:
                    recovery_successful = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        recovery_seconds = time.perf_counter() - recovery_start
        if not recovery_successful:
            raise TimeoutError(
                f"engine did not recover within {args.pg_timeout}s after failed wake"
            )
    else:
        run_api(client.resume_memory_occupation(tags=["weights", "kv_cache"]), args.pg_timeout)

    seconds = time.perf_counter() - start
    after = memory_snapshot(device_index)
    return {
        "seconds": seconds,
        "failed": failed,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "recovery_action": "single_bounded_engine_restart" if failed else None,
        "recovery_attempts": recovery_attempts,
        "recovery_seconds": recovery_seconds,
        "recovery_successful": recovery_successful,
        "before": before,
        "after": after,
        "resumed_cost_bytes": before["free_bytes"] - after["free_bytes"],
    }


async def generate_request(
    client: httpx.AsyncClient,
    server_url: str,
    request_id: int,
    args: argparse.Namespace,
    *,
    use_router: bool = False,
) -> dict:
    submitted = time.perf_counter()
    payload = {
        "input_ids": FIXED_TOKENS,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": False,
        },
        "return_logprob": True,
    }
    base_url = args.router_url if use_router else server_url
    try:
        response = await client.post(f"{base_url}/generate", json=payload, timeout=args.request_timeout)
        response.raise_for_status()
        data = response.json()
        meta = data.get("meta_info", {})
        completed = time.perf_counter()
        output_logprobs = meta.get("output_token_logprobs", [])
        return {
            "request_id": request_id,
            "server_url": server_url,
            "use_router": use_router,
            "submitted_monotonic": submitted,
            "completed_monotonic": completed,
            "latency_seconds": completed - submitted,
            "status": "completed",
            "error_type": None,
            "error_message": None,
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "finish_reason": meta.get("finish_reason"),
            "weight_version": meta.get("weight_version"),
            "weight_versions": meta.get("weight_versions", []),
            "num_retractions": meta.get("num_retractions", 0),
            "time_stats": meta.get("time_stats", {}),
            "output_tokens": [int(item[1]) for item in output_logprobs],
            "output_log_probs": [float(item[0]) for item in output_logprobs],
        }
    except Exception as exc:
        completed = time.perf_counter()
        return {
            "request_id": request_id,
            "server_url": server_url,
            "use_router": use_router,
            "submitted_monotonic": submitted,
            "completed_monotonic": completed,
            "latency_seconds": completed - submitted,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "weight_version": None,
            "weight_versions": [],
            "num_retractions": None,
            "time_stats": {},
            "output_tokens": [],
            "output_log_probs": [],
        }


async def router_list(client: httpx.AsyncClient, router_url: str) -> list[str]:
    response = await client.get(f"{router_url}/list_workers", timeout=10.0)
    response.raise_for_status()
    return response.json()["urls"]


async def router_add(client: httpx.AsyncClient, router_url: str, worker_url: str) -> None:
    response = await client.post(f"{router_url}/add_worker", params={"url": worker_url}, timeout=10.0)
    response.raise_for_status()

async def router_remove(client: httpx.AsyncClient, router_url: str, worker_url: str) -> None:
    response = await client.post(f"{router_url}/remove_worker", params={"url": worker_url}, timeout=10.0)
    response.raise_for_status()


def merge_cycles(rank0_cycles: list[dict], rank1_cycles: list[dict]) -> list[dict]:
    merged = []
    for rank0_cycle, rank1_cycle in zip(rank0_cycles, rank1_cycles):
        record = dict(rank0_cycle)
        record["update"]["b"] = rank1_cycle["update"]["b"]
        record["checksum"]["b"] = rank1_cycle["checksum"]["b"]
        record["output"]["b"] = rank1_cycle["output"]["b"]
        record["engine_weight_version"]["b"] = rank1_cycle["engine_weight_version"]["b"]
        record["requests"].extend(rank1_cycle["requests"])
        record["version_consistent"] = (
            str(record["engine_weight_version"]["a"]) == str(record["cycle"])
            and str(record["engine_weight_version"]["b"]) == str(record["cycle"])
        )
        merged.append(record)
    return merged


def timing_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_seconds": None, "min_seconds": None, "max_seconds": None}
    return {
        "count": len(values),
        "mean_seconds": sum(values) / len(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def request_version_count(request: dict) -> int:
    return len({span.get("version") for span in request.get("weight_versions", [])})


def build_summary(
    rank: int,
    local_rank: int,
    cycles: list[dict],
    connect_seconds_a: float,
    connect_seconds_b: float,
    args: argparse.Namespace,
) -> dict:
    requests = [result for cycle in cycles for result in cycle["requests"]]
    request_failures = [
        {
            "cycle": cycle["cycle"],
            "request_id": result["request_id"],
            "phase": result["phase"],
            "error_type": result["error_type"],
            "error_message": result["error_message"],
        }
        for cycle in cycles
        for result in cycle["requests"]
        if result["status"] == "failed"
    ]
    timeout_causes = [
        failure
        for failure in request_failures
        if "timeout" in failure["error_type"].lower() or "timeout" in failure["error_message"].lower()
    ]
    update_failures = [
        {
            "cycle": cycle["cycle"],
            "engine": engine,
            "failure_type": cycle["update"][engine]["failure_type"],
            "failure_message": cycle["update"][engine]["failure_message"],
        }
        for cycle in cycles
        for engine in ("a", "b")
        if cycle["update"][engine]["failed"]
    ]
    phase_counts = {
        phase: sum(result["phase"] == phase for result in requests)
        for phase in ("before_release", "during_release", "after_wake")
    }
    version_span_counts = {}
    for result in requests:
        count = request_version_count(result)
        version_span_counts[str(count)] = version_span_counts.get(str(count), 0) + 1

    update_timing_keys = (
        "pause_seconds",
        "begin_seconds",
        "broadcast_seconds",
        "end_seconds",
        "version_seconds",
        "resume_seconds",
        "total_seconds",
    )
    update_phase_seconds = {
        engine: {
            key: timing_stats([cycle["update"][engine]["timings"][key] for cycle in cycles])
            for key in update_timing_keys
        }
        for engine in ("a", "b")
    }
    failed_wake_cycle = next((cycle["cycle"] for cycle in cycles if cycle["wake"]["failed"]), None)
    failed_update_cycle = next(
        (cycle["cycle"] for cycle in cycles if cycle["update"]["a"]["failed"] or cycle["update"]["b"]["failed"]),
        None,
    )
    wake_recovery_successful = None
    if failed_wake_cycle is not None and failed_wake_cycle < len(cycles):
        wake_recovery_successful = not cycles[failed_wake_cycle]["wake"]["failed"]
    update_recovery_successful = None
    if failed_update_cycle is not None and failed_update_cycle < len(cycles):
        update_recovery_successful = not (
            cycles[failed_update_cycle]["update"]["a"]["failed"]
            or cycles[failed_update_cycle]["update"]["b"]["failed"]
        )

    return {
        "rank": rank,
        "local_rank": local_rank,
        "cycles": cycles,
        "connect_update_group_seconds": {
            "a": connect_seconds_a,
            "b": connect_seconds_b,
        },
        "real_weight_update_events": len(cycles) * 2,
        "successful_updates": sum(
            not cycle["update"][engine]["failed"] for cycle in cycles for engine in ("a", "b")
        ),
        "failed_updates": sum(
            cycle["update"][engine]["failed"] for cycle in cycles for engine in ("a", "b")
        ),
        "failed_update_cycle": failed_update_cycle,
        "update_recovery_successful": update_recovery_successful,
        "failed_wake_cycle": failed_wake_cycle,
        "wake_recovery_successful": wake_recovery_successful,
        "admitted_requests": sum(result["prompt_tokens"] is not None for result in requests),
        "completed_requests": sum(result["status"] == "completed" for result in requests),
        "failed_requests": sum(result["status"] == "failed" for result in requests),
        "request_phase_counts": phase_counts,
        "request_failures": request_failures,
        "timeout_causes": timeout_causes,
        "request_weight_version_span_counts": version_span_counts,
        "mixed_weight_version_requests": sum(request_version_count(result) > 1 for result in requests),
        "version_consistent_cycles": sum(cycle["version_consistent"] for cycle in cycles),
        "unique_actor_overall_checksums": len(
            {cycle["checksum"]["a"]["actor_overall_checksum"] for cycle in cycles}
        ),
        "unique_engine_a_overall_checksums": len(
            {cycle["checksum"]["a"]["engine_overall_checksum"] for cycle in cycles}
        ),
        "unique_engine_b_overall_checksums": len(
            {cycle["checksum"]["b"]["engine_overall_checksum"] for cycle in cycles}
        ),
        "unique_train_losses": len({cycle["train"]["loss"] for cycle in cycles}),
        "checksum_mismatch_cycles": sum(
            cycle["checksum"][engine]["mismatch_count"] > 0 for cycle in cycles for engine in ("a", "b")
        ),
        "all_engine_tensors_checked_cycles": sum(
            cycle["checksum"][engine]["all_engine_tensors_checked"]
            for cycle in cycles
            for engine in ("a", "b")
        ),
        "expected_engine_checksum_match_cycles": sum(
            cycle["checksum"][engine]["expected_engine_overall_checksum"]
            == cycle["checksum"][engine]["engine_overall_checksum"]
            for cycle in cycles
            for engine in ("a", "b")
        ),
        "output_equal_cycles": sum(
            cycle["output"][engine]["tokens_equal"] for cycle in cycles for engine in ("a", "b")
        ),
        "release_positive_cycles": sum(cycle["release"]["released_bytes"] > 0 for cycle in cycles),
        "router_admission_cycles": sum(cycle["router_admission"] for cycle in cycles),
        "update_failures": update_failures,
        "gpu_phase_timings": {
            "train_forward_ms": timing_stats([cycle["train"]["forward_ms"] for cycle in cycles]),
            "train_backward_ms": timing_stats([cycle["train"]["backward_ms"] for cycle in cycles]),
            "train_optimizer_ms": timing_stats([cycle["train"]["optimizer_ms"] for cycle in cycles]),
            "release_seconds": timing_stats([cycle["release"]["seconds"] for cycle in cycles]),
            "wake_seconds": timing_stats([cycle["wake"]["seconds"] for cycle in cycles]),
            "control_forward_seconds": timing_stats([cycle["control"]["seconds"] for cycle in cycles]),
            "update_phase_seconds": update_phase_seconds,
            "transfer_bytes_per_update": [
                cycle["update"][engine]["timings"]["transfer_bytes"]
                for cycle in cycles
                for engine in ("a", "b")
            ],
        },
        "baseline": args.baseline,
    }


async def main_async(args: argparse.Namespace, rank: int, local_rank: int) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.train()
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    if rank == 0:
        client = SGLangApiClient(server_url=args.server_a_url)
        server_url = args.server_a_url
        update_port = args.update_a_port
        group_suffix = "a"
        engine_label = "a"
        engine_device = 1
    else:
        client = SGLangApiClient(server_url=args.server_b_url)
        server_url = args.server_b_url
        update_port = args.update_b_port
        group_suffix = "b"
        engine_label = "b"
        engine_device = 0

    dist.barrier()
    group, group_name, connect_seconds = connect_update_group(
        client,
        update_port,
        args,
        group_suffix,
        local_rank,
    )
    dist.barrier()

    cycles = []
    previous_weight_norm = None
    for cycle_index in range(args.cycles):
        cycle_number = cycle_index + 1
        cycle_start = time.perf_counter()
        requests = []
        train_metrics = None
        control_metrics = None
        release_metrics = None
        wake_metrics = None
        update_result = None
        checksum_result = None
        output_result = None
        engine_version = None
        router_before_urls = None
        router_during_urls = None
        router_after_urls = None
        router_admission = False

        if rank == 0:
            async with httpx.AsyncClient(timeout=args.request_timeout) as http_client:
                before_a = await generate_request(http_client, args.server_a_url, 0, args)
                before_a["phase"] = "before_release"
                requests.append(before_a)

                router_before_urls = await router_list(http_client, args.router_url)
                await router_remove(http_client, args.router_url, args.server_a_url)
                router_during_urls = await router_list(http_client, args.router_url)

                during_tasks = [
                    asyncio.create_task(
                        generate_request(
                            http_client,
                            args.server_b_url,
                            1 + request_id,
                            args,
                            use_router=True,
                        )
                    )
                    for request_id in range(args.concurrency)
                ]
                during_results = await asyncio.gather(*during_tasks)
                for result in during_results:
                    result["phase"] = "during_release"
                requests.extend(during_results)

            release_metrics = release_engine(client, args, engine_device)
        else:
            async with httpx.AsyncClient(timeout=args.request_timeout) as http_client:
                before_b = await generate_request(http_client, args.server_b_url, 0, args)
                before_b["phase"] = "before_release"
                requests.append(before_b)

        dist.barrier()
        train_metrics = train_step(model, ddp_model, optimizer, device)
        dist.barrier()

        control_tokens, control_log_probs, control_seconds = control_forward(model, device)
        control_for_tokens = control_log_probs_for_tokens(model, control_tokens, device)
        control_metrics = {
            "tokens": control_tokens,
            "log_probs": control_log_probs,
            "seconds": control_seconds,
        }

        if rank == 1:
            update_result = update_weights(
                model,
                client,
                group,
                group_name,
                cycle_number,
                args,
                inject_failure=(cycle_number == args.fail_update_cycle and not args.baseline),
                device_index=local_rank,
            )
        dist.barrier()

        if rank == 0:
            wake_metrics = wake_engine(
                client,
                args,
                server_url,
                engine_device,
                inject_failure=(cycle_number == args.fail_wake_cycle and not args.baseline),
            )
            if wake_metrics["failed"]:
                dist.destroy_process_group(group)
                group, group_name, reconnect_seconds = connect_update_group(
                    client,
                    update_port,
                    args,
                    group_suffix,
                    local_rank,
                )
                wake_metrics["recovery_reconnect_seconds"] = reconnect_seconds
        dist.barrier()

        if rank == 0:
            update_result = update_weights(
                model,
                client,
                group,
                group_name,
                cycle_number,
                args,
                inject_failure=False,
                device_index=local_rank,
            )
        dist.barrier()

        engine_version = run_api(client.get_weight_version(), args.pg_timeout)
        checksum_raw = run_api(client.check_weights(action="checksum"), args.pg_timeout)
        checksum_result = compare_checksums(model, checksum_raw)

        async with httpx.AsyncClient(timeout=args.request_timeout) as http_client:
            after_request = await generate_request(http_client, server_url, 2 + args.concurrency, args)
            after_request["phase"] = "after_wake"
            requests.append(after_request)

            if rank == 0:
                await router_add(http_client, args.router_url, args.server_a_url)
                router_after_urls = await router_list(http_client, args.router_url)
                router_after = await generate_request(
                    http_client,
                    args.server_a_url,
                    3 + args.concurrency,
                    args,
                    use_router=True,
                )
                router_after["phase"] = "after_wake"
                requests.append(router_after)

        output_result = {
            "tokens_equal": after_request["output_tokens"] == control_tokens,
            "max_abs_log_prob_difference": max(
                abs(left - right)
                for left, right in zip(after_request["output_log_probs"], control_for_tokens)
            )
            if after_request["output_log_probs"]
            else None,
            "tokens": after_request["output_tokens"],
        }

        weight_norm = float(
            sum(parameter.detach().float().pow(2).sum() for parameter in model.parameters()).cpu()
        )
        weight_norm_delta = None if previous_weight_norm is None else abs(weight_norm - previous_weight_norm)
        previous_weight_norm = weight_norm

        cycle_record = {
            "cycle": cycle_number,
            "train": train_metrics,
            "control": control_metrics,
            "release": release_metrics,
            "wake": wake_metrics,
            "update": {engine_label: update_result},
            "checksum": {engine_label: checksum_result},
            "output": {engine_label: output_result},
            "engine_weight_version": {engine_label: engine_version},
            "version_consistent": str(engine_version) == str(cycle_number),
            "weight_norm": weight_norm,
            "weight_norm_delta": weight_norm_delta,
            "router_before_urls": router_before_urls,
            "router_during_urls": router_during_urls,
            "router_after_urls": router_after_urls,
            "router_admission": router_admission,
            "requests": requests,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        cycles.append(cycle_record)
        dist.barrier()

    if rank == 1:
        rank1_path = Path(args.output_path).with_name("rank1_cycles.json")
        rank1_path.write_text(json.dumps(cycles, indent=2, sort_keys=True) + "\n")
    dist.barrier()

    if rank == 0:
        rank1_path = Path(args.output_path).with_name("rank1_cycles.json")
        rank1_cycles = json.loads(rank1_path.read_text())
        merged_cycles = merge_cycles(cycles, rank1_cycles)
        for record in merged_cycles:
            all_requests = record["requests"]
            record["router_admission"] = (
                set(record["router_before_urls"]) == {args.server_a_url, args.server_b_url}
                and set(record["router_during_urls"]) == {args.server_b_url}
                and set(record["router_after_urls"]) == {args.server_a_url, args.server_b_url}
                and all(result["status"] == "completed" for result in all_requests)
            )

        run_api(client.destroy_weights_update_group(group_name), args.pg_timeout)
        summary = build_summary(
            rank,
            local_rank,
            merged_cycles,
            connect_seconds,
            None,
            args,
        )
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        run_api(client.destroy_weights_update_group(group_name), args.pg_timeout)
    dist.barrier()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "0"
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{args.default_port}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=args.pg_timeout),
    )
    asyncio.run(main_async(args, rank, local_rank))


if __name__ == "__main__":
    main()
