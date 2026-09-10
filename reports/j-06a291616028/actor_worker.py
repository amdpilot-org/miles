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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--update-port", type=int, required=True)
    parser.add_argument("--default-port", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--pg-timeout", type=float, default=120.0)
    parser.add_argument("--fail-cycle", type=int, default=16)
    return parser.parse_args()


def run_api(coroutine, timeout):
    future = async_utils.submit(coroutine)
    return future.result(timeout=timeout)


def local_checksums(model):
    checksums = {}
    for name, tensor in model.state_dict().items():
        if _is_non_persistent_buffer_name(name):
            continue
        if getattr(tensor, "_skip_weight_check", False):
            continue
        checksums[name] = _hash_tensor(tensor.detach())
    return checksums


def engine_compatible_checksums(model):
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


def compare_checksums(model, engine_result):
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


def train_step(model, ddp_model, optimizer, args, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 32), device=device)
    optimizer.zero_grad(set_to_none=True)
    output = ddp_model(input_ids=input_ids, labels=input_ids)
    loss = output.loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "seconds": time.perf_counter() - start,
    }


def connect_update_group(client, args):
    torch.cuda.set_device(0)
    group_name = f"j06-update-{os.getpid()}"
    start = time.perf_counter()
    print(f"connect_update_group: submitting engine join group_name={group_name}", flush=True)
    future = async_utils.submit(
        client.init_weights_update_group(
            "127.0.0.1",
            args.update_port,
            1,
            2,
            group_name,
            backend="nccl",
        )
    )
    print("connect_update_group: waiting actor rank0 NCCL rendezvous", flush=True)
    group = init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{args.update_port}",
        world_size=2,
        rank=0,
        group_name=group_name,
        timeout=timedelta(seconds=args.pg_timeout),
    )
    print("connect_update_group: actor rank0 NCCL rendezvous complete", flush=True)
    future.result(timeout=args.pg_timeout)
    print("connect_update_group: engine join response complete", flush=True)
    return group, group_name, time.perf_counter() - start


def update_weights(model, client, group, group_name, version, args, inject_failure):
    torch.cuda.set_device(0)
    named_tensors = list(model.state_dict().items())
    names = [name for name, _ in named_tensors]
    dtypes = [str(tensor.dtype).replace("torch.", "") for _, tensor in named_tensors]
    shapes = [list(tensor.shape) for _, tensor in named_tensors]
    transfer_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)

    timings = {"transfer_bytes": transfer_bytes}
    update_start = time.perf_counter()
    print(f"update_weights: pause begin version={version}", flush=True)
    run_api(client.pause_generation(mode="in_place"), args.pg_timeout)
    print("update_weights: pause complete", flush=True)
    timings["pause_seconds"] = time.perf_counter() - update_start

    begin_start = time.perf_counter()
    print("update_weights: begin_weight_update", flush=True)
    run_api(client.begin_weight_update(selector="all", sync_base=True), args.pg_timeout)
    print("update_weights: begin_weight_update complete", flush=True)
    timings["begin_seconds"] = time.perf_counter() - begin_start

    broadcast_start = time.perf_counter()
    print("update_weights: submitting engine receive request", flush=True)
    future = async_utils.submit(
        client.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=False,
            weight_version=str(version),
            selector="all",
        )
    )
    print("update_weights: broadcasting tensors", flush=True)
    handles = []
    for _, tensor in named_tensors:
        tensor = tensor.detach()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()
    torch.cuda.synchronize()
    print("update_weights: broadcast handles complete", flush=True)
    future.result(timeout=args.pg_timeout)
    print("update_weights: engine receive response complete", flush=True)
    timings["broadcast_seconds"] = time.perf_counter() - broadcast_start

    end_start = time.perf_counter()
    failed = False
    failure_type = None
    failure_message = None
    try:
        expected = {"bogus-adapter": {}} if inject_failure else None
        print(f"update_weights: end_weight_update inject_failure={inject_failure}", flush=True)
        run_api(client.end_weight_update(expected_lora_checksums=expected), args.pg_timeout)
        print("update_weights: end_weight_update complete", flush=True)
    except Exception as exc:
        failed = True
        failure_type = type(exc).__name__
        failure_message = str(exc)[:1000]
    timings["end_seconds"] = time.perf_counter() - end_start

    version_start = time.perf_counter()
    print("update_weights: publishing version", flush=True)
    run_api(client.update_weight_version(weight_version=str(version), abort_all_requests=False), args.pg_timeout)
    print("update_weights: version published", flush=True)
    timings["version_seconds"] = time.perf_counter() - version_start

    resume_start = time.perf_counter()
    print("update_weights: resuming generation", flush=True)
    run_api(client.continue_generation(), args.pg_timeout)
    print("update_weights: generation resumed", flush=True)
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


async def generate_request(client, prompt, request_id, args):
    submitted = time.perf_counter()
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": False,
        },
    }
    try:
        response = await client.post("/generate", json=payload, timeout=args.request_timeout)
        response.raise_for_status()
        data = response.json()
        meta = data.get("meta_info", {})
        completed = time.perf_counter()
        return {
            "request_id": request_id,
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
        }
    except Exception as exc:
        completed = time.perf_counter()
        return {
            "request_id": request_id,
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
        }


def classify_request(request, update):
    if request["completed_monotonic"] < update["start_monotonic"]:
        return "before_update"
    if request["submitted_monotonic"] > update["end_monotonic"]:
        return "after_update"
    return "during_update"


def timing_stats(values):
    if not values:
        return {"count": 0, "mean_seconds": None, "min_seconds": None, "max_seconds": None}
    return {
        "count": len(values),
        "mean_seconds": sum(values) / len(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def request_version_count(request):
    return len({span.get("version") for span in request.get("weight_versions", [])})


def build_summary(rank, local_rank, cycles, connect_seconds):
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
            "failure_type": cycle["update"]["failure_type"],
            "failure_message": cycle["update"]["failure_message"],
        }
        for cycle in cycles
        if cycle["update"]["failed"]
    ]
    phase_counts = {
        phase: sum(result["phase"] == phase for result in requests)
        for phase in ("before_update", "during_update", "after_update")
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
        key: timing_stats([cycle["update"]["timings"][key] for cycle in cycles])
        for key in update_timing_keys
    }
    failed_cycle = next((cycle["cycle"] for cycle in cycles if cycle["update"]["failed"]), None)
    recovery_successful = None
    if failed_cycle is not None and failed_cycle < len(cycles):
        recovery_successful = not cycles[failed_cycle]["update"]["failed"]
    return {
        "rank": rank,
        "local_rank": local_rank,
        "cycles": cycles,
        "connect_update_group_seconds": connect_seconds,
        "real_weight_update_events": len(cycles),
        "successful_updates": sum(not cycle["update"]["failed"] for cycle in cycles),
        "failed_updates": sum(cycle["update"]["failed"] for cycle in cycles),
        "failed_update_cycle": failed_cycle,
        "recovery_successful": recovery_successful,
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
            {cycle["checksum"]["actor_overall_checksum"] for cycle in cycles}
        ),
        "unique_engine_overall_checksums": len(
            {cycle["checksum"]["engine_overall_checksum"] for cycle in cycles}
        ),
        "unique_train_losses": len({cycle["train"]["loss"] for cycle in cycles}),
        "checksum_mismatch_cycles": sum(cycle["checksum"]["mismatch_count"] > 0 for cycle in cycles),
        "all_engine_tensors_checked_cycles": sum(cycle["checksum"]["all_engine_tensors_checked"] for cycle in cycles),
        "expected_engine_checksum_match_cycles": sum(
            cycle["checksum"]["expected_engine_overall_checksum"]
            == cycle["checksum"]["engine_overall_checksum"]
            for cycle in cycles
        ),
        "update_failures": update_failures,
        "gpu_phase_timings": {
            "train_seconds": timing_stats([cycle["train"]["seconds"] for cycle in cycles]),
            "update_phase_seconds": update_phase_seconds,
            "transfer_bytes_per_update": [
                cycle["update"]["timings"]["transfer_bytes"] for cycle in cycles
            ],
        },
    }


async def main_async(args, rank, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.train()
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-6)
    client = SGLangApiClient(server_url=args.server_url)

    dist.barrier()

    group = None
    group_name = None
    connect_seconds = None
    if rank == 0:
        group, group_name, connect_seconds = connect_update_group(client, args)
    dist.barrier()

    cycles = []
    previous_weight_norm = None
    for cycle in range(args.cycles):
        train_metrics = train_step(model, ddp_model, optimizer, args, device)
        dist.barrier()
        weight_norm = float(sum(p.detach().float().pow(2).sum() for p in model.parameters()).cpu())
        weight_norm_delta = None if previous_weight_norm is None else abs(weight_norm - previous_weight_norm)
        previous_weight_norm = weight_norm

        generation_results = []
        update_result = None
        if rank == 0:
            prompt_texts = [
                "Explain why request admission must be observable during distributed weight updates.",
                "Summarize the tradeoff between pausing generation and preserving in-flight requests.",
                "Describe how weight versions prevent mixed-policy training data.",
                "Explain a bounded recovery after a failed weight-update finalization.",
                "Summarize why NCCL rendezvous timeouts must be explicit.",
                "Describe how checksums verify a distributed weight transfer.",
                "Explain why request latency should be measured across update windows.",
                "Summarize the value of recording timeout causes rather than only counts.",
                "Explain why pre-update admission is part of rollout observability.",
                "Describe why post-update admission verifies recovery.",
                "Summarize how bounded failures differ from unbounded retries.",
                "Explain why update timing should include every GPU phase.",
            ]
            async with httpx.AsyncClient(base_url=args.server_url, timeout=args.request_timeout) as http_client:
                before_tasks = [
                    asyncio.create_task(generate_request(http_client, prompt, request_id, args))
                    for request_id, prompt in enumerate(prompt_texts[:2])
                ]
                before_results = await asyncio.gather(*before_tasks, return_exceptions=False)

                during_prompts = [
                    prompt_texts[(request_id + 2) % len(prompt_texts)]
                    for request_id in range(args.concurrency)
                ]
                during_tasks = [
                    asyncio.create_task(
                        generate_request(http_client, prompt, 2 + request_id, args)
                    )
                    for request_id, prompt in enumerate(during_prompts)
                ]
                update_result = await asyncio.to_thread(
                    update_weights,
                    model,
                    client,
                    group,
                    group_name,
                    cycle + 1,
                    args,
                    inject_failure=(cycle + 1 == args.fail_cycle),
                )
                during_results = await asyncio.gather(*during_tasks, return_exceptions=False)

                after_request_id = 2 + args.concurrency
                after_tasks = [
                    asyncio.create_task(
                        generate_request(http_client, prompt, after_request_id + request_id, args)
                    )
                    for request_id, prompt in enumerate(prompt_texts[2:4])
                ]
                after_results = await asyncio.gather(*after_tasks, return_exceptions=False)
                generation_results = before_results + during_results + after_results
            for result in generation_results:
                result["phase"] = classify_request(result, update_result)
            engine_version = run_api(client.get_weight_version(), args.pg_timeout)
            checksum_result = run_api(client.check_weights(action="checksum"), args.pg_timeout)
            checksum_comparison = compare_checksums(model, checksum_result)
            version_consistent = str(engine_version) == str(update_result["version"])
            cycle_record = {
                "cycle": cycle + 1,
                "train": train_metrics,
                "weight_norm": weight_norm,
                "weight_norm_delta": weight_norm_delta,
                "update": update_result,
                "engine_weight_version": engine_version,
                "version_consistent": version_consistent,
                "checksum": checksum_comparison,
                "requests": generation_results,
            }
            cycles.append(cycle_record)
            print(json.dumps(cycle_record, sort_keys=True), flush=True)
        else:
            cycle_record = {
                "cycle": cycle + 1,
                "train": train_metrics,
                "weight_norm": weight_norm,
                "weight_norm_delta": weight_norm_delta,
            }
            cycles.append(cycle_record)
        dist.barrier()

    if rank == 0:
        run_api(client.destroy_weights_update_group(group_name), args.pg_timeout)
        dist.destroy_process_group(group)
        summary = build_summary(rank, local_rank, cycles, connect_seconds)
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def main():
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
