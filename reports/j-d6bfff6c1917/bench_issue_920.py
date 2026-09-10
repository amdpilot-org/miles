#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.types import Sample


@dataclass
class RequestRecord:
    mode: str
    wave: int
    request_index: int
    path: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    queue_time: float
    e2e_latency: float
    compute_time: float
    prefill_time: float
    decode_time: float
    output_ids: list[int]
    content: str | None
    token_correct: bool
    token_count_correct: bool
    finish_reason: str


def build_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        hf_checkpoint=args.model_path,
        chat_template_path=None,
        sglang_router_ip=args.sglang_host,
        sglang_router_port=args.sglang_port,
        sglang_model_routers=None,
        sglang_router_policy="random",
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=64,
        rollout_max_context_len=args.context_length,
        rollout_max_response_len=args.short_max_tokens,
        rollout_temperature=0.0,
        rollout_top_p=1.0,
        rollout_top_k=1,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        reward_key=None,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        custom_generate_function_path=None,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        use_distributed_post=False,
        rollout_stop=[],
        rollout_stop_token_ids=[],
        rollout_skip_special_tokens=True,
    )


def phase_times(meta: dict[str, Any]) -> tuple[float, float]:
    forward_entry = float(meta.get("forward_entry_time") or 0.0)
    prefill_finished = float(meta.get("prefill_finished_time") or 0.0)
    request_finished = float(meta.get("request_finished_ts") or 0.0)
    prefill_time = max(0.0, prefill_finished - forward_entry)
    decode_time = max(0.0, request_finished - prefill_finished)
    return prefill_time, decode_time


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("count", "mean", "p50", "p90", "p95", "p99", "min", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(records: list[RequestRecord]) -> dict[str, Any]:
    return {
        "requests": len(records),
        "token_correct": sum(record.token_correct for record in records),
        "token_count_correct": sum(record.token_count_correct for record in records),
        "latency": {
            name: summarize([getattr(record, name) for record in records])
            for name in ("queue_time", "e2e_latency", "compute_time", "prefill_time", "decode_time")
        },
    }


async def direct_prefill(
    client: httpx.AsyncClient,
    state: GenerateState,
    args: argparse.Namespace,
    input_ids: list[int],
    expected_ids: list[int],
    mode: str,
    wave: int,
    request_index: int,
) -> RequestRecord:
    sample = Sample(prompt="", index=request_index)
    payload, halt_status = compute_request_payload(
        state.args,
        input_ids=input_ids,
        sampling_params={"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True},
    )
    if payload is None:
        raise RuntimeError(f"prefill payload was truncated: {halt_status}")
    response = await client.post(f"http://{args.sglang_host}:{args.sglang_port}/generate", json=payload)
    response.raise_for_status()
    output = response.json()
    await update_sample_from_response(state.args, sample, payload=payload, output=output)
    meta = output["meta_info"]
    prefill_time, decode_time = phase_times(meta)
    output_ids = list(output["output_ids"])
    queue_time = float(meta["queue_time"])
    e2e_latency = float(meta["e2e_latency"])
    return RequestRecord(
        mode=mode,
        wave=wave,
        request_index=request_index,
        path="miles_rollout_generate",
        prompt_tokens=int(meta["prompt_tokens"]),
        completion_tokens=int(meta["completion_tokens"]),
        cached_tokens=int(meta.get("cached_tokens", 0)),
        queue_time=queue_time,
        e2e_latency=e2e_latency,
        compute_time=max(0.0, e2e_latency - queue_time),
        prefill_time=prefill_time,
        decode_time=decode_time,
        output_ids=output_ids,
        content=None,
        token_correct=output_ids == expected_ids,
        token_count_correct=int(meta["completion_tokens"]) == 1,
        finish_reason=str(meta["finish_reason"]["type"]),
    )


async def session_decode(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    expected_content: str,
    expected_ids: list[int],
    expected_completion_tokens: int,
    mode: str,
    wave: int,
    request_index: int,
) -> RequestRecord:
    session_response = await client.post(f"http://{args.session_host}:{args.session_port}/sessions")
    session_response.raise_for_status()
    session_id = session_response.json()["session_id"]
    payload = {
        "model": "Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Say exactly: PASS"}],
        "max_tokens": args.short_max_tokens,
        "temperature": 0.0,
    }
    response = await client.post(
        f"http://{args.session_host}:{args.session_port}/sessions/{session_id}/v1/chat/completions",
        json=payload,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    meta = choice["meta_info"]
    prefill_time, decode_time = phase_times(meta)
    output_ids = [int(row[1]) for row in meta.get("output_token_logprobs", [])]
    content = choice["message"]["content"]
    queue_time = float(meta["queue_time"])
    e2e_latency = float(meta["e2e_latency"])
    return RequestRecord(
        mode=mode,
        wave=wave,
        request_index=request_index,
        path="miles_session_chat",
        prompt_tokens=int(meta["prompt_tokens"]),
        completion_tokens=int(meta["completion_tokens"]),
        cached_tokens=int(meta.get("cached_tokens", 0)),
        queue_time=queue_time,
        e2e_latency=e2e_latency,
        compute_time=max(0.0, e2e_latency - queue_time),
        prefill_time=prefill_time,
        decode_time=decode_time,
        output_ids=output_ids,
        content=content,
        token_correct=content == expected_content and output_ids == expected_ids,
        token_count_correct=int(meta["completion_tokens"]) == expected_completion_tokens,
        finish_reason=str(meta["finish_reason"]["type"]),
    )


async def run_waves(
    client: httpx.AsyncClient,
    state: GenerateState,
    args: argparse.Namespace,
    input_ids: list[int],
    expected_prefill_ids: list[int],
    expected_session: tuple[str, list[int], int],
) -> tuple[list[RequestRecord], list[dict[str, Any]]]:
    records: list[RequestRecord] = []
    wave_summaries: list[dict[str, Any]] = []
    for wave in range(args.waves):
        for mode in ("prefill_only", "decode_only", "concurrent"):
            started = time.perf_counter()
            if mode == "prefill_only":
                wave_records = [
                    await direct_prefill(
                        client, state, args, input_ids, expected_prefill_ids, mode, wave, 0
                    )
                ]
            elif mode == "decode_only":
                wave_records = await asyncio.gather(
                    *(
                        session_decode(client, args, *expected_session, mode, wave, index)
                        for index in range(args.short_decodes_per_wave)
                    )
                )
            else:
                prefill = direct_prefill(
                    client, state, args, input_ids, expected_prefill_ids, mode, wave, 0
                )
                decodes = [
                    session_decode(client, args, *expected_session, mode, wave, index + 1)
                    for index in range(args.short_decodes_per_wave)
                ]
                wave_records = list(await asyncio.gather(prefill, *decodes))
            records.extend(wave_records)
            wave_summaries.append(
                {
                    "mode": mode,
                    "wave": wave,
                    "wall_time": time.perf_counter() - started,
                    "requests": len(wave_records),
                }
            )
            print(
                f"{mode} wave {wave + 1}/{args.waves}: "
                f"{wave_summaries[-1]['wall_time']:.3f}s",
                flush=True,
            )
    return records, wave_summaries


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    state = GenerateState(build_args(args))
    long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 2000
    input_ids = state.tokenizer.encode(long_text, add_special_tokens=False)[: args.long_context_tokens]
    if len(input_ids) != args.long_context_tokens:
        raise RuntimeError(f"encoded only {len(input_ids)} long-context tokens")
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.short_decodes_per_wave + 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        preflight = await direct_prefill(
            client, state, args, input_ids, [], "warmup", -1, 0
        )
        expected_prefill_ids = preflight.output_ids
        session_warmup = await session_decode(
            client, args, "", [], 0, "warmup", -1, 0
        )
        expected_session = (
            session_warmup.content or "",
            session_warmup.output_ids,
            session_warmup.completion_tokens,
        )
        records, wave_summaries = await run_waves(
            client,
            state,
            args,
            input_ids,
            expected_prefill_ids,
            expected_session,
        )
    result = {
        "metadata": {
            "model_path": args.model_path,
            "context_length": args.context_length,
            "long_context_tokens": args.long_context_tokens,
            "waves": args.waves,
            "short_decodes_per_wave": args.short_decodes_per_wave,
            "short_max_tokens": args.short_max_tokens,
            "expected_prefill_ids": expected_prefill_ids,
            "expected_session": expected_session,
        },
        "aggregate": {
            mode: aggregate([record for record in records if record.mode == mode])
            for mode in ("prefill_only", "decode_only", "concurrent")
        },
        "path_aggregate": {
            f"{mode}:{path}": aggregate(
                [
                    record
                    for record in records
                    if record.mode == mode and record.path == path
                ]
            )
            for mode in ("prefill_only", "decode_only", "concurrent")
            for path in ("miles_rollout_generate", "miles_session_chat")
        },
        "wave_summaries": wave_summaries,
        "records": [asdict(record) for record in records],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--sglang-port", type=int, default=31111)
    parser.add_argument("--session-host", default="127.0.0.1")
    parser.add_argument("--session-port", type=int, default=31113)
    parser.add_argument("--model-path", default="/job/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--long-context-tokens", type=int, default=16384)
    parser.add_argument("--waves", type=int, default=32)
    parser.add_argument("--short-decodes-per-wave", type=int, default=8)
    parser.add_argument("--short-max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default="/job/artifacts/issue-920/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = asyncio.run(async_main(parsed))
    for mode, values in result["aggregate"].items():
        print(mode, json.dumps(values["latency"], sort_keys=True))
