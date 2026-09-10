import argparse
import asyncio
import json
import statistics
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import sglang
import torch
import transformers

from miles.rollout import sglang_rollout
from miles.rollout.base_types import Sample
from miles.utils.http_utils import init_http_client


def build_args(settings: argparse.Namespace) -> Namespace:
    return Namespace(
        hf_checkpoint=settings.model_path,
        chat_template_path=None,
        rollout_temperature=settings.temperature,
        rollout_top_p=settings.top_p,
        rollout_top_k=settings.top_k,
        rollout_max_response_len=settings.max_new_tokens,
        rollout_stop=[],
        rollout_stop_token_ids=[],
        rollout_skip_special_tokens=True,
        rollout_seed=settings.seed,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=settings.groups * settings.samples_per_group,
        sglang_dp_size=1,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip=settings.host,
        sglang_router_port=settings.port,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
        custom_generate_function_path=None,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        num_layers=0,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        use_distributed_post=False,
        ci_test=False,
    )


def make_samples(groups: int, samples_per_group: int, prompts: list[list[str]]) -> list[list[Sample]]:
    samples = []
    for group_index in range(groups):
        group = []
        for sample_index in range(samples_per_group):
            group.append(
                Sample(
                    group_index=group_index,
                    index=group_index * samples_per_group + sample_index,
                    rollout_id=group_index,
                    prompt=prompts[group_index][sample_index],
                    reward=0.0,
                )
            )
        samples.append(group)
    return samples


def make_prompts(groups: int, samples_per_group: int, run_index: int) -> list[list[str]]:
    return [
        [
            f"Rollout grouping probe {run_index}/{group_index}/{sample_index}: reply briefly."
            for sample_index in range(samples_per_group)
        ]
        for group_index in range(groups)
    ]


def summarize_samples(samples: list[Sample]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "status_counts": dict(Counter(sample.status.value for sample in samples)),
        "response_lengths": [sample.response_length for sample in samples],
        "total_output_tokens": sum(sample.response_length for sample in samples),
    }


async def run_mode(
    args: Namespace,
    prompts: list[list[str]],
    settings: argparse.Namespace,
    mode: str,
) -> tuple[list[Sample], dict[str, Any]]:
    state = sglang_rollout.GenerateState(args)
    state.reset()
    state.dp_counts = [0] * (args.sglang_dp_size or 1)
    groups = make_samples(settings.groups, settings.samples_per_group, prompts)
    flattened = [sample for group in groups for sample in group]
    object_ids = [id(sample) for sample in flattened]
    request_count = 0
    original_post = sglang_rollout.post
    original_can_batch = sglang_rollout._can_batch_generate_group

    async def counted_post(*post_args: Any, **post_kwargs: Any) -> Any:
        nonlocal request_count
        request_count += 1
        return await original_post(*post_args, **post_kwargs)

    sglang_rollout.post = counted_post
    if mode == "individual":
        sglang_rollout._can_batch_generate_group = lambda *_args, **_kwargs: False

    sampling_params = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "max_new_tokens": settings.max_new_tokens,
        "stop": [],
        "stop_token_ids": [],
        "skip_special_tokens": True,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    try:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                sglang_rollout.generate_and_rm_group(
                    args,
                    group,
                    sampling_params.copy(),
                    evaluation=True,
                )
                for group in groups
            )
        )
        latency_s = time.perf_counter() - started
    finally:
        sglang_rollout.post = original_post
        sglang_rollout._can_batch_generate_group = original_can_batch

    result_samples = [sample for group in results for sample in group]
    identity_preserved = [id(sample) for sample in result_samples] == object_ids
    summary = {
        "mode": mode,
        "latency_s": latency_s,
        "requests": request_count,
        "identity_preserved": identity_preserved,
        **summarize_samples(result_samples),
    }
    return result_samples, summary


def compare_modes(individual: list[Sample], grouped: list[Sample], settings: argparse.Namespace) -> dict[str, Any]:
    assert len(individual) == len(grouped)
    token_differences = []
    response_differences = []
    logprob_differences = []
    for left, right in zip(individual, grouped, strict=True):
        assert left.index == right.index
        assert left.group_index == right.group_index
        token_differences.append(0 if left.tokens == right.tokens else 1)
        response_differences.append(0 if left.response == right.response else 1)
        assert len(left.rollout_log_probs or []) == len(right.rollout_log_probs or [])
        logprob_differences.extend(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left.rollout_log_probs or [], right.rollout_log_probs or [], strict=True
            )
        )

    max_logprob_difference = max(logprob_differences, default=0.0)
    mean_logprob_difference = statistics.fmean(logprob_differences) if logprob_differences else 0.0
    return {
        "sample_identity_by_index": all(
            left.index == right.index for left, right in zip(individual, grouped, strict=True)
        ),
        "tokens_aligned": not any(token_differences),
        "token_mismatch_count": sum(token_differences),
        "response_mismatch_count": sum(response_differences),
        "responses_aligned": all(
            left.response == right.response for left, right in zip(individual, grouped, strict=True)
        ),
        "statuses_aligned": all(left.status == right.status for left, right in zip(individual, grouped, strict=True)),
        "response_lengths_aligned": all(
            left.response_length == right.response_length for left, right in zip(individual, grouped, strict=True)
        ),
        "logprobs_aligned": max_logprob_difference <= settings.logprob_tolerance,
        "max_abs_logprob_difference": max_logprob_difference,
        "mean_abs_logprob_difference": mean_logprob_difference,
        "response_cap_respected": all(
            sample.response_length <= settings.max_new_tokens for sample in individual + grouped
        ),
        "truncated_samples_reach_cap": all(
            sample.response_length == settings.max_new_tokens
            for sample in individual + grouped
            if sample.status == Sample.Status.TRUNCATED
        ),
    }


async def run_probe(settings: argparse.Namespace) -> dict[str, Any]:
    args = build_args(settings)
    init_http_client(args)
    state = sglang_rollout.GenerateState(args)
    state.reset()

    for warmup_index in range(settings.warmups):
        warmup_settings = argparse.Namespace(**{**vars(settings), "groups": 1})
        prompts = make_prompts(1, warmup_settings.samples_per_group, warmup_index)
        await run_mode(args, prompts, warmup_settings, "individual")
        await run_mode(args, prompts, warmup_settings, "grouped")

    runs = []
    for run_index in range(settings.runs):
        prompts = make_prompts(settings.groups, settings.samples_per_group, run_index)
        individual, individual_summary = await run_mode(args, prompts, settings, "individual")
        grouped, grouped_summary = await run_mode(args, prompts, settings, "grouped")
        runs.append(
            {
                "run": run_index,
                "individual": individual_summary,
                "grouped": grouped_summary,
                "parity": compare_modes(individual, grouped, settings),
            }
        )

    individual_latencies = [run["individual"]["latency_s"] for run in runs]
    grouped_latencies = [run["grouped"]["latency_s"] for run in runs]
    return {
        "schema_version": 1,
        "settings": {
            "model_path": settings.model_path,
            "host": settings.host,
            "port": settings.port,
            "groups": settings.groups,
            "samples_per_group": settings.samples_per_group,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "seed": settings.seed,
            "runs": settings.runs,
            "warmups": settings.warmups,
            "logprob_tolerance": settings.logprob_tolerance,
        },
        "imports": {
            "torch": {"version": torch.__version__, "path": torch.__file__},
            "sglang": {"version": sglang.__version__, "path": sglang.__file__},
            "transformers": {"version": transformers.__version__, "path": transformers.__file__},
        },
        "runs": runs,
        "summary": {
            "individual_latency_median_s": statistics.median(individual_latencies),
            "grouped_latency_median_s": statistics.median(grouped_latencies),
            "latency_speedup_median": statistics.median(individual_latencies) / statistics.median(grouped_latencies),
            "individual_requests_total": sum(run["individual"]["requests"] for run in runs),
            "grouped_requests_total": sum(run["grouped"]["requests"] for run in runs),
            "all_runs_functional_parity": all(
                all(value for key, value in run["parity"].items() if key != "max_abs_logprob_difference")
                for run in runs
            ),
        },
    }


def parse_settings() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare individual and prompt-group-batched Miles SGLang rollout.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1567)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    settings = parse_settings()
    result = asyncio.run(run_probe(settings))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if settings.output:
        settings.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
