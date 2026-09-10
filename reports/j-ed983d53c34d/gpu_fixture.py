from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch import nn

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import Sample
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state


CASES = ("no_abort", "group_drop", "sample_refill", "survivor_keep", "remove_sample")
GROUP_SIZES = (2, 3, 4, 8)
STEPS = 128
MAX_GROUP_ATTEMPTS = 4
SAMPLE_RETRY_CAP = 2
VOCAB_SIZE = 32
HIDDEN_SIZE = 16
LEARNING_RATE = 0.01
PROCESS_GROUP_TIMEOUT_SECONDS = 30


def _module_record(name: str) -> dict[str, Any]:
    module = importlib.import_module(name)
    return {
        "name": name,
        "file": getattr(module, "__file__", None),
        "paths": list(getattr(module, "__path__", [])),
    }


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def _abort_probability(response_length: int) -> float:
    return 0.01 + 0.01 * response_length


def _make_attempt(
    rng: random.Random,
    *,
    case: str,
    step: int,
    group_ordinal: int,
    slot: int,
    attempt_number: int,
    source: str,
) -> tuple[Sample, dict[str, Any]]:
    response_length = rng.randint(2, 8)
    true_reward = 0.2 + 0.12 * response_length + rng.uniform(-0.08, 0.08)
    probability = _abort_probability(response_length)
    aborted = case != "no_abort" and rng.random() < probability
    zero_loss = case == "remove_sample" and aborted
    prompt = f"prompt-{step}-{group_ordinal}"
    tokens = [1, 2] + [
        3 + ((step + group_ordinal + slot + attempt_number + position) * 7) % (VOCAB_SIZE - 3)
        for position in range(response_length)
    ]
    sample = Sample(
        group_index=step * 100 + group_ordinal,
        index=step * 10_000 + group_ordinal * 100 + slot,
        rollout_id=step * 1_000_000 + group_ordinal * 10_000 + slot * 100 + attempt_number,
        prompt=prompt,
        tokens=tokens,
        response=f"response-{response_length}",
        response_length=response_length,
        label="synthetic",
        reward=0.0 if zero_loss else true_reward,
        loss_mask=[0 if zero_loss else 1 for _ in range(response_length)],
        remove_sample=zero_loss,
        status=Sample.Status.ABORTED if aborted else Sample.Status.COMPLETED,
    )
    identity = {
        "case": case,
        "step": step,
        "group_ordinal": group_ordinal,
        "slot": slot,
        "attempt_number": attempt_number,
        "source": source,
        "sample_index": sample.index,
        "rollout_id": sample.rollout_id,
        "response_length": response_length,
        "true_reward": true_reward,
        "abort_probability": probability,
        "aborted": aborted,
        "zero_loss": zero_loss,
    }
    sample.metadata = identity
    sample.train_metadata = identity
    return sample, identity


def _record_attempts(
    records: list[dict[str, Any]],
    samples: list[Sample],
    identities: list[dict[str, Any]],
    *,
    retained: bool,
) -> None:
    for sample, identity in zip(samples, identities, strict=True):
        record = dict(identity)
        record["retained"] = retained
        record["status"] = sample.status.value
        record["remove_sample"] = sample.remove_sample
        records.append(record)


def _generate_group(
    rng: random.Random,
    *,
    case: str,
    step: int,
    group_ordinal: int,
    group_size: int,
    group_attempt: int,
) -> tuple[list[Sample], list[dict[str, Any]], list[dict[str, Any]]]:
    source = "initial" if group_attempt == 0 else "group_retry"
    samples: list[Sample] = []
    identities: list[dict[str, Any]] = []
    for slot in range(group_size):
        sample, identity = _make_attempt(
            rng,
            case=case,
            step=step,
            group_ordinal=group_ordinal,
            slot=slot,
            attempt_number=group_attempt,
            source=source,
        )
        samples.append(sample)
        identities.append(identity)
    return samples, identities, [identity for identity in identities if identity["aborted"]]


def _refill_samples(
    rng: random.Random,
    *,
    case: str,
    step: int,
    group_ordinal: int,
    group_attempt: int,
    aborted_slots: list[int],
    records: list[dict[str, Any]],
    accounting: Counter[str],
) -> tuple[dict[int, Sample], bool]:
    replacements: dict[int, Sample] = {}
    exhausted = False
    for slot in aborted_slots:
        for retry in range(1, SAMPLE_RETRY_CAP + 1):
            sample, identity = _make_attempt(
                rng,
                case=case,
                step=step,
                group_ordinal=group_ordinal,
                slot=slot,
                attempt_number=group_attempt * 10 + retry,
                source="sample_retry",
            )
            accounting["sample_retry_attempts"] += 1
            _record_attempts(records, [sample], [identity], retained=False)
            if identity["aborted"]:
                accounting["aborted_sample_retry_samples"] += 1
                continue
            accounting["sample_retry_success"] += 1
            replacements[slot] = sample
            break
        if slot not in replacements:
            accounting["sample_retry_cap_exhausted"] += 1
            exhausted = True
    return replacements, exhausted


def _generate_groups(
    rng: random.Random,
    *,
    case: str,
    step: int,
) -> tuple[list[list[Sample]], Counter[str], list[dict[str, Any]]]:
    groups: list[list[Sample]] = []
    accounting: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    group_ordinal = 0
    for target_size in GROUP_SIZES:
        retained_group: list[Sample] | None = None
        for group_attempt in range(MAX_GROUP_ATTEMPTS):
            samples, identities, aborted_identities = _generate_group(
                rng,
                case=case,
                step=step,
                group_ordinal=group_ordinal,
                group_size=target_size,
                group_attempt=group_attempt,
            )
            accounting["attempts"] += len(samples)
            if group_attempt == 0:
                accounting["initial_attempts"] += len(samples)
                accounting["aborted_initial_samples"] += len(aborted_identities)
            else:
                accounting["group_retry_attempts"] += len(samples)
                accounting["aborted_group_retry_samples"] += len(aborted_identities)
            _record_attempts(records, samples, identities, retained=False)
            aborted_slots = [identity["slot"] for identity in aborted_identities]
            if case == "no_abort":
                retained_group = samples
                accounting["retained_original_samples"] += len(samples)
                break
            if case == "group_drop":
                if not aborted_slots:
                    retained_group = samples
                    if group_attempt == 0:
                        accounting["retained_original_samples"] += len(samples)
                    else:
                        accounting["retained_group_retry_samples"] += len(samples)
                    break
                accounting["discarded_groups"] += 1
            elif case == "remove_sample":
                retained_group = samples
                accounting["retained_original_samples"] += len(samples)
                accounting["zeroed_remove_samples"] += len(aborted_slots)
                break
            elif case == "survivor_keep":
                survivors = [sample for sample in samples if sample.status != Sample.Status.ABORTED]
                if len(survivors) >= 2:
                    retained_group = survivors
                    accounting["retained_original_samples"] += len(survivors)
                    accounting["survivor_dropped_samples"] += len(aborted_slots)
                    break
                accounting["discarded_groups"] += 1
            elif case == "sample_refill":
                if not aborted_slots:
                    retained_group = samples
                    if group_attempt == 0:
                        accounting["retained_original_samples"] += len(samples)
                    else:
                        accounting["retained_group_retry_samples"] += len(samples)
                    break
                replacements, exhausted = _refill_samples(
                    rng,
                    case=case,
                    step=step,
                    group_ordinal=group_ordinal,
                    group_attempt=group_attempt,
                    aborted_slots=aborted_slots,
                    records=records,
                    accounting=accounting,
                )
                if not exhausted:
                    retained_group = [
                        replacements.get(slot, sample)
                        for slot, sample in enumerate(samples)
                        if slot not in replacements or replacements[slot].status != Sample.Status.ABORTED
                    ]
                    accounting["retained_original_samples"] += len(samples) - len(replacements)
                    accounting["retained_sample_retry_samples"] += len(replacements)
                    break
                accounting["discarded_groups"] += 1
            group_ordinal += 1
        if retained_group is not None:
            groups.append(retained_group)
            retained_ids = {sample.rollout_id for sample in retained_group}
            for record in records:
                if record["rollout_id"] in retained_ids:
                    record["retained"] = True
        else:
            accounting["unfilled_prompt_groups"] += 1
        group_ordinal += 1
    return groups, accounting, records


def _make_fixture_args() -> Any:
    args = make_args(
        advantage_estimator="grpo",
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        rewards_normalization=True,
        grpo_std_normalization=False,
        use_kl_loss=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_opsm=False,
        use_opd=False,
        use_rollout_logprobs=False,
        skip_actor_forward_only=False,
        normalize_advantages=False,
        calculate_per_token_loss=False,
        qkv_format="thd",
        true_on_policy_mode=True,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        bf16=False,
        fp16=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        dump_details=None,
        custom_pg_loss_reducer_function_path=None,
        custom_tis_function_path=None,
        loss_type="policy_loss",
        global_batch_size=sum(GROUP_SIZES),
        use_dynamic_global_batch_size=False,
        disable_rollout_trim_samples=True,
        reward_key=None,
        n_samples_per_prompt=max(GROUP_SIZES),
        rollout_batch_size=len(GROUP_SIZES),
    )
    return args


def _group_records(groups: list[list[Sample]], train_data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    start = 0
    for group in groups:
        end = start + len(group)
        raw_rewards = train_data["raw_reward"][start:end]
        normalized_rewards = train_data["rewards"][start:end]
        records.append(
            {
                "group_index": group[0].group_index,
                "size": len(group),
                "raw_rewards": raw_rewards,
                "raw_baseline": statistics.fmean(raw_rewards),
                "normalized_rewards": normalized_rewards,
                "normalized_baseline": statistics.fmean(normalized_rewards),
                "sample_indices": train_data["sample_indices"][start:end],
                "rollout_ids": train_data["rollout_ids"][start:end],
            }
        )
        start = end
    return records


def _assert_train_data(groups: list[list[Sample]], train_data: dict[str, Any], case: str) -> None:
    assert train_data["prompt_group_sizes"] == [len(group) for group in groups]
    assert len(set(train_data["rollout_ids"])) == len(train_data["rollout_ids"])
    start = 0
    for group in groups:
        end = start + len(group)
        normalized = train_data["rewards"][start:end]
        assert abs(statistics.fmean(normalized)) < 1e-5
        start = end
    expected_status = (
        {Sample.Status.COMPLETED, Sample.Status.ABORTED}
        if case == "remove_sample"
        else {Sample.Status.COMPLETED}
    )
    assert all(sample.status in expected_status for group in groups for sample in group)
    if case == "remove_sample":
        assert all(
            sample.loss_mask is not None and (any(sample.loss_mask) or sample.remove_sample)
            for group in groups
            for sample in group
        )
    else:
        assert all(any(sample.loss_mask or []) for group in groups for sample in group)


def _run_training_step(
    *,
    args: Any,
    case: str,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conversion_start = time.perf_counter()
    groups, accounting, attempt_records = _generate_groups(rng, case=case, step=step)
    assert groups, f"case={case} step={step} produced no groups"
    flat_samples, metadata = postprocess_rollout_data(args, groups, train_parallel_config={"dp_size": 1})
    metadata["prompt_group_sizes"] = [len(group) for group in groups]
    train_data = convert_samples_to_train_data(
        args,
        flat_samples,
        metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    conversion_ms = (time.perf_counter() - conversion_start) * 1000
    _assert_train_data(groups, train_data, case)

    torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    unconcat_tokens = [torch.tensor(sample.tokens, dtype=torch.long, device=device) for sample in flat_samples]
    total_lengths = [len(sample.tokens) for sample in flat_samples]
    tokens = torch.cat(unconcat_tokens, dim=0)
    full_logits = model(tokens).unsqueeze(0)
    old_log_probs = get_log_probs_and_entropy(
        full_logits.detach(),
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=train_data["response_lengths"],
        with_entropy=False,
        entropy_requires_grad=False,
    )["log_probs"]
    loss_masks = [torch.tensor(mask, dtype=torch.float32, device=device) for mask in train_data["loss_masks"]]
    rollout_mask_sums = torch.tensor(train_data["rollout_mask_sums"], dtype=torch.float32, device=device)
    rollout_data = {
        "log_probs": old_log_probs,
        "rewards": train_data["rewards"],
        "response_lengths": train_data["response_lengths"],
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "max_seq_lens": None,
        "rollout_mask_sums": rollout_mask_sums,
    }
    compute_advantages_and_returns(args, rollout_data)
    sum_of_sample_mean = get_sum_of_sample_mean(
        rollout_data["total_lengths"],
        rollout_data["response_lengths"],
        rollout_data["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        None,
        denominators=rollout_mask_sums,
    )
    batch = {
        "unconcat_tokens": unconcat_tokens,
        "response_lengths": rollout_data["response_lengths"],
        "total_lengths": rollout_data["total_lengths"],
        "loss_masks": rollout_data["loss_masks"],
        "log_probs": old_log_probs,
        "advantages": rollout_data["advantages"],
        "rollout_mask_sums": rollout_mask_sums,
    }
    loss, _ = policy_loss_function(args, batch, full_logits, sum_of_sample_mean)
    assert torch.isfinite(loss)
    loss.backward()
    torch.cuda.synchronize(device)
    forward_backward_ms = (time.perf_counter() - forward_start) * 1000

    collective_start = time.perf_counter()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    flat_gradients = torch.cat([gradient.reshape(-1) for gradient in gradients])
    before_collective = flat_gradients.detach().clone()
    dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    collective_ms = (time.perf_counter() - collective_start) * 1000
    torch.testing.assert_close(flat_gradients, before_collective)

    optimizer_start = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    optimizer_ms = (time.perf_counter() - optimizer_start) * 1000

    advantages = [advantage[0].item() for advantage in rollout_data["advantages"]]
    for advantage, reward in zip(rollout_data["advantages"], train_data["rewards"], strict=True):
        assert advantage.numel() > 0
        torch.testing.assert_close(
            advantage,
            torch.full_like(advantage, reward),
            rtol=0.0,
            atol=1e-6,
        )
    sample_loss_weights = [
        mask_sum / max(rollout_mask_sum, 1)
        for mask_sum, rollout_mask_sum in zip(
            [sum(mask) for mask in train_data["loss_masks"]],
            train_data["rollout_mask_sums"],
            strict=True,
        )
    ]
    result = {
        "step": step,
        "case": case,
        "group_sizes": [len(group) for group in groups],
        "retained_samples": len(flat_samples),
        "accounting": dict(accounting),
        "groups": _group_records(groups, train_data),
        "sample_indices": train_data["sample_indices"],
        "rollout_ids": train_data["rollout_ids"],
        "raw_rewards": train_data["raw_reward"],
        "normalized_rewards": train_data["rewards"],
        "advantages": advantages,
        "rollout_mask_sums": train_data["rollout_mask_sums"],
        "sample_loss_weights": sample_loss_weights,
        "loss": loss.item(),
        "gradient_norm": flat_gradients.norm().item(),
        "timings_ms": {
            "rollout_conversion": conversion_ms,
            "forward_backward": forward_backward_ms,
            "collective": collective_ms,
            "optimizer": optimizer_ms,
        },
    }
    return result, attempt_records


def _summarize_attempts(records: list[dict[str, Any]]) -> dict[str, Any]:
    retained = [record for record in records if record["retained"]]
    aborted = [record for record in records if record["aborted"]]
    lengths = sorted({record["response_length"] for record in records})
    abort_rate_by_length = {
        length: (
            sum(record["aborted"] for record in records if record["response_length"] == length)
            / sum(record["response_length"] == length for record in records)
        )
        for length in lengths
    }
    mean_length = statistics.fmean(record["response_length"] for record in records)
    mean_abort_length = statistics.fmean(record["response_length"] for record in aborted) if aborted else None
    mean_retained_length = statistics.fmean(record["response_length"] for record in retained)
    slope_numerator = sum(
        (record["response_length"] - mean_length) * (float(record["aborted"]) - (len(aborted) / len(records)))
        for record in records
    )
    slope_denominator = sum((record["response_length"] - mean_length) ** 2 for record in records)
    return {
        "attempts": len(records),
        "aborts": len(aborted),
        "abort_rate": len(aborted) / len(records),
        "abort_rate_by_response_length": abort_rate_by_length,
        "abort_probability_slope_per_response_length": slope_numerator / slope_denominator,
        "mean_attempt_response_length": mean_length,
        "mean_aborted_response_length": mean_abort_length,
        "mean_retained_response_length": mean_retained_length,
        "retained_length_bias_vs_all_attempts": mean_retained_length - mean_length,
        "mean_attempt_true_reward": statistics.fmean(record["true_reward"] for record in records),
        "mean_retained_true_reward": statistics.fmean(record["true_reward"] for record in retained),
        "retained_reward_bias_vs_all_attempts": statistics.fmean(record["true_reward"] for record in retained)
        - statistics.fmean(record["true_reward"] for record in records),
    }


def _parameter_norm(model: nn.Module) -> float:
    return sum(parameter.detach().square().sum().item() for parameter in model.parameters())


def _run_case(
    *,
    case: str,
    args: Any,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)).to(device)
    model.load_state_dict(initial_state)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    rng = random.Random(seed)
    step_results: list[dict[str, Any]] = []
    attempt_records: list[dict[str, Any]] = []
    initial_norm = _parameter_norm(model)
    parameter_norms: dict[int, float] = {0: initial_norm}
    for step in range(STEPS):
        result, records = _run_training_step(
            args=args,
            case=case,
            device=device,
            model=model,
            optimizer=optimizer,
            step=step,
            rng=rng,
        )
        step_results.append(result)
        attempt_records.extend(records)
        if step in {0, 32, 64, 96, 127}:
            parameter_norms[step + 1] = _parameter_norm(model)
    final_norm = parameter_norms[128]
    assert abs(final_norm - initial_norm) > 1e-8
    timing_keys = ("rollout_conversion", "forward_backward", "collective", "optimizer")
    timings = {
        key: {
            "mean": statistics.fmean(result["timings_ms"][key] for result in step_results),
            "min": min(result["timings_ms"][key] for result in step_results),
            "max": max(result["timings_ms"][key] for result in step_results),
        }
        for key in timing_keys
    }
    accounting_totals = Counter()
    for result in step_results:
        accounting_totals.update(result["accounting"])
    return {
        "case": case,
        "seed": seed,
        "optimizer_steps": STEPS,
        "parameter_norms": parameter_norms,
        "initial_parameter_norm": initial_norm,
        "final_parameter_norm": final_norm,
        "parameter_norm_drift": final_norm - initial_norm,
        "accounting": dict(accounting_totals),
        "attempt_bias": _summarize_attempts(attempt_records),
        "mean_loss": statistics.fmean(result["loss"] for result in step_results),
        "mean_gradient_norm": statistics.fmean(result["gradient_norm"] for result in step_results),
        "timings_ms": timings,
        "steps": step_results,
    }


def _run_fixture(output: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this fixture requires exactly one assigned CUDA/ROCm GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous_path = Path("/tmp") / f"miles-j-ed983d53c34d-{uuid.uuid4().hex}"
    rendezvous_path.parent.mkdir(parents=True, exist_ok=True)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        world_size=1,
        rank=0,
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    try:
        make_parallel_state(is_pp_last_stage=True)
        args = _make_fixture_args()
        torch.manual_seed(2800)
        initial_model = nn.Sequential(nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, VOCAB_SIZE))
        initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
        results = [
            _run_case(
                case=case,
                args=args,
                device=device,
                initial_state=initial_state,
                seed=2800 + case_index,
            )
            for case_index, case in enumerate(CASES)
        ]
        baseline = results[0]
        for result in results[1:]:
            result["baseline_comparison"] = {
                "mean_loss_delta": result["mean_loss"] - baseline["mean_loss"],
                "mean_gradient_norm_delta": result["mean_gradient_norm"] - baseline["mean_gradient_norm"],
                "retained_length_bias_delta": result["attempt_bias"]["retained_length_bias_vs_all_attempts"]
                - baseline["attempt_bias"]["retained_length_bias_vs_all_attempts"],
                "retained_reward_bias_delta": result["attempt_bias"]["retained_reward_bias_vs_all_attempts"]
                - baseline["attempt_bias"]["retained_reward_bias_vs_all_attempts"],
            }
        report = {
            "status": "passed",
            "runtime": {
                "python": str(__import__("sys").version),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "gpu": {
                    "device": str(device),
                    "name": torch.cuda.get_device_name(device),
                    "capability": torch.cuda.get_device_capability(device),
                    "count": torch.cuda.device_count(),
                },
                "distributed": {
                    "backend": dist.get_backend(),
                    "world_size": dist.get_world_size(),
                    "rendezvous": str(rendezvous_path),
                    "timeout_seconds": PROCESS_GROUP_TIMEOUT_SECONDS,
                },
                "downloads_bytes": 0,
                "model": "synthetic local random initialization",
            },
            "commits": {
                "base": _git_output("rev-parse", "main"),
                "tested": _git_output("rev-parse", "HEAD"),
                "branch": _git_output("branch", "--show-current"),
                "dirty": bool(_git_output("status", "--porcelain")),
            },
            "imports": {
                "miles": _module_record("miles"),
                "miles_rollout_conversion": _module_record("miles.ray.rollout.rollout_data_conversion"),
                "miles_train_data_conversion": _module_record("miles.ray.rollout.train_data_conversion"),
                "miles_loss": _module_record("miles.backends.training_utils.loss"),
                "miles_policy_loss": _module_record("miles.backends.training_utils.loss_hub.losses"),
                "miles_cp_utils": _module_record("miles.backends.training_utils.cp_utils"),
                "torch": _module_record("torch"),
                "torch_native": torch._C.__file__,
            },
            "configuration": {
                "cases": list(CASES),
                "group_sizes": list(GROUP_SIZES),
                "steps_per_case": STEPS,
                "total_optimizer_steps": STEPS * len(CASES),
                "max_group_attempts": MAX_GROUP_ATTEMPTS,
                "sample_retry_cap": SAMPLE_RETRY_CAP,
                "abort_probability": "0.01 + 0.01 * response_length",
                "learning_rate": LEARNING_RATE,
            },
            "results": results,
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()
        rendezvous_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _run_fixture(arguments.output)


if __name__ == "__main__":
    main()
