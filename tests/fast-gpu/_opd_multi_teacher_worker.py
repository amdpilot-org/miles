from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import miles.rollout.on_policy_distillation as opd
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.utils.types import Sample


PROMPT_LENGTH = 4
RESPONSE_LENGTH = 4
VOCAB_SIZE = 32
HIDDEN_SIZE = 16
MAX_SEQUENCE_LENGTH = 16
TOP_K = 4
OPD_COEFFICIENT = 0.25
STEPS = 256
SWITCH_STEP = 128
RECOVERY_STEP = 192
CHECK_INTERVAL = 16
PROCESS_GROUP_TIMEOUT_SECONDS = 120

TEACHER_URLS = {
    "math": "http://teacher-a/generate",
    "code": "http://teacher-b/generate",
}


@dataclass
class WorkerState:
    teachers: dict[str, nn.Module]
    teacher_name: str
    expected_url: str
    calls: dict[str, int]
    world_size: int
    optimizer: torch.optim.Optimizer


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.position_embedding = nn.Embedding(MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence_length = tokens.shape[-1]
        positions = torch.arange(sequence_length, device=tokens.device)
        embeddings = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.output(embeddings)


def _make_model(seed: int, device: torch.device) -> TinyCausalModel:
    torch.manual_seed(seed)
    return TinyCausalModel().to(device)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _assert_placement(model: nn.Module, device: torch.device, label: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.device != device:
            raise AssertionError(f"{label} parameter {name} is on {parameter.device}, expected {device}")


def _assert_frozen(model: nn.Module, snapshot: dict[str, torch.Tensor], label: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            raise AssertionError(f"{label} parameter {name} unexpectedly requires grad")
        if parameter.grad is not None:
            raise AssertionError(f"{label} parameter {name} unexpectedly has a gradient")
        if not torch.equal(parameter.detach(), snapshot[name]):
            raise AssertionError(f"{label} parameter {name} changed")


def _teacher_name(step: int) -> str:
    if step < SWITCH_STEP:
        return "math"
    if step < RECOVERY_STEP:
        return "code"
    return "math"


def _opd_args() -> argparse.Namespace:
    return argparse.Namespace(
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=OPD_COEFFICIENT,
        opd_log_prob_top_k=TOP_K,
        opd_top_k_strategy="only-student",
        opd_reward_weight_mode="student_p",
        opd_topk_per_position=False,
        opd_teacher_urls=[f"{name}={url}" for name, url in TEACHER_URLS.items()],
        opd_teacher_key="opd_teacher",
        rm_url="http://single-teacher/generate",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=31000,
        sglang_router_request_timeout_secs=30,
        reward_key=None,
    )


def _make_tokens(step: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(2025 + step)
    return torch.randint(
        1,
        VOCAB_SIZE,
        (PROMPT_LENGTH + RESPONSE_LENGTH,),
        generator=generator,
        device=device,
    )


def _response_logits(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(tokens)
    start = PROMPT_LENGTH - 1
    return logits[start : start + RESPONSE_LENGTH]


def _student_top_logprobs(model: nn.Module, tokens: torch.Tensor) -> list[list[list[Any]]]:
    log_probs = torch.log_softmax(_response_logits(model, tokens), dim=-1)
    values, token_ids = torch.topk(log_probs, TOP_K, dim=-1)
    return [
        [[float(logprob), int(token_id)] for logprob, token_id in zip(position_values, position_ids)]
        for position_values, position_ids in zip(values.tolist(), token_ids.tolist())
    ]


def _sampled_log_probs(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(_response_logits(model, tokens), dim=-1)
    response_tokens = tokens[PROMPT_LENGTH:]
    return log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)


def _selected_token_ids(student_top: list[list[list[Any]]]) -> list[list[int]]:
    return [[int(entry[1]) for entry in entries] for entries in student_top]


def _teacher_response(
    model: nn.Module,
    tokens: torch.Tensor,
    token_ids: list[int],
) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(tokens)
        log_probs = torch.log_softmax(logits, dim=-1)
        requested = torch.tensor(token_ids, device=tokens.device, dtype=torch.long)
        selected = log_probs[:, requested]
        entries: list[list[list[Any]] | None] = [None]
        for position in range(1, tokens.numel()):
            entries.append(
                [
                    [float(logprob), int(token_id)]
                    for logprob, token_id in zip(selected[position - 1].tolist(), token_ids)
                ]
            )
    return {"meta_info": {"input_token_ids_logprobs": entries}}


def _set_step_context(
    tokens: torch.Tensor,
    teacher_name: str,
    state: WorkerState,
) -> None:
    state.teacher_name = teacher_name
    state.expected_url = TEACHER_URLS[teacher_name]

    async def hook(
        url: str,
        payload: dict[str, Any],
        timeout_secs: int | float | None = None,
    ) -> dict[str, Any]:
        if url != state.expected_url:
            raise AssertionError(f"teacher identity mismatch: expected {state.expected_url}, got {url}")
        if payload["input_ids"] != tokens.tolist():
            raise AssertionError("teacher scoring received different input_ids")
        state.calls[state.teacher_name] += 1
        return _teacher_response(
            state.teachers[url],
            tokens,
            payload["token_ids_logprob"],
        )

    opd._post_json = hook


def _expected_reverse_kl(
    student_top: list[list[list[Any]]],
    student_model: nn.Module,
    teacher_model: nn.Module,
    tokens: torch.Tensor,
) -> torch.Tensor:
    selected_ids = _selected_token_ids(student_top)
    with torch.no_grad():
        student_logits = _response_logits(student_model, tokens)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_logits = _response_logits(teacher_model, tokens)
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
        reverse_kls = []
        for position, ids in enumerate(selected_ids):
            student_values = [entry[0] for entry in student_top[position]]
            weights = torch.softmax(torch.tensor(student_values, device=tokens.device), dim=0)
            student_selected = torch.stack([student_log_probs[position, token_id] for token_id in ids])
            teacher_values = torch.stack([teacher_log_probs[position, token_id] for token_id in ids])
            reverse_kls.append(float(torch.dot(weights, student_selected - teacher_values)))
        return torch.tensor(reverse_kls, dtype=torch.float32, device=tokens.device)


def _assert_selected_logprobs(
    reward_payload: dict[str, Any],
    teacher_model: nn.Module,
    tokens: torch.Tensor,
    selected_ids: list[list[int]],
) -> None:
    response_entries = reward_payload["teacher"]["meta_info"]["input_token_ids_logprobs"][1:][-RESPONSE_LENGTH :]
    with torch.no_grad():
        teacher_log_probs = torch.log_softmax(_response_logits(teacher_model, tokens), dim=-1)
    for position, (entries, expected_ids) in enumerate(zip(response_entries, selected_ids, strict=True)):
        entry_map = {int(entry[1]): float(entry[0]) for entry in entries}
        if not set(expected_ids).issubset(entry_map):
            raise AssertionError(f"teacher selected-token mismatch at response position {position}")
        for token_id in expected_ids:
            expected = float(teacher_log_probs[position, token_id])
            logprob = entry_map[token_id]
            if not math.isclose(logprob, expected, rel_tol=1e-6, abs_tol=1e-7):
                raise AssertionError(f"teacher logprob mismatch for token {token_id}")


def _expected_gradients(
    student: DistributedDataParallel,
    tokens: torch.Tensor,
    advantages: torch.Tensor,
) -> list[torch.Tensor]:
    reference = copy.deepcopy(student.module)
    sampled_log_probs = _sampled_log_probs(reference, tokens)
    loss = -(advantages * sampled_log_probs).mean()
    return list(torch.autograd.grad(loss, reference.parameters()))


def _assert_gradient_synchronization(
    student: DistributedDataParallel,
    world_size: int,
) -> None:
    flattened = torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in student.module.parameters()],
    )
    gathered = [torch.empty_like(flattened) for _ in range(world_size)]
    dist.all_gather(gathered, flattened)
    for rank, gradient in enumerate(gathered[1:], start=1):
        torch.testing.assert_close(gradient, gathered[0], rtol=2e-5, atol=2e-6)


def _record_phase(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _run_step(
    step: int,
    student: DistributedDataParallel,
    state: WorkerState,
    device: torch.device,
) -> tuple[float, float, float]:
    tokens = _make_tokens(step, device)
    teacher_name = _teacher_name(step)
    _set_step_context(tokens, teacher_name, state)
    teachers = state.teachers
    args = _opd_args()

    score_start = torch.cuda.Event(enable_timing=True)
    score_end = torch.cuda.Event(enable_timing=True)
    score_start.record()

    student_top = _student_top_logprobs(student.module, tokens)
    sample = Sample(
        tokens=tokens.tolist(),
        response_length=RESPONSE_LENGTH,
        metadata={
            "opd_teacher": teacher_name,
            "opd_student_top_logprobs": student_top,
        },
    )
    resolved_url = opd._teacher_url_for_sample(args, sample)
    if resolved_url != TEACHER_URLS[teacher_name]:
        raise AssertionError(f"route resolved {resolved_url}, expected {TEACHER_URLS[teacher_name]}")
    reward_payload = asyncio.run(opd.reward_func(args, sample))
    sample.reward = reward_payload
    opd.post_process_rewards(args, [sample])
    sample.validate()

    expected_reverse_kl = _expected_reverse_kl(
        student_top,
        student.module,
        teachers[TEACHER_URLS[teacher_name]],
        tokens,
    )
    torch.testing.assert_close(
        sample.opd_reverse_kl.to(device=device),
        expected_reverse_kl,
        rtol=1e-5,
        atol=1e-6,
    )
    _assert_selected_logprobs(
        reward_payload,
        teachers[TEACHER_URLS[teacher_name]],
        tokens,
        _selected_token_ids(student_top),
    )

    advantages = torch.ones(RESPONSE_LENGTH, device=device)
    current_log_probs = _sampled_log_probs(student, tokens)
    rollout_data = {"opd_reverse_kl": [sample.opd_reverse_kl]}
    advantage_tensors = [advantages]
    apply_opd_kl_to_advantages(args, rollout_data, advantage_tensors, [current_log_probs])
    advantages = advantage_tensors[0]
    expected_advantages = torch.ones(RESPONSE_LENGTH, device=device) - OPD_COEFFICIENT * expected_reverse_kl
    torch.testing.assert_close(advantages, expected_advantages, rtol=1e-5, atol=1e-6)
    score_end.record()
    score_ms = _record_phase(score_start, score_end)

    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_start.record()
    loss = -(advantages * current_log_probs).mean()
    loss.backward()
    backward_end.record()
    backward_ms = _record_phase(backward_start, backward_end)

    if step % CHECK_INTERVAL == 0 or step in (SWITCH_STEP, RECOVERY_STEP):
        expected_gradients = _expected_gradients(student, tokens, advantages)
        for parameter, expected_gradient in zip(student.module.parameters(), expected_gradients, strict=True):
            torch.testing.assert_close(parameter.grad, expected_gradient, rtol=2e-5, atol=2e-6)
        _assert_gradient_synchronization(student, state.world_size)

    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record()
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    optimizer_end.record()
    optimizer_ms = _record_phase(optimizer_start, optimizer_end)

    return score_ms, backward_ms, optimizer_ms


def _initialize_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise AssertionError(f"fixture requires exactly 2 ranks, got {world_size}")
    if torch.cuda.device_count() != 2:
        raise AssertionError(f"fixture requires exactly 2 GPUs, got {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
        device_id=device,
    )
    return rank, local_rank, world_size, device


def _build_worker(
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    DistributedDataParallel,
    TinyCausalModel,
    TinyCausalModel,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    WorkerState,
]:
    student_model = _make_model(2025, device)
    student = DistributedDataParallel(student_model, device_ids=[local_rank], broadcast_buffers=False)
    teacher_a = _make_model(3101, device)
    teacher_b = _make_model(3102, device)
    for teacher in (teacher_a, teacher_b):
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    teachers = {TEACHER_URLS["math"]: teacher_a, TEACHER_URLS["code"]: teacher_b}

    _assert_placement(student.module, device, "student")
    _assert_placement(teacher_a, device, "teacher-a")
    _assert_placement(teacher_b, device, "teacher-b")
    teacher_snapshots = {
        TEACHER_URLS["math"]: _snapshot(teacher_a),
        TEACHER_URLS["code"]: _snapshot(teacher_b),
    }
    initial_student = _snapshot(student.module)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    state = WorkerState(
        teachers=teachers,
        teacher_name="math",
        expected_url=TEACHER_URLS["math"],
        calls={"math": 0, "code": 0},
        world_size=world_size,
        optimizer=optimizer,
    )
    return student, teacher_a, teacher_b, teacher_snapshots, initial_student, state


def _run_training_loop(
    steps: int,
    student: DistributedDataParallel,
    teacher_a: TinyCausalModel,
    teacher_b: TinyCausalModel,
    teacher_snapshots: dict[str, torch.Tensor],
    state: WorkerState,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int]]:
    phase_ms = {"score": 0.0, "backward": 0.0, "optimizer": 0.0}
    previous_teacher = "math"
    route_switches = []
    route_recoveries = []

    for step in range(steps):
        teacher_name = _teacher_name(step)
        if step == SWITCH_STEP:
            if teacher_name == previous_teacher:
                raise AssertionError("expected routing switch")
            route_switches.append(step)
        if step == RECOVERY_STEP:
            if teacher_name == previous_teacher:
                raise AssertionError("expected routing recovery")
            route_recoveries.append(step)
        score_ms, backward_ms, optimizer_ms = _run_step(step, student, state, device)
        phase_ms["score"] += score_ms
        phase_ms["backward"] += backward_ms
        phase_ms["optimizer"] += optimizer_ms
        previous_teacher = teacher_name

        if step % CHECK_INTERVAL == 0 or step in (SWITCH_STEP, RECOVERY_STEP):
            _assert_frozen(teacher_a, teacher_snapshots[TEACHER_URLS["math"]], "teacher-a")
            _assert_frozen(teacher_b, teacher_snapshots[TEACHER_URLS["code"]], "teacher-b")
    return phase_ms, route_switches, route_recoveries


def _final_student_delta(
    student: DistributedDataParallel,
    initial_student: dict[str, torch.Tensor],
) -> float:
    final_student = _snapshot(student.module)
    student_delta = max(
        (final_value - initial_value).abs().max().item()
        for final_value, initial_value in zip(final_student.values(), initial_student.values(), strict=True)
    )
    if student_delta == 0.0:
        raise AssertionError("student weights did not update")
    if not all(torch.isfinite(value).all() for value in final_student.values()):
        raise AssertionError("student weights became non-finite")
    return student_delta


def _build_result(
    steps: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    state: WorkerState,
    route_switches: list[int],
    route_recoveries: list[int],
    student_delta: float,
    phase_ms: dict[str, float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "steps": steps,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": torch.cuda.get_device_name(device),
        "route_schedule": {
            "teacher_a_steps": SWITCH_STEP + (steps - RECOVERY_STEP),
            "teacher_b_steps": RECOVERY_STEP - SWITCH_STEP,
            "switch_step": SWITCH_STEP,
            "recovery_step": RECOVERY_STEP,
        },
        "teacher_calls": state.calls,
        "route_switches": route_switches,
        "route_recoveries": route_recoveries,
        "student_weight_delta": student_delta,
        "timings": {
            "wall_seconds": elapsed_seconds,
            "score_ms": phase_ms["score"],
            "backward_ms": phase_ms["backward"],
            "optimizer_ms": phase_ms["optimizer"],
        },
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank, local_rank, world_size, device = _initialize_distributed()

    student, teacher_a, teacher_b, teacher_snapshots, initial_student, state = _build_worker(
        local_rank,
        world_size,
        device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    phase_ms, route_switches, route_recoveries = _run_training_loop(
        args.steps,
        student,
        teacher_a,
        teacher_b,
        teacher_snapshots,
        state,
        device,
    )

    torch.cuda.synchronize(device)
    student_delta = _final_student_delta(student, initial_student)
    _assert_frozen(teacher_a, teacher_snapshots[TEACHER_URLS["math"]], "teacher-a")
    _assert_frozen(teacher_b, teacher_snapshots[TEACHER_URLS["code"]], "teacher-b")

    elapsed_seconds = time.perf_counter() - started
    result = _build_result(
        args.steps,
        local_rank,
        world_size,
        device,
        state,
        route_switches,
        route_recoveries,
        student_delta,
        phase_ms,
        elapsed_seconds,
    )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()
