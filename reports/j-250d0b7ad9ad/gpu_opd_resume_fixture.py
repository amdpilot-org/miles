#!/usr/bin/env python3
"""Exercise frozen-teacher restoration with the native Torch memory saver."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch_memory_saver import configure_subprocess


CYCLES = 3


def _run_child() -> None:
    output_dir = Path(os.environ.get("MILES_OPD_FIXTURE_DIR", "/job/.cache/opd-resume-fixture"))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1958)
    device = torch.device("cuda")
    teacher_path = output_dir / "teacher.pt"
    student_path = output_dir / "student.pt"
    missing_path = output_dir / "does-not-exist.pt"
    inputs = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 1.5, -2.0, 0.25]], device=device)

    from torch_memory_saver import torch_memory_saver

    with torch_memory_saver.region(tag="teacher_weights", enable_cpu_backup=False):
        teacher = nn.Linear(4, 3).to(device)
    with torch_memory_saver.region(tag="student_weights", enable_cpu_backup=False):
        student = nn.Linear(4, 3).to(device)

    torch.save(teacher.state_dict(), teacher_path)
    torch.save(student.state_dict(), student_path)
    teacher_expected = _load_state(teacher_path, device)
    teacher_output_expected = _outputs(teacher, inputs)

    records = []
    for cycle in range(1, CYCLES + 1):
        with torch.no_grad():
            student.weight.add_(0.125)
            student.bias.add_(0.0625)
        torch.save(student.state_dict(), student_path)
        student_expected = _load_state(student_path, device)
        student_output_expected = _outputs(student, inputs)

        torch_memory_saver.pause(tag="teacher_weights")
        torch_memory_saver.pause(tag="student_weights")
        torch_memory_saver.resume(tag="teacher_weights")
        torch_memory_saver.resume(tag="student_weights")

        resume_without_backup_zeroed = (
            _all_zero(teacher) and _all_zero(student) and _uniform(_outputs(teacher, inputs))
        )
        failed_load_rejected = False
        try:
            _load_state(missing_path, device)
        except FileNotFoundError:
            failed_load_rejected = True

        _load_into(teacher, teacher_path, device)
        _load_into(student, student_path, device)
        teacher_state = {name: tensor.detach() for name, tensor in teacher.state_dict().items()}
        student_state = {name: tensor.detach() for name, tensor in student.state_dict().items()}
        teacher_output = _outputs(teacher, inputs)
        student_output = _outputs(student, inputs)

        teacher_restored = _same_state(teacher_state, teacher_expected)
        student_restored = _same_state(student_state, student_expected)
        outputs_restored = torch.equal(teacher_output, teacher_output_expected) and torch.equal(
            student_output, student_output_expected
        )
        opd_request_admitted = teacher_restored and student_restored and outputs_restored

        records.append(
            {
                "cycle": cycle,
                "resume_without_backup_zeroed": resume_without_backup_zeroed,
                "failed_load_rejected": failed_load_rejected,
                "teacher_parameters_restored": teacher_restored,
                "student_parameters_restored": student_restored,
                "teacher_and_student_outputs_restored": outputs_restored,
                "opd_request_admitted": opd_request_admitted,
            }
        )

    assert all(record["resume_without_backup_zeroed"] for record in records)
    assert all(record["failed_load_rejected"] for record in records)
    assert all(record["opd_request_admitted"] for record in records)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "cycles": records}, indent=2))


def _load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location=device, weights_only=True)


def _load_into(model: nn.Module, path: Path, device: torch.device) -> None:
    model.load_state_dict(_load_state(path, device))


def _outputs(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(inputs).detach().cpu()


def _all_zero(model: nn.Module) -> bool:
    return all(torch.count_nonzero(tensor).item() == 0 for tensor in model.state_dict().values())


def _uniform(outputs: torch.Tensor) -> bool:
    log_probs = torch.log_softmax(outputs, dim=-1)
    return bool(torch.allclose(log_probs, log_probs[0, 0].expand_as(log_probs)))


def _same_state(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> bool:
    return set(actual) == set(expected) and all(
        torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu()) for name in expected
    )


def main() -> None:
    if os.environ.get("MILES_OPD_FIXTURE_CHILD") == "1":
        _run_child()
        return

    with configure_subprocess():
        child_environment = {
            **os.environ,
            "MILES_OPD_FIXTURE_CHILD": "1",
            "MILES_OPD_FIXTURE_DIR": os.environ.get("MILES_OPD_FIXTURE_DIR", "/job/.cache/opd-resume-fixture"),
        }
        completed = subprocess.run(
            [sys.executable, __file__],
            env=child_environment,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
