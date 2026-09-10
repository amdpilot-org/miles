#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import Linear
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.utils import get_binary_path_from_package


def _ensure_memory_saver_preload() -> None:
    preload = os.environ.get("LD_PRELOAD", "")
    if "torch_memory_saver" in preload:
        return
    if os.environ.get("MILES_TMS_PROBE_CHILD") == "1":
        raise RuntimeError("The probe child did not receive the torch_memory_saver preload library")

    library_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
    environment = os.environ.copy()
    environment["LD_PRELOAD"] = f"{library_path}:{preload}" if preload else library_path
    environment["MILES_TMS_PROBE_CHILD"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def _fill_teacher_weights(teacher: Linear) -> None:
    with torch.no_grad():
        teacher.weight.fill_(0.25)
        teacher.bias.fill_(0.5)


def _assert_teacher_matches(teacher: Linear, expected_weight: torch.Tensor, expected_bias: torch.Tensor) -> None:
    assert torch.equal(teacher.weight.detach(), expected_weight)
    assert torch.equal(teacher.bias.detach(), expected_bias)


def _run_probe(cycles: int) -> dict[str, object]:
    if torch.cuda.device_count() < 1:
        raise RuntimeError("This probe requires at least one CUDA/HIP GPU")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    inputs = torch.linspace(-1.0, 1.0, 16, device=device).reshape(4, 4)

    with torch_memory_saver.region(tag="teacher_cpu_backup", enable_cpu_backup=True):
        backed_up_teacher = Linear(4, 3, device=device, dtype=torch.float32)
    _fill_teacher_weights(backed_up_teacher)

    expected_weight = backed_up_teacher.weight.detach().clone()
    expected_bias = backed_up_teacher.bias.detach().clone()
    expected_output = backed_up_teacher(inputs).detach().clone()

    for cycle in range(cycles):
        torch_memory_saver.pause(tag="teacher_cpu_backup")
        torch_memory_saver.resume(tag="teacher_cpu_backup")
        _assert_teacher_matches(backed_up_teacher, expected_weight, expected_bias)
        assert torch.equal(backed_up_teacher(inputs).detach(), expected_output), f"cycle {cycle} changed outputs"

    with torch_memory_saver.region(tag="teacher_no_backup"):
        no_backup_teacher = Linear(4, 3, device=device, dtype=torch.float32)
    _fill_teacher_weights(no_backup_teacher)
    torch_memory_saver.pause(tag="teacher_no_backup")
    torch_memory_saver.resume(tag="teacher_no_backup")
    no_backup_zeroed = (
        torch.count_nonzero(no_backup_teacher.weight).item() == 0
        and torch.count_nonzero(no_backup_teacher.bias).item() == 0
    )
    assert no_backup_zeroed, "The no-backup control unexpectedly preserved teacher parameters"

    with tempfile.TemporaryDirectory(prefix="miles-opd-tms-probe-") as temporary_directory:
        missing_checkpoint = Path(temporary_directory) / "missing-teacher.pt"
        failed_load_blocked = False
        try:
            torch.load(missing_checkpoint, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            failed_load_blocked = True
        assert failed_load_blocked, "A missing checkpoint unexpectedly loaded"

        valid_checkpoint = Path(temporary_directory) / "valid-teacher.pt"
        torch.save(backed_up_teacher.state_dict(), valid_checkpoint)
        restored_state = torch.load(valid_checkpoint, map_location="cpu", weights_only=True)
        no_backup_teacher.load_state_dict(restored_state)
        _assert_teacher_matches(no_backup_teacher, expected_weight, expected_bias)
        assert torch.equal(no_backup_teacher(inputs).detach(), expected_output)

    student = Linear(4, 3, device=device, dtype=torch.float32)
    with torch.no_grad():
        student.weight.copy_(backed_up_teacher.weight)
        student.bias.copy_(backed_up_teacher.bias)
    assert torch.equal(student(inputs).detach(), expected_output)
    _assert_teacher_matches(backed_up_teacher, expected_weight, expected_bias)

    return {
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_module": Path(torch.__file__).resolve().as_posix(),
        "memory_saver_module": Path(sys.modules["torch_memory_saver"].__file__).resolve().as_posix(),
        "memory_saver_preload": os.environ.get("LD_PRELOAD"),
        "cycles": cycles,
        "backed_up_teacher_preserved": True,
        "no_backup_teacher_zeroed": no_backup_zeroed,
        "failed_load_blocked": failed_load_blocked,
        "valid_reload_restored": True,
        "student_update_applied": True,
        "teacher_unchanged_after_student_update": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe frozen-teacher survival across torch_memory_saver cycles")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    _ensure_memory_saver_preload()
    result = _run_probe(cycles=args.cycles)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
