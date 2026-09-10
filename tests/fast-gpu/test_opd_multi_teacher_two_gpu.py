from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

from tests.ci.ci_register import register_rocm_ci


register_rocm_ci(est_time=180, suite="nightly-stage-c-2-gpu-mi350", labels=["opd"])


_WORKER = Path(__file__).with_name("_opd_multi_teacher_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_opd_multi_teacher_two_gpu(tmp_path: Path) -> None:
    if torch.cuda.device_count() != 2:
        raise AssertionError(f"fixture requires exactly 2 GPUs, found {torch.cuda.device_count()}")

    output = tmp_path / "opd-multi-teacher-result.json"
    rendezvous_port = _free_port()
    rendezvous_id = f"miles-opd-{uuid.uuid4().hex}"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(_REPO_ROOT), environment.get("PYTHONPATH", "")]),
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=127.0.0.1:{rendezvous_port}",
        f"--rdzv-id={rendezvous_id}",
        "--rdzv-conf=join_timeout=120,last_call_timeout=10,close_timeout=10",
        "--max-restarts=0",
        f"--log-dir={tmp_path / 'torchrun'}",
        str(_WORKER),
        "--steps",
        "256",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    evidence = json.loads(output.read_text())
    assert evidence["steps"] == 256
    assert evidence["world_size"] == 2
    assert evidence["route_schedule"]["switch_step"] == 128
    assert evidence["route_schedule"]["recovery_step"] == 192
    assert evidence["route_switches"] == [128]
    assert evidence["route_recoveries"] == [192]
    assert evidence["teacher_calls"] == {"math": 192, "code": 64}
    assert evidence["student_weight_delta"] > 0.0
    assert evidence["timings"]["score_ms"] > 0.0
    assert evidence["timings"]["backward_ms"] > 0.0
    assert evidence["timings"]["optimizer_ms"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
