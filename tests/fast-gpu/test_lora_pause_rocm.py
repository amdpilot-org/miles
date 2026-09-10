"""Bounded ROCm validation for the synthetic LoRA pause fixture."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch_memory_saver import configure_subprocess

from tests.ci.ci_register import register_rocm_ci

register_rocm_ci(est_time=60, suite="stage-c-4-gpu-mi350", labels=["amd"])


def test_synthetic_lora_pause_resume_preserves_state_and_releases_memory() -> None:
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.fail("the registered ROCm fixture requires an available HIP GPU")

    fixture = Path(__file__).with_name("lora_pause_fixture.py")
    with configure_subprocess():
        completed = subprocess.run(
            [sys.executable, str(fixture)],
            capture_output=True,
            check=False,
            text=True,
            timeout=180,
        )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["cycle_count"] == 3
    assert result["all_cycles_correct"] is True
    assert result["correctness"] == {
        "base_parameters_frozen": True,
        "adapter_parameters_trainable": True,
        "parameters_preserved": True,
        "forward_output_preserved": True,
        "gradients_preserved": True,
    }
    assert result["gpu"]["name"] == "AMD Instinct MI350X"
    assert result["gpu"]["capability"] == [9, 5]
    assert result["process_group"]["used"] is False

    default_releases = [cycle["default_release_bytes"] for cycle in result["cycles"]]
    grad_buffer_releases = [cycle["grad_buffer_release_bytes"] for cycle in result["cycles"]]
    assert min(default_releases) >= 32 * 1024**2
    assert min(grad_buffer_releases) >= 16 * 1024**2
    assert result["allocator_observations"]["torch_allocated_bytes_unchanged_while_paused"] is True
    assert result["allocator_observations"]["torch_reserved_bytes_unchanged_while_paused"] is True
    assert result["unsupported"]["cpu_backup_backend_mmap_supported"] is False


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
