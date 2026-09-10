"""Two-GPU MoE R3 replay and malformed-split controls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


WORKER = Path(__file__).with_name("_moe_r3_replay_worker.py")
REPO_ROOT = Path(__file__).parents[2]


def _launch_worker(output: Path, control: str = "none") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(WORKER),
            "--output",
            str(output),
            "--control",
            control,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _load_rank_results(output: Path) -> list[dict[str, object]]:
    results = []
    for rank in range(2):
        with (output / f"rank{rank}.json").open() as handle:
            results.append(json.load(handle))
    return results


def test_moe_r3_replay_two_gpus(tmp_path: Path) -> None:
    output = tmp_path / "results"
    result = _launch_worker(output)
    assert result.returncode == 0, result.stdout + result.stderr
    rank_results = _load_rank_results(output)
    assert len(rank_results) == 2
    for rank_result in rank_results:
        assert rank_result["world_size"] == 2
        assert rank_result["cycles"] == 64
        assert rank_result["unique_distributions"] == 64
        assert rank_result["all_to_all_forward_calls"] == 128
        assert rank_result["all_to_all_backward_calls"] == 128
        assert rank_result["split_checks"] == 128
        assert rank_result["token_identity_checks"] == 64
        assert rank_result["gradient_checks"] == 128
        assert rank_result["malformed_controls"] == 2
        assert rank_result["max_output_error"] < 1e-5
        assert rank_result["max_input_gradient_error"] < 1e-5
        assert rank_result["max_weight_gradient_error"] < 1e-5
        assert "dispatch_ms" in rank_result["timings"]
        assert "backward_ms" in rank_result["timings"]


@pytest.mark.parametrize("control", ["local-input", "cross-rank"])
def test_malformed_split_control_fails_promptly(tmp_path: Path, control: str) -> None:
    output = tmp_path / f"control-{control}"
    result = _launch_worker(output, control=control)
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "input split sum" in combined or "split matrices disagree" in combined
    assert "diagnostics=" in combined


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
