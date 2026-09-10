import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

import ray
import sglang
import torch
import transformers
import triton
from safetensors.torch import load_file


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def git_commit(path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_properties() -> list[dict]:
    properties = []
    for index in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(index)
        properties.append(
            {
                "index": index,
                "name": device.name,
                "total_bytes": device.total_memory,
                "gcn_architecture": getattr(device, "gcnArchName", None),
                "compute_capability": [device.major, device.minor],
            }
        )
    return properties


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/job/miles-artifacts/model"))
    parser.add_argument("--repo", type=Path, default=Path("/job/miles"))
    parser.add_argument("--write", type=Path)
    arguments = parser.parse_args()

    cycles = [json.loads(line) for line in (arguments.output_dir / "cycles.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (arguments.output_dir / "cuda_events.jsonl").read_text().splitlines()]
    metadata = json.loads((arguments.output_dir / "metadata.json").read_text())
    log_path = arguments.output_dir.with_suffix(".log")
    grad_norms = [
        float(value)
        for value in re.findall(r"'train/grad_norm': ([0-9.eE+-]+)", log_path.read_text(errors="replace"))
    ]
    model_tensors = load_file(str(arguments.model / "model.safetensors"))
    model_config = json.loads((arguments.model / "config.json").read_text())

    timing = {}
    for phase in ["forward_backward", "grad_clip", "optimizer", "update_weights"]:
        values = [event["elapsed_ms"] for event in events if event["phase"] == phase]
        timing[phase] = distribution(values)
    timing["post_initial_update_weights"] = distribution(
        [event["elapsed_ms"] for event in events if event["phase"] == "update_weights" and event["rollout_id"] >= 0]
    )
    timing["fresh_control_fixed_tokens"] = distribution([cycle["control_cuda_event_ms"] for cycle in cycles])

    wall = {
        key: distribution([cycle[key] for cycle in cycles])
        for key in [
            "rollout_wall_seconds",
            "train_wall_seconds",
            "update_wall_seconds",
            "save_wall_seconds",
            "control_wall_seconds",
            "cycle_wall_seconds",
        ]
    }
    memory = {
        device: {
            "first_gib": cycles[0]["gpu_memory_used_bytes"][device] / 2**30,
            "last_gib": cycles[-1]["gpu_memory_used_bytes"][device] / 2**30,
            "min_gib": min(cycle["gpu_memory_used_bytes"][device] for cycle in cycles) / 2**30,
            "max_gib": max(cycle["gpu_memory_used_bytes"][device] for cycle in cycles) / 2**30,
        }
        for device in ["0", "1"]
    }

    result = {
        "run_id": arguments.output_dir.name,
        "output_dir": str(arguments.output_dir),
        "log_path": str(log_path),
        "status": "passed" if len(cycles) == 64 and all(
            [
                cycle["version_equal"],
                cycle["engine_compare_equal"],
                cycle["fixed_token_output_equal"],
                cycle["optimizer_step_count_equal"],
                cycle["update_event_count_equal"],
            ]
            for cycle in cycles
        ) else "failed",
        "cycles": len(cycles),
        "first_rollout_id": cycles[0]["rollout_id"],
        "last_rollout_id": cycles[-1]["rollout_id"],
        "optimizer_steps": sum(event["phase"] == "optimizer" for event in events),
        "successive_weight_updates": sum(
            event["phase"] == "update_weights" and event["rollout_id"] >= 0 for event in events
        ),
        "versions": {
            "first": cycles[0]["version"],
            "last": cycles[-1]["version"],
            "engine_last": cycles[-1]["engine_version"],
            "all_equal": all(cycle["version_equal"] for cycle in cycles),
        },
        "checksums": {
            "unique_engine": len({cycle["engine_checksum"] for cycle in cycles}),
            "unique_control": len({cycle["control_engine_checksum"] for cycle in cycles}),
            "all_engine_control_equal": all(cycle["engine_compare_equal"] for cycle in cycles),
            "first": cycles[0]["engine_checksum"],
            "last": cycles[-1]["engine_checksum"],
        },
        "fixed_token_outputs": {
            "all_equal": all(cycle["fixed_token_output_equal"] for cycle in cycles),
            "max_abs_log_prob_difference": max(cycle["max_abs_log_prob_difference"] for cycle in cycles),
            "mean_max_abs_log_prob_difference": statistics.mean(
                cycle["max_abs_log_prob_difference"] for cycle in cycles
            ),
        },
        "gradient_norm": distribution(grad_norms) | {"all_nonzero": all(value > 0 for value in grad_norms)},
        "checkpoint_global_step_metadata": {
            "all_observed_zero": all(cycle["checkpoint_global_step"] == 0 for cycle in cycles),
            "expected_final": cycles[-1]["expected_global_step"],
        },
        "event_counts": {
            phase: sum(event["phase"] == phase for event in events)
            for phase in ["forward_backward", "grad_clip", "optimizer", "memory", "update_weights"]
        },
        "cuda_event_timing_ms": timing,
        "wall_seconds": wall,
        "memory_gib": memory,
        "model": {
            "path": str(arguments.model),
            "architecture": model_config["architectures"][0],
            "parameters": sum(tensor.numel() for tensor in model_tensors.values()),
            "vocab_size": model_config["vocab_size"],
            "hidden_size": model_config["hidden_size"],
            "num_hidden_layers": model_config["num_hidden_layers"],
        },
        "environment": {
            "miles_commit": metadata["miles_commit"],
            "miles_path": metadata["miles_path"],
            "sglang_commit": git_commit("/sgl-workspace/sglang"),
            "sglang_path": sglang.__file__,
            "sglang_version": sglang.__version__,
            "aiter_commit": git_commit("/sgl-workspace/aiter"),
            "aiter_path": "/sgl-workspace/aiter/aiter",
            "triton_commit": git_commit("/sgl-workspace/triton-custom"),
            "triton_path": triton.__file__,
            "triton_version": triton.__version__,
            "torch_version": torch.__version__,
            "torch_path": torch.__file__,
            "torch_native_path": torch._C.__file__,
            "hip_version": torch.version.hip,
            "transformers_version": transformers.__version__,
            "transformers_path": transformers.__file__,
            "ray_version": ray.__version__,
            "ray_path": ray.__file__,
            "devices": device_properties(),
        },
        "timeouts": {
            "training_process_group_seconds": metadata["training_pg_timeout_seconds"],
            "update_process_group_default_seconds": metadata["update_pg_default_timeout_seconds"],
            "sglang_distributed_seconds": 600,
            "sglang_watchdog_seconds": 300,
            "nccl_heartbeat_seconds": 600,
            "nccl_socket_milliseconds": 600000,
        },
        "metadata": metadata,
    }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.write:
        arguments.write.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
