#!/usr/bin/env python3
"""Reproduce the ROCm LoRA/Torch Memory Saver pause-resume investigation."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name("results.json")
SUBPROCESS_TIMEOUT_SECONDS = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("parent", "worker", "unsupported-mmap", "unsupported-expandable"),
        default="parent",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--width", type=int, default=16384)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def worker_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TMS_INIT_ENABLE": "1",
            "TMS_INIT_ENABLE_CPU_BACKUP": "1",
            "TMS_INIT_ENABLE_DISK_BACKUP": "0",
        }
    )
    if extra:
        env.update(extra)
    return env


def run_child(mode: str, args: argparse.Namespace, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    from torch_memory_saver import configure_subprocess

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--cycles",
        str(args.cycles),
        "--width",
        str(args.width),
        "--rank",
        str(args.rank),
        "--batch",
        str(args.batch),
        "--device-index",
        str(args.device_index),
    ]
    with configure_subprocess():
        completed = subprocess.run(
            command,
            env=worker_env(extra_env),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"child mode={mode} failed with rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"child mode={mode} produced no JSON output")
    return json.loads(lines[-1])


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_commit(distribution: str) -> str | None:
    try:
        direct_url = importlib.metadata.distribution(distribution).read_text("direct_url.json")
        if direct_url is None:
            return None
        return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except Exception:
        return None


def process_rss_bytes() -> int:
    with Path("/proc/self/status").open() as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS was unavailable")


def memory_snapshot() -> dict[str, int]:
    import torch

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "process_rss_bytes": process_rss_bytes(),
    }


def cpu_snapshot(tensor) -> Any:
    return tensor.detach().cpu().clone()


def comparison(actual, expected) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu()
    difference = (actual_cpu.float() - expected.float()).abs()
    return {
        "bitwise_equal": bool(actual_cpu.equal(expected)),
        "max_abs_difference": float(difference.max().item()) if difference.numel() else 0.0,
    }


def unsupported_mmap() -> dict[str, Any]:
    import torch
    import torch_memory_saver as torch_memory_saver_module
    from torch_memory_saver import torch_memory_saver

    try:
        with torch_memory_saver.region(
            tag="unsupported-mmap",
            enable_cpu_backup=True,
            cpu_backup_backend="mmap",
        ):
            torch.empty(1, device="cuda")
    except ValueError as error:
        return {
            "probe": "cpu_backup_backend=mmap",
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    raise AssertionError("ROCm unexpectedly accepted cpu_backup_backend=mmap")


def unsupported_expandable() -> dict[str, Any]:
    import torch
    from torch_memory_saver import torch_memory_saver

    try:
        with torch_memory_saver.region(tag="unsupported-expandable", enable_cpu_backup=True):
            torch.empty(1, device="cuda")
    except RuntimeError as error:
        return {
            "probe": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    raise AssertionError("TMS unexpectedly accepted expandable_segments")


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))

    import megatron
    import miles
    import sglang
    import sgl_kernel
    import torch
    import torch._C
    import torch_memory_saver as torch_memory_saver_module
    from torch_memory_saver import torch_memory_saver
    from torch_memory_saver.utils import get_binary_path_from_package

    if args.cycles < 1:
        raise ValueError("--cycles must be positive")
    if args.width < 1 or args.rank < 1 or args.batch < 1:
        raise ValueError("--width, --rank, and --batch must be positive")

    started_at = time.monotonic()
    torch.manual_seed(0)
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    properties = torch.cuda.get_device_properties(args.device_index)

    class SmallLoRA(torch.nn.Module):
        def __init__(self, width: int, rank: int):
            super().__init__()
            self.base = torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16, device=device)
            self.adapter_a = torch.nn.Parameter(torch.empty(rank, width, dtype=torch.bfloat16, device=device))
            self.adapter_b = torch.nn.Parameter(torch.empty(width, rank, dtype=torch.bfloat16, device=device))
            with torch.no_grad():
                self.base.weight.normal_(0.0, 0.01)
                self.adapter_a.normal_(0.0, 0.02)
                self.adapter_b.normal_(0.0, 0.02)
            self.base.weight.requires_grad_(False)

        def forward(self, inputs):
            return self.base(inputs) + (inputs @ self.adapter_a.transpose(0, 1) @ self.adapter_b.transpose(0, 1))

    cycles = []
    with torch_memory_saver.region(
        tag="lora",
        enable_cpu_backup=True,
        cpu_backup_backend="pinned",
    ):
        model = SmallLoRA(args.width, args.rank)
        inputs = torch.randn(args.batch, args.width, dtype=torch.bfloat16, device=device)
        outputs = model(inputs)
        loss = outputs.square().mean()
        loss.backward()
        torch.cuda.synchronize()

        expected = {
            "base_weight": cpu_snapshot(model.base.weight),
            "adapter_a": cpu_snapshot(model.adapter_a),
            "adapter_b": cpu_snapshot(model.adapter_b),
            "inputs": cpu_snapshot(inputs),
            "outputs": cpu_snapshot(outputs),
            "adapter_a_grad": cpu_snapshot(model.adapter_a.grad),
            "adapter_b_grad": cpu_snapshot(model.adapter_b.grad),
        }
        tracked_tensors = [
            model.base.weight,
            model.adapter_a,
            model.adapter_b,
            inputs,
            outputs,
            model.adapter_a.grad,
            model.adapter_b.grad,
        ]

        for cycle in range(1, args.cycles + 1):
            before = memory_snapshot()
            torch_memory_saver.pause(tag="lora")
            paused = memory_snapshot()
            torch_memory_saver.resume(tag="lora")
            resumed = memory_snapshot()

            parameter_comparisons = {
                "base_weight": comparison(model.base.weight, expected["base_weight"]),
                "adapter_a": comparison(model.adapter_a, expected["adapter_a"]),
                "adapter_b": comparison(model.adapter_b, expected["adapter_b"]),
            }
            output_comparisons = {
                "existing_output": comparison(outputs, expected["outputs"]),
            }
            gradient_comparisons = {
                "adapter_a_grad": comparison(model.adapter_a.grad, expected["adapter_a_grad"]),
                "adapter_b_grad": comparison(model.adapter_b.grad, expected["adapter_b_grad"]),
            }
            fresh_outputs = model(inputs)
            output_comparisons["fresh_forward_output"] = comparison(fresh_outputs, expected["outputs"])
            del fresh_outputs
            torch.cuda.synchronize()

            all_comparisons = {**parameter_comparisons, **output_comparisons, **gradient_comparisons}
            cycles.append(
                {
                    "cycle": cycle,
                    "before_pause": before,
                    "paused": paused,
                    "resumed": resumed,
                    "observed_device_free_delta_bytes": paused["device_free_bytes"] - before["device_free_bytes"],
                    "observed_process_rss_delta_bytes": paused["process_rss_bytes"] - before["process_rss_bytes"],
                    "torch_allocator_counters_unchanged_across_pause": (
                        before["torch_allocated_bytes"] == paused["torch_allocated_bytes"]
                        and before["torch_reserved_bytes"] == paused["torch_reserved_bytes"]
                    ),
                    "parameter_comparisons": parameter_comparisons,
                    "output_comparisons": output_comparisons,
                    "gradient_comparisons": gradient_comparisons,
                    "all_bitwise_equal": all(value["bitwise_equal"] for value in all_comparisons.values()),
                }
            )

        retain_cpu_backup = bool(torch_memory_saver.retain_cpu_backup)

    result = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_at, 6),
        "case": {
            "kind": "synthetic_single_gpu_lora",
            "cycles": args.cycles,
            "width": args.width,
            "rank": args.rank,
            "batch": args.batch,
            "seed": 0,
            "device_index": args.device_index,
            "assigned_gpu_count": torch.cuda.device_count(),
            "distributed_case": "single_rank_no_process_group",
            "process_group": None,
            "public_model_downloads_bytes": 0,
        },
        "device": {
            "name": properties.name,
            "gcn_arch_name": properties.gcnArchName,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        },
        "versions": {
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "torch_cuda_allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "torch_memory_saver": importlib.metadata.version("torch_memory_saver"),
            "torch_memory_saver_commit": package_commit("torch_memory_saver"),
            "sglang": getattr(sglang, "__version__", None),
            "megatron_core": importlib.metadata.version("megatron-core"),
            "miles_source_commit": git_commit(),
        },
        "imported_paths": {
            "miles": miles.__file__,
            "sglang": sglang.__file__,
            "megatron_paths": list(megatron.__path__),
            "sgl_kernel": sgl_kernel.__file__,
            "torch": torch.__file__,
            "torch_memory_saver": torch_memory_saver_module.__file__,
        },
        "native_module_paths": {
            "torch_c": torch._C.__file__,
            "torch_memory_saver_preload": str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload")),
            "torch_memory_saver_torch": str(get_binary_path_from_package("torch_memory_saver_hook_mode_torch")),
        },
        "allocator": {
            "tms_hook_mode": torch_memory_saver._impl._hook_mode,
            "tms_region_tag": "lora",
            "tms_cpu_backup_backend": "pinned",
            "tms_retain_cpu_backup": retain_cpu_backup,
            "expandable_segments": "disabled",
        },
        "tensor_bytes": {
            "base_weight": model.base.weight.nbytes,
            "adapter_a": model.adapter_a.nbytes,
            "adapter_b": model.adapter_b.nbytes,
            "inputs": inputs.nbytes,
            "outputs": outputs.nbytes,
            "adapter_a_grad": model.adapter_a.grad.nbytes,
            "adapter_b_grad": model.adapter_b.grad.nbytes,
            "tracked_tensors_total": sum(tensor.nbytes for tensor in tracked_tensors),
        },
        "initial": {
            "loss": float(loss.detach().cpu()),
            "base_grad_is_none": model.base.weight.grad is None,
        },
        "cycles": cycles,
        "all_cycles_bitwise_equal": all(cycle["all_bitwise_equal"] for cycle in cycles),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    unsupported = {
        "mmap": run_child("unsupported-mmap", args),
        "expandable_segments": run_child(
            "unsupported-expandable",
            args,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ),
    }
    result = run_child("worker", args)
    result["unsupported_allocator_probes"] = unsupported
    result["finished_at_utc"] = utc_now()
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "parent":
        result = run_parent(args)
    elif args.mode == "worker":
        result = run_worker(args)
    elif args.mode == "unsupported-mmap":
        result = unsupported_mmap()
    else:
        result = unsupported_expandable()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
