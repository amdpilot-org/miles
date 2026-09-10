#!/usr/bin/env python3
"""Run and record the focused gated CanonicalLoRA GPU fixture."""

from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/fast/backends/megatron_utils/test_canonical_lora_gate.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "results.json"
LOG_PATH = Path(__file__).resolve().parent / "run.log"


def _package_identity(distribution: str, module: str) -> dict[str, str | None]:
    try:
        package_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    try:
        spec = importlib.util.find_spec(module)
        module_path = spec.origin if spec is not None else None
    except (ImportError, AttributeError):
        module_path = None
    return {"version": package_version, "module_path": module_path}


def _native_module_path(module: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module)
        return spec.origin if spec is not None else None
    except (ImportError, AttributeError):
        return None


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _gpu_identity() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("This fixture requires an available CUDA/ROCm GPU.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "assigned_device_count": torch.cuda.device_count(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": properties.name,
        "uuid": str(properties.uuid),
        "gcn_arch_name": properties.gcnArchName,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "torch_cuda_version": torch.version.cuda,
    }


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--noconftest",
        "-q",
        str(TEST_PATH),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
        output = completed.stdout
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return_code = 124

    LOG_PATH.write_text(output, encoding="utf-8")
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "command": command,
        "process_timeout_seconds": 300,
        "process_group_timeout_seconds": 60,
        "gpu": _gpu_identity(),
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "dirty_files": _git_output("status", "--short"),
        },
        "packages": {
            "miles": _package_identity("miles", "miles"),
            "megatron_bridge": _package_identity("megatron-bridge", "megatron.bridge"),
            "sglang": _package_identity("sglang", "sglang"),
            "torch": _package_identity("torch", "torch"),
            "transformer_engine": _package_identity("transformer-engine", "transformer_engine"),
        },
        "native_modules": {
            "torch._C": _native_module_path("torch._C"),
            "sgl_kernel.common_ops": _native_module_path("sgl_kernel.common_ops"),
            "aiter.jit.module_aiter_core": _native_module_path("aiter.jit.module_aiter_core"),
        },
        "reduced_case": {
            "hidden_size": 8,
            "attention_heads": 6,
            "query_groups": 2,
            "head_size": 4,
            "lora_rank": 4,
            "lora_alpha": 4,
            "base_output_width": 64,
            "upstream_adapter_output_width": 40,
            "candidate_query_gate_adapter_width": 48,
            "key_adapter_width": 8,
            "value_adapter_width": 8,
            "packed_order": "Q heads, Gate heads, K, V per query group",
        },
        "coverage": {
            "upstream_gated_failure": True,
            "canonical_hf_target_conversion": True,
            "candidate_forward_parity": True,
            "candidate_gradient_parity": True,
            "base_adapter_state_boundaries": True,
            "non_gated_regression": True,
        },
        "not_proven": [
            "full Qwen3.5 or Qwen3.6 model loading",
            "tensor-parallel sizes greater than one",
            "distributed checkpoint resharding",
            "SGLang LoRA publication end to end",
            "performance equivalence with MI355X",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
