#!/usr/bin/env python3
"""Run a synthetic LoRA pause/resume cycle on one ROCm GPU and emit JSON."""

import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.utils import get_binary_path_from_package


CYCLE_COUNT = 3
DEFAULT_TAG = "default"
GRAD_BUFFER_TAG = "grad_buffer"
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))


def _rss_bytes() -> int:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable in /proc/self/status")


def _memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    stats = torch.cuda.memory_stats(0)
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "torch_allocated_bytes": stats["allocated_bytes.all.current"],
        "torch_reserved_bytes": stats["reserved_bytes.all.current"],
        "rss_bytes": _rss_bytes(),
    }


def _tensor_nbytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _assert_preload_hook() -> str:
    preload = os.environ.get("LD_PRELOAD", "")
    paths = [path for path in preload.split(":") if "torch_memory_saver" in path]
    if len(paths) != 1:
        raise RuntimeError(
            "TorchMemorySaver preload mode requires exactly one TMS path in LD_PRELOAD; "
            f"got {paths!r} from {preload!r}"
        )
    return paths[0]


def _probe_mmap_support() -> dict[str, object]:
    try:
        with torch_memory_saver.region(
            tag="mmap_probe",
            enable_cpu_backup=True,
            cpu_backup_backend="mmap",
        ):
            pass
    except ValueError as error:
        return {
            "cpu_backup_backend_mmap_supported": False,
            "error": str(error),
        }
    return {"cpu_backup_backend_mmap_supported": True, "error": None}


def _module_path(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin is not None:
        return str(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).resolve())
    return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _build_model(
    device: torch.device,
) -> tuple[torch.Tensor, torch.nn.Linear, torch.nn.Linear, torch.nn.Linear, torch.Tensor]:
    inputs = torch.randn((128, 4096), device=device, dtype=torch.bfloat16)
    with torch_memory_saver.region(tag=DEFAULT_TAG, enable_cpu_backup=True):
        base = torch.nn.Linear(4096, 8192, device=device, dtype=torch.bfloat16)
        adapter_a = torch.nn.Linear(4096, 64, bias=False, device=device, dtype=torch.bfloat16)
        adapter_b = torch.nn.Linear(64, 8192, bias=False, device=device, dtype=torch.bfloat16)

    for parameter in base.parameters():
        parameter.requires_grad_(False)

    with torch_memory_saver.region(tag=GRAD_BUFFER_TAG):
        output = base(inputs) + adapter_b(adapter_a(inputs)) * 0.5
        loss = output.float().square().mean()
        loss.backward()

    return inputs, base, adapter_a, adapter_b, output


def _expected_state(
    base: torch.nn.Linear,
    adapter_a: torch.nn.Linear,
    adapter_b: torch.nn.Linear,
    output: torch.Tensor,
) -> tuple[dict[str, int], tuple[torch.Tensor, ...], torch.Tensor, tuple[torch.Tensor, ...]]:
    torch.cuda.synchronize()
    baseline = _memory_snapshot()
    expected_parameters = tuple(
        parameter.detach().cpu()
        for parameter in (base.weight, base.bias, adapter_a.weight, adapter_b.weight)
    )
    expected_output = output.detach().cpu()
    expected_gradients = tuple(
        parameter.grad.detach().cpu() for parameter in (adapter_a.weight, adapter_b.weight)
    )
    torch.cuda.synchronize()
    return baseline, expected_parameters, expected_output, expected_gradients


def _run_cycle(
    *,
    cycle_index: int,
    device: torch.device,
    inputs: torch.Tensor,
    base: torch.nn.Linear,
    adapter_a: torch.nn.Linear,
    adapter_b: torch.nn.Linear,
    baseline: dict[str, int],
    expected_parameters: tuple[torch.Tensor, ...],
    expected_output: torch.Tensor,
    expected_gradients: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    torch_memory_saver.pause(tag=GRAD_BUFFER_TAG)
    grad_buffer_paused = _memory_snapshot()
    torch_memory_saver.pause(tag=DEFAULT_TAG)
    default_paused = _memory_snapshot()

    empty_cache_before = torch.cuda.mem_get_info(device)[0]
    torch.cuda.empty_cache()
    empty_cache_after = torch.cuda.mem_get_info(device)[0]

    torch_memory_saver.resume(tag=DEFAULT_TAG)
    torch_memory_saver.resume(tag=GRAD_BUFFER_TAG)
    resumed = _memory_snapshot()

    actual_output = base(inputs) + adapter_b(adapter_a(inputs)) * 0.5
    torch.cuda.synchronize()
    actual_parameters = tuple(
        parameter.detach().cpu()
        for parameter in (base.weight, base.bias, adapter_a.weight, adapter_b.weight)
    )
    actual_gradients = tuple(
        parameter.grad.detach().cpu() for parameter in (adapter_a.weight, adapter_b.weight)
    )

    return {
        "cycle": cycle_index,
        "grad_buffer_release_bytes": grad_buffer_paused["free_bytes"] - baseline["free_bytes"],
        "default_release_bytes": default_paused["free_bytes"]
        - grad_buffer_paused["free_bytes"],
        "empty_cache_release_bytes": empty_cache_after - empty_cache_before,
        "resume_cost_bytes": resumed["free_bytes"] - default_paused["free_bytes"],
        "torch_allocated_bytes_unchanged_while_paused": default_paused[
            "torch_allocated_bytes"
        ]
        == baseline["torch_allocated_bytes"],
        "torch_reserved_bytes_unchanged_while_paused": default_paused[
            "torch_reserved_bytes"
        ]
        == baseline["torch_reserved_bytes"],
        "rss_increase_bytes": default_paused["rss_bytes"] - baseline["rss_bytes"],
        "parameters_preserved": all(
            torch.equal(expected, actual)
            for expected, actual in zip(expected_parameters, actual_parameters)
        ),
        "output_preserved": torch.equal(expected_output, actual_output.cpu()),
        "gradients_preserved": all(
            torch.equal(expected, actual)
            for expected, actual in zip(expected_gradients, actual_gradients)
        ),
    }


def _software_snapshot(preload_path: str) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "torch_module": str(Path(torch.__file__).resolve()),
        "torch_memory_saver": _package_version("torch_memory_saver"),
        "torch_memory_saver_module": str(
            Path(__import__("torch_memory_saver").__file__).resolve()
        ),
        "miles_module": _module_path("miles"),
        "sglang_module": _module_path("sglang"),
        "megatron_module": _module_path("megatron"),
        "native_preload_hook": str(
            Path(get_binary_path_from_package("torch_memory_saver_hook_mode_preload")).resolve()
        ),
        "native_torch_hook": str(
            Path(get_binary_path_from_package("torch_memory_saver_hook_mode_torch")).resolve()
        ),
        "active_preload_hook": str(Path(preload_path).resolve()),
    }


def main() -> None:
    preload_path = _assert_preload_hook()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This fixture requires an available ROCm HIP device")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(20250910)
    inputs, base, adapter_a, adapter_b, output = _build_model(device)
    baseline, expected_parameters, expected_output, expected_gradients = _expected_state(
        base, adapter_a, adapter_b, output
    )
    cycles = [
        _run_cycle(
            cycle_index=cycle_index,
            device=device,
            inputs=inputs,
            base=base,
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            baseline=baseline,
            expected_parameters=expected_parameters,
            expected_output=expected_output,
            expected_gradients=expected_gradients,
        )
        for cycle_index in range(CYCLE_COUNT)
    ]

    all_cycles_correct = all(
        cycle["parameters_preserved"]
        and cycle["output_preserved"]
        and cycle["gradients_preserved"]
        for cycle in cycles
    )
    result = {
        "all_cycles_correct": all_cycles_correct,
        "cycle_count": CYCLE_COUNT,
        "cycles": cycles,
        "correctness": {
            "base_parameters_frozen": not any(
                parameter.requires_grad for parameter in base.parameters()
            ),
            "adapter_parameters_trainable": all(
                parameter.requires_grad
                for parameter in (*adapter_a.parameters(), *adapter_b.parameters())
            ),
            "parameters_preserved": all(cycle["parameters_preserved"] for cycle in cycles),
            "forward_output_preserved": all(cycle["output_preserved"] for cycle in cycles),
            "gradients_preserved": all(cycle["gradients_preserved"] for cycle in cycles),
        },
        "memory": {
            "baseline": baseline,
            "parameter_bytes": _tensor_nbytes(
                (base.weight, base.bias, adapter_a.weight, adapter_b.weight)
            ),
            "gradient_bytes": _tensor_nbytes((adapter_a.weight, adapter_b.weight)),
        },
        "allocator_observations": {
            "torch_allocated_bytes_unchanged_while_paused": all(
                cycle["torch_allocated_bytes_unchanged_while_paused"] for cycle in cycles
            ),
            "torch_reserved_bytes_unchanged_while_paused": all(
                cycle["torch_reserved_bytes_unchanged_while_paused"] for cycle in cycles
            ),
            "empty_cache_released_inactive_segment_bytes": [
                cycle["empty_cache_release_bytes"] for cycle in cycles
            ],
        },
        "unsupported": _probe_mmap_support(),
        "gpu": {
            "count": torch.cuda.device_count(),
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_bytes": properties.total_memory,
        },
        "software": _software_snapshot(preload_path),
        "process_group": {
            "used": False,
            "timeout_seconds": None,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
