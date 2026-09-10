#!/usr/bin/env python3
"""Compare standalone and Ray-colocated ROCm all-reduce graph capture."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


WORLD_SIZE = 2
PROCESS_GROUP_TIMEOUT_SECONDS = 300
RAY_TIMEOUT_SECONDS = 900
SUBPROCESS_TIMEOUT_SECONDS = 900
MAX_CUSTOM_ALL_REDUCE_BYTES = 8 * 1024 * 1024
MODEL_INPUT_FEATURES = 512
MODEL_HIDDEN_FEATURES = 1024
WORKLOAD_TOKEN_COUNTS = tuple(range(64, 2049, 64))
WORKLOAD_HIDDEN_FEATURES = 8192
WORKLOAD_SHAPES = [
    (token_count, hidden_features)
    for token_count in WORKLOAD_TOKEN_COUNTS
    for hidden_features in (WORKLOAD_HIDDEN_FEATURES,)
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _cuda_event_pair() -> Tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _event_elapsed(
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
) -> float:
    return float(start_event.elapsed_time(end_event))


def _tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().double().sum().item())


def _tensor_max_abs_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _worker_environment(
    rank: int,
    master_port: int,
    result_path: Path,
    *,
    colocated: bool,
) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(WORLD_SIZE),
            "RANK": str(rank),
            "LOCAL_RANK": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "GLOO_SOCKET_IFNAME": "lo",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_DEBUG": "WARN",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "1",
            "SGLANG_USE_AITER_AR": "0",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "0",
            "ROCM_QUICK_REDUCE_QUANTIZATION": "FP",
            "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "0",
            "ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": "64",
            "AITER_CUSTOM_AR_MAX_SIZE": str(MAX_CUSTOM_ALL_REDUCE_BYTES),
            "AITER_CUSTOM_AR_MIN_SIZE": "0",
            "PYTHONPATH": ":".join(
                [
                    "/sgl-workspace/sglang/python",
                    "/sgl-workspace/aiter",
                    environment.get("PYTHONPATH", ""),
                ]
            ),
            "FIXTURE_RESULT_PATH": str(result_path),
            "FIXTURE_CONTEXT": "ray_colocated" if colocated else "standalone",
        }
    )
    memory_saver_hook = (
        "/opt/venv/lib/python3.10/site-packages/"
        "torch_memory_saver_hook_mode_preload.abi3.so"
    )
    if Path(memory_saver_hook).exists():
        existing_preload = environment.get("LD_PRELOAD", "")
        environment["LD_PRELOAD"] = (
            f"{memory_saver_hook}:{existing_preload}"
            if existing_preload
            else memory_saver_hook
        )
    return environment


def _pause_resume_probe(device: torch.device) -> Dict[str, Any]:
    preload = os.environ.get("LD_PRELOAD", "")
    supported = "torch_memory_saver_hook_mode_preload" in preload
    if not supported:
        return {
            "supported": False,
            "passed": False,
            "reason": "torch_memory_saver preload hook is absent from this launch context",
        }

    from torch_memory_saver import torch_memory_saver

    probe_tag = "rocm_colocate_fixture_probe"
    with torch_memory_saver.region(
        tag=probe_tag,
        enable_cpu_backup=True,
    ):
        probe_tensor = torch.arange(16, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    expected_tensor = probe_tensor.detach().clone()
    torch_memory_saver.pause(tag=probe_tag)
    torch_memory_saver.resume(tag=probe_tag)
    torch.cuda.synchronize()
    return {
        "supported": True,
        "passed": bool(torch.equal(probe_tensor, expected_tensor)),
        "reason": "pause/resume preserved tagged GPU memory",
    }


def _initialize_worker() -> Tuple[torch.device, dist.ProcessGroup]:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=WORLD_SIZE,
        rank=rank,
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    gloo_group = dist.new_group(
        backend="gloo",
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    return device, gloo_group


def _select_custom_all_reduce(
    gloo_group: dist.ProcessGroup,
    device: torch.device,
) -> Tuple[Any, Dict[str, Any]]:
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        qr_rocm_arch_available,
    )
    from sglang.srt.environ import envs
    from sglang.srt.utils import is_cuda, is_hip

    communicator = QuickAllReduce(
        group=gloo_group,
        device=device,
    )
    metadata = {
        "selected_class": type(communicator).__name__,
        "selected_module": type(communicator).__module__,
        "quick_all_reduce_arch_available": bool(qr_rocm_arch_available()),
        "quick_all_reduce_quantization": os.environ.get(
            "ROCM_QUICK_REDUCE_QUANTIZATION",
            "NONE",
        ),
        "quick_all_reduce_max_size_bytes": int(
            communicator.qr_max_size
        ),
        "sglang_is_cuda": bool(is_cuda()),
        "sglang_is_hip": bool(is_hip()),
        "sglang_v2_env": bool(envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get()),
        "sglang_v2_dispatched": type(communicator).__name__ == "CustomAllReduceV2",
        "disabled": bool(communicator.disabled),
    }
    return communicator, metadata


def _capture_replay_all_reduce(
    communicator: Any,
    gloo_group: dist.ProcessGroup,
    device: torch.device,
    shape: Tuple[int, int],
    cycle_index: int,
) -> Dict[str, Any]:
    rank = dist.get_rank()
    element_count = shape[0] * shape[1]
    graph_input = torch.empty(
        element_count,
        dtype=torch.float16,
        device=device,
    )
    graph_output = torch.empty_like(graph_input)
    graph_input.fill_(1.0)
    communicator.quick_all_reduce(graph_input, out=graph_output)
    torch.cuda.synchronize()
    dist.barrier(group=gloo_group)

    capture_stream = torch.cuda.Stream(device=device)
    capture_start, capture_end = _cuda_event_pair()

    torch.cuda.synchronize()
    dist.barrier(group=gloo_group)
    capture_start.record(capture_stream)
    with torch.cuda.stream(capture_stream):
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph, stream=capture_stream):
            communicator.quick_all_reduce(graph_input, out=graph_output)
    capture_end.record(capture_stream)
    torch.cuda.synchronize()
    dist.barrier(group=gloo_group)

    generator = torch.Generator(device=device)
    generator.manual_seed(211_000 + cycle_index * 100 + rank)
    graph_input.copy_(
        torch.randint(
            1,
            17,
            (element_count,),
            generator=generator,
            device=device,
        ).to(torch.float16)
    )
    reference_tensor = graph_input.detach().clone()

    replay_start, replay_end = _cuda_event_pair()
    torch.cuda.synchronize()
    replay_start.record()
    cuda_graph.replay()
    replay_end.record()
    torch.cuda.synchronize()
    dist.all_reduce(reference_tensor)
    torch.cuda.synchronize()

    assert graph_output is not None, "Quick all-reduce returned no output"
    maximum_difference = _tensor_max_abs_difference(graph_output, reference_tensor)
    output_equal = bool(torch.equal(graph_output, reference_tensor))
    return {
        "cycle": cycle_index,
        "shape": list(shape),
        "elements": element_count,
        "bytes": element_count * graph_input.element_size(),
        "capture_ms": _event_elapsed(capture_start, capture_end),
        "replay_ms": _event_elapsed(replay_start, replay_end),
        "output_equal": output_equal,
        "max_abs_difference": maximum_difference,
        "output_checksum": _tensor_checksum(graph_output),
    }


def _training_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    communicator: Any,
    device: torch.device,
    cycle_index: int,
) -> Dict[str, Any]:
    rank = dist.get_rank()
    token_count = WORKLOAD_SHAPES[cycle_index][0]
    generator = torch.Generator(device=device)
    generator.manual_seed(311_000 + cycle_index * 100 + rank)
    input_tensor = torch.randn(
        (token_count, MODEL_INPUT_FEATURES),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    target_tensor = torch.randn(
        (token_count, MODEL_INPUT_FEATURES),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    forward_start, forward_end = _cuda_event_pair()
    torch.cuda.synchronize()
    forward_start.record()
    output_tensor = model(input_tensor)
    loss_tensor = torch.nn.functional.mse_loss(output_tensor, target_tensor)
    loss_tensor.backward()
    forward_end.record()
    torch.cuda.synchronize()

    allreduce_start, allreduce_end = _cuda_event_pair()
    allreduce_start.record()
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_half = parameter.grad.to(torch.float16)
        if communicator.should_quick_allreduce(gradient_half):
            reduced_gradient = communicator.quick_all_reduce(gradient_half)
            parameter.grad.copy_(reduced_gradient.to(torch.float32))
        else:
            dist.all_reduce(parameter.grad)
    allreduce_end.record()
    torch.cuda.synchronize()

    optimizer_start, optimizer_end = _cuda_event_pair()
    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(loss_tensor.detach().item()),
        "forward_backward_ms": _event_elapsed(forward_start, forward_end),
        "gradient_allreduce_ms": _event_elapsed(allreduce_start, allreduce_end),
        "optimizer_ms": _event_elapsed(optimizer_start, optimizer_end),
    }


def _weight_state(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    ).cpu()


def _run_worker() -> Dict[str, Any]:
    device, gloo_group = _initialize_worker()
    rank = dist.get_rank()
    pause_resume = _pause_resume_probe(device)
    communicator, path_metadata = _select_custom_all_reduce(
        gloo_group,
        device,
    )
    assert not communicator.disabled, "Aiter custom all-reduce initialized disabled"

    torch.manual_seed(2026)
    model = torch.nn.Sequential(
        torch.nn.Linear(MODEL_INPUT_FEATURES, MODEL_HIDDEN_FEATURES),
        torch.nn.GELU(),
        torch.nn.Linear(MODEL_HIDDEN_FEATURES, MODEL_INPUT_FEATURES),
    ).to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    cycle_metrics: List[Dict[str, Any]] = []
    for cycle_index, shape in enumerate(WORKLOAD_SHAPES):
        collective_metric = _capture_replay_all_reduce(
            communicator,
            gloo_group,
            device,
            shape,
            cycle_index,
        )
        training_metric = _training_update(
            model,
            optimizer,
            communicator,
            device,
            cycle_index,
        )
        weight_tensor = _weight_state(model)
        gathered_weights = [torch.empty_like(weight_tensor) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_weights, weight_tensor, group=gloo_group)
        weights_equal = all(
            torch.equal(weight_tensor, gathered_weight)
            for gathered_weight in gathered_weights
        )
        cycle_metrics.append(
            {
                **collective_metric,
                **training_metric,
                "weights_equal_across_ranks": weights_equal,
                "weight_checksum": _tensor_checksum(weight_tensor),
            }
        )

    dist.barrier(group=gloo_group)
    communicator.close()
    result = {
        "context": os.environ["FIXTURE_CONTEXT"],
        "rank": rank,
        "world_size": WORLD_SIZE,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cycle_count": len(cycle_metrics),
        "cycle_metrics": cycle_metrics,
        "pause_resume": pause_resume,
        "custom_all_reduce_path": path_metadata,
        "final_weight_checksum": cycle_metrics[-1]["weight_checksum"],
    }
    dist.destroy_process_group()
    return result


def _write_worker_result(result: Dict[str, Any]) -> None:
    result_path = Path(os.environ["FIXTURE_RESULT_PATH"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _run_standalone() -> List[Dict[str, Any]]:
    master_port = _free_port()
    with tempfile.TemporaryDirectory(prefix="rocm-allreduce-") as temporary_directory:
        result_paths = [
            Path(temporary_directory) / f"rank-{rank}.json"
            for rank in range(WORLD_SIZE)
        ]
        processes = []
        for rank, result_path in enumerate(result_paths):
            environment = _worker_environment(
                rank,
                master_port,
                result_path,
                colocated=False,
            )
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, "--mode", "worker"],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for rank, process in enumerate(processes):
            stdout, stderr = process.communicate(timeout=SUBPROCESS_TIMEOUT_SECONDS)
            if process.returncode != 0:
                raise RuntimeError(
                    f"standalone rank {rank} failed with code {process.returncode}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
        return [json.loads(result_path.read_text()) for result_path in result_paths]


def _run_ray_colocated() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    master_port = _free_port()
    os.environ["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] = "0"
    os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "1"
    os.environ["SGLANG_USE_AITER_AR"] = "0"
    os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "0"
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "FP"
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "0"
    os.environ["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = "64"
    os.environ["AITER_CUSTOM_AR_MAX_SIZE"] = str(MAX_CUSTOM_ALL_REDUCE_BYTES)
    os.environ["AITER_CUSTOM_AR_MIN_SIZE"] = "0"
    memory_saver_hook = (
        "/opt/venv/lib/python3.10/site-packages/"
        "torch_memory_saver_hook_mode_preload.abi3.so"
    )
    if Path(memory_saver_hook).exists():
        existing_preload = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = (
            f"{memory_saver_hook}:{existing_preload}"
            if existing_preload
            else memory_saver_hook
        )
    ray.init(
        num_cpus=2,
        num_gpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level="WARNING",
    )
    placement_group = ray.util.placement_group(
        [{"CPU": 1, "GPU": 1} for _ in range(WORLD_SIZE)],
        strategy="PACK",
    )
    ray.get(placement_group.ready(), timeout=60)

    class ColocatedPeer:
        def run(self) -> Dict[str, Any]:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
            torch.manual_seed(2027)
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
            ).to(device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            start_event, end_event = _cuda_event_pair()
            start_event.record()
            for cycle_index in range(32):
                input_tensor = torch.randn(
                    (32, 128),
                    device=device,
                    dtype=torch.float32,
                )
                target_tensor = torch.randn(
                    (32, 128),
                    device=device,
                    dtype=torch.float32,
                )
                loss_tensor = torch.nn.functional.mse_loss(
                    model(input_tensor),
                    target_tensor,
                )
                loss_tensor.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            end_event.record()
            torch.cuda.synchronize()
            return {
                "device_name": torch.cuda.get_device_name(device),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "iterations": 32,
                "gpu_ms": _event_elapsed(start_event, end_event),
                "final_loss": float(loss_tensor.detach().item()),
            }

    class DistributedWorker:
        def run(self, rank: int, master_port: int) -> Dict[str, Any]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(master_port)
            os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = "0"
            os.environ["FIXTURE_CONTEXT"] = "ray_colocated"
            return _run_worker()

    ColocatedPeerActor = ray.remote(ColocatedPeer)
    DistributedWorkerActor = ray.remote(DistributedWorker)

    worker_actors = []
    peer_actors = []
    for bundle_index in range(WORLD_SIZE):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=bundle_index,
        )
        worker_actors.append(
            DistributedWorkerActor.options(
                num_cpus=0.5,
                num_gpus=0.5,
                scheduling_strategy=scheduling_strategy,
            ).remote()
        )
        peer_actors.append(
            ColocatedPeerActor.options(
                num_cpus=0.5,
                num_gpus=0.5,
                scheduling_strategy=scheduling_strategy,
            ).remote()
        )

    peer_refs = [peer.run.remote() for peer in peer_actors]
    worker_refs = [
        worker.run.remote(rank, master_port)
        for rank, worker in enumerate(worker_actors)
    ]
    worker_results = ray.get(worker_refs, timeout=RAY_TIMEOUT_SECONDS)
    peer_results = ray.get(peer_refs, timeout=RAY_TIMEOUT_SECONDS)
    ray.shutdown()
    return worker_results, peer_results


def _summarize_context(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rank_zero = results[0]
    capture_times = [cycle["capture_ms"] for cycle in rank_zero["cycle_metrics"]]
    replay_times = [cycle["replay_ms"] for cycle in rank_zero["cycle_metrics"]]
    forward_times = [
        cycle["forward_backward_ms"] for cycle in rank_zero["cycle_metrics"]
    ]
    allreduce_times = [
        cycle["gradient_allreduce_ms"] for cycle in rank_zero["cycle_metrics"]
    ]
    optimizer_times = [cycle["optimizer_ms"] for cycle in rank_zero["cycle_metrics"]]
    return {
        "cycle_count": rank_zero["cycle_count"],
        "all_output_equal": all(cycle["output_equal"] for cycle in rank_zero["cycle_metrics"]),
        "maximum_output_difference": max(
            cycle["max_abs_difference"] for cycle in rank_zero["cycle_metrics"]
        ),
        "weights_equal_across_ranks": all(
            cycle["weights_equal_across_ranks"]
            for cycle in rank_zero["cycle_metrics"]
        ),
        "capture_ms_mean": sum(capture_times) / len(capture_times),
        "capture_ms_max": max(capture_times),
        "replay_ms_mean": sum(replay_times) / len(replay_times),
        "replay_ms_max": max(replay_times),
        "forward_backward_ms_mean": sum(forward_times) / len(forward_times),
        "gradient_allreduce_ms_mean": sum(allreduce_times) / len(allreduce_times),
        "optimizer_ms_mean": sum(optimizer_times) / len(optimizer_times),
        "final_weight_checksum": rank_zero["final_weight_checksum"],
        "pause_resume": rank_zero["pause_resume"],
        "custom_all_reduce_path": rank_zero["custom_all_reduce_path"],
    }


def _compare_contexts(
    standalone: Sequence[Dict[str, Any]],
    colocated: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    standalone_cycles = standalone[0]["cycle_metrics"]
    colocated_cycles = colocated[0]["cycle_metrics"]
    output_differences = [
        abs(
            standalone_cycle["output_checksum"]
            - colocated_cycle["output_checksum"]
        )
        for standalone_cycle, colocated_cycle in zip(
            standalone_cycles,
            colocated_cycles,
            strict=True,
        )
    ]
    weight_difference = abs(
        standalone[0]["final_weight_checksum"]
        - colocated[0]["final_weight_checksum"]
    )
    return {
        "cycle_count": len(standalone_cycles),
        "maximum_output_checksum_difference": max(output_differences),
        "final_weight_checksum_difference": weight_difference,
        "outputs_match": max(output_differences) == 0.0,
        "weights_match": weight_difference == 0.0,
    }


def _collect_source_metadata() -> Dict[str, Any]:
    def git_commit(path: str) -> Optional[str]:
        try:
            output = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return output.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import aiter
    import sglang

    return {
        "miles_commit": git_commit("/job/JOB_WORKDIR/miles"),
        "sglang_commit": git_commit("/sgl-workspace/sglang"),
        "aiter_commit": git_commit("/sgl-workspace/aiter"),
        "sglang_path": sglang.__file__,
        "aiter_path": aiter.__file__,
        "quick_all_reduce_python_path": (
            "/sgl-workspace/sglang/python/sglang/srt/distributed/"
            "device_communicators/quick_all_reduce.py"
        ),
        "quick_all_reduce_native_source": (
            "/sgl-workspace/sglang/python/sglang/kernels/aot/csrc/allreduce/"
            "quick_all_reduce.hip"
        ),
        "quick_all_reduce_native_header": (
            "/sgl-workspace/sglang/python/sglang/kernels/aot/csrc/allreduce/"
            "quick_all_reduce_hip.h"
        ),
        "quick_all_reduce_native_extension": (
            "/opt/venv/lib/python3.10/site-packages/sgl_kernel/"
            "common_ops.cpython-310-x86_64-linux-gnu.so"
        ),
        "torch_path": torch.__file__,
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "ray", "worker"], required=True)
    arguments = parser.parse_args()

    if arguments.mode == "worker":
        result = _run_worker()
        _write_worker_result(result)
        return

    standalone_results = _run_standalone()
    colocated_results, peer_results = _run_ray_colocated()
    report = {
        "source_metadata": _collect_source_metadata(),
        "standalone": {
            "summary": _summarize_context(standalone_results),
            "ranks": list(standalone_results),
        },
        "ray_colocated": {
            "summary": _summarize_context(colocated_results),
            "ranks": list(colocated_results),
            "colocated_peers": peer_results,
        },
        "context_comparison": _compare_contexts(
            standalone_results,
            colocated_results,
        ),
    }
    output_path = Path(
        "/job/JOB_WORKDIR/miles/reports/j-211b8b17dedb/results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["context_comparison"], indent=2))


if __name__ == "__main__":
    main()
