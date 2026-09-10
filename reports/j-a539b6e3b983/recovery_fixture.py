from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx
import torch
import uvicorn

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.cell_state import CellAddrInfo
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.ray.train.group import TrainerController
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.retry_utils import NonRetryableError

import recovery_worker
from recovery_worker import _worker_main

OLD_OUTPUT = recovery_worker.OLD_OUTPUT
NEW_OUTPUT = recovery_worker.NEW_OUTPUT
NEW_VERSION = recovery_worker.NEW_VERSION
OLD_WEIGHT_VALUE = recovery_worker.OLD_WEIGHT_VALUE
NEW_WEIGHT_VALUE = recovery_worker.NEW_WEIGHT_VALUE
REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _serialize_weight(value: float) -> str:
    buffer = io.BytesIO()
    torch.save({"weight": torch.tensor([value])}, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FixtureActorCell:
    def __init__(self, serialized_weight: str) -> None:
        self.cell_id = "actor-cell-0"
        self.cell_index = 0
        self.is_alive = True
        self.serialized_weight = serialized_weight
        self.should_fail = False

    async def execute(self, fn_name: str, **kwargs: Any) -> list[str]:
        if fn_name != "update_weights":
            raise RuntimeError(f"fixture actor cell does not implement {fn_name}")
        if self.should_fail:
            raise NonRetryableError("injected fixture actor weight-update failure")
        info = kwargs["info"]
        await asyncio.gather(
            *[
                engine.update_weights_from_tensor(
                    [self.serialized_weight],
                    weight_version=NEW_VERSION,
                )
                for engine in info.rollout_engines
            ]
        )
        return [NEW_VERSION]


class FixtureRun:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, step: str, **details: Any) -> None:
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "step": step,
                **details,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {"events": self.events}


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_external=False,
        check_weight_update_equal=False,
        check_weight_update_skip_list=[],
        debug_rollout_only=False,
        debug_train_only=False,
        colocate=False,
        use_miles_router=True,
        ft_components=[],
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        indep_dp=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        witness_buffer_size=1,
        enable_witness=False,
        save_debug_event_data=None,
        ci_ft_test_actions=None,
    )


def _start_worker(
    context: mp.context.BaseContext,
    gpu_id: int,
    worker_id: str,
    ready_queue: mp.Queue,
) -> tuple[mp.process.BaseProcess, dict[str, Any]]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = context.Process(
        target=_worker_main,
        args=(gpu_id, worker_id, OLD_WEIGHT_VALUE, NEW_WEIGHT_VALUE, port, ready_queue),
        name=f"fixture-worker-{worker_id}",
    )
    process.daemon = True
    process.start()
    if previous_visible_devices is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices

    try:
        status, payload = ready_queue.get(timeout=60.0)
    except queue.Empty as exc:
        process.terminate()
        process.join(1.0)
        raise TimeoutError(f"worker {worker_id} did not report readiness within 60s") from exc
    if status != "ready":
        process.terminate()
        process.join(1.0)
        raise RuntimeError(f"worker {worker_id} failed to start: {payload}")
    return process, payload


async def _start_router() -> tuple[MilesRouter, uvicorn.Server, asyncio.Task[None], int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        router_port = listener.getsockname()[1]
    router = MilesRouter(
        MilesRouterConfig(
            host="127.0.0.1",
            port=router_port,
            max_connections=16,
            timeout=10.0,
            health_check_interval=3600.0,
            health_check_failure_threshold=100,
        )
    )
    startup_event = asyncio.Event()
    router.app.router.on_startup.append(startup_event.set)
    config = uvicorn.Config(router.app, host="127.0.0.1", port=router_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve(), name="fixture-miles-router")
    await asyncio.wait_for(startup_event.wait(), timeout=15.0)
    return router, server, server_task, router_port


async def _stop_worker(worker: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(f"{worker['url']}/fixture/shutdown")
        response.raise_for_status()


async def _router_workers(client: httpx.AsyncClient, router_port: int) -> list[str]:
    response = await client.get(f"http://127.0.0.1:{router_port}/list_workers")
    response.raise_for_status()
    return response.json()["urls"]


async def _generate(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    response = await client.post(f"{url}/generate", json={"text": "fixture probe"})
    response.raise_for_status()
    return response.json()


async def _run(output_path: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 assigned GPUs, found {torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("MI350X" not in name for name in gpu_names):
        raise RuntimeError(f"expected AMD Instinct MI350X GPUs, found {gpu_names}")

    evidence = FixtureRun()
    evidence.record(
        "environment",
        assigned_gpu_count=2,
        gpu_names=gpu_names,
        torch_version=torch.__version__,
        torch_module=torch.__file__,
        torch_native_module=getattr(torch._C, "__file__", None),
        miles_module=__import__("miles").__file__,
        sglang_module=__import__("sglang").__file__,
        sglang_version=__import__("sglang").__version__,
        megatron_path=list(__import__("megatron").__path__),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        process_group_created=False,
    )

    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    workers: list[dict[str, Any]] = []
    processes: list[mp.process.BaseProcess] = []
    router: MilesRouter | None = None
    router_server: uvicorn.Server | None = None
    router_task: asyncio.Task[None] | None = None
    router_port = 0
    rollout_server: RolloutServer | None = None
    original_compute_addr_info = ServerCell._compute_addr_info
    worker_urls_by_cell_id: dict[str, str] = {}

    async def fixture_compute_addr_info(cell: ServerCell) -> CellAddrInfo:
        worker_url = worker_urls_by_cell_id[cell.meta.cell_id]
        return CellAddrInfo(server_url=worker_url, bootstrap_port=None, gate_url=worker_url)

    ServerCell._compute_addr_info = fixture_compute_addr_info

    try:
        for gpu_id in range(2):
            worker_id = f"worker-{gpu_id}"
            process, worker_info = _start_worker(context, gpu_id, worker_id, ready_queue)
            processes.append(process)
            worker = {**worker_info, "url": f"http://127.0.0.1:{worker_info['port']}"}
            workers.append(worker)
            worker_urls_by_cell_id[f"rollout-cell-{gpu_id}"] = worker["url"]
            evidence.record(
                "worker_started",
                worker_id=worker_id,
                gpu_id=gpu_id,
                gpu_name=worker["gpu_name"],
                gpu_uuid=worker["gpu_uuid"],
                url=worker["url"],
            )

        router, router_server, router_task, router_port = await _start_router()
        evidence.record("router_started", url=f"http://127.0.0.1:{router_port}")

        args = _make_args()
        inference_controller = InferenceController(args)
        rollout_server = RolloutServer(
            server_cells={},
            args=args,
            context_lock=inference_controller.context_lock,
            router_ip="127.0.0.1",
            router_port=router_port,
            model_name="fixture-model",
            update_weights=True,
            expected_num_cells=2,
        )
        inference_controller.servers = {"rollout": rollout_server}

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            for gpu_id, worker in enumerate(workers):
                cell_id = f"rollout-cell-{gpu_id}"
                metadata = ServerCellMetadata(
                    model_id="fixture-model",
                    worker_type="regular",
                    cell_id=cell_id,
                    num_gpus_per_engine=1,
                    gpu_offset=gpu_id,
                    sglang_api_key=None,
                    worker_name=f"fixture-worker-{gpu_id}",
                    needs_offload=False,
                    update_weights=True,
                    workers_hash=f"initial-{gpu_id}",
                )
                async with rollout_server.context_lock:
                    await rollout_server.add_cell(metadata)
                cell = rollout_server.server_cells[cell_id]
                await cell.tick()
                direct_output = await _generate(client, worker["url"])
                evidence.record(
                    "recovered_engine_pending_weights",
                    cell_id=cell_id,
                    state=type(cell._state).__name__,
                    direct_output=direct_output,
                )

            initial_router_workers = await _router_workers(client, router_port)
            evidence.record("router_before_first_update", workers=initial_router_workers)
            if initial_router_workers:
                raise AssertionError("updatable engines were registered before the first weight update")

            rejected = await client.post(
                f"http://127.0.0.1:{router_port}/generate",
                json={"text": "must not be admitted"},
            )
            evidence.record(
                "traffic_rejected_before_synchronization",
                status_code=rejected.status_code,
                response_body=rejected.text,
            )
            if rejected.status_code == 200:
                raise AssertionError("router admitted traffic before synchronized engines existed")

            trainer = TrainerController(
                args,
                inference_controller=inference_controller,
                rollout_executor=None,
                role="actor",
                with_ref=False,
            )
            actor_cell = FixtureActorCell(_serialize_weight(NEW_WEIGHT_VALUE))
            trainer._cells_by_id[actor_cell.cell_id] = actor_cell

            first_version = await trainer.update_weights(rollout_id=1)
            evidence.record("first_update_completed", returned_weight_version=first_version)
            if first_version != NEW_VERSION:
                raise AssertionError(f"expected {NEW_VERSION}, got {first_version}")

            first_router_workers = await _router_workers(client, router_port)
            evidence.record("router_after_first_update", workers=first_router_workers)
            if set(first_router_workers) != {worker["url"] for worker in workers}:
                raise AssertionError("both synchronized engines were not registered")

            first_traffic = await asyncio.gather(
                *[_generate(client, f"http://127.0.0.1:{router_port}") for _ in range(4)]
            )
            evidence.record("first_synchronized_traffic", outputs=first_traffic)
            if any(item["text"] != NEW_OUTPUT or item["weight_version"] != NEW_VERSION for item in first_traffic):
                raise AssertionError("router admitted an unsynchronized weight version after first update")
            if {item["worker_id"] for item in first_traffic} != {worker["worker_id"] for worker in workers}:
                raise AssertionError("first synchronized traffic did not exercise both GPUs")

            recovered_gpu_id = 1
            recovered_cell_id = f"rollout-cell-{recovered_gpu_id}"
            async with rollout_server.context_lock:
                await rollout_server.remove_cell(recovered_cell_id)
            await _stop_worker(workers[recovered_gpu_id])
            processes[recovered_gpu_id].join(10.0)
            if processes[recovered_gpu_id].exitcode != 0:
                raise RuntimeError("old rollout worker did not exit cleanly")
            evidence.record("old_engine_stopped", cell_id=recovered_cell_id)

            replacement_process, replacement_info = _start_worker(
                context,
                recovered_gpu_id,
                "worker-1-replacement",
                ready_queue,
            )
            processes[recovered_gpu_id] = replacement_process
            replacement_worker = {
                **replacement_info,
                "url": f"http://127.0.0.1:{replacement_info['port']}",
            }
            workers[recovered_gpu_id] = replacement_worker
            worker_urls_by_cell_id[recovered_cell_id] = replacement_worker["url"]
            replacement_metadata = ServerCellMetadata(
                model_id="fixture-model",
                worker_type="regular",
                cell_id=recovered_cell_id,
                num_gpus_per_engine=1,
                gpu_offset=recovered_gpu_id,
                sglang_api_key=None,
                worker_name="fixture-worker-1-replacement",
                needs_offload=False,
                update_weights=True,
                workers_hash="replacement-1",
            )
            async with rollout_server.context_lock:
                await rollout_server.add_cell(replacement_metadata)
            replacement_cell = rollout_server.server_cells[recovered_cell_id]
            await replacement_cell.tick()
            replacement_direct_output = await _generate(client, replacement_worker["url"])
            recovery_router_workers = await _router_workers(client, router_port)
            evidence.record(
                "replacement_recovered_before_update",
                cell_id=recovered_cell_id,
                state=type(replacement_cell._state).__name__,
                direct_output=replacement_direct_output,
                router_workers=recovery_router_workers,
            )
            if replacement_cell.is_serving or replacement_worker["url"] in recovery_router_workers:
                raise AssertionError("recovered engine was admitted before its weight update")

            recovery_traffic = await asyncio.gather(
                *[_generate(client, f"http://127.0.0.1:{router_port}") for _ in range(4)]
            )
            evidence.record("traffic_during_recovery_window", outputs=recovery_traffic)
            if any(item["worker_id"] != "worker-0" for item in recovery_traffic):
                raise AssertionError("router admitted the unsynchronized replacement during recovery")
            if any(item["text"] != NEW_OUTPUT or item["weight_version"] != NEW_VERSION for item in recovery_traffic):
                raise AssertionError("router served an unintended weight version during recovery")

            actor_cell.should_fail = True
            try:
                await trainer.update_weights(rollout_id=2)
            except NonRetryableError as exc:
                inference_controller.context_lock.reattach()
                inference_controller.context_lock.release()
                evidence.record(
                    "negative_failed_update_left_replacement_unavailable",
                    error=str(exc),
                    replacement_state=type(replacement_cell._state).__name__,
                    router_workers=await _router_workers(client, router_port),
                )
            else:
                raise AssertionError("injected update failure unexpectedly succeeded")
            finally:
                actor_cell.should_fail = False
            if replacement_cell.is_serving:
                raise AssertionError("failed update incorrectly marked replacement as serving")

            recovered_version = await trainer.update_weights(rollout_id=3)
            evidence.record("recovery_update_completed", returned_weight_version=recovered_version)
            if recovered_version != NEW_VERSION:
                raise AssertionError(f"expected {NEW_VERSION}, got {recovered_version}")

            final_router_workers = await _router_workers(client, router_port)
            evidence.record("router_after_recovery_update", workers=final_router_workers)
            if set(final_router_workers) != {worker["url"] for worker in workers}:
                raise AssertionError("both synchronized engines were not registered after recovery")

            final_traffic = await asyncio.gather(
                *[_generate(client, f"http://127.0.0.1:{router_port}") for _ in range(4)]
            )
            evidence.record("final_synchronized_traffic", outputs=final_traffic)
            if any(item["text"] != NEW_OUTPUT or item["weight_version"] != NEW_VERSION for item in final_traffic):
                raise AssertionError("router admitted an unsynchronized weight version after recovery")
            if {item["worker_id"] for item in final_traffic} != {worker["worker_id"] for worker in workers}:
                raise AssertionError("final synchronized traffic did not exercise both GPUs")

            async with rollout_server.context_lock:
                await rollout_server.dispose()
            evidence.record("rollout_server_disposed", cell_ids=list(rollout_server.server_cells))

    finally:
        ServerCell._compute_addr_info = original_compute_addr_info
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evidence.as_dict(), indent=2) + "\n", encoding="utf-8")
        if rollout_server is not None:
            try:
                for cell in list(rollout_server.server_cells.values()):
                    await cell.dispose()
                rollout_server.server_cells.clear()
            except Exception:
                pass
        for worker, process in zip(workers, processes):
            if process.is_alive():
                try:
                    await _stop_worker(worker)
                except Exception:
                    pass
                process.join(5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)
        for process in processes:
            if process.is_alive():
                process.join(1.0)
        if router_server is not None and router_task is not None:
            router_server.should_exit = True
            await asyncio.wait_for(router_task, timeout=10.0)
        if router is not None:
            await router.client.aclose()

    return evidence.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports/j-a539b6e3b983/evidence.json",
        help="path for the ordered JSON evidence record",
    )
    args = parser.parse_args()
    result = asyncio.run(_run(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
