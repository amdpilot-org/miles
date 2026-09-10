#!/usr/bin/env python3
"""Two-GPU actor/rollout recovery-order fixture for Miles issue 1724.

Run with:
    /opt/venv/bin/python -m torch.distributed.run --nproc-per-node=2 \
        reports/j-c433160fb517/recovery_fixture.py

The fixture uses the real Miles ServerCell, RolloutServer, InferenceController, and
SGLang API/router clients. Only worker-address discovery is replaced with fixed local
URLs; the local HTTP servers implement the small subset needed by those clients.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


def use_readonly_aiter_configs() -> None:
    """Read Aiter's installed config files without writing its shared /tmp cache."""
    import aiter.jit.core as aiter_core

    aiter_core.AITER_CONFIGS.get_config_file = lambda env_name, default_file, tuned_file_name: os.getenv(
        env_name, default_file
    )


use_readonly_aiter_configs()

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.cell_state import CellAddrInfo
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock


OLD_WEIGHT = 7.0
NEW_WEIGHT = 13.0
OLD_VERSION = "old-v1"
NEW_VERSION = "new-v2"
HOST = "127.0.0.1"
ENGINE_PORTS = (31000, 31001)
ROUTER_PORT = 32000
ENGINE_URLS = tuple(f"http://{HOST}:{port}" for port in ENGINE_PORTS)
ROUTER_URL = f"http://{HOST}:{ROUTER_PORT}"
PROCESS_GROUP_TIMEOUT_SECONDS = 90
HTTP_TIMEOUT_SECONDS = 15
EVIDENCE_PATH = Path(__file__).with_name("evidence.jsonl")


@dataclasses.dataclass
class Evidence:
    path: Path
    sequence: int = 0

    def record(self, event: str, **fields: Any) -> None:
        self.sequence += 1
        entry = {"sequence": self.sequence, "event": event, **fields}
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def check(self, condition: bool, event: str, **fields: Any) -> None:
        self.record(event, passed=bool(condition), **fields)
        if not condition:
            raise AssertionError(event)



@dataclasses.dataclass
class EngineState:
    model: torch.nn.Module
    version: str
    fail_next: bool = False
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def set_weight(self, weight: float, version: str) -> None:
        with self.lock, torch.no_grad():
            self.model.weight.fill_(weight)
            self.version = version

    def output(self) -> float:
        with self.lock, torch.no_grad():
            return float(self.model(torch.tensor([1.0], device=self.model.weight.device)).item())

    def serialized_weight(self) -> str:
        with self.lock, torch.no_grad():
            payload = io.BytesIO()
            torch.save({"weight": self.model.weight.detach().cpu()}, payload)
        return base64.b64encode(payload.getvalue()).decode("ascii")


def make_engine_app(state: EngineState) -> FastAPI:
    ready = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ready.set()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/gate/activate")
    async def activate_gate() -> dict[str, bool]:
        return {"success": True}

    @app.get("/health_generate")
    async def health_generate() -> dict[str, bool]:
        return {"success": True}

    @app.get("/flush_cache")
    async def flush_cache() -> dict[str, bool]:
        return {"success": True}

    @app.get("/model_info")
    async def model_info() -> dict[str, str]:
        with state.lock:
            return {"weight_version": state.version}

    @app.post("/generate")
    async def generate() -> dict[str, float | str]:
        return {"output": state.output(), "weight_version": state.version}

    @app.post("/update_weights_from_tensor")
    async def update_weights(payload: dict[str, Any]) -> dict[str, bool | str]:
        if state.fail_next:
            state.fail_next = False
            raise HTTPException(status_code=500, detail="injected weight-sync failure")
        serialized = payload["serialized_named_tensors"][0]
        loaded = torch.load(
            io.BytesIO(base64.b64decode(serialized)),
            map_location=state.model.weight.device,
            weights_only=True,
        )
        with state.lock, torch.no_grad():
            state.model.weight.copy_(loaded["weight"].to(state.model.weight.device))
            state.version = str(payload["weight_version"])
        return {"success": True, "weight_version": state.version}

    @app.post("/test/restart_with_old_weights")
    async def restart_with_old_weights() -> dict[str, bool | str]:
        state.set_weight(OLD_WEIGHT, OLD_VERSION)
        return {"success": True, "weight_version": state.version}

    @app.post("/test/fail_next_update")
    async def fail_next_update() -> dict[str, bool]:
        state.fail_next = True
        return {"success": True}

    app.state.ready = ready
    return app


@dataclasses.dataclass
class RouterState:
    workers: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)
    next_id: int = 1


def make_router_app(state: RouterState) -> FastAPI:
    ready = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ready.set()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/workers")
    async def add_worker(payload: dict[str, Any]) -> dict[str, int | str]:
        for worker_id, worker in state.workers.items():
            if worker["url"] == payload["url"]:
                return {"id": worker_id, "url": worker["url"]}
        worker_id = state.next_id
        state.next_id += 1
        state.workers[worker_id] = {"id": worker_id, "url": payload["url"], "worker_type": payload["worker_type"]}
        return {"id": worker_id, "url": payload["url"]}

    @app.get("/workers")
    async def workers() -> dict[str, list[dict[str, Any]]]:
        return {"workers": list(state.workers.values())}

    @app.delete("/workers/{worker_id}")
    async def remove_worker(worker_id: int) -> dict[str, bool]:
        if worker_id not in state.workers:
            raise HTTPException(status_code=404, detail="worker not found")
        del state.workers[worker_id]
        return {"success": True}

    @app.post("/generate")
    async def generate() -> dict[str, Any]:
        if not state.workers:
            return JSONResponse(status_code=503, content={"detail": "no workers registered"})
        worker = next(iter(state.workers.values()))
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{worker['url']}/generate")
            response.raise_for_status()
            return {**response.json(), "worker_url": worker["url"]}

    app.state.ready = ready
    return app


async def start_server(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await app.state.ready.wait()
    server.serve_task = task
    return server


async def stop_server(server: uvicorn.Server) -> None:
    server.should_exit = True
    task = getattr(server, "serve_task", None)
    if task is not None:
        await task


def install_fixed_addresses() -> None:
    async def compute_addr_info(self: ServerCell) -> CellAddrInfo:
        engine_url = ENGINE_URLS[self.meta.gpu_offset]
        return CellAddrInfo(server_url=engine_url, bootstrap_port=None, gate_url=engine_url)

    ServerCell._compute_addr_info = compute_addr_info


def fixture_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_external=False,
        check_weight_update_equal=False,
        debug_rollout_only=False,
        colocate=False,
        ft_components=[],
        use_miles_router=False,
    )


def cell_metadata(gpu_offset: int) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id=f"engine-{gpu_offset}",
        num_gpus_per_engine=1,
        gpu_offset=gpu_offset,
        sglang_api_key=None,
        worker_name=f"fixture-engine-{gpu_offset}",
        needs_offload=False,
        update_weights=True,
        workers_hash="fixture-workers-v1",
    )


async def add_ready_cell(rollout_server: RolloutServer, gpu_offset: int) -> ServerCell:
    metadata = cell_metadata(gpu_offset)
    await rollout_server.add_cell(metadata)
    cell = rollout_server.server_cells[metadata.cell_id]
    await cell.tick()
    return cell


async def router_traffic() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{ROUTER_URL}/generate")
        response.raise_for_status()
        return response.json()


async def registered_worker_urls() -> list[str]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{ROUTER_URL}/workers")
        response.raise_for_status()
        return sorted(worker["url"] for worker in response.json()["workers"])


async def engine_versions(clients: list[SGLangApiClient]) -> list[str]:
    return [str(await client.get_weight_version()) for client in clients]


async def broadcast_actor_weight(weight: float) -> None:
    command = torch.tensor([1], dtype=torch.int64)
    await asyncio.to_thread(torch.distributed.broadcast, command, src=0)
    tensor = torch.tensor([weight], dtype=torch.float32)
    await asyncio.to_thread(torch.distributed.broadcast, tensor, src=0)


async def worker_loop(actor_model: torch.nn.Module) -> None:
    while True:
        command = torch.empty(1, dtype=torch.int64)
        await asyncio.to_thread(torch.distributed.broadcast, command, src=0)
        if int(command.item()) == 0:
            return
        weight = torch.empty(1, dtype=torch.float32)
        await asyncio.to_thread(torch.distributed.broadcast, weight, src=0)
        with torch.no_grad():
            actor_model.weight.copy_(weight)


async def push_weights(
    controller: InferenceController,
    actor_model: torch.nn.Module,
    weight: float,
    version: str,
) -> None:
    await broadcast_actor_weight(weight)
    with torch.no_grad():
        actor_model.weight.fill_(weight)
    info = await controller.start_update_weights()
    serialized = [io.BytesIO()]
    torch.save({"weight": torch.tensor([weight])}, serialized[0])
    encoded = base64.b64encode(serialized[0].getvalue()).decode("ascii")
    await asyncio.gather(
        *[
            client.update_weights_from_tensor(
                serialized_named_tensors=[encoded],
                load_format="fixture",
                weight_version=version,
            )
            for client in info.rollout_engines
        ]
    )
    await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)


async def recover_engine_one(rollout_server: RolloutServer, context_lock: ContextLock) -> ServerCell:
    async with context_lock:
        await rollout_server.remove_cell("engine-1")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{ENGINE_URLS[1]}/test/restart_with_old_weights")
        response.raise_for_status()
    async with context_lock:
        return await add_ready_cell(rollout_server, 1)


def record_environment(evidence: Evidence) -> None:
    import megatron.core
    import sglang

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    evidence.record(
        "environment",
        git_commit=commit,
        world_size=int(os.environ["WORLD_SIZE"]),
        gpu_count=torch.cuda.device_count(),
        gpu_names=[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        torch_version=torch.__version__,
        torch_path=torch.__file__,
        torch_native_path=torch._C.__file__,
        sglang_version=sglang.__version__,
        sglang_path=sglang.__file__,
        megatron_path=megatron.core.__file__,
        miles_path=__import__("miles").__file__,
    )


async def coordinator(actor_model: torch.nn.Module, engine_model: torch.nn.Module) -> None:
    EVIDENCE_PATH.unlink(missing_ok=True)
    evidence = Evidence(EVIDENCE_PATH)
    record_environment(evidence)
    install_fixed_addresses()

    engine_state = EngineState(model=engine_model, version=OLD_VERSION)
    router_state = RouterState()
    engine_server = await start_server(make_engine_app(engine_state), HOST, ENGINE_PORTS[0])
    router_server = await start_server(make_router_app(router_state), HOST, ROUTER_PORT)

    readiness = torch.empty(1, dtype=torch.int64)
    await asyncio.to_thread(torch.distributed.broadcast, readiness, src=1)
    evidence.check(int(readiness.item()) == 1, "rank_one_engine_ready", readiness=int(readiness.item()))

    args = fixture_args()
    controller = InferenceController(args)
    rollout_server = RolloutServer(
        server_cells={},
        args=args,
        context_lock=controller.context_lock,
        router_ip=HOST,
        router_port=ROUTER_PORT,
        model_name="default",
        update_weights=True,
    )
    controller.servers = {"default": rollout_server}

    async with controller.context_lock:
        cell_zero = await add_ready_cell(rollout_server, 0)
        cell_one = await add_ready_cell(rollout_server, 1)
    evidence.check(
        cell_zero.is_pending_weights and cell_one.is_pending_weights,
        "initial_cells_pending_before_first_sync",
        cell_zero_state=type(cell_zero._state).__name__,
        cell_one_state=type(cell_one._state).__name__,
    )
    evidence.check(
        await registered_worker_urls() == [],
        "no_traffic_admitted_before_first_sync",
        registered_workers=await registered_worker_urls(),
    )

    await push_weights(controller, actor_model, NEW_WEIGHT, NEW_VERSION)
    traffic = await router_traffic()
    evidence.check(
        traffic["output"] == NEW_WEIGHT and traffic["weight_version"] == NEW_VERSION,
        "first_sync_admits_new_weights",
        output=traffic["output"],
        weight_version=traffic["weight_version"],
        worker_url=traffic["worker_url"],
    )

    cell_one = await recover_engine_one(rollout_server, controller.context_lock)
    direct_old = await SGLangApiClient(server_url=ENGINE_URLS[1]).get_weight_version()
    traffic_during_recovery = await router_traffic()
    evidence.check(
        cell_one.is_pending_weights
        and direct_old == OLD_VERSION
        and traffic_during_recovery["output"] == NEW_WEIGHT
        and traffic_during_recovery["weight_version"] == NEW_VERSION,
        "recovered_engine_unregistered_until_sync",
        recovered_state=type(cell_one._state).__name__,
        recovered_direct_version=direct_old,
        router_output=traffic_during_recovery["output"],
        router_version=traffic_during_recovery["weight_version"],
        router_worker_url=traffic_during_recovery["worker_url"],
    )

    await push_weights(controller, actor_model, NEW_WEIGHT, NEW_VERSION)
    versions = await engine_versions([SGLangApiClient(url) for url in ENGINE_URLS])
    traffic_after_sync = await router_traffic()
    evidence.check(
        versions == [NEW_VERSION, NEW_VERSION]
        and traffic_after_sync["output"] == NEW_WEIGHT
        and traffic_after_sync["weight_version"] == NEW_VERSION,
        "recovered_engine_admitted_after_sync",
        engine_versions=versions,
        router_output=traffic_after_sync["output"],
        router_version=traffic_after_sync["weight_version"],
    )

    cell_one = await recover_engine_one(rollout_server, controller.context_lock)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        await client.post(f"{ENGINE_URLS[1]}/test/fail_next_update")
    await broadcast_actor_weight(NEW_WEIGHT)
    with torch.no_grad():
        actor_model.weight.fill_(NEW_WEIGHT)
    info = await controller.start_update_weights()
    encoded = io.BytesIO()
    torch.save({"weight": torch.tensor([NEW_WEIGHT])}, encoded)
    encoded_text = base64.b64encode(encoded.getvalue()).decode("ascii")
    update_results = await asyncio.gather(
        *[
            client.update_weights_from_tensor(
                serialized_named_tensors=[encoded_text],
                load_format="fixture",
                weight_version=NEW_VERSION,
            )
            for client in info.rollout_engines
        ],
        return_exceptions=True,
    )
    failed = any(isinstance(result, BaseException) for result in update_results)
    failed_version = str(await SGLangApiClient(server_url=ENGINE_URLS[1]).get_weight_version())
    traffic_after_failure = await router_traffic()
    evidence.check(
        failed
        and cell_one.is_pending_weights
        and failed_version == OLD_VERSION
        and traffic_after_failure["output"] == NEW_WEIGHT
        and traffic_after_failure["weight_version"] == NEW_VERSION,
        "failed_sync_leaves_recovered_engine_unadmitted",
        update_failures=[type(result).__name__ for result in update_results],
        recovered_state=type(cell_one._state).__name__,
        recovered_direct_version=failed_version,
        router_output=traffic_after_failure["output"],
        router_version=traffic_after_failure["weight_version"],
        router_worker_url=traffic_after_failure["worker_url"],
    )

    controller.context_lock.reattach()
    controller.context_lock.release()
    await controller.dispose()
    await stop_server(router_server)
    await stop_server(engine_server)
    command = torch.tensor([0], dtype=torch.int64)
    await asyncio.to_thread(torch.distributed.broadcast, command, src=0)
    evidence.record("fixture_complete", passed=True)


async def rank_one(actor_model: torch.nn.Module, engine_model: torch.nn.Module) -> None:
    engine_state = EngineState(model=engine_model, version=OLD_VERSION)
    engine_server = await start_server(make_engine_app(engine_state), HOST, ENGINE_PORTS[1])
    readiness = torch.tensor([1], dtype=torch.int64)
    await asyncio.to_thread(torch.distributed.broadcast, readiness, src=1)
    await worker_loop(actor_model)
    await stop_server(engine_server)


async def async_main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"fixture requires exactly 2 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    device = torch.cuda.current_device()
    actor_model = torch.nn.Linear(1, 1, bias=False).to(device)
    engine_model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        actor_model.weight.fill_(OLD_WEIGHT)
        engine_model.weight.fill_(OLD_WEIGHT)
    if rank == 0:
        await coordinator(actor_model, engine_model)
    else:
        await rank_one(actor_model, engine_model)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(async_main())
