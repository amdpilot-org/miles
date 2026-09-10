from __future__ import annotations

import base64
import io
import multiprocessing as mp
import threading
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request


OLD_OUTPUT = "old"
NEW_OUTPUT = "new"
OLD_VERSION = "old-v0"
NEW_VERSION = "new-v1"
OLD_WEIGHT_VALUE = 1.0
NEW_WEIGHT_VALUE = 2.0


def _worker_main(
    gpu_id: int,
    worker_id: str,
    old_value: float,
    new_value: float,
    port: int,
    ready_queue: mp.Queue,
) -> None:
    device = torch.device("cuda", 0)
    weight = torch.tensor([old_value], device=device)
    weight_version = OLD_VERSION
    stop_event = threading.Event()
    startup_event = threading.Event()

    app = FastAPI()
    app.router.on_startup.append(startup_event.set)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health_generate")
    async def health_generate() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/gate/activate")
    async def activate_gate() -> dict[str, str]:
        return {"status": "activated"}

    @app.get("/model_info")
    async def model_info() -> dict[str, str]:
        return {"weight_version": weight_version}

    @app.get("/get_weight_version")
    async def get_weight_version() -> dict[str, str]:
        return {"weight_version": weight_version}

    @app.post("/generate")
    async def generate(request: Request) -> dict[str, Any]:
        await request.json()
        value = weight.item()
        output = OLD_OUTPUT if value == old_value else NEW_OUTPUT
        return {
            "text": output,
            "value": value,
            "weight_version": weight_version,
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "gpu_name": torch.cuda.get_device_name(0),
        }

    @app.post("/update_weights_from_tensor")
    async def update_weights_from_tensor(request: Request) -> dict[str, Any]:
        nonlocal weight, weight_version
        payload = await request.json()
        serialized = payload["serialized_named_tensors"][0]
        state = torch.load(
            io.BytesIO(base64.b64decode(serialized)),
            map_location="cpu",
            weights_only=True,
        )
        weight.copy_(state["weight"].to(device))
        weight_version = payload.get("weight_version", NEW_VERSION)
        return {"success": True, "weight_version": weight_version}

    @app.post("/fixture/shutdown")
    async def shutdown_worker() -> dict[str, str]:
        stop_event.set()
        return {"status": "stopping"}

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name=f"fixture-worker-{worker_id}", daemon=True)
    server_thread.start()
    if not startup_event.wait(15.0):
        ready_queue.put_nowait(("error", {"worker_id": worker_id, "error": "worker startup timed out"}))
        return

    ready_queue.put_nowait(
        (
            "ready",
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "port": port,
                "torch_version": torch.__version__,
            },
        )
    )

    stop_event.wait()
    server.should_exit = True
    server_thread.join(5.0)
