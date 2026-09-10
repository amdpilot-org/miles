#!/usr/bin/env python3
"""Launch the reduced two-GPU Miles rollout lifecycle investigation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import torch
import transformers
import sglang
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from torch_memory_saver.utils import get_binary_path_from_package


REPO_ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/health_generate", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise RuntimeError(f"SGLang server did not become healthy in {timeout}s: {last_error}")


def wait_for_router(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/list_workers", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise RuntimeError(f"Miles router did not become healthy in {timeout}s: {last_error}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def git_commit(path: str) -> str:
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_environment(model_path: Path, output_dir: Path) -> dict:
    torch_memory_saver_preload = Path(
        get_binary_path_from_package("torch_memory_saver_hook_mode_preload")
    ).resolve()
    return {
        "runtime_image": "amdpilotv2/miles-job:gbt350-d957-20260909",
        "python": "/opt/venv/bin/python",
        "miles_commit": git_commit(str(REPO_ROOT)),
        "miles_source": str(REPO_ROOT / "miles"),
        "sglang_commit": git_commit("/sgl-workspace/sglang"),
        "sglang_source": "/sgl-workspace/sglang/python/sglang",
        "sglang_version": sglang.__version__,
        "torch_version": torch.__version__,
        "torch_source": torch.__file__,
        "hip_version": torch.version.hip,
        "transformers_version": transformers.__version__,
        "transformers_source": transformers.__file__,
        "torch_memory_saver_version": package_version("torch_memory_saver"),
        "torch_memory_saver_preload": str(torch_memory_saver_preload),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
            }
            for index in range(torch.cuda.device_count())
        ],
        "model_path": str(model_path),
        "model_bytes_on_disk": directory_size(model_path),
        "downloaded_bytes": directory_size(model_path),
        "output_dir": str(output_dir),
    }


def start_sglang_server(
    *,
    model_path: str,
    port: int,
    nccl_port: int,
    device: int,
    enable_memory_saver: bool,
    log_path: Path,
    preload_path: str,
) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device),
            "HF_HOME": "/job/cache/hf",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "INFO",
            "SGLANG_USE_AITER": "0",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
            "NCCL_SOCKET_TIMEOUT_MS": "600000",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        "/opt/venv/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--nccl-port",
        str(nccl_port),
        "--mem-fraction-static",
        "0.20",
        "--disable-cuda-graph",
        *(["--enable-memory-saver", "--enable-weights-cpu-backup"] if enable_memory_saver else []),
        "--attention-backend",
        "triton",
        "--dtype",
        "bfloat16",
        "--enable-metrics",
        "--enable-request-time-stats-logging",
        "--decode-log-interval",
        "1",
    ]
    log_file = log_path.open("w")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )


def start_recovery_monitor(
    *,
    process: subprocess.Popen,
    stop_event: threading.Event,
    output_dir: Path,
    model_path: str,
    port: int,
    nccl_port: int,
    device: int,
    enable_memory_saver: bool,
    preload_path: str,
) -> tuple[dict, threading.Thread, dict]:
    holder = {"process": process}
    restart_count = {"value": 0}

    def monitor() -> None:
        current = process
        while not stop_event.is_set():
            current.wait()
            if stop_event.is_set() or restart_count["value"] >= 1:
                break
            restart_count["value"] += 1
            time.sleep(1.0)
            current = start_sglang_server(
                model_path=model_path,
                port=port,
                nccl_port=nccl_port,
                device=device,
                enable_memory_saver=enable_memory_saver,
                log_path=output_dir / f"sglang-a-recovery-{restart_count['value']}.log",
                preload_path=preload_path,
            )
            holder["process"] = current

    thread = threading.Thread(target=monitor, name="sglang-a-recovery", daemon=True)
    thread.start()
    return holder, thread, restart_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(Path(__file__).resolve().parent / "model"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--fail-wake-cycle", type=int, default=16)
    parser.add_argument("--fail-update-cycle", type=int, default=24)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--pg-timeout", type=float, default=300.0)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).resolve()

    server_a_port = free_port()
    server_b_port = free_port()
    router_port = free_port()
    torchrun_port = free_port()
    update_a_port = free_port()
    update_b_port = free_port()
    nccl_a_port = free_port()
    nccl_b_port = free_port()
    default_port = free_port()

    server_a_url = f"http://127.0.0.1:{server_a_port}"
    server_b_url = f"http://127.0.0.1:{server_b_port}"
    router_url = f"http://127.0.0.1:{router_port}"

    preload_path = str(
        Path(get_binary_path_from_package("torch_memory_saver_hook_mode_preload")).resolve()
    )

    router_config = MilesRouterConfig(
        host="127.0.0.1",
        port=router_port,
        max_connections=64,
        timeout=30.0,
        health_check_interval=300.0,
        health_check_failure_threshold=100,
    )
    router = MilesRouter(router_config, verbose=False)
    router_server = UvicornThreadServer(router.app, host=router_config.host, port=router_config.port)
    router_server.start()

    server_a_holder = None
    server_a_recovery_thread = None
    server_a_restart_count = None
    server_a_stop_event = None
    server_b = None
    actor = None
    actor_log = None
    actor_file = None
    try:
        wait_for_router(router_url, 30.0)
        server_a_process = start_sglang_server(
            model_path=str(model_path),
            port=server_a_port,
            nccl_port=nccl_a_port,
            device=1,
            enable_memory_saver=True,
            log_path=output_dir / "sglang-a.log",
            preload_path=preload_path,
        )
        server_a_stop_event = threading.Event()
        server_a_holder, server_a_recovery_thread, server_a_restart_count = start_recovery_monitor(
            process=server_a_process,
            stop_event=server_a_stop_event,
            output_dir=output_dir,
            model_path=str(model_path),
            port=server_a_port,
            nccl_port=nccl_a_port,
            device=1,
            enable_memory_saver=True,
            preload_path=preload_path,
        )
        server_b = start_sglang_server(
            model_path=str(model_path),
            port=server_b_port,
            nccl_port=nccl_b_port,
            device=0,
            enable_memory_saver=False,
            log_path=output_dir / "sglang-b.log",
            preload_path=preload_path,
        )
        wait_for_server(server_a_url, args.server_start_timeout)
        wait_for_server(server_b_url, args.server_start_timeout)

        with httpx.Client(timeout=10.0) as client:
            for worker_url in (server_a_url, server_b_url):
                response = client.post(f"{router_url}/add_worker", params={"url": worker_url})
                response.raise_for_status()

        actor_environment = os.environ.copy()
        actor_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "HIP_VISIBLE_DEVICES": "0,1",
                "ROCR_VISIBLE_DEVICES": "0,1",
                "HF_HOME": "/job/cache/hf",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_SOCKET_IFNAME": "lo",
                "GLOO_SOCKET_IFNAME": "lo",
                "NCCL_P2P_DISABLE": "1",
                "NCCL_DEBUG": "INFO",
                "SGLANG_USE_AITER": "0",
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
                "NCCL_SOCKET_TIMEOUT_MS": "600000",
                "TORCHELASTIC_USE_AGENT_STORE": "0",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        actor_command = [
            "/opt/venv/bin/torchrun",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=127.0.0.1:{torchrun_port}",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            str(default_port),
            str(Path(__file__).resolve().with_name("actor_worker.py")),
            "--model-path",
            str(model_path),
            "--server-a-url",
            server_a_url,
            "--server-b-url",
            server_b_url,
            "--router-url",
            router_url,
            "--output-path",
            str(output_dir / "summary.json"),
            "--update-a-port",
            str(update_a_port),
            "--update-b-port",
            str(update_b_port),
            "--default-port",
            str(default_port),
            "--cycles",
            str(args.cycles),
            "--concurrency",
            str(args.concurrency),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--request-timeout",
            str(args.request_timeout),
            "--pg-timeout",
            str(args.pg_timeout),
            "--fail-wake-cycle",
            str(args.fail_wake_cycle),
            "--fail-update-cycle",
            str(args.fail_update_cycle),
        ]
        if args.baseline:
            actor_command.append("--baseline")

        actor_log = output_dir / "actor.log"
        actor_file = actor_log.open("w")
        actor = subprocess.Popen(
            actor_command,
            stdout=actor_file,
            stderr=subprocess.STDOUT,
            env=actor_environment,
            start_new_session=True,
        )
        actor_returncode = actor.wait()
        if actor_returncode != 0:
            raise RuntimeError(f"actor worker exited with status {actor_returncode}")

        environment = build_environment(model_path, output_dir)
        environment["wake_recovery"] = {
            "restart_count": server_a_restart_count["value"],
            "restart_limit": 1,
            "health_timeout_seconds": args.pg_timeout,
        }
        (output_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )
    finally:
        if actor is not None and actor.poll() is None:
            stop_process(actor)
        if server_a_stop_event is not None:
            server_a_stop_event.set()
        if server_a_holder is not None:
            stop_process(server_a_holder["process"])
        if server_a_recovery_thread is not None:
            server_a_recovery_thread.join(timeout=2.0)
        if server_b is not None:
            stop_process(server_b)
        router_server.stop()
        if actor_file is not None:
            actor_file.close()


if __name__ == "__main__":
    main()
