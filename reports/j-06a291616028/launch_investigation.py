import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(url, timeout):
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
        time.sleep(1.0)
    raise RuntimeError(f"SGLang server did not become healthy in {timeout}s: {last_error}")


def stop_process(process):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/job/cache/models/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--fail-cycle", type=int, default=16)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_port = free_port()
    torchrun_port = free_port()
    update_port = free_port()
    nccl_port = free_port()
    default_port = free_port()
    server_url = f"http://127.0.0.1:{server_port}"

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "HF_HOME": "/job/cache/hf",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "NCCL_DEBUG": "INFO",
            "SGLANG_USE_AITER": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    server_command = [
        "/opt/venv/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(server_port),
        "--tp-size",
        "1",
        "--nccl-port",
        str(nccl_port),
        "--mem-fraction-static",
        "0.35",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--enable-request-time-stats-logging",
        "--decode-log-interval",
        "1",
    ]
    server_log = output_dir / "sglang.log"
    server_file = server_log.open("w")
    server = subprocess.Popen(
        server_command,
        stdout=server_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )

    actor = None
    actor_log = None
    actor_file = None
    try:
        wait_for_server(server_url, args.server_start_timeout)
        actor_environment = os.environ.copy()
        actor_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "HF_HOME": "/job/cache/hf",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_SOCKET_IFNAME": "lo",
                "GLOO_SOCKET_IFNAME": "lo",
                "NCCL_DEBUG": "INFO",
                "SGLANG_USE_AITER": "0",
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
            str(Path(__file__).with_name("actor_worker.py")),
            "--model-path",
            args.model_path,
            "--server-url",
            server_url,
            "--output-path",
            str(output_dir / "summary.json"),
            "--update-port",
            str(update_port),
            "--default-port",
            str(default_port),
            "--cycles",
            str(args.cycles),
            "--concurrency",
            str(args.concurrency),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--fail-cycle",
            str(args.fail_cycle),
        ]
        actor_log = output_dir / "actor.log"
        actor_file = actor_log.open("w")
        actor = subprocess.Popen(
            actor_command,
            stdout=actor_file,
            stderr=subprocess.STDOUT,
            env=actor_environment,
            start_new_session=True,
        )
        return_code = actor.wait()
        if return_code != 0:
            raise RuntimeError(f"actor worker exited with status {return_code}")
    finally:
        if actor is not None and actor.poll() is None:
            stop_process(actor)
        if actor_file is not None:
            actor_file.close()
        stop_process(server)
        server_file.close()

    metadata = {
        "server_url": server_url,
        "ports": {
            "sglang_http": server_port,
            "torchrun_c10d": torchrun_port,
            "weight_update_nccl": update_port,
            "sglang_nccl": nccl_port,
            "torchrun_default_process_group": default_port,
        },
        "cycles": args.cycles,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "fail_cycle": args.fail_cycle,
    }
    (output_dir / "launch_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
