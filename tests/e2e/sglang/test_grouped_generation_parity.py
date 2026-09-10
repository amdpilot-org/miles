import asyncio
import importlib.util
import math
import os
import signal
import subprocess
import sys
import time
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from tests.ci.ci_register import register_rocm_ci


def _set_private_aiter_config() -> None:
    spec = importlib.util.find_spec("aiter")
    if spec is None or spec.origin is None:
        return
    config = Path(spec.origin).resolve().parents[1] / "configs" / "bf16_tuned_gemm.csv"
    if config.is_file():
        os.environ.setdefault("AITER_CONFIG_GEMM_BF16", str(config))


_set_private_aiter_config()

import miles.utils.http_utils as http_utils  # noqa: E402
from miles.rollout.sglang_rollout import generate_and_rm, generate_and_rm_group  # noqa: E402
from miles.utils.http_utils import find_available_port, init_http_client  # noqa: E402
from miles.utils.types import Sample  # noqa: E402


register_rocm_ci(est_time=240, suite="nightly-stage-c-4-gpu-mi350", labels=["sglang"])

SEED = 1567
GROUPS = 64
SAMPLES_PER_GROUP = 4
MAX_NEW_TOKENS = 16
REPETITIONS = 3
STARTUP_TIMEOUT_SECONDS = 180
SHUTDOWN_TIMEOUT_SECONDS = 15
TOKENIZER_SOURCE_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DEFAULT_MODEL_PATH = Path("/tmp/miles-grouped-generation-model")
DEFAULT_HF_CACHE = Path("/tmp/miles-grouped-generation-hf-cache")


@dataclass
class SGLangServer:
    process: subprocess.Popen
    host: str
    port: int
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _prepare_model() -> Path:
    override = os.environ.get("MILES_GROUPED_GENERATION_MODEL_PATH")
    model_path = Path(override) if override else DEFAULT_MODEL_PATH
    required = [model_path / "config.json", model_path / "model.safetensors", model_path / "tokenizer.json"]
    if all(path.is_file() for path in required):
        return model_path

    model_path.mkdir(parents=True, exist_ok=True)
    hf_cache = Path(os.environ.get("MILES_GROUPED_GENERATION_HF_CACHE", DEFAULT_HF_CACHE))
    tokenizer_source = snapshot_download(
        TOKENIZER_SOURCE_ID,
        cache_dir=hf_cache,
        allow_patterns=["tokenizer*", "special_tokens_map.json"],
    )
    AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(model_path)

    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=None,
        torch_dtype="bfloat16",
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(model_path, safe_serialization=True)
    return model_path


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def _wait_for_ready(server: SGLangServer) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = "no health check attempted"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(
                f"SGLang server exited with code {server.process.returncode}. Log tail:\n{_log_tail(server.log_path)}"
            )
        try:
            response = requests.get(f"{server.base_url}/health", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(
        f"SGLang server not healthy after {STARTUP_TIMEOUT_SECONDS}s: {last_error}. "
        f"Log tail:\n{_log_tail(server.log_path)}"
    )


def _start_server(model_path: Path) -> SGLangServer:
    host = "127.0.0.1"
    port = find_available_port(31200)
    log_path = Path(f"/tmp/miles-grouped-generation-sglang-{port}.log")
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--context-length",
        "128",
        "--mem-fraction-static",
        "0.10",
        "--max-running-requests",
        "64",
        "--max-total-tokens",
        "4096",
        "--disable-radix-cache",
        "--cuda-graph-backend-decode=disabled",
        "--cuda-graph-backend-prefill=disabled",
        "--dist-timeout",
        "60",
        "--random-seed",
        str(SEED),
        "--log-level",
        "info",
    ]
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    server = SGLangServer(process=process, host=host, port=port, log_path=log_path)
    try:
        _wait_for_ready(server)
    except Exception:
        _stop_server(server)
        raise
    finally:
        log_file.close()
    return server


def _stop_server(server: SGLangServer) -> None:
    if server.process.poll() is not None:
        return
    process_group = os.getpgid(server.process.pid)
    os.killpg(process_group, signal.SIGTERM)
    try:
        server.process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process_group, signal.SIGKILL)
        server.process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


@pytest.fixture(scope="module")
def grouped_generation_server():
    model_path = _prepare_model()
    server = _start_server(model_path)
    try:
        yield server
    finally:
        _stop_server(server)


def _make_args(model_path: Path, server: SGLangServer) -> Namespace:
    return Namespace(
        hf_checkpoint=str(model_path),
        chat_template_path=None,
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=0.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=MAX_NEW_TOKENS,
        rollout_stop=[],
        rollout_stop_token_ids=[],
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        sglang_dp_size=1,
        rollout_seed=SEED,
        n_samples_per_prompt=SAMPLES_PER_GROUP,
        custom_generate_function_path=None,
        sglang_router_policy="round_robin",
        sglang_router_ip=server.host,
        sglang_router_port=server.port,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        ci_test=False,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        sglang_speculative_algorithm=None,
        num_layers=2,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        use_distributed_post=False,
    )


def _sampling(max_new_tokens: int = MAX_NEW_TOKENS, stop_token_ids: list[int] | None = None) -> dict:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": max_new_tokens,
        "stop": [],
        "stop_token_ids": stop_token_ids or [],
        "skip_special_tokens": True,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def _make_groups() -> list[list[Sample]]:
    return [
        [
            Sample(
                prompt=f"Grouped rollout parity {group_index}",
                group_index=group_index,
                index=group_index * SAMPLES_PER_GROUP + sample_index,
                reward=0.0,
            )
            for sample_index in range(SAMPLES_PER_GROUP)
        ]
        for group_index in range(GROUPS)
    ]


async def _run_individual(args: Namespace) -> list[Sample]:
    samples = [deepcopy(sample) for group in _make_groups() for sample in group]
    return list(
        await asyncio.gather(*[generate_and_rm(args, sample, _sampling()) for sample in samples])
    )


async def _run_grouped(args: Namespace) -> list[Sample]:
    groups = await asyncio.gather(
        *[generate_and_rm_group(args, group, _sampling()) for group in _make_groups()]
    )
    return [sample for group in groups for sample in group]


def _assert_parity(individual: list[Sample], grouped: list[Sample]) -> None:
    individual_by_key = {(sample.group_index, sample.index): sample for sample in individual}
    grouped_by_key = {(sample.group_index, sample.index): sample for sample in grouped}
    assert individual_by_key.keys() == grouped_by_key.keys()
    for key, expected in individual_by_key.items():
        actual = grouped_by_key[key]
        assert expected.tokens == actual.tokens
        assert expected.response == actual.response
        assert expected.response_length == actual.response_length
        assert expected.status == actual.status
        assert len(expected.rollout_log_probs) == len(actual.rollout_log_probs)
        for expected_logprob, actual_logprob in zip(expected.rollout_log_probs, actual.rollout_log_probs):
            assert math.isclose(expected_logprob, actual_logprob, abs_tol=1e-6)


async def _termination_parity(args: Namespace) -> None:
    probe = Sample(prompt="Grouped rollout parity termination", group_index=999, index=9990, reward=0.0)
    probe = await generate_and_rm(args, probe, _sampling(max_new_tokens=1))
    stop_token = probe.tokens[-1]

    individual = [
        Sample(prompt=probe.prompt, group_index=999, index=9990 + index, reward=0.0)
        for index in range(SAMPLES_PER_GROUP)
    ]
    grouped = deepcopy(individual)
    individual = await asyncio.gather(
        *[generate_and_rm(args, sample, _sampling(stop_token_ids=[stop_token])) for sample in individual]
    )
    grouped = await generate_and_rm_group(args, grouped, _sampling(stop_token_ids=[stop_token]))
    _assert_parity(list(individual), grouped)
    assert all(sample.status == Sample.Status.COMPLETED for sample in individual)
    assert all(sample.tokens[-1] == stop_token for sample in individual)


def test_grouped_generation_parity_and_latency(grouped_generation_server):
    model_path = _prepare_model()
    args = _make_args(model_path, grouped_generation_server)
    init_http_client(args)

    async def run() -> None:
        warmup = _make_groups()[:1]
        await asyncio.gather(
            *[generate_and_rm(args, sample, _sampling()) for group in warmup for sample in group]
        )
        await generate_and_rm_group(args, deepcopy(warmup[0]), _sampling())

        timings = {}
        results = {}
        for mode, runner in (("individual", _run_individual), ("grouped", _run_grouped)):
            timings[mode] = []
            for _ in range(REPETITIONS):
                start = time.perf_counter()
                results[mode] = await runner(args)
                timings[mode].append(time.perf_counter() - start)

        _assert_parity(results["individual"], results["grouped"])
        await _termination_parity(args)

        individual_median = sorted(timings["individual"])[1]
        grouped_median = sorted(timings["grouped"])[1]
        print(
            f"grouped generation latency: individual={individual_median:.6f}s "
            f"grouped={grouped_median:.6f}s speedup={individual_median / grouped_median:.3f}x",
            flush=True,
        )
        await http_utils._http_client.aclose()

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
