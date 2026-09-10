#!/usr/bin/env python3
from __future__ import annotations

import gc
import inspect
import json
import os
import subprocess
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("SGLANG_USE_AITER", "0")

import torch
import torch.distributed as dist
import torch_memory_saver
from torch_memory_saver.utils import get_binary_path_from_package
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


CYCLES = 32
POOL_TOKENS = 131_072
BATCH_TOKENS = 4_096
HEAD_NUM = 8
HEAD_DIM = 128
VOCAB_SIZE = 4_096
HIDDEN_SIZE = 2_048
PROCESS_GROUP_TIMEOUT_SECONDS = 60


def git_commit(path: str) -> str:
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def synchronize() -> None:
    torch.cuda.synchronize()


def phase_timer() -> Any:
    synchronize()
    return time.perf_counter()


def gpu_memory() -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
    }


def allocator_state(allocator: SWATokenToKVPoolAllocator) -> dict[str, int]:
    return {
        "available": allocator.available_size(),
        "full_available": allocator.full_available_size(),
        "swa_available": allocator.swa_available_size(),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def model_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(tensor_bytes(parameter) for parameter in model.parameters())


def finite_float(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def main() -> None:
    torch.manual_seed(0)
    if "torch_memory_saver_hook_mode_preload" not in os.environ.get("LD_PRELOAD", ""):
        raise RuntimeError(
            "run with LD_PRELOAD=/opt/venv/lib/python3.10/site-packages/"
            "torch_memory_saver_hook_mode_preload.abi3.so"
        )
    torch.cuda.set_device(0)

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one assigned GPU, got {torch.cuda.device_count()}")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 5):
        raise RuntimeError(f"expected gfx950/MI350X capability, got {capability}")

    process_group_file = Path(tempfile.gettempdir()) / f"miles-swa-tms-{os.getpid()}-{time.time_ns()}.store"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{process_group_file}",
        world_size=1,
        rank=0,
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )

    generation_model = torch.nn.Sequential(
        torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE),
        torch.nn.GELU(),
        torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE),
    ).to("cuda")
    generation_model.eval()

    pool = SWAKVPool(
        size=POOL_TOKENS,
        size_swa=POOL_TOKENS,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cuda",
        enable_memory_saver=True,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=POOL_TOKENS,
        size_swa=POOL_TOKENS,
        page_size=1,
        dtype=torch.bfloat16,
        device="cuda",
        kvcache=pool,
        need_sort=False,
    )
    adapter = TorchMemorySaverAdapter.create(enable=True)

    records: list[dict[str, Any]] = []
    for cycle in range(CYCLES):
        record: dict[str, Any] = {"cycle": cycle}

        start = phase_timer()
        indices = allocator.alloc(BATCH_TOKENS)
        if indices is None:
            raise RuntimeError("SWA allocator unexpectedly returned no indices")
        swa_indices = pool.full_to_swa_index_mapping[indices]
        if int(swa_indices.min().item()) <= 0 or int(swa_indices.max().item()) <= 0:
            raise RuntimeError("SWA allocator produced invalid full-to-SWA mappings")
        synchronize()
        record["allocate_seconds"] = time.perf_counter() - start
        record["allocator_after_allocate"] = allocator_state(allocator)

        tokens = (torch.arange(BATCH_TOKENS, device="cuda") + cycle) % VOCAB_SIZE
        with torch.no_grad():
            logits = generation_model(tokens)
            generated_tokens = logits.argmax(dim=-1)
        synchronize()
        expected_logits = logits.detach().clone()
        expected_tokens = generated_tokens.detach().clone()

        full_k = pool.full_kv_pool.k_buffer[0]
        full_v = pool.full_kv_pool.v_buffer[0]
        swa_k = pool.swa_kv_pool.k_buffer[0]
        swa_v = pool.swa_kv_pool.v_buffer[0]
        full_k[indices] = torch.randn(BATCH_TOKENS, HEAD_NUM, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        full_v[indices] = torch.randn(BATCH_TOKENS, HEAD_NUM, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        swa_k[swa_indices] = torch.randn(BATCH_TOKENS, HEAD_NUM, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        swa_v[swa_indices] = torch.randn(BATCH_TOKENS, HEAD_NUM, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        synchronize()

        start = phase_timer()
        allocator.free(indices)
        synchronize()
        record["release_allocator_seconds"] = time.perf_counter() - start
        record["allocator_after_release"] = allocator_state(allocator)

        memory_before_pause = gpu_memory()
        start = phase_timer()
        adapter.pause("kv_cache")
        synchronize()
        record["pause_seconds"] = time.perf_counter() - start
        memory_after_pause = gpu_memory()
        record["memory_before_pause"] = memory_before_pause
        record["memory_after_pause"] = memory_after_pause
        released_gib = memory_after_pause["free_gib"] - memory_before_pause["free_gib"]
        record["released_gib"] = released_gib

        training_model = torch.nn.Sequential(
            torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE),
            torch.nn.GELU(),
            torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE),
        ).to("cuda")
        training_model.train()
        optimizer = torch.optim.SGD(training_model.parameters(), lr=0.01)
        parameter_bytes = model_parameter_bytes(training_model)
        record["training_parameter_mib"] = parameter_bytes / 2**20
        if released_gib * 2**30 < parameter_bytes:
            raise RuntimeError("training model did not fit in released GPU memory")

        start = phase_timer()
        training_tokens = (torch.arange(BATCH_TOKENS, device="cuda") + cycle + 1) % VOCAB_SIZE
        training_logits = training_model(training_tokens)
        loss = torch.nn.functional.cross_entropy(
            training_logits,
            (training_tokens + 1) % VOCAB_SIZE,
        )
        loss.backward()
        synchronize()
        record["train_forward_backward_seconds"] = time.perf_counter() - start
        if not finite_float(loss.detach()):
            raise RuntimeError("training loss was non-finite")

        local_gradient_norm = torch.sqrt(
            sum(parameter.grad.pow(2).sum() for parameter in training_model.parameters())
        )
        if not finite_float(local_gradient_norm.detach()):
            raise RuntimeError("training gradient norm was non-finite")

        start = phase_timer()
        for parameter in training_model.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        synchronize()
        collective_gradient_norm = torch.sqrt(
            sum(parameter.grad.pow(2).sum() for parameter in training_model.parameters())
        )
        if not torch.allclose(local_gradient_norm, collective_gradient_norm):
            raise RuntimeError("NCCL all-reduce changed a world-size-1 gradient")
        record["collective_seconds"] = time.perf_counter() - start

        weight_before = next(training_model.parameters()).detach().clone()
        start = phase_timer()
        optimizer.step()
        synchronize()
        record["optimizer_update_seconds"] = time.perf_counter() - start
        weight_after = next(training_model.parameters()).detach()
        update_norm = torch.linalg.vector_norm(weight_after - weight_before)
        if not finite_float(update_norm.detach()) or float(update_norm.item()) <= 0.0:
            raise RuntimeError("optimizer update was empty or non-finite")

        record["loss"] = float(loss.detach().item())
        record["local_gradient_norm"] = float(local_gradient_norm.detach().item())
        record["collective_gradient_norm"] = float(collective_gradient_norm.detach().item())
        record["optimizer_update_norm"] = float(update_norm.detach().item())

        del optimizer, training_logits, loss, training_model
        gc.collect()
        synchronize()

        start = phase_timer()
        adapter.resume("kv_cache")
        synchronize()
        record["resume_seconds"] = time.perf_counter() - start
        record["memory_after_resume"] = gpu_memory()

        with torch.no_grad():
            resumed_logits = generation_model(tokens)
            resumed_tokens = resumed_logits.argmax(dim=-1)
        synchronize()
        if not torch.equal(expected_tokens, resumed_tokens):
            raise RuntimeError(f"token output changed across release/resume in cycle {cycle}")
        if not torch.equal(expected_logits, resumed_logits):
            raise RuntimeError(f"logit output changed across release/resume in cycle {cycle}")
        record["token_output_continuous"] = True
        record["logit_output_continuous"] = True

        if allocator_state(allocator) != {
            "available": POOL_TOKENS,
            "full_available": POOL_TOKENS,
            "swa_available": POOL_TOKENS,
        }:
            raise RuntimeError(f"allocator state drifted in cycle {cycle}")

        records.append(record)
        print(
            f"cycle={cycle:02d} released={released_gib:.3f}GiB "
            f"loss={record['loss']:.6f} grad={record['local_gradient_norm']:.6f} "
            f"update={record['optimizer_update_norm']:.6f}",
            flush=True,
        )

    dist.destroy_process_group()
    if process_group_file.exists():
        process_group_file.unlink()

    del allocator, pool, generation_model, adapter
    gc.collect()
    synchronize()

    sglang_path = Path(inspect.getfile(SWAKVPool)).resolve()
    torch_memory_saver_path = Path(inspect.getfile(torch_memory_saver)).resolve()
    native_hook_path = Path(
        get_binary_path_from_package("torch_memory_saver_hook_mode_preload")
    ).resolve()
    summary = {
        "cycles": CYCLES,
        "device": {
            "name": device_name,
            "capability": list(capability),
            "count": torch.cuda.device_count(),
        },
        "versions": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "sglang_commit": git_commit("/sgl-workspace/sglang"),
            "miles_commit": git_commit("/job/miles"),
        },
        "paths": {
            "sglang_swa_pool": str(sglang_path),
            "torch_memory_saver_python": str(torch_memory_saver_path),
            "torch_memory_saver_native": str(native_hook_path),
        },
        "process_group": {
            "backend": dist.get_backend() if dist.is_initialized() else "nccl",
            "world_size": 1,
            "timeout_seconds": PROCESS_GROUP_TIMEOUT_SECONDS,
            "rendezvous_file": str(process_group_file),
        },
        "records": records,
    }

    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
