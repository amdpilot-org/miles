from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import socket
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TIMEOUT_SECONDS = 30


def _configure_aiter_configs() -> None:
    configs = {
        "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
        "AITER_CONFIG_GROUPED_FMOE": "tuned_grouped_fmoe.csv",
        "AITER_CONFIG_A8W8_BATCHED_GEMM": "a8w8_tuned_batched_gemm.csv",
        "AITER_CONFIG_BF16_BATCHED_GEMM": "bf16_tuned_batched_gemm.csv",
    }
    for name, filename in configs.items():
        os.environ.setdefault(name, f"/sgl-workspace/aiter/aiter/configs/{filename}")


def _write_tiny_model(model_path: Path) -> None:
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 16,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 64,
        "torch_dtype": "bfloat16",
    }
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _construct_cpu_replica(model_path: Path):
    from sglang.srt import server_args as server_args_module
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import RankParallelismConfig
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

    server_args = ServerArgs(
        model_path=str(model_path),
        nnodes=2,
        tp_size=2,
        ep_size=2,
        device="cpu",
    )
    publish(server_args, role="scheduler")
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    parallelism_config = RankParallelismConfig(
        tp_size=2,
        tp_rank=0,
        ep_size=2,
        ep_rank=0,
        world_size=2,
        global_rank=0,
        local_rank=0,
    )
    protocol = UpdateWeightP2P.__new__(UpdateWeightP2P)
    protocol._shared_params_dict = {}
    model = protocol._create_cpu_replica(
        parallelism_config,
        str(model_path),
        server_args,
        first_engine_rank=True,
    )
    return model, get_parallel().nnodes


def _tensor_metadata(state_dict: dict[str, torch.Tensor]) -> list[dict]:
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        for name, tensor in state_dict.items()
    ]


def _transfer_state_dict(
    rank: int,
    state_dict: dict[str, torch.Tensor] | None,
    metadata: list[dict],
    device: torch.device,
    nccl_group: dist.ProcessGroup,
) -> tuple[list[bool], int]:
    received: dict[str, torch.Tensor] = {}
    equal: list[bool] = []
    transferred_bytes = 0
    for entry in metadata:
        name = entry["name"]
        shape = entry["shape"]
        dtype = getattr(torch, entry["dtype"])
        if rank == 0:
            tensor = state_dict[name].detach().to(device=device)
            dist.send(tensor, dst=1, group=nccl_group)
            echo = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(echo, src=1, group=nccl_group)
            transferred_bytes += tensor.numel() * tensor.element_size()
            equal.append(torch.equal(tensor, echo))
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tensor, src=0, group=nccl_group)
            dist.send(tensor, dst=0, group=nccl_group)
            received[name] = tensor
            equal.append(True)
    torch.cuda.synchronize()
    return equal, transferred_bytes


def _mooncake_hip_probe(
    rank: int,
    world_size: int,
    device: torch.device,
    gloo_group: dist.ProcessGroup,
) -> dict:
    from mooncake.engine import TransferEngine

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        hostname = sock.getsockname()[0]

    engine = TransferEngine()
    initialize_rc = engine.initialize(hostname, "P2PHANDSHAKE", "hip", "")
    rpc_port = engine.get_rpc_port()
    source = torch.full((1024,), float(rank + 1), dtype=torch.float32, device=device)
    target = torch.zeros(1024, dtype=torch.float32, device=device)
    nbytes = source.numel() * source.element_size()
    source_rc = engine.register_memory(source.data_ptr(), nbytes, f"cuda:{rank}")
    target_rc = engine.register_memory(target.data_ptr(), nbytes, f"cuda:{rank}")
    ports = [None] * world_size
    dist.all_gather_object(ports, rpc_port, group=gloo_group)
    probe_rc = None
    write_rc = None
    if rank == 0:
        session = f"{hostname}:{ports[1]}"
        probe_rc = engine.send_probe(session)
        write_rc = engine.transfer_sync_write(
            session,
            source.data_ptr(),
            target.data_ptr(),
            nbytes,
        )
        torch.cuda.synchronize()
    write_result = [write_rc]
    dist.broadcast_object_list(write_result, src=0, group=gloo_group)
    dist.barrier(group=gloo_group)
    engine.unregister_memory(source.data_ptr())
    engine.unregister_memory(target.data_ptr())
    return {
        "hostname": hostname,
        "initialize_rc": initialize_rc,
        "rpc_port": rpc_port,
        "register_rc": [source_rc, target_rc],
        "probe_rc": probe_rc,
        "write_rc": write_result[0],
        "supported": write_result[0] == 0,
    }


def _module_path(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return str(spec.origin) if spec and spec.origin else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_evidence() -> dict:
    return {
        "python": sys.executable,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
                "total_mib": torch.cuda.get_device_properties(index).total_memory // 1024**2,
            }
            for index in range(torch.cuda.device_count())
        ],
        "imports": {
            "miles": _module_path("miles"),
            "sglang": _module_path("sglang"),
            "torch": _module_path("torch"),
            "torch._C": _module_path("torch._C"),
            "mooncake.engine": _module_path("mooncake.engine"),
            "transformer_engine": _module_path("transformer_engine"),
            "megatron.core": _module_path("megatron.core"),
            "aiter": _module_path("aiter"),
        },
        "versions": {
            name: _package_version(name)
            for name in ("miles", "sglang", "torch", "transformer_engine", "megatron-core")
        },
    }


def main() -> None:
    _configure_aiter_configs()
    timeout = timedelta(seconds=TIMEOUT_SECONDS)
    dist.init_process_group(backend="nccl", timeout=timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2 or torch.cuda.device_count() < 2:
        raise RuntimeError("This fixture requires exactly two assigned GPUs.")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    gloo_group = dist.new_group(backend="gloo", timeout=timeout)
    nccl_group = dist.new_group(backend="nccl", timeout=timeout)

    probe = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(probe, group=nccl_group)
    if probe.item() != 3:
        raise RuntimeError(f"Unexpected NCCL probe result: {probe.item()}")

    model = None
    state_dict = None
    restored_nnodes = None
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="j-42ae02873e45-") as temporary_directory:
            model_path = Path(temporary_directory) / "tiny-qwen3-moe"
            _write_tiny_model(model_path)
            model, restored_nnodes = _construct_cpu_replica(model_path)
            state_dict = model.state_dict()
            metadata = _tensor_metadata(state_dict)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            state_dict_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in state_dict.values()
            )
            model_class = model.__class__.__name__
        payload = [
            metadata,
            parameter_count,
            state_dict_bytes,
            model_class,
            restored_nnodes,
        ]
    else:
        payload = [None, None, None, None, None]
    dist.broadcast_object_list(payload, src=0, group=gloo_group)
    metadata, parameter_count, state_dict_bytes, model_class, restored_nnodes = payload

    equality, transferred_bytes = _transfer_state_dict(
        rank,
        state_dict,
        metadata,
        device,
        nccl_group,
    )
    all_equal = all(equality)
    gathered_equal = [None] * world_size
    dist.all_gather_object(gathered_equal, all_equal, group=gloo_group)

    mooncake = _mooncake_hip_probe(rank, world_size, device, gloo_group)
    runtime = _runtime_evidence()
    result = {
        "status": "pass" if all(gathered_equal) else "fail",
        "issue": "radixark/miles#2856",
        "timeout_seconds": TIMEOUT_SECONDS,
        "world_size": world_size,
        "process_groups": {
            "default_backend": dist.get_backend(),
            "gloo_backend": dist.get_backend(gloo_group),
            "nccl_backend": dist.get_backend(nccl_group),
        },
        "cpu_replica": {
            "constructed": model_class == "Qwen3MoeForCausalLM",
            "model_class": model_class,
            "parameter_count": parameter_count,
            "state_dict_bytes": state_dict_bytes,
            "tp_size": 2,
            "ep_size": 2,
            "published_nnodes": 2,
            "restored_nnodes": restored_nnodes,
        },
        "nccl_p2p": {
            "tensors": len(metadata),
            "transferred_bytes": transferred_bytes,
            "probe": probe.item(),
            "all_ranks_equal": gathered_equal,
        },
        "mooncake_hip": mooncake,
        "runtime": runtime,
    }

    output_path = os.environ.get("FIXTURE_OUTPUT")
    if output_path and rank == 0:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rank": rank, **result}, indent=2), flush=True)

    dist.barrier(group=gloo_group)
    dist.destroy_process_group()
    if not all(gathered_equal):
        raise RuntimeError("Reconstructed weights differ on at least one rank.")


if __name__ == "__main__":
    main()
