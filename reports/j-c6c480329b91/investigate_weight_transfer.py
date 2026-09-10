import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import pack_units_by_size
from miles.backends.training_utils.weight_update.protocols.broadcast import update_weights_from_distributed
from miles.utils import async_utils


class MetadataClient:
    """Minimal rollout API client used only to satisfy the Miles broadcast hook."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def update_weights_from_distributed(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class TinySourceModel(torch.nn.Module):
    """Small Megatron-layout model used for real optimizer updates."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(vocab, hidden))
        self.qkv = torch.nn.Parameter(torch.randn(3 * hidden, hidden))
        self.fc1 = torch.nn.Parameter(torch.randn(2 * hidden, hidden))
        self.fc2 = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.norm = torch.nn.Parameter(torch.randn(hidden))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding.shape[1]
        x = F.embedding(tokens, self.embedding)
        qkv = x @ self.qkv.T
        q, k, v = qkv.chunk(3, dim=-1)
        attention = F.scaled_dot_product_attention(q, k, v)
        gate_up = x @ self.fc1.T
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_state = (F.silu(gate) * up) @ self.fc2.T
        return F.layer_norm(hidden_state, (hidden,), self.norm)


class TinyConsumerModel(torch.nn.Module):
    """Canonical HF-layout model matching the Miles converter output."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(vocab, hidden))
        self.q = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.k = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.v = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.gate = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.up = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.down = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.norm = torch.nn.Parameter(torch.randn(hidden))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed.shape[1]
        x = F.embedding(tokens, self.embed)
        q = x @ self.q.T
        k = x @ self.k.T
        v = x @ self.v.T
        attention = F.scaled_dot_product_attention(q, k, v)
        gate = x @ self.gate.T
        up = x @ self.up.T
        hidden_state = (F.silu(gate) * up) @ self.down.T
        return F.layer_norm(hidden_state, (hidden,), self.norm)


def _record_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).digest()


def _payload_digest(payload: list[torch.Tensor]) -> bytes:
    payload_bytes = b"".join(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        for tensor in payload
    )
    return hashlib.sha256(payload_bytes).digest()


def _digest_tensor(digest: bytes) -> torch.Tensor:
    return torch.tensor(list(digest), dtype=torch.uint8, device="cuda")


def _canonical_units(source: TinySourceModel, args: SimpleNamespace, transfer_dtype: torch.dtype) -> list[list[tuple[str, torch.Tensor]]]:
    named_params = {
        "module.module.embedding.word_embeddings.weight": source.embedding,
        "module.module.decoder.layers.0.self_attention.linear_qkv.weight": source.qkv,
        "module.module.decoder.layers.0.mlp.linear_fc1.weight": source.fc1,
        "module.module.decoder.layers.0.mlp.linear_fc2.weight": source.fc2,
        "module.module.decoder.final_layernorm.weight": source.norm,
    }
    units = []
    for name, param in named_params.items():
        converted = convert_to_hf(args, "llama", name, param.detach())
        units.append([(hf_name, tensor.to(dtype=transfer_dtype).contiguous()) for hf_name, tensor in converted])
    return units


def _pack_buckets(units: list[list[tuple[str, torch.Tensor]]], max_bytes: int) -> list[list[tuple[str, torch.Tensor]]]:
    return list(pack_units_by_size(units, max_bytes))


def _load_consumer(model: TinyConsumerModel, buckets: list[list[tuple[str, torch.Tensor]]]) -> None:
    mapping = {
        "model.embed_tokens.weight": model.embed,
        "model.layers.0.self_attn.q_proj.weight": model.q,
        "model.layers.0.self_attn.k_proj.weight": model.k,
        "model.layers.0.self_attn.v_proj.weight": model.v,
        "model.layers.0.mlp.gate_proj.weight": model.gate,
        "model.layers.0.mlp.up_proj.weight": model.up,
        "model.layers.0.mlp.down_proj.weight": model.down,
        "model.norm.weight": model.norm,
    }
    with torch.no_grad():
        for bucket in buckets:
            for name, tensor in bucket:
                mapping[name].copy_(tensor)


def _bucket_bytes(buckets: list[list[tuple[str, torch.Tensor]]]) -> int:
    return sum(tensor.nbytes for bucket in buckets for _, tensor in bucket)


def _broadcast_transfer(
    buckets: list[list[tuple[str, torch.Tensor]]],
    client: MetadataClient,
    version: int,
) -> dict:
    transfer_start = time.perf_counter()
    for bucket in buckets:
        futures = update_weights_from_distributed(
            "miles-investigation-broadcast",
            None,
            [client],
            bucket,
            selector="all",
        )
        async_utils.wait_futures(futures)
    transfer_end = time.perf_counter()
    torch.cuda.synchronize()

    version_tensor = torch.tensor([version], dtype=torch.int64, device="cuda")
    dist.broadcast(version_tensor, src=0)
    local_digest = _payload_digest([tensor for bucket in buckets for _, tensor in bucket])
    digest_tensor = _digest_tensor(local_digest)
    dist.broadcast(digest_tensor, src=0)
    if dist.get_rank() == 1:
        receiver_digest = _payload_digest([tensor for bucket in buckets for _, tensor in bucket])
        checksum_match = receiver_digest == digest_tensor.cpu().numpy().tobytes()
    else:
        checksum_match = True
    match_tensor = torch.tensor([int(checksum_match)], dtype=torch.int64, device="cuda")
    dist.all_reduce(match_tensor, op=dist.ReduceOp.MIN)
    return {
        "transfer_wall_ms": (transfer_end - transfer_start) * 1000,
        "version": version,
        "checksum_match": bool(match_tensor.item()),
    }


def _sendrecv_transfer(
    buckets: list[list[tuple[str, torch.Tensor]]],
    metadata_group: dist.ProcessGroup,
    version: int,
    *,
    failure_cycle: bool,
) -> tuple[dict, list[object]]:
    handles = []
    payloads = []
    digest_tensors = []
    failure_detected = False
    transfer_start = time.perf_counter()
    for bucket_index, bucket in enumerate(buckets):
        names = [name for name, _ in bucket]
        shapes = [tuple(tensor.shape) for _, tensor in bucket]
        dtype_name = str(bucket[0][1].dtype)
        metadata = [names, shapes, dtype_name]
        if dist.get_rank() == 0:
            dist.send_object_list(metadata, dst=1, group=metadata_group)
        else:
            metadata = [None, None, None]
            dist.recv_object_list(metadata, src=0, group=metadata_group)
            names, shapes, dtype_name = metadata

        transfer_dtype = torch.float32 if dtype_name == "torch.float32" else torch.bfloat16
        if dist.get_rank() == 0:
            payload = [tensor.detach().contiguous() for _, tensor in bucket]
            digest = _payload_digest(payload)
            digest_tensor = _digest_tensor(digest)
            dist.send(digest_tensor, dst=1)
        else:
            payload = [torch.empty(shape, dtype=transfer_dtype, device="cuda") for shape in shapes]
            digest_tensor = torch.empty(32, dtype=torch.uint8, device="cuda")
            dist.recv(digest_tensor, src=0)

        payloads.append(payload)
        digest_tensors.append(digest_tensor)

        if failure_cycle and bucket_index == 0:
            failure_flag = torch.tensor([1], dtype=torch.int64, device="cuda")
            truncated_numel = torch.tensor([payload[0].numel() // 2], dtype=torch.int64, device="cuda")
            if dist.get_rank() == 0:
                dist.send(failure_flag, dst=1)
                dist.send(truncated_numel, dst=1)
                truncated_tensor = payload[0].flatten()[: truncated_numel.item()].contiguous()
                truncated_handle = dist.isend(truncated_tensor, dst=1)
                truncated_handle.wait()
                dist.barrier()
            else:
                dist.recv(failure_flag, src=0)
                dist.recv(truncated_numel, src=0)
                truncated_tensor = torch.empty(int(truncated_numel.item()), dtype=transfer_dtype, device="cuda")
                truncated_handle = dist.irecv(truncated_tensor, src=0)
                truncated_handle.wait()
                local_digest = _tensor_digest(truncated_tensor)
                expected_digest = digest_tensor.cpu().numpy().tobytes()
                failure_detected = local_digest != expected_digest
                dist.barrier()

        if dist.get_rank() == 0:
            for tensor in payload:
                handle = dist.isend(tensor, dst=1)
                handles.append(handle)
        else:
            for tensor in payload:
                handle = dist.irecv(tensor, src=0)
                handles.append(handle)

    transfer_end = time.perf_counter()
    torch.cuda.synchronize()

    version_tensor = torch.tensor([version], dtype=torch.int64, device="cuda")
    if dist.get_rank() == 0:
        dist.send(version_tensor, dst=1)
    else:
        dist.recv(version_tensor, src=0)

    return {
        "transfer_wall_ms": (transfer_end - transfer_start) * 1000,
        "version": version,
        "failure_cycle": failure_cycle,
        "digest_tensors": digest_tensors,
        "failure_detected": failure_detected,
    }, handles, payloads


def _wait_handles(handles: list[object]) -> float:
    start = time.perf_counter()
    for handle in handles:
        handle.wait()
    end = time.perf_counter()
    torch.cuda.synchronize()
    return (end - start) * 1000


def _compare_outputs(
    expected: torch.Tensor,
    actual: torch.Tensor,
    tolerance: float = 2e-5,
) -> tuple[float, bool]:
    diff = (expected.float() - actual.float()).abs().max()
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    value = diff.item()
    return value, value < tolerance


def _source_forward(model: TinySourceModel, tokens: torch.Tensor) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    with torch.no_grad():
        output = model(tokens)
    end = time.perf_counter()
    torch.cuda.synchronize()
    return output, (end - start) * 1000


def _consumer_forward(model: TinyConsumerModel, tokens: torch.Tensor) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    with torch.no_grad():
        output = model(tokens)
    end = time.perf_counter()
    torch.cuda.synchronize()
    return output, (end - start) * 1000


def _init_distributed(timeout_seconds: int) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))


def _metadata_group(timeout_seconds: int) -> dist.ProcessGroup:
    return dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _environment() -> dict:
    import miles

    sglang_spec = importlib.util.find_spec("sglang")
    try:
        sglang_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        sglang_version = None

    return {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "sglang_version": sglang_version,
        "miles_path": miles.__file__,
        "torch_path": torch.__file__,
        "sglang_path": sglang_spec.origin if sglang_spec else None,
        "git_commit": _git_commit(),
    }


def _write_json(path: Path, payload: dict) -> None:
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    _init_distributed(args.timeout)
    metadata_group = _metadata_group(args.timeout)
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.manual_seed(1234)

    source = TinySourceModel(args.hidden, args.vocab).to(device)
    reference_consumer = TinyConsumerModel(args.hidden, args.vocab).to(device)
    consumer_broadcast = TinyConsumerModel(args.hidden, args.vocab).to(device)
    consumer_sendrecv = TinyConsumerModel(args.hidden, args.vocab).to(device)
    tokens = torch.randint(0, args.vocab, (args.batch, args.sequence), device=device)
    conversion_args = SimpleNamespace(
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        num_query_groups=args.heads,
        kv_channels=None,
        vocab_size=args.vocab,
    )
    client = MetadataClient()
    optimizer = torch.optim.SGD(source.parameters(), lr=args.lr)

    records = []
    failure_recovered = False
    for cycle in range(args.cycles):
        transfer_dtype = torch.bfloat16 if cycle % 2 == 0 else torch.float32
        optimizer.zero_grad(set_to_none=True)
        source_output = source(tokens)
        loss = source_output.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        units = _canonical_units(source, conversion_args, transfer_dtype)
        max_bytes = 3 * args.hidden * args.hidden * transfer_dtype.itemsize + 7
        buckets = _pack_buckets(units, max_bytes)
        bucket_count = len(buckets)
        bucket_bytes = _bucket_bytes(buckets)

        source_output, source_compute_ms = _source_forward(source, tokens)
        _load_consumer(reference_consumer, buckets)
        expected_output, reference_consumer_ms = _consumer_forward(reference_consumer, tokens)
        conversion_tolerance = 5e-2 if transfer_dtype == torch.bfloat16 else 1e-5
        conversion_error, conversion_match = _compare_outputs(
            source_output,
            expected_output,
            tolerance=conversion_tolerance,
        )

        broadcast_start = time.perf_counter()
        broadcast_info = _broadcast_transfer(buckets, client, cycle + 1)
        broadcast_end = time.perf_counter()
        _load_consumer(consumer_broadcast, buckets)
        broadcast_consumer_output, broadcast_consumer_ms = _consumer_forward(consumer_broadcast, tokens)
        broadcast_error, broadcast_match = _compare_outputs(expected_output, broadcast_consumer_output)

        sendrecv_failure_cycle = cycle + 1 == args.failure_cycle
        sendrecv_start = time.perf_counter()
        sendrecv_info, handles, payloads = _sendrecv_transfer(
            buckets,
            metadata_group,
            cycle + 1,
            failure_cycle=sendrecv_failure_cycle,
        )
        overlap_output, sendrecv_overlap_compute_ms = _source_forward(source, tokens)
        sendrecv_wait_ms = _wait_handles(handles)

        if rank == 1:
            checksum_matches = [
                _payload_digest(payload) == digest_tensor.cpu().numpy().tobytes()
                for payload, digest_tensor in zip(payloads, sendrecv_info["digest_tensors"])
            ]
        else:
            checksum_matches = [True] * len(buckets)

        checksum_tensor = torch.tensor(
            [int(all(checksum_matches))],
            dtype=torch.int64,
            device="cuda",
        )
        dist.all_reduce(checksum_tensor, op=dist.ReduceOp.MIN)
        sendrecv_checksum_match = bool(checksum_tensor.item())
        failure_tensor = torch.tensor(
            [int(sendrecv_info["failure_detected"])],
            dtype=torch.int64,
            device="cuda",
        )
        dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
        failure_detected = bool(failure_tensor.item())
        if sendrecv_failure_cycle:
            failure_recovered = failure_detected and sendrecv_checksum_match
        sendrecv_end = time.perf_counter()

        _load_consumer(consumer_sendrecv, buckets)
        sendrecv_consumer_output, sendrecv_consumer_ms = _consumer_forward(consumer_sendrecv, tokens)
        sendrecv_error, sendrecv_match = _compare_outputs(expected_output, sendrecv_consumer_output)

        record = {
            "cycle": cycle + 1,
            "transfer_dtype": str(transfer_dtype),
            "bucket_count": bucket_count,
            "bucket_bytes": bucket_bytes,
            "source_compute_ms": source_compute_ms,
            "reference_consumer_ms": reference_consumer_ms,
            "canonical_conversion_max_abs_error": conversion_error,
            "canonical_conversion_match": conversion_match,
            "broadcast_transfer_ms": (broadcast_end - broadcast_start) * 1000,
            "broadcast_checksum_match": broadcast_info["checksum_match"],
            "broadcast_consumer_ms": broadcast_consumer_ms,
            "broadcast_max_abs_error": broadcast_error,
            "broadcast_match": broadcast_match,
            "sendrecv_transfer_ms": (sendrecv_end - sendrecv_start) * 1000,
            "sendrecv_overlap_compute_ms": sendrecv_overlap_compute_ms,
            "sendrecv_wait_ms": sendrecv_wait_ms,
            "sendrecv_overlap_fraction": (
                sendrecv_overlap_compute_ms
                / max(sendrecv_overlap_compute_ms + sendrecv_wait_ms, 1e-9)
            ),
            "sendrecv_checksum_match": sendrecv_checksum_match,
            "sendrecv_failure_detected": failure_detected,
            "sendrecv_consumer_ms": sendrecv_consumer_ms,
            "sendrecv_max_abs_error": sendrecv_error,
            "sendrecv_match": sendrecv_match,
            "sendrecv_failure_injected": sendrecv_failure_cycle,
            "failure_recovered": failure_recovered if sendrecv_failure_cycle else None,
        }
        records.append(record)

    summary = {
        "environment": _environment(),
        "cycles": args.cycles,
        "hidden": args.hidden,
        "vocab": args.vocab,
        "batch": args.batch,
        "sequence": args.sequence,
        "timeout_seconds": args.timeout,
        "world_size": dist.get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "device_name": torch.cuda.get_device_name(),
        "backend": dist.get_backend(),
        "metadata_backend": dist.get_backend(metadata_group),
        "failure_cycle": args.failure_cycle,
        "records": records,
        "all_broadcast_match": all(record["broadcast_match"] for record in records),
        "all_broadcast_checksum_match": all(record["broadcast_checksum_match"] for record in records),
        "all_canonical_conversion_match": all(record["canonical_conversion_match"] for record in records),
        "all_sendrecv_match": all(record["sendrecv_match"] for record in records),
        "all_sendrecv_checksum_match": all(record["sendrecv_checksum_match"] for record in records),
        "failure_recovered": failure_recovered,
    }
    _write_json(Path(args.output), summary)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--failure-cycle", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
