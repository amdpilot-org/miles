import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import ray
import sglang
import torch
import transformers
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from sglang.srt.managers.mm_utils import tensor_hash
from transformers import AutoConfig, AutoModelForCausalLM

from miles.ray.placement_group import create_rollout_components, create_training_models
from miles.ray.wiring import launch_worker_manager
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.logging_utils import configure_logger
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking


FIXED_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
FIXED_OUTPUT_TOKENS = 4


def git_commit(path: str) -> str:
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def state_checksum(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def extract_model_state(payload: dict) -> dict:
    state = payload.get("model_state", payload)
    if "model" in state:
        state = state["model"]
    return {name: tensor for name, tensor in state.items() if isinstance(tensor, torch.Tensor)}


def load_fresh_control(model_path: str, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, str]:
    dcp_to_torch_save(str(checkpoint_path), str(checkpoint_path.with_suffix(".pt")))
    payload = torch.load(checkpoint_path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    state = extract_model_state(payload)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float16)
    model.eval()
    return model, state, state_checksum(model.state_dict())


def engine_state_for_checksum(state: dict) -> dict:
    engine_state = dict(state)
    for name in list(engine_state):
        if name.endswith(".self_attn.q_proj.weight"):
            prefix = name[: -len("q_proj.weight")]
            engine_state[f"{prefix}qkv_proj.weight"] = torch.cat(
                [
                    engine_state.pop(f"{prefix}q_proj.weight"),
                    engine_state.pop(f"{prefix}k_proj.weight"),
                    engine_state.pop(f"{prefix}v_proj.weight"),
                ],
                dim=0,
            )
            engine_state[f"{prefix}qkv_proj.bias"] = torch.cat(
                [
                    engine_state.pop(f"{prefix}q_proj.bias"),
                    engine_state.pop(f"{prefix}k_proj.bias"),
                    engine_state.pop(f"{prefix}v_proj.bias"),
                ],
                dim=0,
            )
        elif name.endswith(".mlp.gate_proj.weight"):
            prefix = name[: -len("gate_proj.weight")]
            engine_state[f"{prefix}gate_up_proj.weight"] = torch.cat(
                [
                    engine_state.pop(f"{prefix}gate_proj.weight"),
                    engine_state.pop(f"{prefix}up_proj.weight"),
                ],
                dim=0,
            )
    return engine_state


def engine_compatible_checksum(state: dict, engine_checksum: object, device: torch.device) -> dict:
    records = []

    def flatten_checksum(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                flatten_checksum(item)
        else:
            records.append(value)

    flatten_checksum(engine_checksum)
    if len(records) != 1:
        raise RuntimeError(f"expected one rollout engine checksum record, got {len(records)}")
    engine_record = records[0]
    if "ranks" in engine_record:
        engine_ranks = engine_record["ranks"]
        if len(engine_ranks) != 1:
            raise RuntimeError(f"expected one rollout engine rank, got {len(engine_ranks)}")
        engine_rank = engine_ranks[0]
        engine_tensor_checksums = engine_rank["checksums"]
        engine_overall_checksum = engine_rank["per_gpu_checksum"]
        engine_per_engine_checksum = engine_record["per_engine_checksum"]
    else:
        engine_tensor_checksums = engine_record["checksums"]
        engine_overall_checksum = engine_record["per_gpu_checksum"]
        engine_per_engine_checksum = engine_overall_checksum
    control_state = engine_state_for_checksum(state)
    control_tensor_checksums = {}
    for name, expected_checksum in engine_tensor_checksums.items():
        if name not in control_state:
            raise RuntimeError(f"fresh control checkpoint is missing engine tensor {name!r}")
        tensor = control_state[name].detach().to(device=device, dtype=torch.float16).contiguous()
        control_tensor_checksums[name] = f"{tensor_hash(tensor):016x}"
    digest = hashlib.sha256()
    for name in sorted(control_tensor_checksums):
        digest.update(name.encode())
        digest.update(control_tensor_checksums[name].encode())
    return {
        "per_tensor_equal": control_tensor_checksums == engine_tensor_checksums,
        "per_tensor_checksums": control_tensor_checksums,
        "overall_checksum": digest.hexdigest(),
        "engine_overall_checksum": engine_overall_checksum,
        "per_engine_checksum": hashlib.sha256(digest.hexdigest().encode()).hexdigest(),
        "engine_per_engine_checksum": engine_per_engine_checksum,
    }


def control_forward(model: torch.nn.Module, device: torch.device) -> tuple[list[int], list[float], float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    sequence = list(FIXED_TOKENS)
    tokens: list[int] = []
    log_probs: list[float] = []

    start_event.record()
    with torch.no_grad():
        for _ in range(FIXED_OUTPUT_TOKENS):
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[:, -1].float()
            log_prob = torch.log_softmax(logits, dim=-1).max()
            token = int(logits.argmax(dim=-1).item())
            tokens.append(token)
            log_probs.append(float(log_prob.item()))
            sequence.append(token)
    end_event.record()
    end_event.synchronize()
    return tokens, log_probs, start_event.elapsed_time(end_event)


def control_log_probs_for_tokens(model: torch.nn.Module, output_tokens: list[int], device: torch.device) -> list[float]:
    sequence = list(FIXED_TOKENS)
    values: list[float] = []
    with torch.no_grad():
        for token in output_tokens:
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[:, -1].float()
            values.append(float(torch.log_softmax(logits, dim=-1)[0, token].item()))
            sequence.append(token)
    return values


def gpu_memory_trend() -> dict[str, int]:
    result = {}
    for device in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result[str(device)] = total_bytes - free_bytes
    return result


def write_json(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")


def read_timing_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as timing_file:
        return [json.loads(line) for line in timing_file if line.strip()]


async def run(args) -> None:
    output_dir = Path(os.environ["J_INVESTIGATION_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    args.cuda_event_timing_path = str(output_dir / "cuda_events.jsonl")
    Path(args.cuda_event_timing_path).write_text("", encoding="utf-8")
    results_path = output_dir / "cycles.jsonl"
    results_path.write_text("", encoding="utf-8")

    configure_logger(args, source=MainProcessIdentity())
    worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    actor_model, critic_model = await create_training_models(args, inference_controller, rollout_executor)
    async with inference_controller.context_lock:
        updatable_server = next(server for server in inference_controller.servers.values() if server.update_weights)
        api_client = updatable_server.api_clients[0]
    control_device = torch.device(0)
    torch.cuda.set_device(control_device)

    metadata = {
        "miles_commit": git_commit("/job/miles"),
        "sglang_commit": git_commit(str(Path(sglang.__file__).parents[2])),
        "miles_path": "/job/miles/miles",
        "torch_path": torch.__file__,
        "sglang_path": sglang.__file__,
        "transformers_path": transformers.__file__,
        "ray_path": ray.__file__,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "transformers_version": transformers.__version__,
        "sglang_version": sglang.__version__,
        "ray_version": ray.__version__,
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "training_pg_timeout_seconds": args.distributed_timeout_minutes * 60,
        "update_pg_default_timeout_seconds": 1800,
        "ray_tmpdir": os.environ.get("RAY_TMPDIR"),
        "run_uuid": args.run_uuid,
        "num_rollout_per_epoch": num_rollout_per_epoch,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    initial_version = await actor_model.update_weights(rollout_id=-1)
    await rollout_executor.set_weight_version.remote(initial_version)

    try:
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            cycle_start = time.perf_counter()
            await inference_controller.prepare_rollout(rollout_id)
            rollout_start = time.perf_counter()
            rollout_data_pack = await rollout_executor.get.remote(rollout_id)
            rollout_seconds = time.perf_counter() - rollout_start

            train_start = time.perf_counter()
            await actor_model.train(rollout_id, rollout_data_pack)
            train_seconds = time.perf_counter() - train_start
            remove_rollout_data_refs(args, rollout_data_pack)
            await actor_model.clear_memory()

            update_start = time.perf_counter()
            version = await actor_model.update_weights(rollout_id=rollout_id)
            update_seconds = time.perf_counter() - update_start
            await rollout_executor.set_weight_version.remote(version)
            engine_version = await api_client.get_weight_version()
            engine_checksum = await inference_controller.check_weights("checksum")

            save_start = time.perf_counter()
            await actor_model.save_model(rollout_id)
            save_seconds = time.perf_counter() - save_start
            checkpoint_dir = Path(args.save) / f"iter_{rollout_id + 1:07d}"
            checkpoint_metadata = json.loads((checkpoint_dir / "meta.json").read_text())
            checkpoint_global_step = int(checkpoint_metadata["global_step"])
            timing_records = read_timing_records(Path(args.cuda_event_timing_path))
            optimizer_steps_this_cycle = sum(
                record["phase"] == "optimizer" and record["rollout_id"] == rollout_id
                for record in timing_records
            )
            cumulative_optimizer_steps = sum(record["phase"] == "optimizer" for record in timing_records)
            update_events_this_cycle = sum(
                record["phase"] == "update_weights" and record["rollout_id"] == rollout_id
                for record in timing_records
            )
            cumulative_weight_updates = sum(
                record["phase"] == "update_weights" and record["rollout_id"] >= 0 for record in timing_records
            )

            control_start = time.perf_counter()
            control_model, control_state, control_checksum = load_fresh_control(
                args.hf_checkpoint, checkpoint_dir / "model", control_device
            )
            control_engine_checksum = engine_compatible_checksum(control_state, engine_checksum, control_device)
            control_tokens, control_log_probs, control_cuda_ms = control_forward(control_model, control_device)
            payload = {
                "input_ids": FIXED_TOKENS,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": FIXED_OUTPUT_TOKENS},
                "return_logprob": True,
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{api_client.server_url}/generate", json=payload)
                response.raise_for_status()
                engine_output = response.json()
            output_logprobs = engine_output["meta_info"]["output_token_logprobs"]
            engine_tokens = [int(item[1]) for item in output_logprobs]
            engine_log_probs = [float(item[0]) for item in output_logprobs]
            control_for_engine = control_log_probs_for_tokens(control_model, engine_tokens, control_device)
            log_prob_differences = [abs(left - right) for left, right in zip(engine_log_probs, control_for_engine)]
            control_seconds = time.perf_counter() - control_start
            del control_model
            torch.cuda.empty_cache()

            expected_version = rollout_id + 2
            expected_step = (rollout_id + 1) * args.num_steps_per_rollout
            record = {
                "rollout_id": rollout_id,
                "version": version,
                "engine_version": engine_version,
                "version_equal": str(version) == str(engine_version) == str(expected_version),
                "engine_compare_equal": (
                    control_engine_checksum["per_tensor_equal"]
                    and control_engine_checksum["overall_checksum"] == control_engine_checksum["engine_overall_checksum"]
                    and control_engine_checksum["per_engine_checksum"] == control_engine_checksum["engine_per_engine_checksum"]
                ),
                "engine_checksum": control_engine_checksum["engine_overall_checksum"],
                "control_checksum": control_checksum,
                "control_engine_checksum": control_engine_checksum["overall_checksum"],
                "checkpoint_global_step": checkpoint_global_step,
                "expected_global_step": expected_step,
                "global_step_metadata_equal": checkpoint_global_step == expected_step,
                "optimizer_steps_this_cycle": optimizer_steps_this_cycle,
                "cumulative_optimizer_steps": cumulative_optimizer_steps,
                "optimizer_step_count_equal": optimizer_steps_this_cycle == args.num_steps_per_rollout,
                "update_events_this_cycle": update_events_this_cycle,
                "cumulative_weight_updates": cumulative_weight_updates,
                "update_event_count_equal": update_events_this_cycle == 1,
                "fixed_tokens": FIXED_TOKENS,
                "engine_tokens": engine_tokens,
                "control_tokens": control_tokens,
                "fixed_token_output_equal": engine_tokens == control_tokens,
                "engine_log_probs": engine_log_probs,
                "control_log_probs": control_log_probs,
                "max_abs_log_prob_difference": max(log_prob_differences),
                "mean_abs_log_prob_difference": sum(log_prob_differences) / len(log_prob_differences),
                "control_cuda_event_ms": control_cuda_ms,
                "rollout_wall_seconds": rollout_seconds,
                "train_wall_seconds": train_seconds,
                "update_wall_seconds": update_seconds,
                "save_wall_seconds": save_seconds,
                "control_wall_seconds": control_seconds,
                "cycle_wall_seconds": time.perf_counter() - cycle_start,
                "gpu_memory_used_bytes": gpu_memory_trend(),
            }
            write_json(results_path, record)
            if not all(
                [
                    record["version_equal"],
                    record["optimizer_step_count_equal"],
                    record["update_event_count_equal"],
                    record["engine_compare_equal"],
                    record["fixed_token_output_equal"],
                ]
            ):
                raise RuntimeError(f"cycle {rollout_id} failed checks: {record}")
            if record["max_abs_log_prob_difference"] > 0.05:
                raise RuntimeError(f"cycle {rollout_id} log-prob drift exceeded tolerance: {record}")
            if rollout_id > 0:
                shutil.rmtree(Path(args.save) / f"iter_{rollout_id:07d}", ignore_errors=True)
    finally:
        await rollout_executor.dispose.remote()
        await inference_controller.dispose()
        await actor_model.dispose()
        if critic_model is not None:
            await critic_model.dispose()
        finish_tracking()
        del worker_manager


if __name__ == "__main__":
    arguments = parse_args()
    try:
        asyncio.run(run(arguments))
    finally:
        finish_tracking()
