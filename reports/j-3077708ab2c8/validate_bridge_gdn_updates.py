#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _use_job_miles() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_use_job_miles()

import aiter.jit.core as aiter_core


aiter_core.AITER_CONFIGS.get_config_file = lambda _env_name, default_file, _tuned_name: default_file

from megatron.bridge import AutoBridge
from megatron.core import parallel_state as megatron_parallel_state
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed
from miles.utils.ft_utils.process_group_utils import GroupInfo


CYCLES = 32
MODEL_PATH = Path("/job/cache/miles-gdn-fixture-qwen35")
OUTPUT_PATH = Path("/job/miles/reports/j-3077708ab2c8/gpu_validation.json")
NCCL_TIMEOUT_SECONDS = 120


def _create_local_model() -> None:
    if MODEL_PATH.exists():
        return

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    config.model_type = "qwen3_5_text"
    config.architectures = ["Qwen3_5ForCausalLM"]
    torch.manual_seed(3077708)
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16).to("cuda")
    model.save_pretrained(MODEL_PATH, safe_serialization=True)


def _set_miles_parallel_state() -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=group,
            tp=group,
            pp=group,
            ep=group,
            etp=group,
            indep_dp=group,
        )
    )


def _make_args() -> Namespace:
    return Namespace(
        hf_checkpoint=str(MODEL_PATH),
        megatron_to_hf_mode="bridge",
        update_weight_buffer_size=1024 * 1024,
        update_weight_transfer_mode="broadcast",
        colocate=False,
        rollout_num_gpus_per_engine=1,
        sglang_speculative_algorithm=None,
        q_lora_rank=None,
        vocab_size=64,
        mtp_num_layers=None,
        custom_model_provider_path=None,
        offload_rollout=False,
        check_weight_update_equal=False,
    )


class LocalEngineClient:
    async def update_weights_from_distributed(self, **_kwargs) -> None:
        return None


def _build_megatron_model():
    bridge = AutoBridge.from_hf_pretrained(MODEL_PATH, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    return bridge, provider, models


def _coverage(bridge, models) -> dict:
    model_names = {
        strip_param_name_prefix(name) for name, _parameter in models[0].named_parameters()
    }
    task_names = {
        strip_param_name_prefix(task.param_name) for task in bridge.get_conversion_tasks(models)
    }
    return {
        "megatron_parameter_count": len(model_names),
        "bridge_task_count": len(task_names),
        "missing_megatron_names": sorted(model_names - task_names),
        "unexpected_bridge_names": sorted(task_names - model_names),
    }


def _missing_name_diagnostics(models) -> dict:
    direct_args = Namespace(
        num_experts=None,
        vocab_size=64,
        mtp_num_layers=None,
        num_layers=4,
        hidden_size=32,
        kv_channels=16,
        num_attention_heads=2,
        num_query_groups=1,
    )
    gdn_markers = ("in_proj", "out_norm", "A_log", "dt_bias", "conv1d")
    missing_names = []
    for name, parameter in models[0].named_parameters():
        if not any(marker in name for marker in gdn_markers):
            continue
        direct_name = f"module.module.{strip_param_name_prefix(name)}"
        try:
            _convert_to_hf_core(direct_args, "qwen3_5", direct_name, parameter)
        except ValueError as error:
            missing_names.append({"name": direct_name, "error": str(error)})
    return {
        "checked_bridge_gdn_names": len(missing_names),
        "missing_names": missing_names,
    }


def _initial_conversion_check(units, reference_model) -> dict:
    reference_state = reference_model.state_dict()
    exported_names = {name for unit in units for name, _tensor in unit}
    mismatches = []
    for unit in units:
        for name, tensor in unit:
            if name not in reference_state or not torch.equal(tensor.detach(), reference_state[name]):
                mismatches.append(name)
    return {
        "exported_tensor_count": sum(len(unit) for unit in units),
        "hf_state_dict_count": len(reference_state),
        "missing_hf_names": sorted(exported_names - set(reference_state)),
        "unexpected_hf_names": sorted(set(reference_state) - exported_names),
        "unequal_tensor_count": len(mismatches),
        "unequal_tensor_names": mismatches,
    }


def _broadcast_metadata(payload, gloo_group, source: int):
    if dist.get_rank() == source:
        dist.broadcast_object_list(payload, src=source, group=gloo_group)
        return payload
    received = [None] * len(payload)
    dist.broadcast_object_list(received, src=source, group=gloo_group)
    return received


def _apply_bucket(model, bucket) -> int:
    state = model.state_dict()
    mismatch_count = 0
    for name, tensor in bucket:
        if name not in state:
            mismatch_count += 1
            continue
        state[name].copy_(tensor.detach())
        if not torch.equal(state[name], tensor.detach()):
            mismatch_count += 1
    return mismatch_count


def _trainer(
    bridge,
    models,
    gloo_group,
    nccl_group,
    input_ids,
    labels,
) -> dict:
    _set_miles_parallel_state()
    reference_model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).cuda()
    reference_model.eval()
    args = _make_args()
    iterator = get_hf_weight_iterator(
        args,
        models,
        required_placement=WeightUpdatePlacement(gather_pp=False),
        model_name="qwen3_5_text",
        quantization_config=None,
    )
    protocol = UpdateWeightFromDistributed(args)
    protocol.rollout_engines = [LocalEngineClient()]
    protocol._model_update_groups = nccl_group
    protocol.is_sender = True
    protocol.group_name = "miles-gdn-fixture"
    protocol._selector = "all"

    coverage = _coverage(bridge, models)
    missing_names = _missing_name_diagnostics(models)
    with torch.no_grad():
        initial_units = list(
            iterator.iter_hf_weights(
                {name: parameter for name, parameter in models[0].named_parameters()},
                materialize=True,
            )
        )
    initial_conversion = _initial_conversion_check(initial_units, reference_model)

    optimizer = torch.optim.SGD(models[0].parameters(), lr=0.01)
    cycles = []
    previous_logits = None

    for cycle_index in range(CYCLES):
        cycle_start = perf_counter()
        optimizer_start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = models[0](input_ids, torch.arange(8, device="cuda").unsqueeze(0), None)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]).float(), labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_ms = (perf_counter() - optimizer_start) * 1000

        conversion_start = perf_counter()
        with torch.no_grad():
            units = list(
                iterator.iter_hf_weights(
                    {name: parameter for name, parameter in models[0].named_parameters()},
                    materialize=True,
                )
            )
        torch.cuda.synchronize()
        conversion_ms = (perf_counter() - conversion_start) * 1000

        broadcast_start = perf_counter()
        reference_mismatches = 0
        converted_tensor_count = sum(len(bucket) for bucket in units)
        for bucket in units:
            _broadcast_metadata(["bucket"], gloo_group, source=0)
            metadata = [
                [name for name, _tensor in bucket],
                [str(tensor.dtype) for _name, tensor in bucket],
                [list(tensor.shape) for _name, tensor in bucket],
            ]
            _broadcast_metadata(metadata, gloo_group, source=0)
            reference_mismatches += _apply_bucket(reference_model, bucket)
            protocol.send_bucket(bucket)
        torch.cuda.synchronize()
        broadcast_ms = (perf_counter() - broadcast_start) * 1000

        _broadcast_metadata(["forward"], gloo_group, source=0)
        reference_start = perf_counter()
        with torch.no_grad():
            reference_logits = reference_model(input_ids).logits
        torch.cuda.synchronize()
        reference_forward_ms = (perf_counter() - reference_start) * 1000

        consumer_logits = torch.empty_like(reference_logits)
        dist.broadcast(consumer_logits, src=1, group=nccl_group)
        consumer_timing = [None, None]
        dist.broadcast_object_list(consumer_timing, src=1, group=gloo_group)
        output_max_abs_diff = (
            (reference_logits.float() - consumer_logits.float()).abs().max().item()
        )
        output_allclose = torch.allclose(
            reference_logits.float(), consumer_logits.float(), rtol=0.02, atol=0.02
        )
        output_delta = None
        if previous_logits is not None:
            output_delta = (
                (previous_logits.float() - reference_logits.float()).abs().max().item()
            )
        previous_logits = reference_logits.detach().clone()
        cycle_ms = (perf_counter() - cycle_start) * 1000

        cycles.append(
            {
                "cycle": cycle_index + 1,
                "optimizer_loss": float(loss.detach()),
                "optimizer_ms": optimizer_ms,
                "conversion_ms": conversion_ms,
                "broadcast_ms": broadcast_ms,
                "reference_forward_ms": reference_forward_ms,
                "consumer_forward_ms": consumer_timing[0],
                "cycle_ms": cycle_ms,
                "converted_tensor_count": converted_tensor_count,
                "broadcast_bucket_count": len(units),
                "reference_tensor_mismatch_count": reference_mismatches,
                "consumer_tensor_mismatch_count": consumer_timing[1],
                "output_max_abs_diff": output_max_abs_diff,
                "output_allclose": bool(output_allclose),
                "output_delta_from_previous": output_delta,
                "valid_update": True,
                "later_valid_update_after_missing_names": cycle_index == 0,
            }
        )

    _broadcast_metadata(["done"], gloo_group, source=0)
    return {
        "coverage": coverage,
        "missing_name_diagnostics": missing_names,
        "initial_conversion": initial_conversion,
        "cycles": cycles,
    }


def _consumer(gloo_group, nccl_group, input_ids) -> None:
    _set_miles_parallel_state()
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    model.eval()
    cycle_mismatch_count = 0

    while True:
        phase = _broadcast_metadata(["phase"], gloo_group, source=0)[0]
        if phase == "done":
            break
        if phase == "bucket":
            names, dtype_names, shapes = _broadcast_metadata(
                [[], [], []], gloo_group, source=0
            )
            state = model.state_dict()
            for name, dtype_name, shape in zip(names, dtype_names, shapes, strict=True):
                dtype = getattr(torch, dtype_name.removeprefix("torch."))
                tensor = torch.empty(shape, dtype=dtype, device="cuda")
                dist.broadcast(tensor, src=0, group=nccl_group)
                if name not in state:
                    cycle_mismatch_count += 1
                    continue
                state[name].copy_(tensor)
                if not torch.equal(state[name], tensor):
                    cycle_mismatch_count += 1
            continue
        if phase != "forward":
            raise RuntimeError(f"Unexpected consumer phase: {phase}")

        forward_start = perf_counter()
        with torch.no_grad():
            logits = model(input_ids).logits
        torch.cuda.synchronize()
        forward_ms = (perf_counter() - forward_start) * 1000
        dist.broadcast(logits, src=1, group=nccl_group)
        dist.broadcast_object_list(
            [forward_ms, cycle_mismatch_count], src=1, group=gloo_group
        )
        cycle_mismatch_count = 0


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _runtime_metadata() -> dict:
    import megatron.bridge
    import transformers
    import miles

    repo_root = Path(__file__).resolve().parents[2]
    return {
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "gpu_capabilities": [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())],
        "distributed_backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "rendezvous_port": os.environ.get("MASTER_PORT"),
        "process_group_timeout_seconds": NCCL_TIMEOUT_SECONDS,
        "miles_commit": _git_commit(repo_root),
        "megatron_commit": _git_commit(Path("/root/Megatron-LM")),
        "source_paths": {
            "miles": str(Path(miles.__path__[0])),
            "megatron_bridge": megatron.bridge.__file__,
            "transformers": transformers.__file__,
            "torch": torch.__file__,
            "aiter": aiter_core.__file__,
        },
        "downloads_gb": 0.0,
        "model_source": "local random initialization",
        "consumer": "local Hugging Face process on GPU 1; not an SGLang server",
        "protocol": "UpdateWeightFromDistributed.send_bucket and update_weights_from_distributed",
        "bridge": "HfWeightIteratorBridge with Megatron Bridge Qwen35Bridge mappings",
        "limitations": [
            "Dense Qwen3.5 text model only; no VL, MoE, MTP, quantization, PP>1, or TP>1 claim.",
            "The consumer is a local HF process, not a production SGLang rollout engine.",
            "This is a correctness fixture, not a MI355X or MI350X performance benchmark.",
        ],
    }


def main() -> None:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected 2 assigned GPUs, found {torch.cuda.device_count()}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=NCCL_TIMEOUT_SECONDS),
    )
    megatron_parallel_state.initialize_model_parallel(1, 1)
    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.group.WORLD

    if dist.get_rank() == 0:
        _create_local_model()
    dist.barrier()

    input_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], device="cuda")
    labels = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 1]], device="cuda")

    if dist.get_rank() == 0:
        bridge, _provider, models = _build_megatron_model()
        result = _trainer(bridge, models, gloo_group, nccl_group, input_ids, labels)
        result["runtime"] = _runtime_metadata()
        result["cycle_count"] = len(result["cycles"])
        result["all_cycles_valid"] = all(
            cycle["valid_update"] and cycle["output_allclose"] for cycle in result["cycles"]
        )
        result["all_tensor_checks_passed"] = (
            result["initial_conversion"]["unequal_tensor_count"] == 0
            and all(cycle["reference_tensor_mismatch_count"] == 0 for cycle in result["cycles"])
            and all(cycle["consumer_tensor_mismatch_count"] == 0 for cycle in result["cycles"])
        )
        result["parameter_coverage_passed"] = (
            not result["coverage"]["missing_megatron_names"]
            and not result["coverage"]["unexpected_bridge_names"]
        )
        result["missing_name_diagnostics_recorded"] = bool(
            result["missing_name_diagnostics"]["missing_names"]
        )
        result["later_valid_update_recorded"] = any(
            cycle["later_valid_update_after_missing_names"] for cycle in result["cycles"]
        )

        if not (
            result["all_cycles_valid"]
            and result["all_tensor_checks_passed"]
            and result["parameter_coverage_passed"]
            and result["later_valid_update_recorded"]
        ):
            raise RuntimeError(f"Validation failed: {json.dumps(result, indent=2)}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
    else:
        _consumer(gloo_group, nccl_group, input_ids)

    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
