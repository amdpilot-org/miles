import argparse
import hashlib
import json
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("gpt-oss", "qwen3-moe"), default="gpt-oss")
    parser.add_argument("--hf-checkpoint", type=Path, default=Path("/tmp/miles-574-tiny-gptoss"))
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/miles-574-results"))
    parser.add_argument("--dump-mismatches", action="store_true")
    parser.add_argument("--skip-cross-rank-checks", action="store_true")
    return parser.parse_args()


def create_tiny_checkpoint(path: Path, architecture: str) -> None:
    from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

    if architecture == "gpt-oss":
        config = GptOssConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            sliding_window=8,
            max_position_embeddings=64,
            num_local_experts=4,
            num_experts_per_tok=2,
            attention_bias=True,
            torch_dtype="float32",
        )
        model = GptOssForCausalLM(config)
    else:
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            rope_theta=1.0,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            torch_dtype="float32",
        )
        model = Qwen3MoeForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)


def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def fill_parameters(model, rank: int, update: int) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            base = stable_hash(name) % 97
            flat = parameter.data.view(-1)
            flat.copy_(
                torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype)
                + (base + rank + update * 7) * 0.01
            )


def fill_hf_parameters(model, update: int) -> None:
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            base = stable_hash(name) % 97
            flat = parameter.data.view(-1)
            flat.copy_(
                torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype)
                + (base + update * 11) * 0.01
            )


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def instrument_collectives(records: list[dict]) -> None:
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    original_tp = MegatronParamMapping.gather_from_tp_ranks
    original_ep = MegatronParamMapping.gather_from_ep_ranks

    def tp_gather(self, tensor):
        records.append({"collective": "tp_all_gather", "param": self.megatron_param})
        return original_tp(self, tensor)

    def ep_gather(self, tensor, module, hf_param_name):
        records.append({"collective": "ep_all_gather", "param": self.megatron_param})
        return original_ep(self, tensor, module, hf_param_name)

    MegatronParamMapping.gather_from_tp_ranks = tp_gather
    MegatronParamMapping.gather_from_ep_ranks = ep_gather


def build_model(hf_checkpoint: Path, ep_size: int, architecture: str):
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 4
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = ep_size
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = True
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.moe_grouped_gemm = architecture == "gpt-oss"
    provider.bf16 = False
    provider.fp16 = False
    provider.params_dtype = torch.float32
    if hasattr(provider, "finalize"):
        provider.finalize()
    return bridge, provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=False,
        init_model_with_meta_device=False,
        mixed_precision_wrapper=None,
    )


def expert_ownership(collective_records: list[dict]) -> list[str]:
    experts = []
    for record in collective_records:
        match = re.search(r"(?:weight|bias)(\d+)$", record["param"])
        if match:
            experts.append(match.group(1))
    return sorted(set(experts))


def normalized_collective_sequence(records: list[dict]) -> list[dict]:
    return [
        {
            "collective": record["collective"],
            "param": re.sub(r"((?:weight|bias))\d+$", r"\1<expert>", record["param"]),
        }
        for record in records
    ]


def local_parameter_digests(model) -> dict[str, str]:
    return {
        name: tensor_digest(parameter)
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank == 0 and not (args.hf_checkpoint / "config.json").exists():
        create_tiny_checkpoint(args.hf_checkpoint, args.architecture)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))

    collective_records = []
    instrument_collectives(collective_records)
    bridge, model_list = build_model(args.hf_checkpoint, args.ep_size, args.architecture)
    model = model_list[0]
    fill_parameters(model, rank, -1)
    hf_model = bridge.hf_pretrained
    _ = hf_model.model
    hf_model._state_dict_accessor = None
    fill_hf_parameters(hf_model, 0)
    bridge._model_bridge.load_weights_hf_to_megatron(hf_model, model_list)

    from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

    iterator_args = argparse.Namespace(
        hf_checkpoint=str(args.hf_checkpoint),
        megatron_to_hf_mode="bridge",
        update_weight_buffer_size=1024 * 1024,
        vocab_size=64,
        custom_model_provider_path=None,
        q_lora_rank=None,
        sglang_speculative_algorithm=None,
    )
    iterator = get_hf_weight_iterator(
        iterator_args,
        model_list,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name="gptoss" if args.architecture == "gpt-oss" else "qwen3moe",
        quantization_config=None,
    )

    round_results = []
    cross_rank_checks = not args.skip_cross_rank_checks
    for update in range(args.rounds):
        load_start = time.monotonic()
        fill_hf_parameters(hf_model, update)
        bridge._model_bridge.load_weights_hf_to_megatron(hf_model, model_list)
        load_seconds = time.monotonic() - load_start
        collective_records.clear()
        export_start = time.monotonic()
        converted = {}
        for unit in iterator.iter_hf_weights(dict(model.named_parameters()), materialize=True):
            for name, tensor in unit:
                converted[name] = tensor.detach()
        export_seconds = time.monotonic() - export_start
        expected = dict(hf_model.state.items())
        missing = sorted(set(expected) - set(converted))
        unexpected = sorted(set(converted) - set(expected))
        if missing or unexpected:
            raise RuntimeError(f"converted tensor names differ: missing={missing}, unexpected={unexpected}")
        mismatches = []
        for name, expected_tensor in expected.items():
            converted_tensor = converted[name].detach().cpu()
            expected_tensor = expected_tensor.detach().cpu()
            if converted_tensor.dtype != expected_tensor.dtype or not torch.equal(converted_tensor, expected_tensor):
                mismatches.append(name)
                if args.dump_mismatches:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"converted": converted_tensor, "expected": expected_tensor},
                        args.output_dir / f"rank{rank}-{name.replace('/', '_')}.pt",
                    )
        if mismatches:
            raise RuntimeError(f"converted tensors differ from HF reference: {mismatches[:5]}")
        converted_digests = {
            name: {"shape": list(tensor.shape), "digest": tensor_digest(tensor)}
            for name, tensor in sorted(converted.items())
        }
        if cross_rank_checks:
            gathered_records = [None] * dist.get_world_size()
            gathered_digests = [None] * dist.get_world_size()
            local_sequence = normalized_collective_sequence(collective_records)
            dist.all_gather_object(gathered_records, local_sequence)
            dist.all_gather_object(gathered_digests, converted_digests)
            if any(record != gathered_records[0] for record in gathered_records[1:]):
                raise RuntimeError("collective sequences differ across ranks")
            if any(digests != gathered_digests[0] for digests in gathered_digests[1:]):
                raise RuntimeError("converted tensor references differ across ranks")
        round_results.append(
            {
                "update": update,
                "load_seconds": load_seconds,
                "export_seconds": export_seconds,
                "collectives": list(collective_records),
                "local_parameters": local_parameter_digests(model),
                "expected_tensors": {
                    name: {"shape": list(tensor.shape), "digest": tensor_digest(tensor)}
                    for name, tensor in sorted(expected.items())
                },
                "tensors": {
                    name: {"shape": list(tensor.shape), "digest": tensor_digest(tensor)}
                    for name, tensor in sorted(converted.items())
                },
            }
        )
        if update == 0:
            references = {name: tensor.clone() for name, tensor in converted.items()}
        else:
            mismatches = [name for name in references if name not in converted]
            mismatches.extend(name for name in converted if name not in references)
            mismatches.extend(
                name for name, tensor in converted.items() if name in references and not torch.equal(tensor, references[name])
            )
            if mismatches:
                raise RuntimeError(f"converted tensors changed unexpectedly: {mismatches[:5]}")

    output = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "ep_size": args.ep_size,
        "rounds": args.rounds,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device_name": torch.cuda.get_device_name(local_rank),
        "device_capability": torch.cuda.get_device_capability(local_rank),
        "visible_device_count": torch.cuda.device_count(),
        "expert_ownership": expert_ownership(round_results[-1]["collectives"]),
        "rounds_result": round_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps(output, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
