#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllGather
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


@dataclass(frozen=True)
class Config:
    mode: str
    cycles: int
    seed: int
    timeout_seconds: int
    learning_rate: float
    vision_regularization: float
    output_dir: Path
    check_every: int


class TinyMultimodalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        image_features: int,
        image_tokens: int,
        text_tokens: int,
        language_blocks: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_tokens = image_tokens
        self.text_tokens = text_tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.vision = nn.Sequential(
            nn.Linear(image_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.language_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(language_blocks)
            ]
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        text_ids: torch.Tensor,
        image: torch.Tensor,
        image_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embedding = self.embedding(text_ids)
        vision_features = self.vision(image)
        fused = torch.cat((text_embedding, image_weight * vision_features), dim=1)
        for language_block in self.language_blocks:
            fused = language_block(fused)
        logits = self.output(fused)
        return logits, vision_features


class Instrumentation:
    def __init__(self, rank: int, world_size: int, output_dir: Path) -> None:
        self.rank = rank
        self.world_size = world_size
        self.output_dir = output_dir
        self.events: list[dict[str, Any]] = []
        self.event_path = output_dir / f"rank_{rank}_events.jsonl"
        self.event_file = self.event_path.open("a", encoding="utf-8")
        self.context: dict[str, Any] = {
            "cycle": None,
            "phase": None,
            "module": None,
        }
        self.original_unshard = FSDPParamGroup.unshard
        self.original_wait_for_unshard = FSDPParamGroup.wait_for_unshard
        self.original_pre_backward = FSDPParamGroup.pre_backward
        self.original_post_backward = FSDPParamGroup.post_backward
        self.original_all_gather = dist.all_gather_into_tensor
        self.original_reduce_scatter = dist.reduce_scatter_tensor

    def append_event(
        self,
        operation: str,
        module: str | None,
        elapsed_ms: float,
        details: dict[str, Any],
    ) -> None:
        event = {
            "rank": self.rank,
            "cycle": self.context["cycle"],
            "phase": self.context["phase"],
            "operation": operation,
            "module": module,
            "elapsed_ms": elapsed_ms,
        }
        event.update(details)
        self.events.append(event)
        self.event_file.write(json.dumps(event, sort_keys=True) + "\n")
        self.event_file.flush()

    def install(self) -> None:
        def module_name(self_group: FSDPParamGroup) -> str:
            return str(self_group._module_fqn or type(self_group).__name__).strip(", ")

        def unshard(self_group: FSDPParamGroup, async_op: bool = False) -> Any:
            module = module_name(self_group)
            previous_module = self.context["module"]
            self.context["module"] = module
            started = time.perf_counter()
            result = self.original_unshard(self_group, async_op)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.append_event(
                "unshard",
                module,
                elapsed_ms,
                {
                    "async_op": async_op,
                    "parameter_count": len(self_group.fsdp_params),
                },
            )
            self.context["module"] = previous_module
            return result

        def wait_for_unshard(self_group: FSDPParamGroup) -> Any:
            module = module_name(self_group)
            previous_module = self.context["module"]
            self.context["module"] = module
            started = time.perf_counter()
            result = self.original_wait_for_unshard(self_group)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.append_event(
                "wait_for_unshard",
                module,
                elapsed_ms,
                {"parameter_count": len(self_group.fsdp_params)},
            )
            self.context["module"] = previous_module
            return result

        def pre_backward(self_group: FSDPParamGroup, default_prefetch: bool, *unused: Any) -> Any:
            module = module_name(self_group)
            previous_module = self.context["module"]
            self.context["module"] = module
            started = time.perf_counter()
            result = self.original_pre_backward(self_group, default_prefetch, *unused)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.append_event(
                "pre_backward",
                module,
                elapsed_ms,
                {"default_prefetch": default_prefetch},
            )
            self.context["module"] = previous_module
            return result

        def post_backward(self_group: FSDPParamGroup, *unused: Any) -> Any:
            module = module_name(self_group)
            previous_module = self.context["module"]
            self.context["module"] = module
            started = time.perf_counter()
            result = self.original_post_backward(self_group, *unused)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.append_event(
                "post_backward",
                module,
                elapsed_ms,
                {"parameter_count": len(self_group.fsdp_params)},
            )
            self.context["module"] = previous_module
            return result

        def all_gather_into_tensor(
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            group: Any = None,
            async_op: bool = False,
        ) -> Any:
            started = time.perf_counter()
            result = self.original_all_gather(
                output_tensor,
                input_tensor,
                group=group,
                async_op=async_op,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.append_event(
                "all_gather_into_tensor",
                self.context["module"],
                elapsed_ms,
                {
                    "input_numel": input_tensor.numel(),
                    "output_numel": output_tensor.numel(),
                    "async_op": async_op,
                },
            )
            return result

        def reduce_scatter_tensor(
            output: torch.Tensor,
            input: torch.Tensor,
            group: Any = None,
            op: Any = None,
            async_op: bool = False,
        ) -> Any:
            started = time.perf_counter()
            result = self.original_reduce_scatter(
                output,
                input,
                group=group,
                op=op,
                async_op=async_op,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.append_event(
                "reduce_scatter_tensor",
                self.context["module"],
                elapsed_ms,
                {
                    "input_numel": input.numel(),
                    "output_numel": output.numel(),
                    "async_op": async_op,
                },
            )
            return result

        FSDPParamGroup.unshard = unshard
        FSDPParamGroup.wait_for_unshard = wait_for_unshard
        FSDPParamGroup.pre_backward = pre_backward
        FSDPParamGroup.post_backward = post_backward
        dist.all_gather_into_tensor = all_gather_into_tensor
        dist.reduce_scatter_tensor = reduce_scatter_tensor

    def uninstall(self) -> None:
        self.event_file.close()
        FSDPParamGroup.unshard = self.original_unshard
        FSDPParamGroup.wait_for_unshard = self.original_wait_for_unshard
        FSDPParamGroup.pre_backward = self.original_pre_backward
        FSDPParamGroup.post_backward = self.original_post_backward
        dist.all_gather_into_tensor = self.original_all_gather
        dist.reduce_scatter_tensor = self.original_reduce_scatter


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("desync", "sync"), required=True)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3011329)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vision-regularization", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-every", type=int, default=8)
    args = parser.parse_args()
    return Config(
        mode=args.mode,
        cycles=args.cycles,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        learning_rate=args.learning_rate,
        vision_regularization=args.vision_regularization,
        output_dir=args.output_dir,
        check_every=args.check_every,
    )


def git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_model(config: Config, device: torch.device) -> TinyMultimodalModel:
    torch.manual_seed(config.seed)
    model = TinyMultimodalModel(
        vocab_size=32,
        hidden_size=16,
        image_features=8,
        image_tokens=4,
        text_tokens=8,
        language_blocks=2,
    ).to(device)
    return model


def create_reference(model: TinyMultimodalModel, device: torch.device) -> TinyMultimodalModel:
    reference = TinyMultimodalModel(
        vocab_size=model.vocab_size,
        hidden_size=model.hidden_size,
        image_features=8,
        image_tokens=model.image_tokens,
        text_tokens=model.text_tokens,
        language_blocks=len(model.language_blocks),
    ).to(device)
    reference.load_state_dict(model.state_dict())
    return reference


def wrap_model(model: TinyMultimodalModel, mesh: Any) -> None:
    fully_shard(model.embedding, mesh=mesh)
    model.embedding._get_fsdp_state()._fsdp_param_group._module_fqn = "embedding"
    fully_shard(model.vision, mesh=mesh)
    model.vision._get_fsdp_state()._fsdp_param_group._module_fqn = "vision"
    for language_block_index, language_block in enumerate(model.language_blocks):
        fully_shard(language_block, mesh=mesh)
        language_block._get_fsdp_state()._fsdp_param_group._module_fqn = f"language_block_{language_block_index}"
    fully_shard(model.output, mesh=mesh)
    model.output._get_fsdp_state()._fsdp_param_group._module_fqn = "output"


def make_batch(
    config: Config,
    rank: int,
    cycle: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1_000_003 * rank + cycle)
    batch_size = 2
    text_ids_cpu = torch.randint(
        0,
        32,
        (batch_size, model_text_tokens()),
        generator=generator,
    )
    image_cpu = torch.randn(
        batch_size,
        model_image_tokens(),
        8,
        generator=generator,
    )
    labels_cpu = torch.randint(
        0,
        32,
        (batch_size, model_text_tokens() + model_image_tokens()),
        generator=generator,
    )
    text_ids = text_ids_cpu.to(device)
    image = image_cpu.to(device)
    labels = labels_cpu.to(device)
    sample_weights = torch.tensor([1.0, 0.0], device=device)
    return {
        "text_ids": text_ids,
        "image": image,
        "labels": labels,
        "sample_weights": sample_weights,
    }


def model_text_tokens() -> int:
    return 8


def model_image_tokens() -> int:
    return 4


def compute_loss(
    model: TinyMultimodalModel,
    batch: dict[str, Any],
    image_weight: float,
    vision_regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.tensor(image_weight, device=batch["image"].device)
    logits, vision_features = model(
        batch["text_ids"],
        batch["image"],
        weight,
    )
    cross_entropy = F.cross_entropy(
        logits.reshape(-1, 32),
        batch["labels"].reshape(-1),
        reduction="none",
    ).reshape(2, -1)
    per_sample_loss = cross_entropy.mean(dim=1)
    weighted_loss = (per_sample_loss * batch["sample_weights"]).sum() / batch["sample_weights"].sum().clamp(min=1.0)
    vision_regularization_loss = vision_features.pow(2).mean()
    loss = weighted_loss + vision_regularization * vision_regularization_loss
    return loss, vision_regularization_loss


def synchronize_and_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def elapsed_ms(started: float) -> float:
    return (synchronize_and_time() - started) * 1000.0


def all_reduce_reference_gradients(reference: TinyMultimodalModel, world_size: int) -> None:
    for parameter in reference.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def compare_gradients(
    model: TinyMultimodalModel,
    reference: TinyMultimodalModel,
) -> dict[str, Any]:
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    parameter_count = 0
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        if parameter.grad is None or reference_parameter.grad is None:
            continue
        fsdp_gradient = parameter.grad.full_tensor()
        reference_gradient = reference_parameter.grad.detach()
        absolute_error = (fsdp_gradient - reference_gradient).abs().max().item()
        reference_norm = reference_gradient.abs().max().item()
        relative_error = absolute_error / max(reference_norm, 1e-12)
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
        parameter_count += 1
    return {
        "parameter_count": parameter_count,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def compare_weights(
    model: TinyMultimodalModel,
    reference: TinyMultimodalModel,
) -> dict[str, Any]:
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    parameter_count = 0
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        fsdp_weight = parameter.full_tensor()
        reference_weight = reference_parameter.detach()
        absolute_error = (fsdp_weight - reference_weight).abs().max().item()
        reference_norm = reference_weight.abs().max().item()
        relative_error = absolute_error / max(reference_norm, 1e-12)
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
        parameter_count += 1
    return {
        "parameter_count": parameter_count,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def collective_signature(events: list[dict[str, Any]], cycle: int, operation: str) -> list[str | None]:
    return [
        event["module"]
        for event in events
        if event["cycle"] == cycle and event["operation"] == operation
    ]


def compare_collective_order(
    rank_events: dict[int, list[dict[str, Any]]],
    cycles: int,
) -> dict[str, Any]:
    operations = ("unshard", "wait_for_unshard", "all_gather_into_tensor", "reduce_scatter_tensor")
    mismatches: list[dict[str, Any]] = []
    for cycle in range(cycles):
        for operation in operations:
            signature_zero = collective_signature(rank_events[0], cycle, operation)
            signature_one = collective_signature(rank_events[1], cycle, operation)
            if signature_zero != signature_one:
                mismatches.append(
                    {
                        "cycle": cycle,
                        "operation": operation,
                        "rank_0": signature_zero,
                        "rank_1": signature_one,
                    }
                )
    return {
        "operations_compared": list(operations),
        "cycles_compared": cycles,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=config.timeout_seconds),
        device_id=device,
    )
    mesh = init_device_mesh("cuda", (world_size,))
    instrumentation = Instrumentation(rank, world_size, config.output_dir)
    instrumentation.install()
    model = create_model(config, device)
    reference = create_reference(model, device)
    wrap_model(model, mesh)
    fsdp_optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=config.learning_rate)
    cycle_results: list[dict[str, Any]] = []
    error_message: str | None = None
    error_type: str | None = None
    try:
        for cycle in range(config.cycles):
            fsdp_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            batch = make_batch(config, rank, cycle, device)
            image_rank = cycle % world_size
            has_image = rank == image_rank
            image_weight = 1.0 if has_image else 0.0
            if config.mode == "desync":
                execute_vision = has_image
            else:
                execute_vision = True
            instrumentation.context["cycle"] = cycle
            instrumentation.context["phase"] = "forward"
            forward_started = synchronize_and_time()
            if execute_vision:
                loss, vision_regularization_loss = compute_loss(
                    model,
                    batch,
                    image_weight,
                    config.vision_regularization,
                )
            else:
                text_embedding = model.embedding(batch["text_ids"])
                fused = text_embedding
                for language_block in model.language_blocks:
                    fused = language_block(fused)
                logits = model.output(fused)
                cross_entropy = F.cross_entropy(
                    logits.reshape(-1, 32),
                    batch["labels"][:, : model_text_tokens()].reshape(-1),
                    reduction="none",
                ).reshape(2, -1)
                per_sample_loss = cross_entropy.mean(dim=1)
                loss = (
                    per_sample_loss * batch["sample_weights"]
                ).sum() / batch["sample_weights"].sum().clamp(min=1.0)
                vision_regularization_loss = torch.tensor(0.0, device=device)
            forward_ms = elapsed_ms(forward_started)
            instrumentation.context["phase"] = "backward"
            backward_started = synchronize_and_time()
            loss.backward()
            backward_ms = elapsed_ms(backward_started)
            reference_started = synchronize_and_time()
            reference_loss, reference_vision_regularization_loss = compute_loss(
                reference,
                batch,
                image_weight,
                config.vision_regularization,
            )
            reference_loss.backward()
            all_reduce_reference_gradients(reference, world_size)
            reference_ms = elapsed_ms(reference_started)
            gradient_comparison = compare_gradients(model, reference)
            fsdp_optimizer.step()
            reference_optimizer.step()
            optimizer_started = synchronize_and_time()
            optimizer_ms = elapsed_ms(optimizer_started)
            weight_comparison = None
            if (cycle + 1) % config.check_every == 0 or cycle + 1 == config.cycles:
                weight_comparison = compare_weights(model, reference)
            cycle_result = {
                "cycle": cycle,
                "rank": rank,
                "has_image": has_image,
                "image_weight": image_weight,
                "execute_vision": execute_vision,
                "masked_sample_index": 1,
                "loss": loss.detach().item(),
                "vision_regularization_loss": vision_regularization_loss.detach().item(),
                "reference_loss": reference_loss.detach().item(),
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "reference_ms": reference_ms,
                "optimizer_ms": optimizer_ms,
                "gradient_comparison": gradient_comparison,
                "weight_comparison": weight_comparison,
            }
            cycle_results.append(cycle_result)
    except BaseException as exception:
        error_type = type(exception).__name__
        error_message = str(exception)
    finally:
        instrumentation.uninstall()
    local_payload = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "nccl_version": torch.cuda.nccl.version(),
        "commit": git_commit(),
        "config": {
            "mode": config.mode,
            "cycles": config.cycles,
            "seed": config.seed,
            "timeout_seconds": config.timeout_seconds,
            "learning_rate": config.learning_rate,
            "vision_regularization": config.vision_regularization,
            "check_every": config.check_every,
        },
        "source_paths": {
            "torch": torch.__file__,
            "fsdp_param_group": inspect.getfile(FSDPParamGroup),
            "fsdp_collectives": inspect.getfile(DefaultAllGather),
        },
        "cycle_results": cycle_results,
        "events": instrumentation.events,
        "error_type": error_type,
        "error_message": error_message,
    }
    local_output = config.output_dir / f"rank_{rank}.json"
    local_output.write_text(json.dumps(local_payload, indent=2, sort_keys=True))
    if dist.is_initialized() and error_type is None:
        dist.barrier()
        if rank == 0:
            gathered_events: dict[int, list[dict[str, Any]]] = {}
            for source_rank in range(world_size):
                rank_path = config.output_dir / f"rank_{source_rank}.json"
                if rank_path.exists():
                    rank_payload = json.loads(rank_path.read_text())
                    gathered_events[source_rank] = rank_payload["events"]
            order_comparison = compare_collective_order(gathered_events, config.cycles)
            combined = {
                "rank_0": local_payload,
                "rank_1": None,
                "collective_order_comparison": order_comparison,
            }
            rank_one_path = config.output_dir / "rank_1.json"
            if rank_one_path.exists():
                combined["rank_1"] = json.loads(rank_one_path.read_text())
            combined_output = config.output_dir / "combined.json"
            combined_output.write_text(json.dumps(combined, indent=2, sort_keys=True))
        dist.barrier()
        dist.destroy_process_group()
    if error_type is not None:
        raise RuntimeError(f"{error_type}: {error_message}")


if __name__ == "__main__":
    main()
