from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import Qwen3MoeForCausalLM

import miles.backends.fsdp_utils.adaptations.specs.qwen3_moe
from miles.backends.fsdp_utils.adaptations import routing_replay
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def initialize_parallel_state() -> None:
    trivial_group = dist.new_group([dist.get_rank()])
    trivial = GroupInfo(rank=0, size=1, group=trivial_group)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def logprob_args(vocab_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        qkv_format="bshd",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=1024,
        vocab_size=vocab_size,
        allgather_cp=False,
    )


def token_logprobs(model, input_ids, args):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids=input_ids).logits
    batch_size, sequence_length = input_ids.shape
    result = get_log_probs_and_entropy(
        logits=logits,
        args=args,
        unconcat_tokens=[row for row in input_ids],
        total_lengths=[sequence_length] * batch_size,
        response_lengths=[16] * batch_size,
        with_entropy=False,
        max_seq_lens=[sequence_length] * batch_size,
    )
    return torch.cat(result["log_probs"])


def tensor_stats(left, right):
    difference = (left.float() - right.float()).abs()
    return {
        "mean_abs": difference.mean().item(),
        "max_abs": difference.max().item(),
    }


def gradient_stats(model, reference_model):
    stats = {"max_abs": 0.0, "mean_abs": 0.0, "count": 0}
    total = 0.0
    count = 0
    for parameter, reference in zip(model.parameters(), reference_model.parameters(), strict=True):
        if parameter.grad is None or reference.grad is None:
            continue
        difference = (parameter.grad.float() - reference.grad.float()).abs()
        stats["max_abs"] = max(stats["max_abs"], difference.max().item())
        total += difference.sum().item()
        count += difference.numel()
    stats["mean_abs"] = total / count if count else 0.0
    stats["count"] = count
    return stats


def parameter_drift(model, reference_model):
    return max(
        (parameter.float() - reference.float()).abs().max().item()
        for parameter, reference in zip(model.parameters(), reference_model.parameters(), strict=True)
    )


def routing_difference(on_routing, off_routing):
    if len(on_routing) != len(off_routing):
        return {"equal": False, "max_abs": 1_000_000.0}
    return {
        "equal": all(torch.equal(left, right) for left, right in zip(on_routing, off_routing)),
        "max_abs": max(
            (left.float() - right.float()).abs().max().item()
            for left, right in zip(on_routing, off_routing)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(20260910)
    initialize_parallel_state()

    compute_dtype = torch.float16
    parameter_dtype = torch.float32
    reference_model = Qwen3MoeForCausalLM.from_pretrained(
        args.model, torch_dtype=parameter_dtype, use_cache=False
    ).to(local_rank)
    reference_model.eval()
    r3_on_model = copy.deepcopy(reference_model)
    r3_off_model = copy.deepcopy(reference_model)
    r3_on_model.train()
    r3_off_model.train()

    replay_args = SimpleNamespace(
        use_rollout_routing_replay=True,
        use_routing_replay=True,
        ci_test=True,
    )
    routing_replay.enable(replay_args)
    stream_count = routing_replay.install(r3_on_model, r3_on_model.config)

    off_routing_capture = []
    for module in r3_off_model.modules():
        if type(module).__name__ == "Qwen3MoeTopKRouter":
            module.register_forward_hook(
                lambda module, inputs, output: off_routing_capture.append(output[2].detach().cpu())
            )

    r3_on_model = DistributedDataParallel(
        r3_on_model, device_ids=[local_rank], broadcast_buffers=False
    )
    r3_off_model = DistributedDataParallel(
        r3_off_model, device_ids=[local_rank], broadcast_buffers=False
    )
    on_optimizer = torch.optim.AdamW(r3_on_model.parameters(), lr=args.learning_rate)
    off_optimizer = torch.optim.AdamW(r3_off_model.parameters(), lr=args.learning_rate)
    logprob_config = logprob_args(reference_model.config.vocab_size)

    generator = torch.Generator(device="cpu").manual_seed(20260910)
    records = []
    phase_seconds = {
        "reference": 0.0,
        "r3_on_logprob": 0.0,
        "r3_on_train": 0.0,
        "r3_off_logprob": 0.0,
        "r3_off_train": 0.0,
        "optimizer": 0.0,
    }
    run_start = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    input_ids = torch.randint(
        0, reference_model.config.vocab_size, (2, 32), generator=generator
    ).to(local_rank)

    for cycle in range(args.cycles):
        reference_model.zero_grad(set_to_none=True)
        phase_start = time.perf_counter()
        with torch.no_grad():
            reference_logprobs = token_logprobs(reference_model, input_ids, logprob_config)
        reference_loss = -token_logprobs(reference_model, input_ids, logprob_config).mean()
        reference_loss.backward()
        torch.cuda.synchronize()
        phase_seconds["reference"] += time.perf_counter() - phase_start

        routing_replay.reset()
        with routing_replay.stage(routing_replay.RECORD), torch.no_grad():
            r3_on_model(input_ids=input_ids)
        on_routing = [
            tensor.detach().cpu()
            for replay in routing_replay_manager_replays()
            for tensor in replay.top_indices_list
        ]

        phase_start = time.perf_counter()
        routing_replay.rewind()
        with routing_replay.stage(routing_replay.REPLAY_FORWARD), torch.no_grad():
            on_logprobs = token_logprobs(r3_on_model, input_ids, logprob_config)
        torch.cuda.synchronize()
        phase_seconds["r3_on_logprob"] += time.perf_counter() - phase_start

        on_optimizer.zero_grad(set_to_none=True)
        phase_start = time.perf_counter()
        routing_replay.rewind()
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            on_training_logprobs = token_logprobs(r3_on_model, input_ids, logprob_config)
            on_loss = -on_training_logprobs.mean()
        with routing_replay.stage(routing_replay.REPLAY_BACKWARD):
            on_loss.backward()
        torch.cuda.synchronize()
        phase_seconds["r3_on_train"] += time.perf_counter() - phase_start

        off_routing_capture.clear()
        phase_start = time.perf_counter()
        with torch.no_grad():
            off_logprobs = token_logprobs(r3_off_model, input_ids, logprob_config)
        off_routing = [tensor.clone() for tensor in off_routing_capture]
        torch.cuda.synchronize()
        phase_seconds["r3_off_logprob"] += time.perf_counter() - phase_start

        off_optimizer.zero_grad(set_to_none=True)
        phase_start = time.perf_counter()
        off_training_logprobs = token_logprobs(r3_off_model, input_ids, logprob_config)
        off_loss = -off_training_logprobs.mean()
        off_loss.backward()
        torch.cuda.synchronize()
        phase_seconds["r3_off_train"] += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        on_optimizer.step()
        off_optimizer.step()
        torch.cuda.synchronize()
        phase_seconds["optimizer"] += time.perf_counter() - phase_start

        on_reference_error = tensor_stats(on_logprobs, reference_logprobs)
        off_reference_error = tensor_stats(off_logprobs, reference_logprobs)
        on_off_error = tensor_stats(on_logprobs, off_logprobs)
        routing = routing_difference(on_routing, off_routing)
        on_gradient = gradient_stats(r3_on_model, reference_model)
        off_gradient = gradient_stats(r3_off_model, reference_model)
        on_off_gradient = tensor_stats(
            torch.cat([parameter.grad.flatten() for parameter in r3_on_model.parameters() if parameter.grad is not None]),
            torch.cat([parameter.grad.flatten() for parameter in r3_off_model.parameters() if parameter.grad is not None]),
        )

        records.append(
            {
                "cycle": cycle,
                "tokens": input_ids.detach().cpu().tolist(),
                "routing": routing,
                "r3_on": {
                    "per_token_abs_error": (on_logprobs - reference_logprobs).abs().detach().cpu().tolist(),
                    "logprob_error": on_reference_error,
                    "gradient_vs_reference": on_gradient,
                    "parameter_drift_max": parameter_drift(r3_on_model, reference_model),
                },
                "r3_off": {
                    "per_token_abs_error": (off_logprobs - reference_logprobs).abs().detach().cpu().tolist(),
                    "logprob_error": off_reference_error,
                    "gradient_vs_reference": off_gradient,
                    "parameter_drift_max": parameter_drift(r3_off_model, reference_model),
                },
                "r3_on_vs_off": {
                    "logprob": on_off_error,
                    "gradient": on_off_gradient,
                },
            }
        )
        routing_replay.reset()

    total_seconds = time.perf_counter() - run_start
    dist.barrier()
    if rank == 0:
        payload = {
            "cycles": args.cycles,
            "world_size": world_size,
            "compute_dtype": str(compute_dtype),
            "parameter_dtype": str(parameter_dtype),
            "r3_streams": stream_count,
            "learning_rate": args.learning_rate,
            "phase_seconds": phase_seconds,
            "total_seconds": total_seconds,
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    dist.destroy_process_group()


def routing_replay_manager_replays():
    from miles.utils.replay_base import routing_replay_manager

    return routing_replay_manager.replays


if __name__ == "__main__":
    main()
