from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sglang import Engine


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--spec", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--nccl-port", type=int, default=29731)
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()

    engine_kwargs = {
        "model_path": str(args.model),
        "dtype": "float16",
        "tp_size": args.tp_size,
        "skip_tokenizer_init": True,
        "mem_fraction_static": 0.25,
        "attention_backend": "triton",
        "moe_runner_backend": "triton",
        "nccl_port": args.nccl_port,
        "dist_timeout": 120,
        "disable_cuda_graph": True,
        "enable_return_routed_experts": True,
        "log_level": "warning",
    }
    if args.spec:
        engine_kwargs.update(
            {
                "speculative_algorithm": "EAGLE",
                "speculative_draft_model_path": str(args.model),
                "speculative_num_steps": 2,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 3,
                "speculative_draft_attention_backend": "triton",
                "speculative_moe_runner_backend": "triton",
            }
        )

    start = time.perf_counter()
    engine = Engine(**engine_kwargs)
    startup_seconds = time.perf_counter() - start

    generator = torch.Generator(device="cpu").manual_seed(20260910)
    records = []
    generate_seconds = 0.0
    for cycle in range(args.cycles):
        input_ids = torch.randint(0, 4096, (16,), generator=generator).tolist()
        request_start = time.perf_counter()
        generate_kwargs = {
            "input_ids": [input_ids],
            "sampling_params": {
                "max_new_tokens": 16,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
        if not args.simple:
            generate_kwargs.update({"return_logprob": True, "return_routed_experts": True})
        result = engine.generate(**generate_kwargs)
        generate_seconds += time.perf_counter() - request_start
        records.append({"cycle": cycle, "input_ids": input_ids, "result": jsonable(result)})

    engine.shutdown()
    payload = {
        "path": "spec_on" if args.spec else "spec_off",
        "cycles": args.cycles,
        "startup_seconds": startup_seconds,
        "generate_seconds": generate_seconds,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
