#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIXTURE_TEXTS = (
    "The quick brown fox jumps over the lazy dog.",
    "A small deterministic fixture measures log probabilities.",
    "Packed sequences preserve boundaries during attention.",
    "Batch invariance requires careful numerical kernels.",
    "The rollout engine and training model share weights.",
    "Numerical evidence should distinguish bias from noise.",
    "Controlled experiments reduce unrelated variation.",
    "Honest negative results still improve engineering.",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/job/cache/models/Qwen3-0.6B")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results.json")))
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def module_identity(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    module = importlib.import_module(name)
    return {
        "path": spec.origin,
        "version": getattr(module, "__version__", None),
    }


def software_identity(model_path):
    modules = {
        name: module_identity(name)
        for name in (
            "torch",
            "transformers",
            "miles",
            "sglang",
            "megatron.core",
            "transformer_engine",
            "flash_attn",
            "triton",
            "aiter",
        )
    }
    modules["torch_native"] = module_identity("torch._C")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": modules,
        "torch_hip_version": torch.version.hip,
        "torch_cuda_version": torch.version.cuda,
        "miles_git_commit": git_commit,
    }


def hardware_identity():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This fixture requires exactly one assigned CUDA/HIP device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device_count": torch.cuda.device_count(),
        "name": properties.name,
        "gcn_arch": properties.gcnArchName,
        "major": properties.major,
        "minor": properties.minor,
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "uuid": str(getattr(properties, "uuid", "")),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_fixture(tokenizer):
    return [tokenizer(text, add_special_tokens=True)["input_ids"] for text in FIXTURE_TEXTS]


def rollout_logprobs(engine, token_sequences, batch_sizes, timeout_seconds):
    results = {}
    for batch_size in batch_sizes:
        if batch_size == 1:
            outputs = []
            for sequence in token_sequences:
                output = engine.generate(
                    input_ids=sequence,
                    sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                    return_logprob=True,
                    logprob_start_len=0,
                )
                outputs.append(output)
            selected = token_sequences
        else:
            selected = token_sequences[:batch_size]
            outputs = engine.generate(
                input_ids=selected,
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                logprob_start_len=0,
            )
        if not isinstance(outputs, list):
            raise RuntimeError(f"Expected a list of rollout outputs, got {type(outputs)}")
        batch_values = {}
        for sequence_index, (sequence, output) in enumerate(zip(selected, outputs, strict=True)):
            entries = output["meta_info"]["input_token_logprobs"]
            if len(entries) != len(sequence):
                raise RuntimeError(
                    f"Rollout logprob length {len(entries)} != token length {len(sequence)}"
                )
            values = [float(entry[0]) for entry in entries[1:]]
            if len(values) != len(sequence) - 1:
                raise RuntimeError("Rollout did not return one logprob per predicted input token")
            batch_values[sequence_index] = values
        results[batch_size] = batch_values
    return results


def load_training_model(model_path, attention_implementation):
    from transformers import AutoModelForCausalLM

    hf_attention_implementation = (
        "flash_attention_2"
        if attention_implementation.startswith("flash_")
        else attention_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=(
            "eager"
            if attention_implementation.startswith("triton")
            else hf_attention_implementation
        ),
        torch_dtype=torch.bfloat16,
    )
    if attention_implementation == "triton":
        from miles.backends.fsdp_utils.sglang_attn_bridge.hf_sglang_triton_patch import (
            apply_sglang_triton_attention_patch,
        )

        patched_layers = apply_sglang_triton_attention_patch(model)
        if patched_layers == 0:
            raise RuntimeError("Miles Triton attention bridge patched zero layers")
    model.to("cuda")
    model.eval()
    return model


def enable_training_batch_invariant_ops():
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    enable_batch_invariant_mode(enable_bmm=False)


def training_logprobs_for_logits(logits, token_sequence, vocab_size):
    from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy

    predicted_logits = logits[0, : len(token_sequence) - 1]
    target_tokens = torch.tensor(token_sequence[1:], dtype=torch.long, device=logits.device)
    log_probs, _ = calculate_log_probs_and_entropy(
        predicted_logits,
        target_tokens,
        None,
        true_on_policy=True,
        vocab_size=vocab_size,
        temperature=1.0,
    )
    return [float(value) for value in log_probs.detach().cpu().tolist()]


def run_unpacked_forward(model, token_sequences, batch_size, pad_token_id, vocab_size):
    if batch_size == 1:
        batches = [[sequence] for sequence in token_sequences]
    else:
        batches = [token_sequences[:batch_size]]
    results = {}
    for batch_index, sequences in enumerate(batches):
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length), pad_token_id, dtype=torch.long, device="cuda"
        )
        attention_mask = torch.zeros(
            (len(sequences), max_length), dtype=torch.long, device="cuda"
        )
        position_ids = torch.zeros(
            (len(sequences), max_length), dtype=torch.long, device="cuda"
        )
        for row_index, sequence in enumerate(sequences):
            input_ids[row_index, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device="cuda"
            )
            attention_mask[row_index, : len(sequence)] = 1
            position_ids[row_index, : len(sequence)] = torch.arange(
                len(sequence), dtype=torch.long, device="cuda"
            )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits
        for local_index, sequence in enumerate(sequences):
            result_index = batch_index if len(sequences) == 1 else local_index
            results[result_index] = training_logprobs_for_logits(
                logits[local_index].unsqueeze(0), sequence, vocab_size
            )
    return results


def run_packed_forward(model, token_sequences, batch_size, vocab_size):
    if batch_size == 1:
        batches = [[sequence] for sequence in token_sequences]
    else:
        batches = [token_sequences[:batch_size]]
    results = {}
    for batch_index, sequences in enumerate(batches):
        flattened = [token for sequence in sequences for token in sequence]
        lengths = [len(sequence) for sequence in sequences]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        input_ids = torch.tensor([flattened], dtype=torch.long, device="cuda")
        position_ids = torch.cat(
            [torch.arange(length, dtype=torch.long, device="cuda") for length in lengths]
        ).unsqueeze(0)
        cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        max_length = max(lengths)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max_length,
                max_length_k=max_length,
            ).logits
        offset = 0
        for local_index, sequence in enumerate(sequences):
            sequence_logits = logits[0, offset : offset + len(sequence)]
            result_index = batch_index if len(sequences) == 1 else local_index
            results[result_index] = training_logprobs_for_logits(
                sequence_logits.unsqueeze(0), sequence, vocab_size
            )
            offset += len(sequence)
    return results


def collect_training_mode(model_path, mode, token_sequences, batch_sizes, vocab_size, pad_token_id):
    model = load_training_model(model_path, mode)
    results = {}
    for batch_size in batch_sizes:
        if mode == "flash_packed":
            results[batch_size] = run_packed_forward(model, token_sequences, batch_size, vocab_size)
        else:
            results[batch_size] = run_unpacked_forward(
                model, token_sequences, batch_size, pad_token_id, vocab_size
            )
    del model
    torch.cuda.empty_cache()
    return results


def difference_metrics(training_values, rollout_values):
    if len(training_values) != len(rollout_values):
        raise RuntimeError(
            f"Cannot compare logprobs of different lengths: {len(training_values)} vs {len(rollout_values)}"
        )
    differences = [train - rollout for train, rollout in zip(training_values, rollout_values, strict=True)]
    absolute = [abs(value) for value in differences]
    return {
        "count": len(differences),
        "signed_mean": sum(differences) / len(differences),
        "absolute_mean": sum(absolute) / len(absolute),
        "absolute_max": max(absolute),
        "rmse": (sum(value * value for value in differences) / len(differences)) ** 0.5,
        "bitwise_equal": differences == [0.0] * len(differences),
    }


def compare_to_individual(results, baseline_batch, batched_batch, selected_indices):
    comparisons = {}
    for sequence_index in selected_indices:
        baseline = results[baseline_batch][sequence_index]
        batched = results[batched_batch][sequence_index]
        comparisons[sequence_index] = difference_metrics(batched, baseline)
    return comparisons


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or sorted(batch_sizes) != batch_sizes or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("Batch sizes must be unique and increasing")
    if batch_sizes[0] != 1 or batch_sizes[-1] > len(FIXTURE_TEXTS):
        raise ValueError("The sweep must start at 1 and fit the fixture")

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    token_sequences = tokenize_fixture(tokenizer)
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_identity(),
        "software": software_identity(args.model_path),
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "architecture": config.architectures,
            "dtype": "bfloat16",
            "vocab_size": config.vocab_size,
            "safetensors_sha256": sha256_file(Path(args.model_path) / "model.safetensors"),
        },
        "settings": {
            "batch_sizes": batch_sizes,
            "temperature": 1.0,
            "rollout_sampling_temperature": 0.0,
            "rollout_attention_backend": "triton",
            "rollout_deterministic": True,
            "rollout_cuda_graph": False,
            "rollout_radix_cache": False,
            "sglang_use_aiter": os.environ.get("SGLANG_USE_AITER"),
            "use_rocm_aiter_rope_backend": os.environ.get("USE_ROCM_AITER_ROPE_BACKEND"),
            "training_batch_invariant_ops": True,
            "training_logprob_path": "miles.backends.training_utils.loss_hub.math_utils.calculate_log_probs_and_entropy",
            "timeout_seconds": args.timeout_seconds,
        },
        "token_sequences": token_sequences,
    }

    from sglang import Engine

    engine = Engine(
        model_path=args.model_path,
        attention_backend="triton",
        enable_deterministic_inference=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        disable_chunked_prefix_cache=True,
        mem_fraction_static=args.mem_fraction_static,
        random_seed=0,
        watchdog_timeout=args.timeout_seconds,
        dist_timeout=args.timeout_seconds,
        skip_server_warmup=True,
        log_level="warning",
    )
    try:
        output["rollout"] = rollout_logprobs(
            engine, token_sequences, batch_sizes, args.timeout_seconds
        )
    finally:
        engine.shutdown()

    enable_training_batch_invariant_ops()
    output["training"] = {}
    for mode in ("flash_unpacked", "flash_packed", "triton_unpacked"):
        output["training"][mode] = collect_training_mode(
            args.model_path,
            mode,
            token_sequences,
            batch_sizes,
            config.vocab_size,
            tokenizer.pad_token_id,
        )

    output_path = Path(args.output)
    write_json(output_path, output)

    comparisons = {}
    rollout_comparisons = {}
    for batch_size in batch_sizes[1:]:
        selected_indices = list(range(batch_size))
        rollout_comparisons[str(batch_size)] = {
            "batch_invariance_vs_individual": {
                str(sequence_index): difference_metrics(
                    output["rollout"][batch_size][sequence_index],
                    output["rollout"][1][sequence_index],
                )
                for sequence_index in selected_indices
            }
        }
    comparisons["rollout"] = rollout_comparisons
    for mode, mode_results in output["training"].items():
        mode_comparisons = {}
        for batch_size in batch_sizes[1:]:
            selected_indices = list(range(batch_size))
            mode_comparisons[str(batch_size)] = {
                "batch_invariance_vs_individual": compare_to_individual(
                    mode_results, 1, batch_size, selected_indices
                ),
                "rollout_difference": {
                    str(sequence_index): difference_metrics(
                        mode_results[batch_size][sequence_index],
                        output["rollout"][batch_size][sequence_index],
                    )
                    for sequence_index in selected_indices
                },
            }
        comparisons[mode] = mode_comparisons
    output["comparisons"] = comparisons
    write_json(output_path, output)
    print(json.dumps(output["comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
