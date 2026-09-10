from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


def build_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=4096,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=64,
        moe_intermediate_size=128,
        num_local_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        torch_dtype="float16",
    )


def add_mtp_weights(path: Path, hidden_size: int) -> None:
    weights = load_file(path / "model.safetensors", device="cpu")
    mtp_prefix = "model.mtp.layers.0."
    layer_prefix = "model.layers.0."

    for name, tensor in list(weights.items()):
        if name.startswith(layer_prefix):
            weights[mtp_prefix + name] = tensor.clone()

    generator = torch.Generator(device="cpu").manual_seed(20260910)
    weights[mtp_prefix + "fc.weight"] = torch.randn(
        (hidden_size, 2 * hidden_size), generator=generator, dtype=torch.float32
    ).to(torch.float16)
    weights[mtp_prefix + "pre_fc_norm_embedding.weight"] = torch.ones(hidden_size, dtype=torch.float16)
    weights[mtp_prefix + "pre_fc_norm_hidden.weight"] = torch.ones(hidden_size, dtype=torch.float16)
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260910)
    config = build_config()
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.num_nextn_predict_layers = 1
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    add_mtp_weights(args.output, config.hidden_size)


if __name__ == "__main__":
    main()
