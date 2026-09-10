#!/usr/bin/env python3
"""Create a tiny local Qwen2 model and GPT-2 tokenizer for GPU fixtures."""

from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(2853)
    config = Qwen2Config(
        vocab_size=50304,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_cache=True,
    )
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"saved {parameter_count} parameters to {output_dir}")


if __name__ == "__main__":
    main()
