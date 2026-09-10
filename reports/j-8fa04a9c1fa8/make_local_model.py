import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

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
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved {sum(p.numel() for p in model.parameters())} parameters to {args.output_dir}")


if __name__ == "__main__":
    main()
