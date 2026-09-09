import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer



MODEL = Path("/job/cache/huggingface/Qwen3-VL-4B-Instruct")
OUTPUT = Path("/job/artifacts/gradient_check.json")


def language_vision_parameters(model):
    language = []
    vision = []
    for name, parameter in model.named_parameters():
        if name.startswith("visual.") or name.startswith("model.visual."):
            vision.append((name, parameter))
        else:
            language.append((name, parameter))
    return language, vision


def run_path(model, input_ids, labels, pixel_values, image_grid_thw, mm_token_type_ids):
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        attention_mask=torch.ones_like(input_ids),
    )
    logits = output.logits[:, :-1, :]
    targets = labels[:, 1:]
    selected = targets.ne(-100)
    token_logits = logits[selected].float()
    token_targets = targets[selected]
    loss = torch.nn.functional.cross_entropy(token_logits, token_targets)
    loss.backward()
    language, vision = language_vision_parameters(model)
    gradients = {}
    for group in (language, vision):
        for name, parameter in group:
            gradients[name] = (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
    language_max = max((gradients[name].abs().max().item() for name, _ in language), default=0.0)
    vision_max = max((gradients[name].abs().max().item() for name, _ in vision), default=0.0)
    return {
        "loss": loss.item(),
        "language_max": language_max,
        "vision_max": vision_max,
        "gradients": gradients,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).cuda()
    model.train()

    messages = [
        {"role": "user", "content": "Return the deterministic answer."},
        {"role": "assistant", "content": "seven"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = processor(text=[prompt_text], return_tensors="pt")["input_ids"][0]
    full_ids = processor(text=[full_text], return_tensors="pt")["input_ids"][0]
    assert prompt_ids.tolist() == full_ids[: prompt_ids.numel()].tolist()

    image = Image.new("RGB", (32, 32), (7, 11, 13))
    dummy_output = processor(images=[image], return_tensors="pt")
    image_tokens = int(dummy_output["image_grid_thw"][0].prod().item()) // (
        processor.image_processor.merge_size**2
    )
    vision_start = model.config.vision_start_token_id
    image_token = model.config.image_token_id
    vision_end = model.config.vision_end_token_id
    appended = [vision_start] + [image_token] * image_tokens + [vision_end]
    input_ids = torch.cat([full_ids, torch.tensor(appended)]).unsqueeze(0).cuda()
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.numel() : full_ids.numel()] = input_ids[
        :, prompt_ids.numel() : full_ids.numel()
    ]

    control = run_path(model, input_ids, labels, None, None, None)
    pixel_values = dummy_output["pixel_values"].cuda().to(model.dtype)
    image_grid_thw = dummy_output["image_grid_thw"].cuda()
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[:, full_ids.numel() + 1 : full_ids.numel() + 1 + image_tokens] = 1
    dummy = run_path(
        model, input_ids, labels, pixel_values, image_grid_thw, mm_token_type_ids
    )

    language_names = {name for name, _ in language_vision_parameters(model)[0]}
    vision_names = {name for name, _ in language_vision_parameters(model)[1]}
    differences = {
        name: (control["gradients"][name] - dummy["gradients"][name]).abs().max().item()
        for name in control["gradients"]
    }
    language_differences = {name: differences[name] for name in language_names}
    vision_differences = {name: differences[name] for name in vision_names}
    result = {
        "checkpoint_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "sequence_length": int(input_ids.numel()),
        "image_tokens": image_tokens,
        "control_loss": control["loss"],
        "dummy_loss": dummy["loss"],
        "loss_difference": abs(control["loss"] - dummy["loss"]),
        "language_parameter_count": len(language_differences),
        "language_max_difference": max(language_differences.values()),
        "language_all_finite": all(
            torch.isfinite(control["gradients"][name]).all().item()
            and torch.isfinite(dummy["gradients"][name]).all().item()
            for name in language_differences
        ),
        "vision_parameter_count": len(vision_differences),
        "vision_max_difference": max(vision_differences.values()),
        "vision_control_max": control["vision_max"],
        "vision_dummy_max": dummy["vision_max"],
        "language_per_parameter_max_difference": language_differences,
        "vision_per_parameter_max_difference": vision_differences,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
