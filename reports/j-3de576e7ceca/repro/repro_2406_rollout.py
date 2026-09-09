import logging

from miles.utils.processing_utils import extract_multimodal_train_inputs, load_processor, load_tokenizer

logger = logging.getLogger(__name__)

TOKENIZER = None
PROCESSOR = None


def _rendered_messages(sample):
    fixture = sample.metadata
    user_content = []
    if fixture["has_image"]:
        user_content.append({"type": "image", "image": fixture["image_path"]})
    user_content.append({"type": "text", "text": fixture["user_text"]})
    return (
        [{"role": "user", "content": user_content}],
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": fixture["assistant_text"]}]},
        ],
    )


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    global TOKENIZER, PROCESSOR
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    samples = data_buffer.get_samples(args.rollout_batch_size)
    processed = []
    for sample_group in samples:
        (sample,) = sample_group
        prompt_messages, full_messages = _rendered_messages(sample)
        prompt_text = TOKENIZER.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = TOKENIZER.apply_chat_template(full_messages, tokenize=False)
        images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
        prompt_output = PROCESSOR(
            text=[prompt_text], images=images, return_tensors="pt"
        )
        full_output = PROCESSOR(text=[full_text], images=images, return_tensors="pt")
        prompt_ids = prompt_output["input_ids"][0]
        full_ids = full_output["input_ids"][0]
        assert prompt_ids.tolist() == full_ids[: prompt_ids.numel()].tolist()

        sample.tokens = full_ids.tolist()
        sample.response_length = full_ids.numel() - prompt_ids.numel()
        sample.loss_mask = [1] * sample.response_length
        sample.reward = 0
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(full_output)
        processed.append(sample)
        logger.info(
            "deterministic_rollout rollout_id=%s fixture=%s image=%s tokens=%s response=%s",
            rollout_id,
            sample.metadata["fixture_id"],
            sample.metadata["has_image"],
            len(sample.tokens),
            sample.response_length,
        )
    return processed
