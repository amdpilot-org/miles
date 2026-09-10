from argparse import Namespace

import pytest

from miles.rollout import sglang_rollout
from miles.utils.types import AdapterRef, Sample


def make_args() -> Namespace:
    return Namespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        sglang_enable_deterministic_inference=False,
        custom_generate_function_path=None,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        num_layers=0,
        sglang_speculative_algorithm=None,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        ci_test=False,
    )


class DummyGenerateState:
    def __init__(self, _args: Namespace) -> None:
        self.processor = None

        class DummyTokenizer:
            def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
                return [len(prompt)]

        self.tokenizer = DummyTokenizer()


def make_output(text: str, token_id: int) -> dict:
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": "length", "length": 1},
            "prompt_tokens": 1,
            "output_token_logprobs": [(-0.5, token_id, None)],
            "completion_tokens": 1,
            "cached_tokens": 0,
            "weight_version": "default",
        },
    }


@pytest.mark.asyncio
async def test_generate_batch_preserves_sample_order(monkeypatch) -> None:
    args = make_args()
    samples = [Sample(index=index, group_index=0, prompt=f"prompt-{index}") for index in range(2)]
    requests = []

    async def fake_post(url, payload, **_kwargs):
        requests.append((url, payload))
        return [make_output("first", 101), make_output("second", 202)]

    monkeypatch.setattr(sglang_rollout, "GenerateState", DummyGenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    result = await sglang_rollout.generate_batch(args, samples, {"max_new_tokens": 1})

    assert result == samples
    assert requests[0][0] == "http://127.0.0.1:30000/generate"
    assert requests[0][1]["input_ids"] == [[8], [8]]
    assert [sample.tokens for sample in result] == [[8, 101], [8, 202]]
    assert [sample.response for sample in result] == ["first", "second"]
    assert [sample.rollout_log_probs for sample in result] == [[-0.5], [-0.5]]
    assert all(sample.status == Sample.Status.TRUNCATED for sample in result)


def test_group_batching_falls_back_for_unsafe_features() -> None:
    args = make_args()
    plain_sample = Sample(index=0, group_index=0, prompt="prompt")
    assert sglang_rollout._can_batch_generate_group(args, [plain_sample])

    unsafe_cases = {
        "custom function": lambda namespace, _sample: setattr(
            namespace, "custom_generate_function_path", "custom.generate"
        ),
        "deterministic inference": lambda namespace, _sample: setattr(
            namespace, "sglang_enable_deterministic_inference", True
        ),
        "routing-key policy": lambda namespace, _sample: setattr(
            namespace, "sglang_router_policy", "consistent_hashing"
        ),
        "lora rollout": lambda namespace, _sample: setattr(namespace, "lora_rank", 8),
        "sample adapter": lambda _namespace, sample: setattr(sample, "adapter", AdapterRef(name="adapter", slot=0)),
        "existing response": lambda _namespace, sample: setattr(sample, "response", "partial"),
        "sample custom function": lambda _namespace, sample: setattr(
            sample, "generate_function_path", "sample.generate"
        ),
        "multimodal input": lambda _namespace, sample: setattr(sample, "multimodal_inputs", {"images": [b"image"]}),
    }
    for case_name, make_unsafe in unsafe_cases.items():
        unsafe_args = make_args()
        unsafe_sample = Sample(index=0, group_index=0, prompt="prompt")
        make_unsafe(unsafe_args, unsafe_sample)
        assert not sglang_rollout._can_batch_generate_group(unsafe_args, [unsafe_sample]), case_name
