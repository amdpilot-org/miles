from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput


@contextmanager
def _noop_timer(_name: str) -> Iterator[None]:
    yield


def test_fsdp_train_debug_rollout_only_returns_a_normal_output(monkeypatch):
    """A debug-rollout-only FSDP step trains nothing yet answers the driver with a NORMAL output."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(offload_train=False, debug_rollout_only=True)
    actor._heartbeat = Mock()
    actor._train_core = Mock()
    actor.wake_up = Mock()
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    monkeypatch.setattr(actor_module, "timer", _noop_timer)
    monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)

    result = actor.train(3, object())

    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_not_called()


def test_vision_collectives_run_dummy_forward_only_for_image_free_rank(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(vision_config={})
    actor._add_dummy_vision_inputs = Mock()
    group = object()
    monkeypatch.setattr(
        actor_module,
        "get_parallel_state",
        lambda: SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_group=lambda: group)),
    )
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(actor_module, "_current_cuda_device", lambda: torch.device("cpu"))

    def all_reduce(flag, op=None, group=None):
        assert op == actor_module.dist.ReduceOp.MAX
        assert group is group
        flag.fill_(1)

    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)

    batch = {"multimodal_train_inputs": {}}
    actor._synchronize_vision_collectives(batch)

    actor._add_dummy_vision_inputs.assert_called_once_with(batch)


def test_vision_collectives_skip_dummy_when_all_ranks_are_image_free(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(vision_config={})
    actor._add_dummy_vision_inputs = Mock()
    group = object()
    monkeypatch.setattr(
        actor_module,
        "get_parallel_state",
        lambda: SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_group=lambda: group)),
    )
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(actor_module, "_current_cuda_device", lambda: torch.device("cpu"))

    def all_reduce(flag, op=None, group=None):
        assert op == actor_module.dist.ReduceOp.MAX
        assert group is group
        flag.fill_(0)

    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)

    batch = {"multimodal_train_inputs": {}}
    actor._synchronize_vision_collectives(batch)

    actor._add_dummy_vision_inputs.assert_not_called()
    assert batch == {"multimodal_train_inputs": {}}


def test_vision_collectives_keep_image_present_rank_unchanged(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(vision_config={})
    actor._add_dummy_vision_inputs = Mock()
    group = object()
    monkeypatch.setattr(
        actor_module,
        "get_parallel_state",
        lambda: SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_group=lambda: group)),
    )
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(actor_module, "_current_cuda_device", lambda: torch.device("cpu"))

    def all_reduce(flag, op=None, group=None):
        assert op == actor_module.dist.ReduceOp.MAX
        assert group is group
        flag.fill_(1)

    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)

    batch = {"multimodal_train_inputs": {"pixel_values": torch.ones(1)}}
    actor._synchronize_vision_collectives(batch)

    actor._add_dummy_vision_inputs.assert_not_called()
    assert batch == {"multimodal_train_inputs": {"pixel_values": torch.ones(1)}}


def test_vision_collectives_skip_sync_for_single_rank_fsdp_group(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(vision_config={})
    actor._add_dummy_vision_inputs = Mock()
    group = object()
    monkeypatch.setattr(
        actor_module,
        "get_parallel_state",
        lambda: SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_group=lambda: group)),
    )
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda _group: 1)

    def all_reduce(_flag, op=None, group=None):
        raise AssertionError("Single-rank FSDP groups should not synchronize image presence")

    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)

    batch = {"multimodal_train_inputs": {}}
    actor._synchronize_vision_collectives(batch)

    actor._add_dummy_vision_inputs.assert_not_called()
    assert batch == {"multimodal_train_inputs": {}}


def test_vision_collectives_skip_sync_for_non_vlm_models(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace()
    actor._add_dummy_vision_inputs = Mock()

    def get_parallel_state():
        raise AssertionError("Non-VLM models should not query FSDP parallel state")

    monkeypatch.setattr(actor_module, "get_parallel_state", get_parallel_state)

    batch = {"multimodal_train_inputs": {}}
    actor._synchronize_vision_collectives(batch)

    actor._add_dummy_vision_inputs.assert_not_called()
    assert batch == {"multimodal_train_inputs": {}}


def test_dummy_vision_inputs_append_zero_loss_tokens(monkeypatch):
    class Processor:
        image_processor = SimpleNamespace(merge_size=2)

        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            return {
                "pixel_values": torch.ones(6, 4),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.int64),
            }

    monkeypatch.setattr(actor_module, "_current_cuda_device", lambda: torch.device("cpu"))

    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(
        vision_start_token_id=10,
        image_token_id=11,
        vision_end_token_id=12,
    )
    actor.processor = Processor()
    actor._dummy_vision_inputs = None
    actor._dummy_vision_token_ids = None
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4]]),
        "full_loss_masks": torch.ones(1, 4, dtype=torch.int64),
        "position_ids": torch.ones(1, 4, dtype=torch.int64),
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [8],
        "response_lengths": [4],
        "max_seq_lens": [8],
        "cu_seqlens": torch.tensor([0, 4, 8]),
        "max_seqlen": 8,
    }

    actor._add_dummy_vision_inputs(batch)

    assert batch["tokens"].tolist() == [[1, 2, 3, 4, 10, 11, 12]]
    assert batch["full_loss_masks"].tolist() == [[1, 1, 1, 1, 0, 0, 0]]
    assert batch["position_ids"] is None
    assert batch["total_lengths"] == [11]
    assert batch["max_seq_lens"] == [11]
    assert batch["cu_seqlens"].tolist() == [0, 7, 11]
    assert batch["max_seqlen"] == 11
    assert batch["multimodal_train_inputs"]["pixel_values"].shape == (4, 4)
    assert batch["multimodal_train_inputs"]["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert batch["multimodal_train_inputs"]["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 0, 1, 0]]


def test_dummy_vision_inputs_allow_missing_thd_max_seq_lens(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._get_dummy_vision_token_ids = lambda: (10, 11, 12)
    actor._get_dummy_vision_inputs = lambda: {
        "pixel_values": torch.ones(4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.int64),
    }
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4]]),
        "full_loss_masks": torch.ones(1, 4, dtype=torch.int64),
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [8],
        "response_lengths": [4],
        "max_seq_lens": [None],
    }

    actor._add_dummy_vision_inputs(batch)

    assert batch["max_seq_lens"] == [None]
    assert batch["tokens"].shape == (1, 7)


def test_dummy_vision_inputs_support_bshd_microbatch_size_two(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._get_dummy_vision_token_ids = lambda: (10, 11, 12)
    actor._get_dummy_vision_inputs = lambda: {
        "pixel_values": torch.ones(4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.int64),
    }
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "full_loss_masks": torch.ones(2, 4, dtype=torch.int64),
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])],
        "total_lengths": [8, 8],
        "response_lengths": [4, 4],
        "max_seq_lens": [8, 8],
    }

    actor._add_dummy_vision_inputs(batch)

    assert batch["tokens"].shape == (2, 7)
    assert batch["full_loss_masks"].tolist() == [[1, 1, 1, 1, 0, 0, 0]] * 2
    assert [tokens.tolist() for tokens in batch["unconcat_tokens"]] == [
        [1, 2, 3, 4, 10, 11, 12],
        [5, 6, 7, 8, 10, 11, 12],
    ]
    assert batch["total_lengths"] == [11, 11]
    assert batch["max_seq_lens"] == [11, 11]
    assert batch["multimodal_train_inputs"]["pixel_values"].shape == (8, 4)
    assert batch["multimodal_train_inputs"]["image_grid_thw"].shape == (2, 3)
    assert batch["multimodal_train_inputs"]["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 0, 1, 0]] * 2


def test_dummy_vision_inputs_support_bshd_unequal_sample_lengths(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._get_dummy_vision_token_ids = lambda: (10, 11, 12)
    actor._get_dummy_vision_inputs = lambda: {
        "pixel_values": torch.ones(4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.int64),
    }
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
        "full_loss_masks": torch.ones(2, 5, dtype=torch.int64),
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])],
        "total_lengths": [8, 7],
        "response_lengths": [4, 3],
        "max_seq_lens": [8, 7],
    }

    actor._add_dummy_vision_inputs(batch)

    assert batch["tokens"].shape == (2, 8)
    assert batch["full_loss_masks"].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]] * 2
    assert [tokens.tolist() for tokens in batch["unconcat_tokens"]] == [
        [1, 2, 3, 4, 10, 11, 12],
        [5, 6, 7, 10, 11, 12],
    ]
    assert batch["total_lengths"] == [11, 10]
    assert batch["max_seq_lens"] == [11, 10]
    assert batch["multimodal_train_inputs"]["pixel_values"].shape == (8, 4)
    assert batch["multimodal_train_inputs"]["image_grid_thw"].shape == (2, 3)
    assert batch["multimodal_train_inputs"]["mm_token_type_ids"].tolist() == [
        [0, 0, 0, 0, 0, 0, 1, 0]
    ] * 2


def test_dummy_vision_inputs_support_thd_microbatch_size_two(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._get_dummy_vision_token_ids = lambda: (10, 11, 12)
    actor._get_dummy_vision_inputs = lambda: {
        "pixel_values": torch.ones(4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.int64),
    }
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        "full_loss_masks": torch.ones(1, 7, dtype=torch.int64),
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])],
        "total_lengths": [8, 7],
        "response_lengths": [4, 3],
        "max_seq_lens": [8, 7],
        "cu_seqlens": torch.tensor([0, 4, 7]),
        "max_seqlen": 4,
    }

    actor._add_dummy_vision_inputs(batch)

    assert batch["tokens"].shape == (1, 13)
    assert batch["full_loss_masks"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    assert [tokens.tolist() for tokens in batch["unconcat_tokens"]] == [
        [1, 2, 3, 4, 10, 11, 12],
        [5, 6, 7, 10, 11, 12],
    ]
    assert batch["total_lengths"] == [11, 10]
    assert batch["max_seq_lens"] == [11, 10]
    assert batch["cu_seqlens"].tolist() == [0, 7, 13]
    assert batch["max_seqlen"] == 7
    assert batch["multimodal_train_inputs"]["pixel_values"].shape == (8, 4)
    assert batch["multimodal_train_inputs"]["image_grid_thw"].shape == (2, 3)
    assert batch["multimodal_train_inputs"]["mm_token_type_ids"].tolist() == [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]


def test_dummy_vision_inputs_support_thd_padding_boundary(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._get_dummy_vision_token_ids = lambda: (10, 11, 12)
    actor._get_dummy_vision_inputs = lambda: {
        "pixel_values": torch.ones(4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.int64),
    }
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0]]),
        "full_loss_masks": torch.ones(1, 9, dtype=torch.int64),
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])],
        "total_lengths": [8, 7],
        "response_lengths": [4, 3],
        "max_seq_lens": [8, 7],
        "cu_seqlens": torch.tensor([0, 4, 7, 9]),
        "max_seqlen": 4,
    }

    actor._add_dummy_vision_inputs(batch)

    assert batch["tokens"].shape == (1, 15)
    assert batch["full_loss_masks"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    assert [tokens.tolist() for tokens in batch["unconcat_tokens"]] == [
        [1, 2, 3, 4, 10, 11, 12],
        [5, 6, 7, 10, 11, 12],
    ]
    assert batch["total_lengths"] == [11, 10]
    assert batch["max_seq_lens"] == [11, 10]
    assert batch["cu_seqlens"].tolist() == [0, 7, 13, 15]
    assert batch["max_seqlen"] == 7
    assert batch["multimodal_train_inputs"]["mm_token_type_ids"].tolist() == [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ]


def test_unsupported_mixed_vlm_fails_before_model_forward():
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.processor = None
    actor._dummy_vision_inputs = None
    actor._dummy_vision_token_ids = None
    actor.hf_config = SimpleNamespace(
        vision_start_token_id=10,
        image_token_id=11,
        vision_end_token_id=12,
    )

    try:
        actor._get_dummy_vision_inputs()
    except RuntimeError as error:
        assert "requires a processor" in str(error)
    else:
        raise AssertionError("Expected an explicit RuntimeError")
