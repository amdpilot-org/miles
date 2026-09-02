from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.fsdp_utils import checkpoint as checkpoint_module


@pytest.fixture
def without_collectives(monkeypatch):
    monkeypatch.setattr(checkpoint_module.torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(checkpoint_module.dist, "barrier", lambda *a, **k: None)


def _actor(**args_overrides) -> SimpleNamespace:
    defaults = dict(start_rollout_id=None, no_load_rng=True)
    defaults.update(args_overrides)
    return SimpleNamespace(global_step=0, micro_step=0, restored_rollout_id=0, args=Namespace(**defaults))


def _finalize(actor: SimpleNamespace, *, metadata: dict, iteration: int | None) -> None:
    checkpoint_module.finalize_load(actor, {"rng": None, "metadata": metadata, "iteration": iteration})


class TestTheRolloutALoadRestoredTo:
    def test_the_metadata_position_wins_over_the_directory_iteration(self, without_collectives):
        """meta.json names the rollout the run was about to start, which the directory number only approximates."""
        actor = _actor()

        _finalize(actor, metadata={"global_step": 4, "next_rollout_id": 12}, iteration=9)

        assert (actor.restored_rollout_id, actor.args.start_rollout_id) == (12, 12)

    def test_a_metadata_without_a_rollout_position_falls_back_to_the_directory_iteration(self, without_collectives):
        """An older meta.json carries the step counters only, and answering 0 would hide the real position."""
        actor = _actor()

        _finalize(actor, metadata={"global_step": 4, "micro_step": 2}, iteration=9)

        assert (actor.restored_rollout_id, actor.args.start_rollout_id) == (9, 9)

    def test_an_unreadable_metadata_falls_back_to_the_directory_iteration(self, without_collectives):
        """A truncated meta.json is read as no metadata at all, and the checkpoint still says where it stands."""
        actor = _actor()

        _finalize(actor, metadata={}, iteration=9)

        assert (actor.restored_rollout_id, actor.args.start_rollout_id) == (9, 9)

    def test_a_requested_start_rollout_id_survives_the_directory_fallback(self, without_collectives):
        """--start-rollout-id is what the run asked for, and the restored position is reported separately."""
        actor = _actor(start_rollout_id=20)

        _finalize(actor, metadata={"global_step": 4}, iteration=9)

        assert (actor.restored_rollout_id, actor.args.start_rollout_id) == (9, 20)

    def test_a_checkpoint_that_says_nothing_leaves_the_actor_where_it_was(self, without_collectives):
        """Neither source knows a position here, so inventing one would move the run."""
        actor = _actor()

        _finalize(actor, metadata={}, iteration=None)

        assert (actor.restored_rollout_id, actor.args.start_rollout_id) == (0, None)
