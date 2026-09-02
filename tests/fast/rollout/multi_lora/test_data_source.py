import logging
from argparse import Namespace

import pytest

from miles.rollout.multi_lora.data_source import MultiLoRAAsyncDataSource


class _FakeSource:
    def __init__(self) -> None:
        self.loaded: list[object] = []

    def load(self, rollout_id=None) -> None:
        self.loaded.append(rollout_id)


@pytest.fixture
def data_source() -> MultiLoRAAsyncDataSource:
    return MultiLoRAAsyncDataSource(Namespace())


class TestRestoringTheDatasetStateOfEveryAdapter:
    def test_a_run_whose_adapters_are_not_reconciled_yet_is_told_why_nothing_was_restored(self, data_source, caplog):
        """create_training_models loads the rollout executor before the adapters are registered."""
        with caplog.at_level(logging.WARNING):
            data_source.load()

        assert "no adapter has been reconciled into a data source yet" in caplog.text

    def test_it_does_not_report_a_configured_run_as_serving_no_adapter(self, data_source, caplog):
        """That is not what happened: the adapters are configured, and reconcile() has not run yet."""
        with caplog.at_level(logging.WARNING):
            data_source.load()

        assert "this run serves no adapter" not in caplog.text

    def test_it_restores_nothing_when_no_source_has_been_reconciled(self, data_source):
        """There is nothing to restore into, so the warning is the whole of what load() does."""
        assert data_source.load(rollout_id=3) is None

    def test_every_reconciled_adapter_has_its_own_dataset_state_restored(self, data_source, caplog):
        """Once reconcile() has filled the sources, this is an ordinary restore of every one of them."""
        data_source.sources = {"solver": _FakeSource(), "verifier": _FakeSource()}

        with caplog.at_level(logging.WARNING):
            data_source.load(rollout_id=3)

        assert [source.loaded for source in data_source.sources.values()] == [[3], [3]]
        assert caplog.text == ""
