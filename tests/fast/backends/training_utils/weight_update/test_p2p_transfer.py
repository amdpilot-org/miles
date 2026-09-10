from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import aiter.jit.core as aiter_core

aiter_core.AITER_CONFIGS.get_config_file = lambda _env_name, default_file, _tuned_file_name: default_file

from miles.backends.training_utils.weight_update.protocols.p2p import (
    P2P_TRAINER_RECREATE_MESSAGE,
    UpdateWeightP2P,
)
from miles.backends.training_utils.weight_update.protocols import p2p as p2p_module
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import (
    P2P_TRANSFER_WAIT_FAILURE_MESSAGE,
    P2PTransferManager,
)


def _failed_future() -> Future:
    future = Future()
    future.set_exception(RuntimeError("synthetic transfer failure"))
    return future


def _make_protocol(transfer_manager: P2PTransferManager) -> UpdateWeightP2P:
    protocol = UpdateWeightP2P.__new__(UpdateWeightP2P)
    protocol.is_sender = True
    protocol.transfer_manager = transfer_manager
    protocol._tensor_update_pending = {}
    protocol._staged_tensors = {}
    return protocol


def test_wait_transfers_raises_and_poisons_the_manager() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=1)
    manager.transfer_futures.append(_failed_future())

    with pytest.raises(RuntimeError, match=P2P_TRANSFER_WAIT_FAILURE_MESSAGE) as error:
        manager.wait_transfers()

    assert str(error.value.__cause__) == "synthetic transfer failure"
    assert manager.failed
    assert not manager.transfer_futures


def test_timeout_retains_unresolved_transfer_ownership() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=0)
    unresolved_future = Future()
    manager.transfer_futures.append(unresolved_future)

    with pytest.raises(RuntimeError, match=P2P_TRANSFER_WAIT_FAILURE_MESSAGE):
        manager.wait_transfers()

    assert manager.failed
    assert manager.transfer_futures == [unresolved_future]


def test_immediate_transfer_batch_uses_the_configured_timeout() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=0)
    unresolved_future = Future()
    manager.transfer_futures.append(unresolved_future)

    with pytest.raises(RuntimeError, match=P2P_TRANSFER_WAIT_FAILURE_MESSAGE):
        manager.wait_transfer_batch([unresolved_future])

    assert manager.failed
    assert manager.transfer_futures == [unresolved_future]


def test_submit_refuses_after_failure() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=1)
    failure = RuntimeError("synthetic transfer failure")
    manager.record_failure("synthetic failure", failure)

    with pytest.raises(RuntimeError, match=P2P_TRANSFER_WAIT_FAILURE_MESSAGE) as error:
        manager.submit(lambda: None)

    assert error.value.__cause__ is failure


def test_after_base_weights_reports_remote_rank_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=1)
    protocol = _make_protocol(manager)
    collective = MagicMock(return_value=False)
    gloo_group = MagicMock()
    monkeypatch.setattr(p2p_module, "collective_bool_and", collective)
    monkeypatch.setattr(p2p_module, "get_gloo_group", lambda: gloo_group)

    with pytest.raises(RuntimeError, match="P2P weight transfer failed on at least one trainer rank"):
        protocol.after_base_weights()

    assert manager.failed
    collective.assert_called_once_with(value=True, group=gloo_group)


def test_send_bucket_stops_loading_after_an_immediate_failure() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=1)
    protocol = _make_protocol(manager)
    protocol._get_transfer_ready_params = lambda _bucket: (["weight"], [("weight", MagicMock())])
    first_replica = MagicMock()
    second_replica = MagicMock()
    protocol._transfer_engine_meta_list = [
        (first_replica, [SimpleNamespace(session_id="first")]),
        (second_replica, [SimpleNamespace(session_id="second")]),
    ]

    def fail_first_transfer(_remote_session, _names) -> None:
        raise RuntimeError("synthetic transfer failure")

    protocol._do_p2p_write_one_session = fail_first_transfer
    bucket = [("weight", MagicMock())]
    protocol.send_bucket(bucket)

    first_replica.load_weights.assert_called_once()
    second_replica.load_weights.assert_not_called()
    assert manager.failed
    assert bucket == []


def test_connect_refuses_to_reuse_a_poisoned_updater() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=1)
    manager.record_failure("synthetic failure", RuntimeError("synthetic transfer failure"))
    protocol = _make_protocol(manager)

    with pytest.raises(RuntimeError, match=P2P_TRAINER_RECREATE_MESSAGE):
        protocol.connect(MagicMock(), None, None, MagicMock(), MagicMock(), "all")


def test_successful_transfers_clear_completed_futures() -> None:
    manager = P2PTransferManager(num_workers=1, transfer_timeout=1)
    manager.submit(lambda: None)

    manager.wait_transfers()

    assert not manager.failed
    assert manager.transfer_futures == []


def test_transfer_failure_skips_weight_finalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "miles.backends.training_utils.weight_update.updater.dist.get_rank", lambda: 0
    )
    monkeypatch.setattr(
        "miles.backends.training_utils.weight_update.updater.dist.barrier", MagicMock()
    )
    monkeypatch.setattr(
        "miles.backends.training_utils.weight_update.updater.get_gloo_group", MagicMock()
    )
    monkeypatch.setattr(
        "miles.backends.training_utils.weight_update.updater.timer",
        MagicMock(return_value=nullcontext()),
    )
    monkeypatch.setattr(
        "miles.backends.training_utils.weight_update.updater.tqdm", MagicMock()
    )
    finalization_calls = {
        "end_weight_update": MagicMock(),
        "set_weight_version": MagicMock(),
        "resume_engines": MagicMock(),
    }
    for name, call in finalization_calls.items():
        monkeypatch.setattr(
            f"miles.backends.training_utils.weight_update.updater.{name}", call
        )

    protocol = _make_protocol(P2PTransferManager(num_workers=1, transfer_timeout=1))
    protocol.begin_sync = MagicMock(return_value=True)
    protocol.send_bucket = MagicMock()
    protocol.after_base_weights = MagicMock(
        side_effect=RuntimeError("P2P weight transfer failed on at least one trainer rank")
    )
    protocol.finalize = MagicMock()
    protocol.after_engines_resumed = MagicMock()
    protocol.use_weight_update_session = True
    protocol.rollout_engines = []
    protocol.group_name = "test-p2p"

    updater = WeightUpdater.__new__(WeightUpdater)
    updater.args = SimpleNamespace(pause_generation_mode="in_place")
    updater.protocol = protocol
    updater.weight_version = 0
    updater.is_lora = False
    updater._hf_weight_iterator = SimpleNamespace(
        placement=protocol.required_placement,
        weight_update_selector="all",
        iter_hf_weights=lambda *_args, **_kwargs: iter([[("weight", MagicMock())]]),
    )
    updater.weights_getter = lambda: {}
    updater._lora_sync_config = None
    updater._registered_adapters = set()
    updater.multi_lora_adapters = None

    with pytest.raises(RuntimeError, match="P2P weight transfer failed"):
        updater.update_weights()

    protocol.finalize.assert_not_called()
    for call in finalization_calls.values():
        call.assert_not_called()
