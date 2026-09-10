from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock


class _RecordingCell:
    def __init__(self, *, cell_id: str, needs_offload: bool, addressable: bool = True):
        self.meta = SimpleNamespace(needs_offload=needs_offload, cell_id=cell_id, num_gpus_per_engine=1, gpu_offset=0)
        self.is_pending_weights_or_serving = addressable
        self.calls: list[tuple[str, dict]] = []
        self.reload_result = {"success": True, "message": "Success"}

    async def offload(self, tags):
        self.calls.append(("offload", dict(tags=tags)))
        return f"offloaded-{self.meta.cell_id}"

    async def onload(self, tags):
        self.calls.append(("onload", dict(tags=tags)))
        return f"onloaded-{self.meta.cell_id}"

    async def reload_weights(self, model_path):
        self.calls.append(("reload_weights", dict(model_path=model_path)))
        return self.reload_result

    async def check_weights(self, action, allow_quant_error, selector, skip_list):
        self.calls.append(
            (
                "check_weights",
                dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list),
            )
        )
        return f"checked-{self.meta.cell_id}"


def _make_server(cells: list[_RecordingCell], **overrides) -> RolloutServer:
    return RolloutServer(
        server_cells={cell.meta.cell_id: cell for cell in cells},
        args=SimpleNamespace(colocate=True),
        context_lock=ContextLock("InferenceController"),
        **overrides,
    )


class TestMemoryFanOut:
    async def test_only_the_cells_sharing_gpus_with_the_trainer_give_memory_back(self):
        """A resident engine told to release would drop the weights nobody is going to reload."""
        colocated = _RecordingCell(cell_id="a", needs_offload=True)
        resident = _RecordingCell(cell_id="b", needs_offload=False)
        srv = _make_server([colocated, resident])

        async with srv.context_lock:
            await srv.offload()

        assert [name for name, _ in colocated.calls] == ["offload"]
        assert resident.calls == []

    @pytest.mark.parametrize("op", ["offload", "onload"])
    async def test_the_requested_tags_reach_every_cell_unchanged(self, op):
        """Resuming the wrong tag set brings back a different slice of the engine's memory."""
        cells = [_RecordingCell(cell_id=str(i), needs_offload=True) for i in range(3)]
        srv = _make_server(cells)

        async with srv.context_lock:
            results = await getattr(srv, op)(tags=["weights"])

        assert results == [f"{op}ed-{i}" for i in range(3)]
        assert all(cell.calls == [(op, dict(tags=["weights"]))] for cell in cells)

    async def test_a_cell_without_an_address_yet_is_left_alone(self):
        """Reconcile can add a gated cell mid-window; dialling it asserts inside the cell and
        takes down a weight update the cell was never part of."""
        gated = _RecordingCell(cell_id="gated", needs_offload=True, addressable=False)
        serving = _RecordingCell(cell_id="serving", needs_offload=True)
        srv = _make_server([gated, serving])

        async with srv.context_lock:
            await srv.offload()

        assert gated.calls == []
        assert [name for name, _ in serving.calls] == ["offload"]

    async def test_a_frozen_offloaded_cell_reloads_after_weight_resume(self):
        cell = _RecordingCell(cell_id="teacher", needs_offload=True)
        srv = _make_server([cell], model_name="teacher", model_path="/teacher", update_weights=False)

        async with srv.context_lock:
            await srv.onload(tags=["weights"])

        assert cell.calls == [
            ("onload", dict(tags=["weights"])),
            ("reload_weights", dict(model_path="/teacher")),
        ]

    async def test_an_updatable_cell_waits_for_actor_weight_sync(self):
        cell = _RecordingCell(cell_id="actor", needs_offload=True)
        srv = _make_server([cell], model_name="actor", model_path="/actor", update_weights=True)

        async with srv.context_lock:
            await srv.onload(tags=["weights"])

        assert cell.calls == [("onload", dict(tags=["weights"]))]

    async def test_a_kv_only_resume_does_not_reload_frozen_weights(self):
        cell = _RecordingCell(cell_id="teacher", needs_offload=True)
        srv = _make_server([cell], model_name="teacher", model_path="/teacher", update_weights=False)

        async with srv.context_lock:
            await srv.onload(tags=["kv_cache"])

        assert cell.calls == [("onload", dict(tags=["kv_cache"]))]

    async def test_a_frozen_offloaded_cell_without_a_model_path_fails_closed(self):
        cell = _RecordingCell(cell_id="teacher", needs_offload=True)
        srv = _make_server([cell], model_name="teacher", model_path=None, update_weights=False)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="no model_path"):
                await srv.onload(tags=["weights"])

        assert cell.calls == [("onload", dict(tags=["weights"]))]

    async def test_a_failed_frozen_weight_reload_fails_closed(self):
        cell = _RecordingCell(cell_id="teacher", needs_offload=True)
        cell.reload_result = {"success": False, "message": "checkpoint unavailable"}
        srv = _make_server([cell], model_name="teacher", model_path="/teacher", update_weights=False)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="checkpoint unavailable"):
                await srv.onload(tags=["weights"])

        assert cell.calls[-1] == ("reload_weights", dict(model_path="/teacher"))


class TestCheckWeightsFanOut:
    async def test_every_addressable_cell_is_checked_with_the_same_arguments(self):
        """Narrowing this to one cell would verify one engine of N and call them all equal."""
        cells = [_RecordingCell(cell_id=str(i), needs_offload=False) for i in range(3)]
        srv = _make_server(cells)

        async with srv.context_lock:
            results = await srv.check_weights(
                action="snapshot", allow_quant_error=True, selector="lora", skip_list=["x"]
            )

        assert results == [f"checked-{i}" for i in range(3)]
        assert all(
            cell.calls
            == [
                (
                    "check_weights",
                    dict(action="snapshot", allow_quant_error=True, selector="lora", skip_list=["x"]),
                )
            ]
            for cell in cells
        )

    async def test_a_cell_without_an_address_yet_is_not_checked(self):
        """The check runs during the weight update window, which a gated cell has not entered."""
        gated = _RecordingCell(cell_id="gated", needs_offload=False, addressable=False)
        srv = _make_server([gated])

        async with srv.context_lock:
            assert await srv.check_weights(action="snapshot") == []

        assert gated.calls == []
