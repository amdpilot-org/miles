# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from tests.e2e.ft.conftest_ft.fault_injection.core import (
    POLL_INTERVAL_SECONDS,
    QUIESCENT_POLLS_REQUIRED,
    InjectionAdmission,
    list_cells,
    run_fault_injection_loop,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import EventLog
from tests.e2e.ft.conftest_ft.fault_injection.views import STALE_STATUS_GRACE_SECONDS, compute_injection_times

from miles.utils.test_utils.polling_worker import PollingWorker

logger = logging.getLogger(__name__)

API_SERVER_PORT: int = 18080
# A pod deletion, the slowest form, cannot be cancelled and is two kubectl calls bounded at a minute.
STOP_AND_JOIN_TIMEOUT_SECONDS: float = 180.0


class FaultInjectorHandle:
    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        mean_interval_seconds_of_cell_type: dict[str, float],
        cell_fault_forms: CellFaultForms,
        get_virtual_cells: Callable[[], list[dict]] | None = None,
        injection_enabled: Callable[[], bool] | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
    ) -> None:
        self.event_log = EventLog()
        self.cell_fault_forms = cell_fault_forms
        self._base_url = base_url
        self._cell_types: set[str] = set(mean_interval_seconds_of_cell_type)
        self._get_virtual_cells: Callable[[], list[dict]] | None = get_virtual_cells
        self._admission = InjectionAdmission(is_open=injection_enabled)

        def inject_until_asked_to_stop(stop_event: threading.Event) -> None:
            run_fault_injection_loop(
                base_url=base_url,
                seed=seed,
                mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
                stop_event=stop_event,
                event_log=self.event_log,
                cell_fault_forms=cell_fault_forms,
                get_virtual_cells=get_virtual_cells,
                injection_admission=self._admission,
                poll_interval_seconds=poll_interval_seconds,
                quiescent_polls_required=quiescent_polls_required,
            )

        self._worker = PollingWorker(name="ft-random-fault-injector", run=inject_until_asked_to_stop)

    def start(self) -> None:
        self._worker.start()

    def observe_a_fault_free_tail(self) -> None:
        self._admission.close_after_any_fault_in_flight()
        deadline = time.monotonic() + STALE_STATUS_GRACE_SECONDS + STOP_AND_JOIN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not (times := compute_injection_times(self.event_log.events)):
                return
            remaining = STALE_STATUS_GRACE_SECONDS - (datetime.now(timezone.utc) - max(times)).total_seconds()
            if remaining <= 0:
                return
            logger.info(f"Observing {remaining:.0f}s more of a fault-free tail before reading the witnesses")
            time.sleep(min(remaining, POLL_INTERVAL_SECONDS))
        logger.warning(
            f"The last accepted injection is still younger than {STALE_STATUS_GRACE_SECONDS}s after waiting out "
            f"the tail, so the recovery witness may read no observation fresh enough to clear it"
        )

    def stop_and_join(self) -> None:
        self._worker.stop_and_join(timeout_seconds=STOP_AND_JOIN_TIMEOUT_SECONDS)
        self._worker.assert_not_running(
            message=(
                f"The fault injector was still mid-injection {STOP_AND_JOIN_TIMEOUT_SECONDS}s after being asked to "
                f"stop: it may still crash a cell nothing will heal, and reading its log would race it"
            )
        )
        self._observe_final_snapshot()

    def _observe_final_snapshot(self) -> None:
        cells = list_cells(base_url=self._base_url, cell_types=self._cell_types)
        if cells is None:
            return
        if self._get_virtual_cells is not None:
            cells.extend(self._get_virtual_cells())
        self.event_log.observe(cells)


def spawn_fault_injector(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
) -> FaultInjectorHandle:
    handle = FaultInjectorHandle(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
        get_virtual_cells=get_virtual_cells,
        injection_enabled=injection_enabled,
        poll_interval_seconds=poll_interval_seconds,
        quiescent_polls_required=quiescent_polls_required,
    )
    handle.start()
    return handle
