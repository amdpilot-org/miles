from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.utils import device_flops, train_metric_utils
from miles.utils.device_flops import local_peak_bf16_tflops, peak_bf16_tflops
from miles.utils.misc import SingletonMeta
from miles.utils.timer import Timer
from miles.utils.train_metric_utils import log_perf_data_raw

# These are the AMD Instinct parts the README lists as supported via ROCm
# ("AMD MI300X, MI325, MI350, and MI355X via ROCm"). ``device_flops`` ships no
# AMD peak-BF16 entries yet, so every one resolves to ``None`` and the repo
# cannot turn a measured TFLOP/s into a utilization figure on these GPUs. See
# reports/j-ce2409d963cf/REPORT.md for an on-metal MI355X GEMM measurement that
# reaches this exact code path. When AMD entries are added to the table, flip
# these expectations to the mapped peak values.
SUPPORTED_AMD_DEVICES = [
    "AMD Instinct MI300X",
    "AMD Instinct MI325",
    "AMD Instinct MI350",
    "AMD Instinct MI355X",
]


@pytest.mark.parametrize("device_name", SUPPORTED_AMD_DEVICES)
def test_supported_amd_devices_are_currently_unmapped(device_name):
    assert peak_bf16_tflops(device_name) is None


def test_local_lookup_returns_none_for_mi355x(monkeypatch):
    monkeypatch.setattr(device_flops, "_current_device_name", lambda: "AMD Instinct MI355X")
    assert local_peak_bf16_tflops() is None


def test_mi355x_silently_omits_mfu_through_the_real_table(monkeypatch):
    """The gap has a user-facing consequence: on an unmapped AMD GPU the peak
    resolves to ``None``, so ``log_perf_data_raw`` omits the MFU metric
    entirely. This wires the characterization to the real code path a Miles
    run hits, not just the lookup function in isolation."""
    calls: list[dict] = []
    monkeypatch.setattr(train_metric_utils.tracking, "log", lambda args, payload, **kw: calls.append(payload))

    # Point the real table lookup at the MI355X (which is unmapped -> None).
    monkeypatch.setattr(device_flops, "_current_device_name", lambda: "AMD Instinct MI355X")

    SingletonMeta._instances.pop(Timer, None)
    timer = Timer()
    timer.seq_lens = [1024, 2048]
    timer.timers = {"actor_train": 2.0}
    try:
        args = SimpleNamespace(
            wandb_always_use_train_step=False,
            mfu_peak_tflops=None,  # no manual override -> falls back to local_peak_bf16_tflops
        )
        log_perf_data_raw(
            rollout_id=0,
            args=args,
            is_primary_rank=True,
            compute_total_fwd_flops=lambda seq_lens: 60.0,
        )
    finally:
        SingletonMeta._instances.pop(Timer, None)

    [payload] = calls
    # Throughput is still logged...
    assert "perf/actor_train_tflops" in payload
    # ...but MFU and its denominator are absent because the peak is None.
    assert "perf/actor_train_mfu" not in payload
    assert "perf/mfu_peak_tflops" not in payload
