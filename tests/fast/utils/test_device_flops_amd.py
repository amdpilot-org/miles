from __future__ import annotations

import pytest

from miles.utils import device_flops
from miles.utils.device_flops import local_peak_bf16_tflops, peak_bf16_tflops

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
