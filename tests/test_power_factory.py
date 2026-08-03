"""Tests for PowerMonitor's total-power arithmetic.

Focused on how integrated versus discrete GPU power combines with the CPU
package reading, which is where double-counting would otherwise inflate every
energy and CO2 figure derived from a snapshot.
"""

import pytest

from warpt.backends.power.factory import PowerMonitor
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)


def _package(watts: float) -> DomainPower:
    """Build a RAPL package-domain reading (contains the iGPU on Intel parts)."""
    return DomainPower(
        domain=PowerDomain.PACKAGE,
        power_watts=watts,
        source=PowerSource.RAPL,
        metadata={"rapl_name": "package-0"},
    )


def _core(watts: float) -> DomainPower:
    """Build a RAPL core-domain reading (cores only — excludes the iGPU)."""
    return DomainPower(
        domain=PowerDomain.CORE,
        power_watts=watts,
        source=PowerSource.RAPL,
        metadata={"rapl_name": "core"},
    )


def _gpu(watts: float, *, integrated: bool, vendor: str = "intel") -> GPUPowerInfo:
    """Build a per-GPU reading tagged with vendor and integrated-ness."""
    return GPUPowerInfo(
        index=0,
        name="Test GPU",
        vendor=vendor,
        power_watts=watts,
        metadata={"integrated": integrated},
    )


@pytest.fixture
def monitor() -> PowerMonitor:
    """Return an uninitialized PowerMonitor (the arithmetic needs no backends)."""
    return PowerMonitor(include_process_attribution=False)


def test_discrete_gpu_adds_to_package(monitor):
    """A discrete GPU sits outside the CPU package, so it is added."""
    total = monitor._calculate_total_power(
        [_package(45.0)], [_gpu(120.0, integrated=False)]
    )
    assert total == pytest.approx(165.0)


def test_integrated_gpu_not_double_counted_with_package(monitor):
    """An integrated GPU is already inside the package reading."""
    total = monitor._calculate_total_power(
        [_package(45.0)], [_gpu(12.0, integrated=True)]
    )
    assert total == pytest.approx(45.0)


def test_integrated_gpu_counted_when_package_missing(monitor):
    """With no package reading, nothing else covers the iGPU — count it."""
    total = monitor._calculate_total_power([], [_gpu(12.0, integrated=True)])
    assert total == pytest.approx(12.0)


def test_integrated_gpu_counted_alongside_core_only(monitor):
    """The CORE domain excludes the iGPU (it lives in uncore), so add it."""
    total = monitor._calculate_total_power(
        [_core(30.0)], [_gpu(12.0, integrated=True)]
    )
    assert total == pytest.approx(42.0)


def test_mixed_integrated_and_discrete_with_package(monitor):
    """Only the discrete card is added on top of the package reading."""
    total = monitor._calculate_total_power(
        [_package(45.0)],
        [
            _gpu(12.0, integrated=True),
            _gpu(120.0, integrated=False, vendor="nvidia"),
        ],
    )
    assert total == pytest.approx(165.0)


def test_gpu_without_integrated_metadata_is_counted(monitor):
    """A backend that omits the flag (e.g. NVML) is treated as discrete."""
    gpu = GPUPowerInfo(index=0, name="NVIDIA GPU", vendor="nvidia", power_watts=200.0)
    assert monitor._calculate_total_power([_package(45.0)], [gpu]) == pytest.approx(
        245.0
    )


def test_no_measurements_returns_none(monitor):
    """With nothing measurable the total is None rather than 0.0."""
    assert monitor._calculate_total_power([], []) is None


def test_integrated_only_with_package_still_reports_package(monitor):
    """Skipping the iGPU must not discard the package measurement itself."""
    total = monitor._calculate_total_power(
        [_package(45.0)], [_gpu(0.0, integrated=True)]
    )
    assert total == pytest.approx(45.0)
