"""Tests for power monitoring Pydantic/dataclass models."""

from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSnapshot,
    PowerSource,
    ProcessPower,
)


def test_domain_power_to_dict():
    """Test DomainPower dictionary conversion."""
    dp = DomainPower(
        domain=PowerDomain.PACKAGE,
        power_watts=45.5,
        energy_joules=1000.0,
        source=PowerSource.RAPL,
    )
    data = dp.to_dict()
    assert data["domain"] == "package"
    assert data["power_watts"] == 45.5
    assert data["energy_joules"] == 1000.0
    assert data["source"] == "rapl"


def test_process_power_calculation():
    """Test ProcessPower total power calculation and rounding."""
    pp = ProcessPower(
        pid=1234,
        name="test_proc",
        cpu_power_watts=10.1234,
        gpu_power_watts=5.5678,
        cpu_percent=50.0,
    )
    # total_power_watts should be calculated in __post_init__
    assert pp.total_power_watts == 10.1234 + 5.5678

    data = pp.to_dict()
    assert data["cpu_power_watts"] == 10.123
    assert data["gpu_power_watts"] == 5.568
    assert data["total_power_watts"] == 15.691


def test_power_snapshot_helpers():
    """Test PowerSnapshot helper methods."""
    snapshot = PowerSnapshot(
        timestamp=1600000000.0,
        domains=[
            DomainPower(PowerDomain.PACKAGE, 50.0),
            DomainPower(PowerDomain.DRAM, 10.0),
        ],
        gpus=[
            GPUPowerInfo(index=0, name="GPU1", power_watts=100.0),
            GPUPowerInfo(index=1, name="GPU2", power_watts=150.0),
        ],
    )

    assert snapshot.get_domain_power(PowerDomain.PACKAGE) == 50.0
    assert snapshot.get_domain_power(PowerDomain.CORE) is None
    assert snapshot.get_cpu_power() == 50.0  # Prefers PACKAGE
    assert snapshot.get_gpu_power() == 250.0
