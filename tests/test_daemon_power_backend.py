"""Tests for DaemonPowerBackend (reads the out-of-process power-daemon)."""

from unittest.mock import MagicMock

from warpt.backends.power.daemon_source import DaemonPowerBackend
from warpt.models.power_models import PowerDomain, PowerSource

# Canned metrics response matching warpt-daemon/src/types.rs (MetricsResponse).
SAMPLE_METRICS = {
    "hostname": "node1",
    "timestamp": 1000,
    "reset_time": 42,
    "error_count": 0,
    "components": {
        "cpu": {
            "watts": 45.0,
            "joules_since_reset": 5000.0,
            "watt_hours_since_reset": 1.39,
            "reset_time": 42,
        },
        "ram": {
            "watts": 8.0,
            "joules_since_reset": 900.0,
            "watt_hours_since_reset": 0.25,
            "reset_time": 42,
        },
        "storage": {
            "watts": 3.0,
            "joules_since_reset": 300.0,
            "watt_hours_since_reset": 0.08,
            "reset_time": 42,
        },
        "accelerators": [
            {
                "id": 0,
                "type": "gpu",
                "model": "NVIDIA A100",
                "watts": 250.0,
                "joules_since_reset": 30000.0,
                "watt_hours_since_reset": 8.33,
                "reset_time": 42,
            },
            {
                "id": 1,
                "type": "gpu",
                "model": "NVIDIA A100",
                "watts": 240.0,
                "joules_since_reset": 29000.0,
                "watt_hours_since_reset": 8.06,
                "reset_time": 42,
            },
        ],
    },
    "total": {
        "watts": 546.0,
        "joules_since_reset": 65200.0,
        "watt_hours_since_reset": 18.1,
    },
}


def _backend_with_metrics(metrics=None):
    backend = DaemonPowerBackend()
    backend._client = MagicMock()
    backend._client.metrics.return_value = metrics or SAMPLE_METRICS
    backend._client.healthz.return_value = True
    return backend


class TestDaemonPowerBackendMapping:
    """Tests that daemon metrics map onto warpt's power models."""

    def test_get_source(self):
        """Source identifier is DAEMON."""
        assert _backend_with_metrics().get_source() == PowerSource.DAEMON

    def test_cpu_and_ram_domains(self):
        """CPU maps to PACKAGE and RAM maps to DRAM with energy counters."""
        readings = _backend_with_metrics().get_power_readings()
        by_domain = {d.domain: d for d in readings}

        cpu = by_domain[PowerDomain.PACKAGE]
        assert cpu.power_watts == 45.0
        assert cpu.energy_joules == 5000.0
        assert cpu.source == PowerSource.DAEMON

        ram = by_domain[PowerDomain.DRAM]
        assert ram.power_watts == 8.0
        assert ram.energy_joules == 900.0

    def test_accelerators_become_gpu_domains(self):
        """Each accelerator becomes a GPU DomainPower with its index/energy."""
        readings = _backend_with_metrics().get_power_readings()
        gpu_domains = [d for d in readings if d.domain == PowerDomain.GPU]
        assert len(gpu_domains) == 2
        assert {d.metadata["gpu_index"] for d in gpu_domains} == {0, 1}
        assert all(d.source == PowerSource.DAEMON for d in gpu_domains)
        assert {d.energy_joules for d in gpu_domains} == {30000.0, 29000.0}

    def test_storage_is_not_emitted_as_domain(self):
        """Storage is excluded from domains (no enum); covered by the total."""
        readings = _backend_with_metrics().get_power_readings()
        domains = {d.domain for d in readings}
        assert PowerDomain.PACKAGE in domains
        assert PowerDomain.DRAM in domains
        assert len(readings) == 4  # cpu + ram + 2 gpus, no storage

    def test_get_gpu_power_info(self):
        """Accelerators map to GPUPowerInfo with index/name/power."""
        gpus = _backend_with_metrics().get_gpu_power_info()
        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA A100"
        assert gpus[0].power_watts == 250.0
        assert gpus[0].metadata["integrated"] is False

    def test_get_total_watts_uses_daemon_total(self):
        """Total comes from the daemon (includes storage), not a domain re-sum."""
        assert _backend_with_metrics().get_total_watts() == 546.0


class TestDaemonPowerBackendAvailability:
    """Tests for daemon health detection."""

    def test_is_available_true_when_healthy(self):
        """Reports available when the health check passes."""
        assert _backend_with_metrics().is_available() is True

    def test_is_available_false_when_unreachable(self):
        """Reports unavailable (no exception) when nothing is listening."""
        backend = DaemonPowerBackend(base_url="http://127.0.0.1:1")
        assert backend.is_available() is False
