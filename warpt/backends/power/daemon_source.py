"""Power backend that reads from the out-of-process warpt power-daemon.

When the Rust ``power-daemon`` REST service is running on localhost, this backend
exposes its per-component readings and exact energy counters through the same
``PowerBackend`` interface used by the in-process native backends. The factory
auto-detects it; if the daemon is unreachable, warpt falls back to native reading.

Distinct from ``daemon.py`` (the in-process sampling daemon) — this file is only
the client-side backend for the external Rust service.
"""

from __future__ import annotations

from warpt.backends.power.base import PowerBackend
from warpt.backends.power.daemon_client import PowerClient, PowerClientError
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)


class DaemonPowerBackend(PowerBackend):
    """Reads power/energy from the warpt power-daemon over HTTP."""

    def __init__(self, base_url: str | None = None) -> None:
        self._client = PowerClient(base_url)

    def is_available(self) -> bool:
        """Return True if the power-daemon answers its health check."""
        return self._client.healthz()

    def get_source(self) -> PowerSource:
        """Return the power source identifier for this backend."""
        return PowerSource.DAEMON

    def _fetch(self) -> dict | None:
        """Read the daemon's metrics once; None if it is unreachable."""
        try:
            return self._client.metrics()
        except PowerClientError:
            return None

    def read_snapshot(
        self,
    ) -> tuple[list[DomainPower], list[GPUPowerInfo], float]:
        """Daemon fetch: (domain readings, GPU info, total watts)."""
        data = self._fetch()
        if data is None:
            return [], [], 0.0
        total = float(data.get("total", {}).get("watts", 0.0))
        return self._readings_from(data), self._gpus_from(data), total

    def get_power_readings(self) -> list[DomainPower]:
        """Map the daemon's per-component metrics into DomainPower readings."""
        data = self._fetch()
        return self._readings_from(data) if data is not None else []

    def _readings_from(self, data: dict) -> list[DomainPower]:
        """Map the daemon's components into per-domain power readings."""
        components = data.get("components", {})
        reset_time = data.get("reset_time")
        readings: list[DomainPower] = []

        cpu = components.get("cpu")
        if cpu is not None:
            readings.append(
                self._component_domain(PowerDomain.PACKAGE, cpu, reset_time)
            )

        ram = components.get("ram")
        if ram is not None:
            readings.append(self._component_domain(PowerDomain.DRAM, ram, reset_time))

        # Storage is included in the daemon's total but has no PowerDomain enum,
        # so it is intentionally not emitted as a domain (see read_snapshot).
        for accel in components.get("accelerators", []):
            readings.append(
                DomainPower(
                    domain=PowerDomain.GPU,
                    power_watts=float(accel.get("watts", 0.0)),
                    energy_joules=accel.get("joules_since_reset"),
                    source=PowerSource.DAEMON,
                    metadata={
                        "daemon": True,
                        "gpu_index": accel.get("id", 0),
                        "model": accel.get("model", ""),
                        "accel_type": accel.get("type", ""),
                        "reset_time": reset_time,
                    },
                )
            )
        return readings

    @staticmethod
    def _component_domain(
        domain: PowerDomain, comp: dict, reset_time: object
    ) -> DomainPower:
        return DomainPower(
            domain=domain,
            power_watts=float(comp.get("watts", 0.0)),
            energy_joules=comp.get("joules_since_reset"),
            source=PowerSource.DAEMON,
            metadata={"daemon": True, "reset_time": reset_time},
        )

    def _gpus_from(self, data: dict) -> list[GPUPowerInfo]:
        """Map the daemon's accelerators into per-GPU power info."""
        gpus: list[GPUPowerInfo] = []
        for accel in data.get("components", {}).get("accelerators", []):
            gpus.append(
                GPUPowerInfo(
                    index=int(accel.get("id", 0)),
                    name=str(accel.get("model", "")),
                    power_watts=float(accel.get("watts", 0.0)),
                    metadata={
                        "integrated": False,
                        "daemon": True,
                        "accel_type": accel.get("type", ""),
                        "energy_joules": accel.get("joules_since_reset"),
                    },
                )
            )
        return gpus
