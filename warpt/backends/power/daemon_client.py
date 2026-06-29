"""Dependency-free HTTP client for the out-of-process warpt power-daemon.

Talks to the Rust ``power-daemon`` REST service (the ``warpt-daemon`` repo) over
localhost. This is a trimmed, vendored copy of that repo's ``power_client.py`` —
only the pieces ``DaemonPowerBackend`` needs. urllib only; no third-party deps,
so warpt stays pure-Python.

The daemon URL defaults to ``http://127.0.0.1:8080`` and honors the
``POWER_DAEMON_URL`` environment variable — that env var is warpt's entire
configuration story for the daemon.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_API_PREFIX = "/public/v1/warpt"


class PowerClientError(Exception):
    """Raised when the power-daemon is unreachable or returns bad data."""


@dataclass(frozen=True)
class PowerReading:
    """Snapshot of total power and cumulative energy at a point in time."""

    timestamp: float
    watts: float
    joules_since_reset: float
    watt_hours_since_reset: float
    reset_time: float
    hostname: str


class PowerClient:
    """Minimal HTTP client for the power-daemon REST API."""

    def __init__(self, base_url: str | None = None, timeout: float = 2.0) -> None:
        url = base_url or os.environ.get("POWER_DAEMON_URL") or DEFAULT_BASE_URL
        self.base_url = url.rstrip("/")
        self._timeout = timeout

    def _get(self, path: str) -> dict:
        """GET a JSON endpoint under the daemon's versioned API prefix."""
        endpoint = f"{self.base_url}{_API_PREFIX}{path}"
        try:
            with urlopen(endpoint, timeout=self._timeout) as response:
                payload: dict = json.loads(response.read().decode("utf-8"))
                return payload
        except URLError as e:
            raise PowerClientError(
                f"Cannot reach power-daemon at {self.base_url}: {e}"
            ) from e
        except (json.JSONDecodeError, ValueError) as e:
            raise PowerClientError(f"Invalid JSON from power-daemon: {e}") from e

    def healthz(self) -> bool:
        """Return True if the daemon answers its health check, False otherwise."""
        try:
            data = self._get("/healthz")
        except PowerClientError:
            return False
        return str(data.get("status", "")).lower() in ("ok", "healthy", "up")

    def metrics(self) -> dict:
        """Return the raw metrics response (full components + total)."""
        return self._get("/power/metrics")

    def current(self) -> PowerReading:
        """Return a reading of total power and cumulative energy."""
        data = self.metrics()
        total = data.get("total", {})
        return PowerReading(
            timestamp=float(data.get("timestamp", time.time())),
            watts=float(total.get("watts", 0.0)),
            joules_since_reset=float(total.get("joules_since_reset", 0.0)),
            watt_hours_since_reset=float(total.get("watt_hours_since_reset", 0.0)),
            reset_time=float(data.get("reset_time", 0.0)),
            hostname=str(data.get("hostname", "unknown")),
        )
