"""VitalsNurse — continuous hardware observation via subprocess polling."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

from warpt.daemon.casefile import CaseFile

DEFAULT_GPU_THRESHOLDS: dict[str, dict[str, float]] = {
    "utilization_percent": {"value": 80.0, "sustained_seconds": 15.0},
    "memory_utilization_percent": {"value": 85.0, "sustained_seconds": 15.0},
    "temperature_c": {"value": 80.0, "sustained_seconds": 15.0},
}


class VitalsNurse:
    """Polls ``warpt monitor --no-tui --json`` and manages heartbeat persistence.

    Parameters
    ----------
    casefile
        CaseFile instance for database writes.
    buffer_size
        Maximum number of snapshots to retain in the ring buffer.
    heartbeat_interval
        Seconds between heartbeat writes to DuckDB.
    poll_interval
        Seconds between subprocess poll reads.
    gpu_thresholds
        Override default GPU threshold configuration.
    """

    def __init__(
        self,
        casefile: CaseFile,
        buffer_size: int = 60,
        heartbeat_interval: float = 30.0,
        poll_interval: float = 5.0,
        gpu_thresholds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._casefile = casefile
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._heartbeat_interval = heartbeat_interval
        self._poll_interval = poll_interval
        self._known_gpus: set[str] = set()
        self._on_threshold_breach: Callable[[dict[str, Any]], None] | None = None
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_heartbeat: float = 0.0
        self._gpu_thresholds = gpu_thresholds or DEFAULT_GPU_THRESHOLDS
        # Tracks when each (metric, gpu_guid) breach started: monotonic time
        self._breach_start: dict[tuple[str, str], float] = {}
        # Tracks which (metric, gpu_guid) breaches have already fired
        self._breach_fired: set[tuple[str, str]] = set()

    def feed_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Ingest a parsed JSON snapshot into the ring buffer.

        Also registers any new GPUs and writes heartbeats on schedule.

        Parameters
        ----------
        snapshot
            Parsed dict from ``ResourceSnapshot.to_dict()`` JSON output.
        """
        self._buffer.append(snapshot)
        self._register_gpus(snapshot)
        self._maybe_write_heartbeat(snapshot)
        self._check_thresholds(snapshot)

    def get_buffer(self) -> list[dict[str, Any]]:
        """Return a copy of the current ring buffer contents.

        Returns
        -------
            List of snapshots, oldest first.
        """
        return list(self._buffer)

    def get_latest(self) -> dict[str, Any] | None:
        """Return the most recent snapshot, or None if buffer is empty."""
        return self._buffer[-1] if self._buffer else None

    def set_on_threshold_breach(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a callback for threshold breach events.

        Parameters
        ----------
        callback
            Called with event data when a threshold breach is detected.
        """
        self._on_threshold_breach = callback

    def start(self) -> None:
        """Start the subprocess and polling thread."""
        self._stop_event.clear()
        self._process = subprocess.Popen(
            ["warpt", "monitor", "--no-tui", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and terminate the subprocess."""
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)
        self._process = None
        self._thread = None

    def _poll_loop(self) -> None:
        """Read JSON lines from the subprocess stdout."""
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
                self.feed_snapshot(snapshot)
            except json.JSONDecodeError:
                continue

    def _check_thresholds(self, snapshot: dict[str, Any]) -> None:
        """Evaluate GPU metrics against configured thresholds."""
        now = time.monotonic()
        for gpu in snapshot.get("gpu_usage", []):
            guid = gpu.get("guid")
            if not guid:
                continue
            for metric, rule in self._gpu_thresholds.items():
                current_value = gpu.get(metric)
                if current_value is None:
                    continue
                key = (metric, guid)
                if current_value > rule["value"]:
                    # Metric is breaching
                    if key not in self._breach_start:
                        self._breach_start[key] = now
                    elapsed = now - self._breach_start[key]
                    if (
                        elapsed >= rule["sustained_seconds"]
                        and key not in self._breach_fired
                    ):
                        self._breach_fired.add(key)
                        event = {
                            "metric": metric,
                            "value": current_value,
                            "threshold": rule["value"],
                            "gpu_guid": guid,
                            "sustained_seconds": elapsed,
                        }
                        self._write_vitals(snapshot, "threshold_breach")
                        if self._on_threshold_breach:
                            self._on_threshold_breach(event)
                else:
                    # Metric dropped below threshold — reset
                    self._breach_start.pop(key, None)
                    self._breach_fired.discard(key)

    def _register_gpus(self, snapshot: dict[str, Any]) -> None:
        """Register any new GPUs found in the snapshot."""
        for gpu in snapshot.get("gpu_usage", []):
            guid = gpu.get("guid")
            if not guid or guid in self._known_gpus:
                continue
            self._known_gpus.add(guid)
            now = datetime.now().isoformat()
            self._casefile.execute(
                """
                INSERT INTO gpu_profiles (gpu_guid, model, vendor, last_seen_at)
                VALUES (?, ?, 'nvidia', ?::TIMESTAMP)
                ON CONFLICT (gpu_guid) DO UPDATE
                SET last_seen_at = ?::TIMESTAMP
                """,
                [guid, gpu.get("model", "Unknown"), now, now],
            )

    def _maybe_write_heartbeat(self, snapshot: dict[str, Any]) -> None:
        """Write a heartbeat row if enough time has elapsed."""
        now = time.monotonic()
        if now - self._last_heartbeat < self._heartbeat_interval:
            return
        self._last_heartbeat = now
        self._write_vitals(snapshot, "heartbeat")

    def _write_vitals(self, snapshot: dict[str, Any], collection_type: str) -> None:
        """Write a snapshot row to the vitals table."""
        gpu_structs = []
        for gpu in snapshot.get("gpu_usage", []):
            gpu_structs.append(
                {
                    "gpu_guid": gpu.get("guid"),
                    "gpu_index": gpu.get("index"),
                    "utilization_pct": gpu.get("utilization_percent"),
                    "mem_utilization_pct": gpu.get("memory_utilization_percent"),
                    "power_w": gpu.get("power_watts"),
                    "temperature_c": None,
                    "mem_used_bytes": None,
                    "mem_total_bytes": None,
                    "throttle_reasons": None,
                }
            )

        self._casefile.execute(
            """
            INSERT INTO vitals (
                ts, cpu_utilization_pct, cpu_power_w,
                mem_total_bytes, mem_available_bytes, mem_wired_bytes,
                mem_utilization_pct, gpus, total_power_w, collection_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot.get("timestamp"),
                snapshot.get("cpu_utilization_percent"),
                snapshot.get("cpu_power_watts"),
                snapshot.get("total_memory_bytes"),
                snapshot.get("available_memory_bytes"),
                snapshot.get("wired_memory_bytes"),
                snapshot.get("memory_utilization_percent"),
                gpu_structs,
                snapshot.get("cpu_power_watts"),
                collection_type,
            ],
        )

    def write_snapshot(self, snapshot: dict[str, Any], collection_type: str) -> None:
        """Write a vitals snapshot immediately (for threshold breaches).

        Parameters
        ----------
        snapshot
            Parsed snapshot dict.
        collection_type
            One of ``'heartbeat'``, ``'threshold_breach'``, ``'on_demand'``.
        """
        self._write_vitals(snapshot, collection_type)
