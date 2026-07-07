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
from warpt.daemon.gpu_fields import SNAPSHOT_TO_DB
from warpt.utils.logger import Logger

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
        heartbeat_interval: float = 10.0,
        poll_interval: float = 5.0,
        gpu_thresholds: dict[str, dict[str, float]] | None = None,
        restart_backoff: float = 1.0,
        max_restart_backoff: float = 30.0,
        healthy_run_seconds: float = 30.0,
        max_consecutive_failures: int = 5,
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
        self._log = Logger.get("daemon.vitals_nurse")
        # Subprocess supervision: restart the monitor if it dies unexpectedly.
        self._restart_backoff = restart_backoff
        self._max_restart_backoff = max_restart_backoff
        self._healthy_run_seconds = healthy_run_seconds
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._last_exit_code: int | None = None
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
        """Start the supervised monitor subprocess (auto-restarts if it dies)."""
        self._stop_event.clear()
        self._consecutive_failures = 0
        self._last_exit_code = None
        self._log.info(
            "VitalsNurse started (poll=%.1fs, heartbeat=%.1fs)",
            self._poll_interval,
            self._heartbeat_interval,
        )
        self._thread = threading.Thread(
            target=self._supervise_loop, name="vitals-supervisor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the supervisor and terminate the subprocess."""
        self._stop_event.set()
        proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread:
            self._thread.join(timeout=5)
        self._process = None
        self._thread = None
        self._log.info("VitalsNurse stopped.")

    def is_healthy(self) -> bool:
        """Return False once the monitor subprocess is repeatedly failing to stay up."""
        return self._consecutive_failures < self._max_consecutive_failures

    def get_health(self) -> dict[str, Any]:
        """Return supervisor health details for status reporting."""
        return {
            "healthy": self.is_healthy(),
            "consecutive_failures": self._consecutive_failures,
            "last_exit_code": self._last_exit_code,
        }

    def _spawn_process(self) -> subprocess.Popen:
        """Spawn the ``warpt monitor`` subprocess. Isolated for testability."""
        return subprocess.Popen(
            ["warpt", "monitor", "--no-tui", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _supervise_loop(self) -> None:
        """Run the monitor subprocess, restarting it with backoff if it exits early.

        A monitor subprocess that dies while the daemon keeps running is a
        production hazard: without supervision the daemon stays "alive" but
        blind, silently observing nothing. This loop detects an unexpected exit,
        logs it, and restarts with exponential backoff — escalating to a
        critical log after repeated rapid failures (``max_consecutive_failures``).
        """
        backoff = self._restart_backoff
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._process = self._spawn_process()
            except Exception:
                self._log.exception("Failed to spawn monitor subprocess")
                self._consecutive_failures += 1
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, self._max_restart_backoff)
                continue

            self._read_stream(self._process)

            if self._stop_event.is_set():
                break

            # Subprocess exited on its own while we were still running.
            self._last_exit_code = self._process.poll()
            ran_for = time.monotonic() - started
            if ran_for >= self._healthy_run_seconds:
                # Ran healthily for a while before dying — not a failure streak.
                self._consecutive_failures = 0
                backoff = self._restart_backoff
            else:
                self._consecutive_failures += 1

            log_at = self._log.error if self.is_healthy() else self._log.critical
            log_at(
                "monitor subprocess exited (code=%s) after %.1fs; "
                "restarting in %.1fs (consecutive_failures=%d)",
                self._last_exit_code,
                ran_for,
                backoff,
                self._consecutive_failures,
            )
            self._stop_event.wait(backoff)
            backoff = min(backoff * 2, self._max_restart_backoff)

    def _read_stream(self, process: subprocess.Popen) -> None:
        """Read and feed JSON snapshot lines from a subprocess until EOF or stop."""
        if process.stdout is None:
            return
        for line in process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError:
                self._log.debug("Skipping unparseable JSON line")
                continue
            self.feed_snapshot(snapshot)

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
                        self._log.warning(
                            "Threshold breach: %s at %.1f "
                            "(threshold %.1f, sustained %.1fs)",
                            metric,
                            current_value,
                            rule["value"],
                            elapsed,
                        )
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
            self._log.info("Registered GPU: %s (%s)", gpu.get("model", "Unknown"), guid)
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
        self._log.debug("Heartbeat written")

    def _write_vitals(self, snapshot: dict[str, Any], collection_type: str) -> None:
        """Write a snapshot row to the vitals table."""
        gpu_structs = []
        for gpu in snapshot.get("gpu_usage", []):
            gpu_struct = {
                db_col: gpu.get(snap_key) for snap_key, db_col in SNAPSHOT_TO_DB.items()
            }
            gpu_struct["throttle_reasons"] = gpu.get("throttle_reasons")
            gpu_structs.append(gpu_struct)

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
                self._compute_total_power(snapshot),
                collection_type,
            ],
        )

    @staticmethod
    def _compute_total_power(snapshot: dict[str, Any]) -> float | None:
        """Sum CPU and per-GPU power draw; None if no component reports power."""
        components = [snapshot.get("cpu_power_watts")]
        components.extend(
            gpu.get("power_watts") for gpu in snapshot.get("gpu_usage", [])
        )
        present = [p for p in components if p is not None]
        return sum(present) if present else None

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
