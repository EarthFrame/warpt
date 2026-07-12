"""ChartNurse — baseline analytics and LLM interpretation for GPU metrics."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from warpt.daemon.agents.prompts import CHART_NURSE_SYSTEM_PROMPT
from warpt.daemon.casefile import CaseFile
from warpt.daemon.gpu_fields import SNAPSHOT_TO_DB
from warpt.daemon.llm.base import LLMProvider
from warpt.utils.logger import Logger


class ChartNurse:
    """Computes baseline analytics from DuckDB and requests LLM interpretation.

    Parameters
    ----------
    casefile
        CaseFile instance for database queries.
    provider
        LLMProvider instance for interpretation.
    """

    def __init__(self, casefile: CaseFile, provider: LLMProvider) -> None:
        self._casefile = casefile
        self._provider = provider
        self._log = Logger.get("daemon.agents.chart_nurse")

    def analyze(
        self, gpu_guid: str, metric: str, current_value: float
    ) -> dict[str, Any]:
        """Run baseline analytics and LLM interpretation for a GPU metric.

        Parameters
        ----------
        gpu_guid
            GPU identifier.
        metric
            The struct field name (e.g. ``"utilization_pct"``).
        current_value
            Current observed value that triggered analysis.

        Returns
        -------
            Analysis dict with baseline, profile, deviation, correlated
            signals, prior cases, event count, interpretation, and model.
        """
        stats = self._gather_analytics(gpu_guid, metric, current_value)

        response = self._provider.generate(
            [{"role": "user", "content": json.dumps(stats, default=str)}],
            system=CHART_NURSE_SYSTEM_PROMPT,
        )

        return {
            **stats,
            "interpretation": response.text,
            "model_used": response.model,
        }

    def analyze_without_llm(
        self, gpu_guid: str, metric: str, current_value: float
    ) -> dict[str, Any]:
        """Return analytics dict without an LLM call.

        Same structure as ``analyze()`` but with ``interpretation=None``
        and ``model_used=None``.
        """
        stats = self._gather_analytics(gpu_guid, metric, current_value)
        return {**stats, "interpretation": None, "model_used": None}

    def _gather_analytics(
        self, gpu_guid: str, metric: str, current_value: float
    ) -> dict[str, Any]:
        """Compute the full analytics payload shared by both analyze paths."""
        baseline = self._rolling_averages(gpu_guid, metric)
        hour_profile = self._hourly_profile(gpu_guid, metric)
        prior_cases = self._prior_cases(gpu_guid)
        event_count = self._event_count_7d(gpu_guid)
        correlated = self._correlated_signals(gpu_guid)

        # Deviation from 1h average
        deviation_pct = None
        if baseline["1h_avg"] and baseline["1h_avg"] > 0:
            deviation_pct = round(
                ((current_value - baseline["1h_avg"]) / baseline["1h_avg"]) * 100,
                1,
            )

        return {
            "gpu_guid": gpu_guid,
            "metric": metric,
            "current_value": current_value,
            "baseline": baseline,
            "current_hour_profile": hour_profile,
            "deviation_pct": deviation_pct,
            "correlated_signals": correlated,
            "prior_cases": prior_cases,
            "event_count_7d": event_count,
        }

    def _rolling_averages(self, gpu_guid: str, metric: str) -> dict[str, float | None]:
        """Compute 1h, 24h, and 7d rolling averages for a GPU metric."""
        column = SNAPSHOT_TO_DB.get(metric, metric)
        result: dict[str, float | None] = {
            "1h_avg": None,
            "24h_avg": None,
            "7d_avg": None,
        }
        intervals = {
            "1h_avg": "1 HOUR",
            "24h_avg": "24 HOURS",
            "7d_avg": "7 DAYS",
        }
        for key, interval in intervals.items():
            rows = self._casefile.query(
                f"""
                SELECT AVG(g.{column})
                FROM vitals, UNNEST(gpus) AS t(g)
                WHERE g.gpu_guid = ?
                  AND ts > current_timestamp - INTERVAL '{interval}'
                """,
                [gpu_guid],
            )
            if rows and rows[0][0] is not None:
                result[key] = round(rows[0][0], 1)
        return result

    def _hourly_profile(self, gpu_guid: str, metric: str) -> dict[str, Any] | None:
        """Compute mean and stddev for the current hour-of-day."""
        column = SNAPSHOT_TO_DB.get(metric, metric)
        current_hour = datetime.now().hour
        rows = self._casefile.query(
            f"""
            SELECT AVG(g.{column}) AS mean, STDDEV(g.{column}) AS stddev
            FROM vitals, UNNEST(gpus) AS t(g)
            WHERE g.gpu_guid = ?
              AND EXTRACT(HOUR FROM ts) = ?
            """,
            [gpu_guid, current_hour],
        )
        if rows and rows[0][0] is not None:
            return {
                "hour": current_hour,
                "mean": round(rows[0][0], 1),
                "stddev": round(rows[0][1], 1) if rows[0][1] is not None else 0.0,
            }
        return None

    # Core metrics surfaced together so the Attending reasons over a
    # coherent picture, not one metric.
    _CORRELATED_COLUMNS = (
        "utilization_pct",
        "mem_utilization_pct",
        "power_w",
        "temperature_c",
    )

    def _correlated_signals(self, gpu_guid: str) -> dict[str, Any]:
        """Collect current vs 1h-baseline for core metrics + recent throttling.

        Returns
        -------
            ``{"metrics": {<db_col>: {"current", "1h_avg", "deviation_pct"}},
            "throttle_reasons_recent": [...]}``
        """
        metrics: dict[str, dict[str, float | None]] = {}
        for column in self._CORRELATED_COLUMNS:
            current = None
            rows = self._casefile.query(
                f"""
                SELECT g.{column}
                FROM vitals, UNNEST(gpus) AS t(g)
                WHERE g.gpu_guid = ?
                ORDER BY ts DESC LIMIT 1
                """,
                [gpu_guid],
            )
            if rows and rows[0][0] is not None:
                current = round(rows[0][0], 1)

            avg_1h = None
            rows = self._casefile.query(
                f"""
                SELECT AVG(g.{column})
                FROM vitals, UNNEST(gpus) AS t(g)
                WHERE g.gpu_guid = ?
                  AND ts > current_timestamp - INTERVAL '1 HOUR'
                """,
                [gpu_guid],
            )
            if rows and rows[0][0] is not None:
                avg_1h = round(rows[0][0], 1)

            deviation_pct = None
            if current is not None and avg_1h and avg_1h > 0:
                deviation_pct = round(((current - avg_1h) / avg_1h) * 100, 1)

            metrics[column] = {
                "current": current,
                "1h_avg": avg_1h,
                "deviation_pct": deviation_pct,
            }

        reasons = self._casefile.query(
            """
            SELECT DISTINCT r.tr
            FROM vitals, UNNEST(gpus) AS t(g),
                 UNNEST(g.throttle_reasons) AS r(tr)
            WHERE g.gpu_guid = ?
              AND ts > current_timestamp - INTERVAL '1 HOUR'
            """,
            [gpu_guid],
        )
        return {
            "metrics": metrics,
            "throttle_reasons_recent": sorted(r[0] for r in reasons),
        }

    def _prior_cases(self, gpu_guid: str) -> list[dict[str, Any]]:
        """List up to 5 most recent cases for this GPU."""
        rows = self._casefile.query(
            """
            SELECT DISTINCT c.case_id, c.title, c.opened_at
            FROM cases c
            JOIN events e ON e.case_id = c.case_id
            WHERE e.gpu_guid = ?
            ORDER BY c.opened_at DESC
            LIMIT 5
            """,
            [gpu_guid],
        )
        return [{"case_id": r[0], "title": r[1], "opened_at": str(r[2])} for r in rows]

    def _event_count_7d(self, gpu_guid: str) -> int:
        """Count events for this GPU in the last 7 days."""
        rows = self._casefile.query(
            """
            SELECT COUNT(*)
            FROM events
            WHERE gpu_guid = ?
              AND ts > current_timestamp - INTERVAL '7 DAYS'
            """,
            [gpu_guid],
        )
        return rows[0][0] if rows else 0
