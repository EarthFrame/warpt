"""Built-in read-only tools for the Attending's evidence-gathering loop.

Five pure reads (historical vitals, current snapshot, prior cases, case
detail, GPU specs) plus one policy-gated probe that reuses the ``warpt
stress`` framework as a diagnostic reproduction tool — never on a busy node.
"""

from __future__ import annotations

from typing import Any, ClassVar

from warpt.daemon.agents.tools.base import Tool, ToolDeniedError, ToolError
from warpt.daemon.agents.tools.registry import ToolRegistry
from warpt.daemon.casefile import CaseFile
from warpt.daemon.gpu_fields import SNAPSHOT_TO_DB
from warpt.daemon.remediation.audit import AuditLog
from warpt.daemon.remediation.base import ProposedAction
from warpt.daemon.remediation.policy import VERDICT_ALLOW, PolicyEngine
from warpt.daemon.vitals_nurse import VitalsNurse

_CASE_COLUMNS = (
    "case_id",
    "title",
    "status",
    "opened_at",
    "closed_at",
    "updated_at",
    "observation",
    "hypothesis",
    "severity",
    "confidence_pct",
    "recommended_action",
    "reasoning_chain",
    "baseline_deviation_pct",
    "report_content",
    "diagnostician_model",
)

# Probe categories map onto the stress framework's TestCategory values.
_PROBE_CATEGORIES = {"gpu": "ACCELERATOR", "cpu": "CPU", "memory": "RAM"}


class QueryHistoricalVitals(Tool):
    """Windowed stats for one GPU metric from the vitals table."""

    name = "query_historical_vitals"
    description = (
        "Statistics (count/min/max/avg/stddev) for one GPU metric over a "
        "time window, from the node's local vitals history."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "gpu_guid": {"type": "string", "description": "GPU identifier"},
            "metric": {
                "type": "string",
                "description": (
                    "Snapshot metric key, e.g. 'utilization_percent', "
                    "'temperature_c', 'power_watts'"
                ),
            },
            "window_hours": {
                "type": "number",
                "description": "Look-back window in hours (default 24)",
            },
        },
        "required": ["gpu_guid", "metric"],
    }

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Compute windowed stats for the requested GPU metric."""
        gpu_guid = args.get("gpu_guid")
        metric = args.get("metric")
        if not gpu_guid or not metric:
            raise ToolError("gpu_guid and metric are required")
        column = SNAPSHOT_TO_DB.get(metric)
        if column is None:
            raise ToolError(
                f"Unknown metric {metric!r} "
                f"(valid: {', '.join(sorted(SNAPSHOT_TO_DB))})"
            )
        window_hours = float(args.get("window_hours", 24))
        rows = self._casefile.query(
            f"""
            SELECT COUNT(g.{column}), MIN(g.{column}), MAX(g.{column}),
                   AVG(g.{column}), STDDEV(g.{column}), MIN(ts), MAX(ts)
            FROM vitals, UNNEST(gpus) AS t(g)
            WHERE g.gpu_guid = ?
              AND ts > current_timestamp - INTERVAL '{window_hours} HOURS'
            """,
            [gpu_guid],
        )
        count, vmin, vmax, avg, stddev, first_ts, last_ts = rows[0]
        return {
            "gpu_guid": gpu_guid,
            "metric": metric,
            "window_hours": window_hours,
            "count": count or 0,
            "min": round(vmin, 2) if vmin is not None else None,
            "max": round(vmax, 2) if vmax is not None else None,
            "avg": round(avg, 2) if avg is not None else None,
            "stddev": round(stddev, 2) if stddev is not None else None,
            "first_ts": str(first_ts) if first_ts is not None else None,
            "last_ts": str(last_ts) if last_ts is not None else None,
        }


class GetCurrentSnapshot(Tool):
    """Latest vitals frame from the in-memory ring buffer."""

    name = "get_current_snapshot"
    description = "The most recent vitals snapshot (CPU, memory, all GPUs)."
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self, vitals_nurse: VitalsNurse) -> None:
        self._vitals_nurse = vitals_nurse

    def run(self, args: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """Return the newest ring-buffer snapshot (null when empty)."""
        return {"snapshot": self._vitals_nurse.get_latest()}


class ListPriorCases(Tool):
    """Recent cases recorded for a GPU."""

    name = "list_prior_cases"
    description = "Most recent prior cases for a GPU (id, title, status)."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "gpu_guid": {"type": "string", "description": "GPU identifier"},
            "limit": {"type": "integer", "description": "Max cases (default 5)"},
        },
        "required": ["gpu_guid"],
    }

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """List up to *limit* recent cases linked to the GPU's events."""
        gpu_guid = args.get("gpu_guid")
        if not gpu_guid:
            raise ToolError("gpu_guid is required")
        limit = int(args.get("limit", 5))
        rows = self._casefile.query(
            """
            SELECT DISTINCT c.case_id, c.title, c.status, c.opened_at,
                            c.hypothesis
            FROM cases c
            JOIN events e ON e.case_id = c.case_id
            WHERE e.gpu_guid = ?
            ORDER BY c.opened_at DESC
            LIMIT ?
            """,
            [gpu_guid, limit],
        )
        return {
            "cases": [
                {
                    "case_id": r[0],
                    "title": r[1],
                    "status": r[2],
                    "opened_at": str(r[3]),
                    "hypothesis": r[4],
                }
                for r in rows
            ]
        }


class GetCase(Tool):
    """Full detail for one case."""

    name = "get_case"
    description = "Full stored detail for a single case by id."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "case_id": {"type": "integer", "description": "Case id"},
        },
        "required": ["case_id"],
    }

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the case row keyed by column name."""
        case_id = args.get("case_id")
        if case_id is None:
            raise ToolError("case_id is required")
        rows = self._casefile.query(
            f"SELECT {', '.join(_CASE_COLUMNS)} FROM cases WHERE case_id = ?",
            [int(case_id)],
        )
        if not rows:
            raise ToolError(f"case {case_id} not found")
        return {
            col: (str(val) if col.endswith("_at") and val is not None else val)
            for col, val in zip(_CASE_COLUMNS, rows[0], strict=False)
        }


class GetGpuSpecs(Tool):
    """Registered hardware profile for a GPU."""

    name = "get_gpu_specs"
    description = "Hardware profile for a GPU (model, memory, power limit)."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "gpu_guid": {"type": "string", "description": "GPU identifier"},
        },
        "required": ["gpu_guid"],
    }

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the gpu_profiles row for the GPU."""
        gpu_guid = args.get("gpu_guid")
        if not gpu_guid:
            raise ToolError("gpu_guid is required")
        rows = self._casefile.query(
            """
            SELECT gpu_guid, model, vendor, memory_total_bytes,
                   compute_capability, pcie_gen, driver_version,
                   power_limit_w, first_seen_at, last_seen_at
            FROM gpu_profiles WHERE gpu_guid = ?
            """,
            [gpu_guid],
        )
        if not rows:
            raise ToolError(f"no profile for GPU {gpu_guid!r}")
        r = rows[0]
        return {
            "gpu_guid": r[0],
            "model": r[1],
            "vendor": r[2],
            "memory_total_bytes": r[3],
            "compute_capability": r[4],
            "pcie_gen": r[5],
            "driver_version": r[6],
            "power_limit_w": r[7],
            "first_seen_at": str(r[8]),
            "last_seen_at": str(r[9]),
        }


class RunDiagnosticProbe(Tool):
    """Policy-gated diagnostic stress probe (idle nodes only)."""

    name = "run_diagnostic_probe"
    description = (
        "Run a short diagnostic stress probe to reproduce a suspected "
        "hardware issue. Only permitted by policy on an idle node; every "
        "request is audited."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["gpu", "cpu", "memory"],
                "description": "Subsystem to probe (default gpu)",
            },
            "duration_s": {
                "type": "integer",
                "description": "Probe duration in seconds (default 10)",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        vitals_nurse: VitalsNurse,
        policy_engine: PolicyEngine,
        audit_log: AuditLog,
        config: dict[str, Any],
    ) -> None:
        self._vitals_nurse = vitals_nurse
        self._policy = policy_engine
        self._audit = audit_log
        probes = (config.get("remediation", {}) or {}).get("probes", {}) or {}
        self._idle_threshold = float(probes.get("idle_threshold_pct", 20.0))
        self._max_duration = int(probes.get("max_duration_s", 30))

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Policy-check, idle-check, audit, then run the probe."""
        category = args.get("category", "gpu")
        if category not in _PROBE_CATEGORIES:
            raise ToolError(
                f"Unknown category {category!r} "
                f"(valid: {', '.join(_PROBE_CATEGORIES)})"
            )
        duration_s = min(int(args.get("duration_s", 10)), self._max_duration)

        action = ProposedAction(
            action_type="diagnostic_probe",
            target=category,
            parameters={"category": category, "duration_s": duration_s},
            reason="attending-ordered diagnostic probe",
        )
        decision = self._policy.evaluate(action)
        self._audit.record(action, decision)
        if decision.verdict != VERDICT_ALLOW:
            raise ToolDeniedError(f"probe denied by policy: {decision.reason}")

        self._check_idle()
        return self._execute(category, duration_s)

    def _check_idle(self) -> None:
        """Refuse to probe unless every GPU is below the idle threshold."""
        latest = self._vitals_nurse.get_latest()
        if latest is None:
            raise ToolDeniedError("no vitals snapshot available; cannot verify idle")
        for gpu in latest.get("gpu_usage", []):
            util = gpu.get("utilization_percent")
            if util is not None and util >= self._idle_threshold:
                raise ToolDeniedError(
                    f"GPU {gpu.get('guid', '?')} is busy "
                    f"({util:.1f}% >= {self._idle_threshold:.1f}% idle "
                    "threshold); refusing to probe"
                )

    def _execute(self, category: str, duration_s: int) -> dict[str, Any]:
        """Run the stress tests for *category* and summarize results."""
        try:
            from warpt.stress.base import TestCategory
            from warpt.stress.registry import TestRegistry
            from warpt.stress.runner import TestRunner
        except ImportError as e:
            raise ToolError(f"stress framework unavailable: {e}") from e

        test_category = TestCategory[_PROBE_CATEGORIES[category]]
        registry = TestRegistry()
        runner = TestRunner()
        runner.add_tests(registry.get_tests_by_category(test_category))
        results = runner.run(duration=duration_s)
        return {
            "status": "completed",
            "category": category,
            "duration_s": duration_s,
            "results": results.results,
            "errors": results.errors,
        }


def build_default_registry(
    *,
    casefile: CaseFile,
    vitals_nurse: VitalsNurse,
    policy_engine: PolicyEngine,
    audit_log: AuditLog,
    config: dict[str, Any],
) -> ToolRegistry:
    """Build the standard Attending tool registry.

    Parameters
    ----------
    casefile
        CaseFile for DB reads.
    vitals_nurse
        VitalsNurse for snapshot access and the idle check.
    policy_engine
        Policy engine gating the diagnostic probe.
    audit_log
        Audit log recording every probe request.
    config
        Full daemon config dict.
    """
    registry = ToolRegistry()
    registry.register(QueryHistoricalVitals(casefile))
    registry.register(GetCurrentSnapshot(vitals_nurse))
    registry.register(ListPriorCases(casefile))
    registry.register(GetCase(casefile))
    registry.register(GetGpuSpecs(casefile))
    registry.register(
        RunDiagnosticProbe(vitals_nurse, policy_engine, audit_log, config)
    )
    return registry
