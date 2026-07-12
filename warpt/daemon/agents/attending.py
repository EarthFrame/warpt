"""Attending — bounded agentic diagnosis loop with tool-based evidence.

The Attending gathers evidence through read-only tools (bounded by
iterations, wall-clock, and the provider's own budget wrapper), then makes a
final schema-constrained conclude call. Every tool invocation is persisted to
``tool_calls``; the recommended action is proposed through the remediation
policy engine and audited — never executed. Without a tool registry the
Attending degrades to the one-shot structured diagnosis (the pipeline's
degradation ladder is unchanged).
"""

from __future__ import annotations

import json
import time
from typing import Any

from warpt.daemon.agents.calibration import calibrate_confidence
from warpt.daemon.agents.prompts import get_prompt, prompt_snapshot
from warpt.daemon.agents.tools.registry import ToolRegistry
from warpt.daemon.casefile import CaseFile
from warpt.daemon.llm.base import LLMError, LLMProvider, LLMSchemaError
from warpt.daemon.remediation.audit import AuditLog
from warpt.daemon.remediation.base import ProposedAction
from warpt.daemon.remediation.policy import PolicyEngine
from warpt.daemon.vitals_nurse import VitalsNurse
from warpt.utils.logger import Logger

_TRIAGE_LABELS = {
    "thermal_power": "Thermal / Power",
    "memory": "Memory",
    "compute": "Compute",
    "storage_io": "Storage / IO",
}

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_WALL_CLOCK_S = 120.0

# Schema-enforced diagnosis shape (validated at the provider layer).
DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        "confidence": {"type": "integer"},
        "recommended_action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string"},
                "target": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["action_type", "description"],
        },
        "reasoning_chain": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "hypothesis",
        "severity",
        "recommended_action",
        "reasoning_chain",
        "evidence",
    ],
    "additionalProperties": False,
}


class Attending:
    """Agentic diagnosis loop with LLM tool use and calibrated confidence.

    Parameters
    ----------
    casefile
        CaseFile instance for database reads/writes.
    provider
        LLMProvider instance for LLM calls.
    vitals_nurse
        VitalsNurse instance for current snapshot access.
    config
        Daemon config dict (``triage_order``, ``attending.*``).
    tool_registry
        Tools for evidence gathering; ``None`` disables the loop and the
        Attending degrades to a one-shot structured diagnosis.
    policy_engine
        Remediation policy engine; with ``audit_log``, recommended actions
        are proposed and recorded (never executed).
    audit_log
        Audit log for proposed actions.
    """

    def __init__(
        self,
        casefile: CaseFile,
        provider: LLMProvider,
        vitals_nurse: VitalsNurse,
        config: dict[str, Any],
        tool_registry: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._casefile = casefile
        self._provider = provider
        self._vitals_nurse = vitals_nurse
        self._config = config
        self._tool_registry = tool_registry
        self._policy_engine = policy_engine
        self._audit_log = audit_log
        attending_cfg = config.get("attending", {}) or {}
        self._max_iterations = int(
            attending_cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        )
        self._max_wall_clock_s = float(
            attending_cfg.get("max_wall_clock_s", DEFAULT_MAX_WALL_CLOCK_S)
        )
        self._log = Logger.get("daemon.agents.attending")

    def diagnose(
        self, chart_nurse_result: dict[str, Any], case_id: int
    ) -> dict[str, Any]:
        """Run the diagnosis loop for a case.

        Parameters
        ----------
        chart_nurse_result
            Structured output from Chart Nurse's ``analyze()``.
        case_id
            Existing case row to update with the diagnosis.

        Returns
        -------
            Diagnosis dict with hypothesis, severity, confidence_pct,
            recommended_action, reasoning_chain, evidence, and tools_used.
        """
        self._log.info("Attending diagnosing case #%d", case_id)

        prompt = get_prompt("attending_system")
        system = self._build_system_prompt(prompt.text)
        snapshot = self._vitals_nurse.get_latest()
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "chart_nurse_analysis": chart_nurse_result,
                        "current_vitals_snapshot": snapshot,
                    },
                    default=str,
                ),
            }
        ]

        tools_used: list[str] = []
        probe_ran = False
        probe_corroborates: bool | None = None
        if self._tool_registry is not None:
            probe_ran, probe_corroborates = self._tool_loop(
                messages, system, case_id, tools_used
            )

        # Final, schema-constrained conclude call. Transport errors bubble
        # to the pipeline's retry/degradation ladder.
        messages.append(
            {
                "role": "user",
                "content": "Conclude now. Produce your final diagnosis.",
            }
        )
        try:
            final = self._provider.generate(
                messages, system=system, response_schema=DIAGNOSIS_SCHEMA
            )
            parsed = final.structured or {}
            diagnosis = self._build_diagnosis(
                parsed,
                chart_nurse_result,
                tools_used,
                probe_ran,
                probe_corroborates,
            )
        except LLMSchemaError as e:
            diagnosis = self._fallback_response(str(e), tools_used)

        self._propose_action(diagnosis, chart_nurse_result, case_id)
        self._update_case(case_id, chart_nurse_result, diagnosis, prompt)

        self._log.info(
            "Diagnosis complete for case #%d: %s",
            case_id,
            diagnosis["hypothesis"],
        )
        return diagnosis

    def _tool_loop(
        self,
        messages: list[dict[str, Any]],
        system: str,
        case_id: int,
        tools_used: list[str],
    ) -> tuple[bool, bool | None]:
        """Run the bounded evidence-gathering loop; mutates *messages*.

        Returns
        -------
            ``(probe_ran, probe_corroborates)`` for calibration.
        """
        assert self._tool_registry is not None
        tool_defs = self._tool_registry.tool_defs()
        started = time.monotonic()
        probe_ran = False
        probe_corroborates: bool | None = None

        for iteration in range(self._max_iterations):
            if time.monotonic() - started >= self._max_wall_clock_s:
                self._log.warning(
                    "Attending wall-clock budget exhausted after %d iterations",
                    iteration,
                )
                break
            try:
                response = self._provider.generate(
                    messages, system=system, tools=tool_defs
                )
            except LLMError:
                if iteration == 0:
                    raise
                self._log.warning(
                    "LLM error mid-loop; concluding with evidence gathered "
                    "so far (%d iterations)",
                    iteration,
                )
                break

            if not response.tool_calls:
                if response.text:
                    messages.append({"role": "assistant", "content": response.text})
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                outcome = self._run_and_record_tool(call, case_id)
                if call.name not in tools_used:
                    tools_used.append(call.name)
                if call.name == "run_diagnostic_probe":
                    if outcome.get("status") == "ok":
                        probe_ran = True
                        result = outcome.get("result", {}) or {}
                        probe_corroborates = bool(result.get("errors"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(outcome, default=str),
                    }
                )
        return probe_ran, probe_corroborates

    def _run_and_record_tool(self, call: Any, case_id: int) -> dict[str, Any]:
        """Execute one tool call and persist it to ``tool_calls``."""
        assert self._tool_registry is not None
        t0 = time.monotonic()
        outcome = self._tool_registry.run(call.name, call.arguments)
        latency_ms = (time.monotonic() - t0) * 1000.0
        self._casefile.execute(
            """
            INSERT INTO tool_calls (case_id, tool_name, arguments, result,
                                    status, latency_ms)
            VALUES (?, ?, ?::JSON, ?::JSON, ?, ?)
            """,
            [
                case_id,
                call.name,
                json.dumps(call.arguments, default=str),
                json.dumps(outcome, default=str),
                outcome.get("status", "error"),
                latency_ms,
            ],
        )
        self._log.debug(
            "Tool %s -> %s (%.0fms)", call.name, outcome.get("status"), latency_ms
        )
        return outcome

    def _build_diagnosis(
        self,
        parsed: dict[str, Any],
        chart_nurse_result: dict[str, Any],
        tools_used: list[str],
        probe_ran: bool,
        probe_corroborates: bool | None,
    ) -> dict[str, Any]:
        """Assemble the diagnosis dict with calibrated confidence."""
        confidence = calibrate_confidence(
            llm_confidence=parsed.get("confidence"),
            deviation_z=self._deviation_z(chart_nurse_result),
            corroborating_signals=self._corroborating_signals(chart_nurse_result),
            probe_ran=probe_ran,
            probe_corroborates=probe_corroborates,
            prior_case_count=len(chart_nurse_result.get("prior_cases", [])),
        )
        return {
            "hypothesis": parsed["hypothesis"],
            "severity": parsed.get("severity", "warning"),
            "confidence_pct": confidence,
            "recommended_action": parsed["recommended_action"],
            "reasoning_chain": parsed.get("reasoning_chain", []),
            "evidence": parsed.get("evidence", []),
            "tools_used": tools_used,
        }

    @staticmethod
    def _deviation_z(chart_nurse_result: dict[str, Any]) -> float | None:
        """Compute |current - hour_mean| / stddev from the hour profile."""
        profile = chart_nurse_result.get("current_hour_profile")
        if not profile:
            return None
        stddev = profile.get("stddev")
        mean = profile.get("mean")
        current = chart_nurse_result.get("current_value")
        if not stddev or mean is None or current is None:
            return None
        return abs(current - mean) / stddev

    @staticmethod
    def _corroborating_signals(chart_nurse_result: dict[str, Any]) -> int:
        """Count corroborating signals from Chart Nurse's correlation data."""
        correlated = chart_nurse_result.get("correlated_signals") or {}
        breach_metric = chart_nurse_result.get("metric")
        count = 0
        for metric, values in (correlated.get("metrics") or {}).items():
            if metric == breach_metric:
                continue
            deviation = (values or {}).get("deviation_pct")
            if deviation is not None and abs(deviation) >= 10:
                count += 1
        if correlated.get("throttle_reasons_recent"):
            count += 1
        return count

    def _build_system_prompt(self, template: str) -> str:
        """Construct the system prompt with triage order from config."""
        triage_order = self._config.get(
            "triage_order", ["thermal_power", "memory", "compute", "storage_io"]
        )
        triage_lines = []
        for i, key in enumerate(triage_order, 1):
            label = _TRIAGE_LABELS.get(key, key)
            triage_lines.append(f"{i}. {label}")
        return template.format(triage_order="\n".join(triage_lines))

    def _fallback_response(self, detail: str, tools_used: list[str]) -> dict[str, Any]:
        """Fallback diagnosis when schema-valid output was exhausted.

        Confidence is ``None`` (stored as NULL) — display sites suppress it.
        """
        self._log.warning("Schema-valid LLM response unavailable, using fallback")
        return {
            "hypothesis": "Unable to parse LLM diagnosis",
            "severity": "warning",
            "confidence_pct": None,
            "recommended_action": {
                "action_type": "investigate",
                "description": "Review Chart Nurse analysis manually",
            },
            "reasoning_chain": [f"LLM returned unparseable response: {detail[:200]}"],
            "evidence": [],
            "tools_used": tools_used,
        }

    def _propose_action(
        self,
        diagnosis: dict[str, Any],
        chart_nurse_result: dict[str, Any],
        case_id: int,
    ) -> None:
        """Propose the recommended action through policy + audit (no exec)."""
        if self._policy_engine is None or self._audit_log is None:
            return
        recommended = diagnosis.get("recommended_action") or {}
        action = ProposedAction(
            action_type=recommended.get("action_type", "investigate"),
            target=recommended.get("target") or chart_nurse_result.get("gpu_guid"),
            parameters={"description": recommended.get("description", "")},
            reason=diagnosis.get("hypothesis", ""),
            case_id=case_id,
        )
        decision = self._policy_engine.evaluate(action)
        self._audit_log.record(action, decision)

    def _update_case(
        self,
        case_id: int,
        chart_nurse_result: dict[str, Any],
        diagnosis: dict[str, Any],
        prompt: Any,
    ) -> None:
        """Write diagnosis fields to the existing case row."""
        observation = json.dumps(chart_nurse_result, default=str)
        baseline_deviation = chart_nurse_result.get("deviation_pct")
        historical_ctx = json.dumps(
            {
                "baseline": chart_nurse_result.get("baseline"),
                "current_hour_profile": chart_nurse_result.get("current_hour_profile"),
                "prior_cases": chart_nurse_result.get("prior_cases"),
                "event_count_7d": chart_nurse_result.get("event_count_7d"),
            },
            default=str,
        )

        self._casefile.execute(
            """
            UPDATE cases SET
                observation = ?,
                hypothesis = ?,
                severity = ?,
                confidence_pct = ?,
                recommended_action = ?,
                reasoning_chain = ?,
                evidence = ?::JSON,
                tools_used = ?::JSON,
                historical_context = ?,
                baseline_deviation_pct = ?,
                diagnostician_model = ?,
                prompt_snapshot = ?::JSON,
                updated_at = current_timestamp
            WHERE case_id = ?
            """,
            [
                observation,
                diagnosis["hypothesis"],
                diagnosis["severity"],
                diagnosis["confidence_pct"],
                json.dumps(diagnosis["recommended_action"], default=str),
                "\n".join(diagnosis["reasoning_chain"]),
                json.dumps(diagnosis["evidence"], default=str),
                json.dumps(diagnosis["tools_used"], default=str),
                historical_ctx,
                baseline_deviation,
                self._provider.model,
                json.dumps(prompt_snapshot(prompt, self._provider.model)),
                case_id,
            ],
        )
        self._log.debug("Case #%d updated with diagnosis", case_id)
