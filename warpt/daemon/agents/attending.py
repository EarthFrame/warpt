"""Attending — Python-orchestrated diagnosis loop using LLM advisory."""

from __future__ import annotations

import json
from typing import Any

from warpt.daemon.agents.prompts import ATTENDING_SYSTEM_PROMPT_TEMPLATE
from warpt.daemon.casefile import CaseFile
from warpt.daemon.llm.base import LLMProvider, LLMSchemaError
from warpt.daemon.vitals_nurse import VitalsNurse
from warpt.utils.logger import Logger

# Confidence is hardcoded to -1.23 as a sentinel value.
# This signals that confidence scoring is not yet calibrated.
#
# Future calculation should consider:
# - Whether an LLM was involved in the diagnosis (vs. raw analytics fallback)
# - Whether stress tests were run (Issue #28) — tests should allow higher confidence
# - Whether the LLM response was well-formed JSON or required fallback parsing
# - Deviation magnitude from baseline (high deviation + consistent reasoning = higher)
#
# Revisit when stress test integration lands (#28) and with real-world calibration data.
CONFIDENCE_SENTINEL = -1.23

_TRIAGE_LABELS = {
    "thermal_power": "Thermal / Power",
    "memory": "Memory",
    "compute": "Compute",
    "storage_io": "Storage / IO",
}

# Schema-enforced diagnosis shape (validated at the provider layer).
# The LLM's self-reported confidence is accepted but not yet trusted —
# CONFIDENCE_SENTINEL is stored until Phase-2 calibration lands.
DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "confidence": {"type": "integer"},
        "recommended_action": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["hypothesis", "recommended_action", "reasoning"],
    "additionalProperties": False,
}


class Attending:
    """Python-orchestrated diagnosis loop with LLM advisory.

    Parameters
    ----------
    casefile
        CaseFile instance for database reads/writes.
    provider
        LLMProvider instance for LLM calls.
    vitals_nurse
        VitalsNurse instance for current snapshot access.
    config
        Daemon config dict (must contain ``triage_order``).
    """

    def __init__(
        self,
        casefile: CaseFile,
        provider: LLMProvider,
        vitals_nurse: VitalsNurse,
        config: dict[str, Any],
    ) -> None:
        self._casefile = casefile
        self._provider = provider
        self._vitals_nurse = vitals_nurse
        self._config = config
        self._log = Logger.get("daemon.agents.attending")

    def diagnose(
        self, chart_nurse_result: dict[str, Any], case_id: int
    ) -> dict[str, Any]:
        """Run the diagnosis loop for a case.

        Parameters
        ----------
        chart_nurse_result
            Structured JSON output from Chart Nurse's ``analyze()``.
        case_id
            Existing case row to update with diagnosis.

        Returns
        -------
            Diagnosis dict with hypothesis, confidence_pct, recommended_action,
            and reasoning.
        """
        self._log.info("Attending diagnosing case #%d", case_id)

        snapshot = self._vitals_nurse.get_latest()
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(chart_nurse_result, snapshot)

        # Structured output is schema-validated at the provider layer
        # (bounded retries there). Transport errors bubble up to the
        # pipeline's retry/degradation ladder.
        try:
            response = self._provider.generate(
                [{"role": "user", "content": user_prompt}],
                system=system_prompt,
                response_schema=DIAGNOSIS_SCHEMA,
            )
            parsed = response.structured or {}
            diagnosis = {
                "hypothesis": parsed["hypothesis"],
                "confidence_pct": CONFIDENCE_SENTINEL,
                "recommended_action": parsed["recommended_action"],
                "reasoning": parsed["reasoning"],
            }
        except LLMSchemaError as e:
            diagnosis = self._fallback_response(str(e))

        # Write diagnosis to case
        self._update_case(case_id, chart_nurse_result, diagnosis)

        self._log.info(
            "Diagnosis complete for case #%d: %s",
            case_id,
            diagnosis["hypothesis"],
        )
        return diagnosis

    def _build_system_prompt(self) -> str:
        """Construct the system prompt with triage order from config."""
        triage_order = self._config.get(
            "triage_order", ["thermal_power", "memory", "compute", "storage_io"]
        )
        triage_lines = []
        for i, key in enumerate(triage_order, 1):
            label = _TRIAGE_LABELS.get(key, key)
            triage_lines.append(f"{i}. {label}")
        triage_text = "\n".join(triage_lines)
        return ATTENDING_SYSTEM_PROMPT_TEMPLATE.format(triage_order=triage_text)

    def _build_user_prompt(
        self, chart_nurse_result: dict[str, Any], snapshot: dict[str, Any] | None
    ) -> str:
        """Build the user prompt combining Chart Nurse data and vitals."""
        prompt_data = {
            "chart_nurse_analysis": chart_nurse_result,
            "current_vitals_snapshot": snapshot,
        }
        return json.dumps(prompt_data, default=str)

    def _fallback_response(self, detail: str) -> dict[str, Any]:
        """Return fallback diagnosis when schema-valid output was exhausted."""
        self._log.warning("Schema-valid LLM response unavailable, using fallback")
        return {
            "hypothesis": "Unable to parse LLM diagnosis",
            "confidence_pct": CONFIDENCE_SENTINEL,
            "recommended_action": "Review Chart Nurse analysis manually",
            "reasoning": f"LLM returned unparseable response: {detail[:200]}",
        }

    def _update_case(
        self,
        case_id: int,
        chart_nurse_result: dict[str, Any],
        diagnosis: dict[str, Any],
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
                confidence_pct = ?,
                recommended_action = ?,
                reasoning_chain = ?,
                historical_context = ?,
                baseline_deviation_pct = ?,
                diagnostician_model = ?,
                updated_at = current_timestamp
            WHERE case_id = ?
            """,
            [
                observation,
                diagnosis["hypothesis"],
                diagnosis["confidence_pct"],
                diagnosis["recommended_action"],
                diagnosis["reasoning"],
                historical_ctx,
                baseline_deviation,
                self._provider.model,
                case_id,
            ],
        )
        self._log.debug("Case #%d updated with diagnosis", case_id)
