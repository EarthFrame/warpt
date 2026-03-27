"""Attending — Python-orchestrated diagnosis loop using LLM advisory."""

from __future__ import annotations

import json
from typing import Any

from warpt.daemon.agents.ollama_client import OllamaClient
from warpt.daemon.agents.prompts import ATTENDING_SYSTEM_PROMPT_TEMPLATE
from warpt.daemon.casefile import CaseFile
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


class Attending:
    """Python-orchestrated diagnosis loop with LLM advisory.

    Parameters
    ----------
    casefile
        CaseFile instance for database reads/writes.
    ollama_client
        OllamaClient instance for LLM calls.
    vitals_nurse
        VitalsNurse instance for current snapshot access.
    config
        Daemon config dict (must contain ``models.attending`` and ``triage_order``).
    """

    def __init__(
        self,
        casefile: CaseFile,
        ollama_client: OllamaClient,
        vitals_nurse: VitalsNurse,
        config: dict[str, Any],
    ) -> None:
        self._casefile = casefile
        self._client = ollama_client
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

        # Retry generate+parse cycle up to 3 times on malformed output.
        # Connection errors (RuntimeError) bubble up immediately.
        max_parse_attempts = 3
        diagnosis = None
        for attempt in range(max_parse_attempts):
            self._log.debug("Calling LLM for diagnosis (attempt %d)", attempt + 1)
            raw_response = self._client.generate(user_prompt, system_prompt)
            diagnosis = self._try_parse(raw_response)
            if diagnosis is not None:
                break
            self._log.warning(
                "Malformed LLM response (attempt %d/%d), retrying",
                attempt + 1,
                max_parse_attempts,
            )

        if diagnosis is None:
            self._log.warning(
                "All %d parse attempts failed, using fallback",
                max_parse_attempts,
            )
            diagnosis = self._fallback_response(raw_response)

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

    def _try_parse(self, raw: str) -> dict[str, Any] | None:
        """Try to parse LLM JSON response. Returns None on failure."""
        try:
            parsed = json.loads(raw)
            return {
                "hypothesis": parsed["hypothesis"],
                "confidence_pct": CONFIDENCE_SENTINEL,
                "recommended_action": parsed["recommended_action"],
                "reasoning": parsed["reasoning"],
            }
        except (json.JSONDecodeError, KeyError):
            return None

    def _fallback_response(self, raw: str) -> dict[str, Any]:
        """Return fallback diagnosis when all parse attempts fail."""
        self._log.warning("Malformed LLM response, using fallback")
        return {
            "hypothesis": "Unable to parse LLM diagnosis",
            "confidence_pct": CONFIDENCE_SENTINEL,
            "recommended_action": "Review Chart Nurse analysis manually",
            "reasoning": f"LLM returned unparseable response: {raw[:200]}",
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
                self._client.model,
                case_id,
            ],
        )
        self._log.debug("Case #%d updated with diagnosis", case_id)
