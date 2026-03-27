"""Intelligence pipeline — Chart Nurse → Attending → Scribe with degradation."""

from __future__ import annotations

import json
import time
from typing import Any

from warpt.daemon.agents.ollama_client import OllamaPermanentError


def run_intelligence_pipeline(
    case_id: int,
    event: dict[str, Any],
    *,
    chart_nurse: Any,
    attending: Any,
    scribe: Any,
    casefile: Any,
    log: Any,
    retries: int = 3,
    backoff: float = 2.0,
) -> None:
    """Run Chart Nurse → Attending → Scribe with retry and graceful degradation.

    Degradation ladder (best → worst):
    1. Full diagnosis — both LLMs succeed
    2. Chart Nurse analysis only — Attending fails
    3. Raw analytics only — Chart Nurse LLM fails
    4. Phase 1 event only — everything fails (case already has event data)
    """
    gpu_guid = event.get("gpu_guid", "")
    metric = event.get("metric", "")
    value = event.get("value", 0.0)

    # --- Stage 1: Chart Nurse ---
    chart_result = None
    llm_succeeded = False

    for attempt in range(retries):
        try:
            chart_result = chart_nurse.analyze(gpu_guid, metric, value)
            llm_succeeded = True
            break
        except OllamaPermanentError:
            log.warning("Chart Nurse permanent error, skipping retries")
            break
        except RuntimeError as e:
            if attempt < retries - 1:
                delay = backoff * (2**attempt)
                log.warning(
                    "Chart Nurse retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    retries,
                    delay,
                    e,
                )
                time.sleep(delay)
            else:
                log.warning("Chart Nurse exhausted %d retries: %s", retries, e)

    if not llm_succeeded:
        # Fall back to analytics without LLM
        chart_result = chart_nurse.analyze_without_llm(gpu_guid, metric, value)
        _write_degraded_observation(
            casefile, case_id, chart_result, "Chart Nurse LLM unavailable"
        )
        scribe.report(case_id)
        return

    # --- Stage 2: Attending (only if Chart Nurse LLM succeeded) ---
    for attempt in range(retries):
        try:
            attending.diagnose(chart_result, case_id)
            break
        except OllamaPermanentError:
            log.warning("Attending permanent error, skipping retries")
            _write_degraded_observation(
                casefile, case_id, chart_result, "Attending unavailable"
            )
            break
        except RuntimeError as e:
            if attempt < retries - 1:
                delay = backoff * (2**attempt)
                log.warning(
                    "Attending retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    retries,
                    delay,
                    e,
                )
                time.sleep(delay)
            else:
                log.warning("Attending exhausted %d retries: %s", retries, e)
                _write_degraded_observation(
                    casefile, case_id, chart_result, "Attending unavailable"
                )

    # --- Stage 3: Scribe always runs ---
    scribe.report(case_id)


def _write_degraded_observation(
    casefile: Any,
    case_id: int,
    chart_result: dict[str, Any],
    note: str,
) -> None:
    """Write partial analysis to case row when a pipeline stage fails."""
    observation = json.dumps(chart_result, default=str)
    casefile.execute(
        """
        UPDATE cases SET
            observation = ?,
            hypothesis = ?,
            recommended_action = 'Review preliminary data manually',
            reasoning_chain = ?,
            baseline_deviation_pct = ?,
            updated_at = current_timestamp
        WHERE case_id = ?
        """,
        [
            observation,
            f"Preliminary analysis ({note})",
            note,
            chart_result.get("deviation_pct"),
            case_id,
        ],
    )
