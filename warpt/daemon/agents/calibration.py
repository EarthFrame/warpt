"""Deterministic confidence calibration for Attending diagnoses.

Retires the ``-1.23`` sentinel: confidence is now computed from the LLM's
self-report blended with evidence quality. The formula is deliberately
simple, additive, and fully deterministic given fixed inputs:

``base``
    The LLM's self-reported confidence clamped to [0, 100]; when the LLM
    reported none, a neutral base of 40.

Adjustments (additive):

- **Deviation z-score** — how far the current value sits from the
  hour-of-day baseline, in stddevs: ``z >= 3`` +15, ``z >= 2`` +10,
  ``z >= 1`` +5, ``z < 1`` -10 (the anomaly is not statistically unusual),
  unknown 0.
- **Corroborating signals** — other metrics deviating alongside the breach
  metric and/or recent throttle reasons: +5 each, capped at +15.
- **Diagnostic probe** — ran and corroborated the hypothesis +15; ran and
  found nothing -20; not run 0.
- **Prior cases** — two or more prior cases on the same GPU +5.

The result clamps to [5, 95]: never certain, never zero. First-cut weights —
revisit with real-world calibration data.
"""

from __future__ import annotations


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def calibrate_confidence(
    *,
    llm_confidence: int | None,
    deviation_z: float | None,
    corroborating_signals: int,
    probe_ran: bool,
    probe_corroborates: bool | None,
    prior_case_count: int,
) -> int:
    """Compute calibrated diagnosis confidence (5-95).

    Parameters
    ----------
    llm_confidence
        The LLM's self-reported confidence (0-100), or ``None``.
    deviation_z
        ``|current - hour_mean| / hour_stddev``, or ``None`` when no
        hour-of-day profile (or zero stddev) exists.
    corroborating_signals
        Count of corroborating signals (other deviating metrics, recent
        throttle reasons).
    probe_ran
        Whether a diagnostic probe executed for this case.
    probe_corroborates
        Whether the probe corroborated the hypothesis; ``None`` when not
        run or inconclusive.
    prior_case_count
        Number of prior cases recorded for the same GPU.
    """
    base = _clamp(llm_confidence, 0, 100) if llm_confidence is not None else 40.0

    adjustment = 0.0
    if deviation_z is not None:
        if deviation_z >= 3:
            adjustment += 15
        elif deviation_z >= 2:
            adjustment += 10
        elif deviation_z >= 1:
            adjustment += 5
        else:
            adjustment -= 10

    adjustment += min(corroborating_signals * 5, 15)

    if probe_ran:
        adjustment += 15 if probe_corroborates else -20

    if prior_case_count >= 2:
        adjustment += 5

    return int(_clamp(base + adjustment, 5, 95))
