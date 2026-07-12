"""Deny-by-default policy engine for proposed actions.

Rules in this phase (nothing executes regardless of verdict):

- ``diagnostic_probe`` — allowed **only** when ``remediation.probes.enabled``
  is truthy in config; denied otherwise. The probe tool additionally enforces
  its own idle check before running.
- Every other ``action_type`` — ``require_approval``. This is the
  deny-by-default posture: LLM-recommended actions are recorded with their
  verdict for the audit trail, and a human (or a future approval flow)
  decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from warpt.daemon.remediation.base import ProposedAction
from warpt.utils.logger import Logger

VERDICT_ALLOW = "allow"
VERDICT_DENY = "deny"
VERDICT_REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyDecision:
    """Outcome of a policy evaluation.

    Parameters
    ----------
    verdict
        One of ``"allow"``, ``"deny"``, ``"require_approval"``.
    reason
        Human-readable explanation for the audit trail.
    """

    verdict: str
    reason: str


class PolicyEngine:
    """Evaluates proposed actions against the daemon's remediation config.

    Parameters
    ----------
    config
        Full daemon config dict; reads the ``remediation`` block.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._remediation = config.get("remediation", {}) or {}
        self._log = Logger.get("daemon.remediation.policy")

    def evaluate(self, action: ProposedAction) -> PolicyDecision:
        """Return the policy verdict for *action*.

        Parameters
        ----------
        action
            The proposed action to evaluate.
        """
        if action.action_type == "diagnostic_probe":
            probes = self._remediation.get("probes", {}) or {}
            if probes.get("enabled"):
                decision = PolicyDecision(
                    verdict=VERDICT_ALLOW,
                    reason="diagnostic probes are enabled in config",
                )
            else:
                decision = PolicyDecision(
                    verdict=VERDICT_DENY,
                    reason=(
                        "diagnostic probes are disabled in config "
                        "(remediation.probes.enabled: false)"
                    ),
                )
        else:
            decision = PolicyDecision(
                verdict=VERDICT_REQUIRE_APPROVAL,
                reason=(
                    f"action type {action.action_type!r} requires approval "
                    "(no actions execute in this phase)"
                ),
            )
        self._log.debug(
            "Policy: %s on %s -> %s (%s)",
            action.action_type,
            action.target,
            decision.verdict,
            decision.reason,
        )
        return decision
