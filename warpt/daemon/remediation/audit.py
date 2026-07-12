"""Audit log — every proposed action is persisted, whatever the verdict."""

from __future__ import annotations

import json

from warpt.daemon.casefile import CaseFile
from warpt.daemon.remediation.base import STATUS_PROPOSED, ProposedAction
from warpt.daemon.remediation.policy import PolicyDecision
from warpt.utils.logger import Logger


class AuditLog:
    """Writes proposed actions and their policy verdicts to the DB.

    Parameters
    ----------
    casefile
        CaseFile instance for database writes.
    """

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile
        self._log = Logger.get("daemon.remediation.audit")

    def record(
        self,
        action: ProposedAction,
        decision: PolicyDecision,
        status: str = STATUS_PROPOSED,
    ) -> int:
        """Persist *action* with its policy *decision*; return the action_id.

        Parameters
        ----------
        action
            The proposed action.
        decision
            The policy verdict for it.
        status
            Lifecycle status — only ``"proposed"`` is written this phase.
        """
        self._casefile.execute(
            """
            INSERT INTO actions (case_id, action_type, target, parameters,
                                 reason, policy_verdict, status)
            VALUES (?, ?, ?, ?::JSON, ?, ?, ?)
            """,
            [
                action.case_id,
                action.action_type,
                action.target,
                json.dumps(action.parameters, default=str),
                action.reason,
                decision.verdict,
                status,
            ],
        )
        rows = self._casefile.query("SELECT max(action_id) FROM actions")
        action_id = rows[0][0]
        self._log.info(
            "Action recorded: #%s %s on %s [verdict=%s]",
            action_id,
            action.action_type,
            action.target,
            decision.verdict,
        )
        return action_id
