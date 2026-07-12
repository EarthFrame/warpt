"""Remediation seam — propose → policy → audit, with ZERO executable actions.

This package deliberately ships **empty of behavior**: it defines the action
lifecycle interface, a deny-by-default policy engine, and an audit log so the
diagnosis pipeline can *propose and record* actions today. Executable
remediation (throttle, cordon, kill, restart) is a later phase — when it
lands, actions implement ``execute()``/``verify()``/``rollback()`` and policy
flips, with no pipeline rewrite.
"""

from warpt.daemon.remediation.audit import AuditLog
from warpt.daemon.remediation.base import (
    STATUS_PROPOSED,
    ProposedAction,
    RemediationAction,
)
from warpt.daemon.remediation.policy import PolicyDecision, PolicyEngine

__all__ = [
    "STATUS_PROPOSED",
    "AuditLog",
    "PolicyDecision",
    "PolicyEngine",
    "ProposedAction",
    "RemediationAction",
]
