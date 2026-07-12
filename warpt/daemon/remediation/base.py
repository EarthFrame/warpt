"""Action lifecycle types for the remediation seam.

Lifecycle: ``propose → policy_check → (approve) → execute → verify →
rollback``. In this phase only the first two steps exist in the pipeline —
actions are proposed and policy-checked, then recorded to the audit log.
**No** ``RemediationAction`` implementations ship yet; that is a deliberate
design decision (see ER_PRODUCTION_PLAN.md §2f), not an omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# The only action status written in this phase. Executable statuses
# (approved/executed/verified/rolled_back) arrive with the actions themselves.
STATUS_PROPOSED = "proposed"


@dataclass
class ProposedAction:
    """An action the diagnosis pipeline proposes — never executes.

    Parameters
    ----------
    action_type
        Kind of action (e.g. ``"diagnostic_probe"``, ``"investigate"``).
    target
        What the action applies to (GPU guid, node, category).
    parameters
        Structured action parameters.
    reason
        Why the action is proposed (typically the diagnosis hypothesis).
    case_id
        Case this proposal belongs to, when applicable.
    """

    action_type: str
    target: str | None
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    case_id: int | None = None


class RemediationAction(Protocol):
    """Protocol future executable actions implement.

    Deliberately has **no implementations** in this phase — the seam exists
    so on-node actions plug in later without a pipeline rewrite.
    """

    @property
    def name(self) -> str:
        """Action name (matches ``ProposedAction.action_type``)."""
        ...

    def execute(self, action: ProposedAction) -> dict[str, Any]:
        """Perform the action; return a structured result."""
        ...

    def verify(self) -> bool:
        """Confirm the action had the intended effect."""
        ...

    def rollback(self) -> None:
        """Undo the action if verification failed."""
        ...
