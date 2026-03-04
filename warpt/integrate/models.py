"""Data models for the integration pipeline.

Defines the structured questions document that enables communication
between the agent, the vendor engineer, and the iteration loop.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QuestionTier(str, Enum):
    """Priority tier for integration questions.

    Parameters
    ----------
    BLOCKING : str
        Cannot proceed without an answer.
    DEFAULTED : str
        Agent made a reasonable default, confirm or override.
    CLARIFICATION_NEEDED : str
        Agent needs more context to make a good decision.
    INFORMATIONAL : str
        FYI for the vendor, no action required.
    """

    BLOCKING = "blocking"
    DEFAULTED = "defaulted"
    CLARIFICATION_NEEDED = "clarification_needed"
    INFORMATIONAL = "informational"


class QuestionStatus(str, Enum):
    """Lifecycle status of a question.

    Parameters
    ----------
    OPEN : str
        Awaiting human response.
    ANSWERED : str
        Human provided an answer, agent has not yet processed.
    IMPLEMENTED : str
        Agent processed the answer into code.
    VERIFIED : str
        Code changes pass tests and linting.
    """

    OPEN = "open"
    ANSWERED = "answered"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


class Question(BaseModel):
    """A single integration question logged by the agent."""

    id: int
    tier: QuestionTier
    status: QuestionStatus = QuestionStatus.OPEN
    title: str
    finding: str = Field(
        description="What the agent found in the SDK docs"
    )
    decision: str = Field(
        description="What decision was made, or why none could be"
    )
    alternatives: str = Field(
        description="Available options or approaches"
    )
    impact: str = Field(
        description="Downstream impact on warpt integration"
    )
    code_reference: str | None = Field(
        default=None,
        description="file:line reference in generated code",
    )
    answer: str | None = Field(
        default=None,
        description="Human-provided answer (filled during iterate)",
    )
    notes: str | None = Field(
        default=None,
        description="Additional notes from human or agent",
    )


class QuestionsDocument(BaseModel):
    """The full questions document, serialized as YAML."""

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metadata: vendor, sdk_source, created, "
            "session_id, pass_number"
        ),
    )
    questions: list[Question] = Field(default_factory=list)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data = self.model_dump(mode="json")
        return yaml.dump(
            data, default_flow_style=False, sort_keys=False
        )

    @classmethod
    def from_yaml(cls, text: str) -> QuestionsDocument:
        """Deserialize from YAML string."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    def next_id(self) -> int:
        """Get the next available question ID."""
        if not self.questions:
            return 1
        return max(q.id for q in self.questions) + 1

    def get_by_status(
        self, status: QuestionStatus
    ) -> list[Question]:
        """Filter questions by status."""
        return [q for q in self.questions if q.status == status]

    def get_by_id(self, question_id: int) -> Question | None:
        """Find a question by ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def summary(self) -> dict[str, int]:
        """Count questions by tier and status."""
        counts: dict[str, int] = {}
        for q in self.questions:
            tier_key = f"{q.tier.value}:{q.status.value}"
            counts[tier_key] = counts.get(tier_key, 0) + 1
        return counts
