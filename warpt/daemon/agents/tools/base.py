"""Tool protocol for agent evidence gathering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class ToolError(RuntimeError):
    """Tool failed to run (bad arguments, missing data, backend error)."""


class ToolDeniedError(ToolError):
    """Tool run was denied by policy or a safety check (e.g. busy GPU)."""


class Tool(ABC):
    """A named, schema-described capability an agent may invoke.

    Subclasses set ``name``, ``description``, and ``input_schema`` (a JSON
    schema for the arguments object) and implement ``run()``.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with *args* and return a JSON-safe result.

        Raises
        ------
        ToolDeniedError
            When policy or a safety check refuses the run.
        ToolError
            On any other failure.
        """
