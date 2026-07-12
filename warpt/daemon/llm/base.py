"""Core types for the LLM provider abstraction.

Message shape is a plain list of ``{"role": ..., "content": ...}`` dicts so
no provider SDK types leak into agent code.

Canonical message shapes (provider-agnostic — each provider maps these to
its own wire format):

- ``{"role": "system"|"user"|"assistant", "content": str}``
- Assistant turn that called tools::

    {"role": "assistant", "content": str,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}]}

- Tool result turn::

    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

where ``content`` on a tool result is the JSON-serialized tool outcome.

``tools`` and ``response_schema`` are mutually exclusive on a single
``generate()`` call — agent loops gather evidence with ``tools`` and then
make a final, schema-constrained conclude call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Retryable LLM failure (connection, timeout, rate limit, 5xx)."""


class LLMPermanentError(LLMError):
    """Non-retryable LLM failure (missing model, bad auth, bad config)."""


class LLMSchemaError(LLMError):
    """Model output failed schema validation after bounded retries.

    Transport succeeded but the model could not produce schema-valid output;
    callers may degrade to a fallback response instead of retrying transport.
    """


@dataclass(frozen=True)
class ToolDef:
    """Definition of a tool the model may call during generation.

    Parameters
    ----------
    name
        Tool name (matches the agent-side tool registry).
    description
        What the tool does — shown to the model.
    input_schema
        JSON schema for the tool's arguments object.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    Parameters
    ----------
    id
        Provider-assigned call id, or a synthesized ``"call_<n>"`` for
        backends that don't issue ids.
    name
        Name of the tool to invoke.
    arguments
        Parsed arguments object for the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Result of a single ``LLMProvider.generate()`` call.

    Parameters
    ----------
    text
        Raw generated text.
    structured
        Parsed object validated against the requested ``response_schema``,
        or ``None`` when no schema was requested.
    model
        Model that produced the response.
    provider
        Configured provider name (e.g. ``"local"``, ``"claude"``).
    input_tokens
        Prompt tokens consumed, if the backend reports usage.
    output_tokens
        Completion tokens generated, if the backend reports usage.
    tool_calls
        Tool invocations the model requested; empty when none.
    stop_reason
        Why generation stopped — ``"tool_use"`` when tool calls were
        returned; otherwise provider-specific or ``None``.
    """

    text: str
    model: str
    provider: str
    structured: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


class LLMProvider(Protocol):
    """Protocol every inference backend implements."""

    @property
    def name(self) -> str:
        """Configured provider name (used in logs and metrics)."""
        ...

    @property
    def model(self) -> str:
        """Model identifier this provider is configured with."""
        ...

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        tools: list[ToolDef] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Generate a completion for *messages*.

        When ``response_schema`` is given, the provider must return an
        ``LLMResponse`` whose ``structured`` field validates against the
        schema, or raise ``LLMError`` after bounded internal retries.

        When ``tools`` is given, the provider may return ``tool_calls``
        instead of (or alongside) text. ``tools`` and ``response_schema``
        are mutually exclusive — passing both raises ``LLMPermanentError``.
        """
        ...


def check_tools_schema_exclusive(
    tools: list[ToolDef] | None, response_schema: dict[str, Any] | None
) -> None:
    """Raise if both ``tools`` and ``response_schema`` were passed.

    Raises
    ------
    LLMPermanentError
        When both are non-None.
    """
    if tools is not None and response_schema is not None:
        raise LLMPermanentError("tools and response_schema are mutually exclusive")


def validate_json_schema(data: Any, schema: dict[str, Any]) -> list[str]:
    """Validate *data* against a minimal JSON-schema subset.

    Supports ``type``, ``properties``, ``required``, ``items``, and ``enum``
    — enough for structured diagnosis output without a jsonschema dependency.

    Returns
    -------
        List of human-readable violations; empty means valid.
    """
    errors: list[str] = []
    _validate(data, schema, "$", errors)
    return errors


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _validate(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS.get(t, lambda _v: True)(data) for t in allowed):
            errors.append(
                f"{path}: expected type {expected}, got {type(data).__name__}"
            )
            return

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} not in enum {schema['enum']}")

    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}: missing required key {key!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                _validate(data[key], subschema, f"{path}.{key}", errors)

    if isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            _validate(item, schema["items"], f"{path}[{i}]", errors)


def parse_structured_text(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse *raw* as JSON and validate against *schema*.

    Raises
    ------
    ValueError
        If the text is not valid JSON, not an object, or fails validation.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    violations = validate_json_schema(data, schema)
    if violations:
        raise ValueError("schema violations: " + "; ".join(violations))
    return data
