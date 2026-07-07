"""Core types for the LLM provider abstraction.

Message shape is a plain list of ``{"role": ..., "content": ...}`` dicts so
no provider SDK types leak into agent code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    """

    text: str
    model: str
    provider: str
    structured: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    """Protocol every inference backend implements.

    ``tools`` support is deliberately absent until the Phase-2 agent loop;
    the seam is ``generate()`` with optional schema-validated output.
    """

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
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Generate a completion for *messages*.

        When ``response_schema`` is given, the provider must return an
        ``LLMResponse`` whose ``structured`` field validates against the
        schema, or raise ``LLMError`` after bounded internal retries.
        """
        ...


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
