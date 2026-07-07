"""Tests for warpt.daemon.llm.base — types, errors, schema validation."""

import pytest

from warpt.daemon.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMResponse,
    parse_structured_text,
    validate_json_schema,
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "confidence": {"type": "integer"},
        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hypothesis", "confidence"],
}


def test_permanent_error_is_llm_error():
    """LLMPermanentError is catchable as LLMError and RuntimeError."""
    err = LLMPermanentError("model not found")
    assert isinstance(err, LLMError)
    assert isinstance(err, RuntimeError)


def test_llm_response_defaults():
    """Optional fields default to None."""
    resp = LLMResponse(text="hi", model="m", provider="p")
    assert resp.structured is None
    assert resp.input_tokens is None
    assert resp.output_tokens is None


def test_validate_json_schema_accepts_valid():
    """Validate json schema accepts valid."""
    data = {
        "hypothesis": "thermal throttling",
        "confidence": 80,
        "severity": "warning",
        "evidence": ["temp 92C"],
    }
    assert validate_json_schema(data, _SCHEMA) == []


def test_validate_json_schema_missing_required():
    """Validate json schema missing required."""
    errors = validate_json_schema({"hypothesis": "x"}, _SCHEMA)
    assert any("confidence" in e for e in errors)


def test_validate_json_schema_wrong_type():
    """Validate json schema wrong type."""
    errors = validate_json_schema({"hypothesis": 42, "confidence": 80}, _SCHEMA)
    assert any("hypothesis" in e for e in errors)


def test_validate_json_schema_enum_violation():
    """Validate json schema enum violation."""
    errors = validate_json_schema(
        {"hypothesis": "x", "confidence": 1, "severity": "catastrophic"}, _SCHEMA
    )
    assert any("enum" in e for e in errors)


def test_validate_json_schema_bad_array_item():
    """Validate json schema bad array item."""
    errors = validate_json_schema(
        {"hypothesis": "x", "confidence": 1, "evidence": ["ok", 7]}, _SCHEMA
    )
    assert any("evidence[1]" in e for e in errors)


def test_validate_json_schema_bool_is_not_integer():
    """Validate json schema bool is not integer."""
    errors = validate_json_schema({"hypothesis": "x", "confidence": True}, _SCHEMA)
    assert any("confidence" in e for e in errors)


def test_parse_structured_text_valid():
    """Parse structured text valid."""
    raw = '{"hypothesis": "load spike", "confidence": 70}'
    assert parse_structured_text(raw, _SCHEMA)["confidence"] == 70


def test_parse_structured_text_invalid_json():
    """Parse structured text invalid json."""
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_structured_text("not json", _SCHEMA)


def test_parse_structured_text_non_object():
    """Parse structured text non object."""
    with pytest.raises(ValueError, match="expected JSON object"):
        parse_structured_text("[1, 2]", _SCHEMA)


def test_parse_structured_text_schema_violation():
    """Parse structured text schema violation."""
    with pytest.raises(ValueError, match="schema violations"):
        parse_structured_text('{"hypothesis": "x"}', _SCHEMA)
