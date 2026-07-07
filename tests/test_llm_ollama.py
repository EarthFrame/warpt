"""Tests for warpt.daemon.llm.providers.ollama."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from warpt.daemon.llm.base import LLMError, LLMPermanentError
from warpt.daemon.llm.providers.ollama import (
    OllamaProvider,
    get_installed_models,
    is_model_installed,
)

_POST = "warpt.daemon.llm.providers.ollama.requests.post"
_GET = "warpt.daemon.llm.providers.ollama.requests.get"


def _chat_response(content, prompt_tokens=10, eval_tokens=5):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_generate_returns_response_on_success():
    """generate() returns text, model, provider, and token usage."""
    provider = OllamaProvider(model="llama3:8b", name="local")

    with patch(_POST, return_value=_chat_response("GPU is overheating")) as post:
        resp = provider.generate(
            [{"role": "user", "content": "What is wrong?"}], system="Be brief."
        )

    assert resp.text == "GPU is overheating"
    assert resp.model == "llama3:8b"
    assert resp.provider == "local"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
    assert resp.structured is None

    payload = post.call_args.kwargs["json"]
    assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
    assert payload["messages"][1] == {"role": "user", "content": "What is wrong?"}
    assert payload["stream"] is False
    assert "format" not in payload


def test_generate_raises_llm_error_on_connection_error():
    """Generate raises llm error on connection error."""
    provider = OllamaProvider(model="llama3:8b")

    with patch(_POST, side_effect=requests.ConnectionError("refused")):
        with pytest.raises(LLMError, match="Cannot connect to Ollama"):
            provider.generate([{"role": "user", "content": "hello"}])


def test_generate_raises_llm_error_on_timeout():
    """Generate raises llm error on timeout."""
    provider = OllamaProvider(model="llama3:8b")

    with patch(_POST, side_effect=requests.Timeout("timed out")):
        with pytest.raises(LLMError, match="timed out"):
            provider.generate([{"role": "user", "content": "hello"}])


def test_generate_raises_permanent_on_404():
    """HTTP 404 (model not pulled) is a permanent, non-retryable error."""
    provider = OllamaProvider(model="nonexistent:model")

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    http_error = requests.HTTPError(response=mock_resp)
    mock_resp.raise_for_status.side_effect = http_error

    with patch(_POST, return_value=mock_resp):
        with pytest.raises(LLMPermanentError, match="not found"):
            provider.generate([{"role": "user", "content": "hello"}])


def test_generate_raises_llm_error_on_malformed_body():
    """Generate raises llm error on malformed body."""
    provider = OllamaProvider(model="llama3:8b")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"unexpected": "shape"}
    mock_resp.raise_for_status = MagicMock()

    with patch(_POST, return_value=mock_resp):
        with pytest.raises(LLMError, match="Unexpected response"):
            provider.generate([{"role": "user", "content": "hello"}])


_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
}


def test_generate_with_schema_returns_structured():
    """response_schema is forwarded as Ollama's format and output is parsed."""
    provider = OllamaProvider(model="llama3:8b")
    content = json.dumps({"verdict": "healthy"})

    with patch(_POST, return_value=_chat_response(content)) as post:
        resp = provider.generate(
            [{"role": "user", "content": "diagnose"}], response_schema=_SCHEMA
        )

    assert resp.structured == {"verdict": "healthy"}
    assert post.call_args.kwargs["json"]["format"] == _SCHEMA


def test_generate_with_schema_retries_then_succeeds():
    """A schema miss is retried; the valid second response is returned."""
    provider = OllamaProvider(model="llama3:8b")
    responses = [
        _chat_response("not json"),
        _chat_response(json.dumps({"verdict": "degraded"})),
    ]

    with patch(_POST, side_effect=responses) as post:
        resp = provider.generate(
            [{"role": "user", "content": "diagnose"}], response_schema=_SCHEMA
        )

    assert resp.structured == {"verdict": "degraded"}
    assert post.call_count == 2


def test_generate_with_schema_exhausts_attempts():
    """Persistent schema misses raise LLMError after 3 attempts."""
    provider = OllamaProvider(model="llama3:8b")

    with patch(_POST, return_value=_chat_response("still not json")) as post:
        with pytest.raises(LLMError, match="schema validation"):
            provider.generate(
                [{"role": "user", "content": "diagnose"}], response_schema=_SCHEMA
            )

    assert post.call_count == 3


def test_get_installed_models_returns_list():
    """Get installed models returns list."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch(_GET, return_value=mock_resp):
        models = get_installed_models()

    assert models == ["llama3:8b", "mistral:7b"]


def test_get_installed_models_returns_none_when_unreachable():
    """Get installed models returns none when unreachable."""
    with patch(_GET, side_effect=requests.ConnectionError("refused")):
        assert get_installed_models() is None


def test_is_model_installed():
    """Is model installed."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
    mock_resp.raise_for_status = MagicMock()

    with patch(_GET, return_value=mock_resp):
        assert is_model_installed("llama3:8b") is True
        assert is_model_installed("mistral:7b") is False
