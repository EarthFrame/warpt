"""Tests for warpt.daemon.agents.ollama_client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from warpt.daemon.agents.ollama_client import (
    OllamaClient,
    OllamaPermanentError,
    get_installed_models,
    retry_generate,
)


def test_generate_returns_text_on_success():
    """generate() returns the response text from Ollama."""
    client = OllamaClient(model="llama3:8b")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "GPU is overheating"}
    mock_resp.raise_for_status = MagicMock()

    target = "warpt.daemon.agents.ollama_client.requests.post"
    with patch(target, return_value=mock_resp):
        result = client.generate("What is wrong with this GPU?")

    assert result == "GPU is overheating"


def test_generate_raises_on_connection_error():
    """generate() raises RuntimeError when Ollama is unreachable."""
    client = OllamaClient(model="llama3:8b")

    with patch(
        "warpt.daemon.agents.ollama_client.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            client.generate("hello")


def test_generate_raises_on_timeout():
    """generate() raises RuntimeError when Ollama times out."""
    client = OllamaClient(model="llama3:8b")

    with patch(
        "warpt.daemon.agents.ollama_client.requests.post",
        side_effect=requests.Timeout("timed out"),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            client.generate("hello")


def test_get_installed_models_returns_list():
    """get_installed_models() returns model names from Ollama."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
    }
    mock_resp.raise_for_status = MagicMock()

    target = "warpt.daemon.agents.ollama_client.requests.get"
    with patch(target, return_value=mock_resp):
        models = get_installed_models()

    assert models == ["llama3:8b", "mistral:7b"]


def test_get_installed_models_returns_none_when_unreachable():
    """get_installed_models() returns None when Ollama can't be reached."""
    with patch(
        "warpt.daemon.agents.ollama_client.requests.get",
        side_effect=requests.ConnectionError("refused"),
    ):
        result = get_installed_models()

    assert result is None


def test_retry_generate_succeeds_on_second_try():
    """retry_generate() retries after failure and returns on success."""
    client = OllamaClient(model="llama3:8b")
    call_count = 0

    def fake_generate(_prompt, system_prompt=None):  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("temporary failure")
        return "recovered answer"

    client.generate = fake_generate

    with patch("warpt.daemon.agents.ollama_client.time.sleep"):
        result = retry_generate(client, "test prompt")

    assert result == "recovered answer"
    assert call_count == 2


def test_retry_generate_exhausts_retries():
    """retry_generate() raises RuntimeError after all retries fail."""
    client = OllamaClient(model="llama3:8b")
    client.generate = MagicMock(side_effect=RuntimeError("always fails"))

    with patch("warpt.daemon.agents.ollama_client.time.sleep"):
        with pytest.raises(RuntimeError, match="always fails"):
            retry_generate(client, "test prompt", retries=3)

    assert client.generate.call_count == 3


def test_permanent_error_not_retried():
    """retry_generate() re-raises OllamaPermanentError without retrying."""
    client = OllamaClient(model="llama3:8b")
    client.generate = MagicMock(side_effect=OllamaPermanentError("model not found"))

    with patch("warpt.daemon.agents.ollama_client.time.sleep"):
        with pytest.raises(OllamaPermanentError, match="model not found"):
            retry_generate(client, "test prompt", retries=3)

    # Only called once — no retries on permanent error
    assert client.generate.call_count == 1


def test_generate_raises_permanent_on_404():
    """generate() raises OllamaPermanentError when Ollama returns HTTP 404."""
    client = OllamaClient(model="nonexistent:model")

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    http_error = requests.HTTPError(response=mock_resp)
    mock_resp.raise_for_status.side_effect = http_error

    with patch(
        "warpt.daemon.agents.ollama_client.requests.post",
        return_value=mock_resp,
    ):
        with pytest.raises(OllamaPermanentError, match="not found"):
            client.generate("hello")
