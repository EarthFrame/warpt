"""Ollama provider — local pulled models behind the LLMProvider protocol."""

from __future__ import annotations

from typing import Any

import requests

from warpt.daemon.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMResponse,
    LLMSchemaError,
    parse_structured_text,
)
from warpt.utils.logger import Logger

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0

OLLAMA_CONNECTION_ERROR = "Cannot connect to Ollama. Is it running? Try: ollama serve"
OLLAMA_TIMEOUT_ERROR = "Ollama request timed out."
OLLAMA_HTTP_ERROR = "Ollama HTTP error: {error}"
OLLAMA_UNEXPECTED_RESPONSE = "Unexpected response from Ollama."
OLLAMA_MODEL_NOT_FOUND = (
    "Model not found (HTTP 404). Is it pulled? Try: ollama pull {model}"
)

_SCHEMA_ATTEMPTS = 3


class OllamaProvider:
    """LLMProvider backed by a local Ollama server (``/api/chat``).

    Parameters
    ----------
    model
        Ollama model name (e.g. ``"llama3:8b"``).
    url
        Base URL for the Ollama server.
    name
        Configured provider name for logs/metrics.
    timeout
        Default request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        url: str = DEFAULT_OLLAMA_URL,
        name: str = "ollama",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._url = url
        self._name = name
        self._timeout = timeout
        self._log = Logger.get("daemon.llm.ollama")

    @property
    def name(self) -> str:
        """Configured provider name."""
        return self._name

    @property
    def model(self) -> str:
        """Configured model identifier."""
        return self._model

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Generate a completion via Ollama's chat endpoint.

        With ``response_schema``, Ollama's structured-output ``format`` field
        constrains generation and the result is locally validated, retrying
        up to 3 attempts on schema misses.
        """
        chat_messages = list(messages)
        if system:
            chat_messages = [{"role": "system", "content": system}, *chat_messages]

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "stream": False,
        }
        if response_schema is not None:
            payload["format"] = response_schema

        attempts = _SCHEMA_ATTEMPTS if response_schema is not None else 1
        last_schema_error: ValueError | None = None
        for attempt in range(attempts):
            text, response = self._post_chat(payload, timeout or self._timeout)
            if response_schema is None:
                return response
            try:
                response.structured = parse_structured_text(text, response_schema)
                return response
            except ValueError as e:
                last_schema_error = e
                self._log.warning(
                    "Ollama schema miss (attempt %d/%d): %s", attempt + 1, attempts, e
                )
        raise LLMSchemaError(
            f"Ollama output failed schema validation after {attempts} attempts: "
            f"{last_schema_error}"
        )

    def _post_chat(
        self, payload: dict[str, Any], timeout: float
    ) -> tuple[str, LLMResponse]:
        """POST to ``/api/chat`` and map transport errors to LLM errors."""
        try:
            resp = requests.post(f"{self._url}/api/chat", json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            text = body["message"]["content"]
        except requests.ConnectionError as e:
            raise LLMError(OLLAMA_CONNECTION_ERROR) from e
        except requests.Timeout as e:
            raise LLMError(OLLAMA_TIMEOUT_ERROR) from e
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise LLMPermanentError(
                    OLLAMA_MODEL_NOT_FOUND.format(model=self._model)
                ) from e
            raise LLMError(OLLAMA_HTTP_ERROR.format(error=e)) from e
        except (KeyError, TypeError, ValueError) as e:
            raise LLMError(OLLAMA_UNEXPECTED_RESPONSE) from e
        except requests.RequestException as e:
            raise LLMError(OLLAMA_HTTP_ERROR.format(error=e)) from e

        return text, LLMResponse(
            text=text,
            model=self._model,
            provider=self._name,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )


def get_installed_models(url: str = DEFAULT_OLLAMA_URL) -> list[str] | None:
    """Query Ollama for installed models.

    Returns
    -------
        List of model names, or ``None`` if Ollama is unreachable.
    """
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.RequestException:
        return None


def is_model_installed(model: str, url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check whether *model* is installed in Ollama."""
    installed = get_installed_models(url)
    if installed is None:
        return False
    return model in installed
