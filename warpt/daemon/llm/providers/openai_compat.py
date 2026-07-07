"""OpenAI-compatible provider — vLLM / TGI / self-hosted inference clusters.

Speaks ``POST {url}/chat/completions`` with optional bearer auth. Structured
output first tries the OpenAI ``response_format`` json_schema shape; backends
that reject it fall back to schema-in-prompt emulation. Output is always
locally validated either way.
"""

from __future__ import annotations

import json
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

DEFAULT_TIMEOUT = 120.0

_SCHEMA_ATTEMPTS = 3

_SCHEMA_PROMPT_SUFFIX = (
    "\n\nRespond with ONLY a JSON object matching this JSON schema "
    "(no prose, no code fences):\n{schema}"
)


class OpenAICompatProvider:
    """LLMProvider for OpenAI-compatible HTTP endpoints.

    Parameters
    ----------
    model
        Model name served by the cluster (e.g. ``"llama3:70b"``).
    url
        Base URL including the API prefix (e.g. ``http://host/v1``).
    api_key
        Optional bearer token resolved from env/secret file.
    name
        Configured provider name for logs/metrics.
    timeout
        Default request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        url: str,
        api_key: str | None = None,
        name: str = "openai_compat",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._name = name
        self._timeout = timeout
        self._log = Logger.get("daemon.llm.openai_compat")
        # Set on the first 4xx rejection of response_format so later calls
        # go straight to prompt emulation.
        self._response_format_unsupported = False

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
        """Generate a completion via ``/chat/completions``.

        With ``response_schema``, tries native ``response_format`` first,
        degrading to schema-in-prompt emulation on backends that reject it;
        output is locally validated with bounded retries either way.
        """
        chat_messages = list(messages)
        if system:
            chat_messages = [{"role": "system", "content": system}, *chat_messages]

        if response_schema is None:
            text, response = self._post(chat_messages, None, timeout)
            return response

        attempts = _SCHEMA_ATTEMPTS
        last_schema_error: ValueError | None = None
        for attempt in range(attempts):
            if self._response_format_unsupported:
                text, response = self._post(
                    _with_schema_prompt(chat_messages, response_schema), None, timeout
                )
            else:
                try:
                    text, response = self._post(chat_messages, response_schema, timeout)
                except LLMPermanentError:
                    # Backend rejected response_format — emulate via prompt.
                    self._response_format_unsupported = True
                    self._log.info(
                        "Backend rejected response_format; falling back to "
                        "schema-in-prompt emulation"
                    )
                    text, response = self._post(
                        _with_schema_prompt(chat_messages, response_schema),
                        None,
                        timeout,
                    )
            try:
                response.structured = parse_structured_text(text, response_schema)
                return response
            except ValueError as e:
                last_schema_error = e
                self._log.warning(
                    "Cluster schema miss (attempt %d/%d): %s", attempt + 1, attempts, e
                )
        raise LLMSchemaError(
            f"Cluster output failed schema validation after {attempts} attempts: "
            f"{last_schema_error}"
        )

    def _post(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None,
        timeout: float | None,
    ) -> tuple[str, LLMResponse]:
        """POST to ``/chat/completions`` and map errors to LLM errors."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = requests.post(
                f"{self._url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout or self._timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
        except requests.ConnectionError as e:
            raise LLMError(f"Cannot connect to inference cluster {self._url}") from e
        except requests.Timeout as e:
            raise LLMError("Inference cluster request timed out") from e
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (400, 401, 403, 404, 422):
                raise LLMPermanentError(
                    f"Inference cluster rejected the request (HTTP {status}): {e}"
                ) from e
            raise LLMError(f"Inference cluster HTTP error: {e}") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError("Unexpected response from inference cluster") from e
        except requests.RequestException as e:
            raise LLMError(f"Inference cluster HTTP error: {e}") from e

        usage = body.get("usage") or {}
        return text, LLMResponse(
            text=text,
            model=body.get("model", self._model),
            provider=self._name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def _with_schema_prompt(
    messages: list[dict[str, str]], schema: dict[str, Any]
) -> list[dict[str, str]]:
    """Append the schema instruction to the last user message (copy)."""
    result = [dict(m) for m in messages]
    suffix = _SCHEMA_PROMPT_SUFFIX.format(schema=json.dumps(schema))
    for msg in reversed(result):
        if msg["role"] == "user":
            msg["content"] = msg["content"] + suffix
            break
    return result
