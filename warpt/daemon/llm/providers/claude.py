"""Claude (Anthropic API) provider — the default cloud backend.

The ``anthropic`` SDK is imported lazily so Ollama-only nodes can run the
daemon without it installed. Structured output uses the API's JSON-schema
enforcement (``output_config.format``), then local validation as a belt.
"""

from __future__ import annotations

import copy
from typing import Any

from warpt.daemon.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMResponse,
    LLMSchemaError,
    parse_structured_text,
)
from warpt.utils.logger import Logger

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0

_SCHEMA_ATTEMPTS = 3

_INSTALL_HINT = (
    "The 'anthropic' package is required for the Claude provider. "
    "Install it with: pip install anthropic"
)


class ClaudeProvider:
    """LLMProvider backed by the Anthropic Messages API.

    Parameters
    ----------
    model
        Anthropic model ID (e.g. ``"claude-sonnet-5"``).
    api_key
        API key resolved from env/secret file (never from config).
    name
        Configured provider name for logs/metrics.
    max_tokens
        Per-response output token cap.
    timeout
        Default request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_key: str | None = None,
        name: str = "claude",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise LLMPermanentError(_INSTALL_HINT) from e

        self._anthropic = anthropic
        self._model = model
        self._name = name
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._log = Logger.get("daemon.llm.claude")
        # Retry policy lives in the budget wrapper, uniform across providers.
        self._client = anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=timeout
        )

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
        """Generate a completion via the Anthropic Messages API.

        With ``response_schema``, output is API-enforced via
        ``output_config.format`` and locally validated, retrying up to 3
        attempts on schema misses.
        """
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if response_schema is not None:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": _strictify(response_schema),
                }
            }

        attempts = _SCHEMA_ATTEMPTS if response_schema is not None else 1
        last_schema_error: ValueError | None = None
        for attempt in range(attempts):
            response = self._create(params, timeout or self._timeout)
            if response_schema is None:
                return response
            try:
                response.structured = parse_structured_text(
                    response.text, response_schema
                )
                return response
            except ValueError as e:
                last_schema_error = e
                self._log.warning(
                    "Claude schema miss (attempt %d/%d): %s", attempt + 1, attempts, e
                )
        raise LLMSchemaError(
            f"Claude output failed schema validation after {attempts} attempts: "
            f"{last_schema_error}"
        )

    def _create(self, params: dict[str, Any], timeout: float) -> LLMResponse:
        """Call messages.create and map SDK exceptions to LLM errors."""
        a = self._anthropic
        try:
            message = self._client.with_options(timeout=timeout).messages.create(
                **params
            )
        except (
            a.AuthenticationError,
            a.PermissionDeniedError,
            a.NotFoundError,
            a.BadRequestError,
        ) as e:
            raise LLMPermanentError(f"Claude API rejected the request: {e}") from e
        except a.RateLimitError as e:
            raise LLMError(f"Claude API rate limited: {e}") from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMError(f"Claude API server error: {e}") from e
            raise LLMPermanentError(f"Claude API error: {e}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"Cannot reach the Claude API: {e}") from e

        if message.stop_reason == "refusal":
            raise LLMPermanentError("Claude declined the request (refusal)")

        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LLMError("Claude returned no text content")

        return LLMResponse(
            text=text,
            model=message.model,
            provider=self._name,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* with ``additionalProperties: false`` on objects.

    The Anthropic structured-output API requires it on every object schema.
    """
    result = copy.deepcopy(schema)
    _add_strict(result)
    return result


def _add_strict(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for value in node.values():
            _add_strict(value)
    elif isinstance(node, list):
        for item in node:
            _add_strict(item)
