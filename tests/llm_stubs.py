"""Shared fake LLM provider for agent tests."""

from __future__ import annotations

from typing import Any

from warpt.daemon.llm.base import LLMResponse, LLMSchemaError, parse_structured_text


class FakeProvider:
    """In-memory LLMProvider double.

    Parameters
    ----------
    responses
        A single response text, or a sequence consumed one per call (the
        last entry repeats). Entries that are exceptions are raised.
    model
        Model name reported in responses.
    name
        Provider name reported in responses.
    """

    def __init__(
        self,
        responses: str | list[str | Exception] | None = None,
        model: str = "fake-model",
        name: str = "fake",
    ) -> None:
        if responses is None or isinstance(responses, str):
            responses = [responses or ""]
        self._responses: list[str | Exception] = list(responses)
        self._model = model
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Configured provider name."""
        return self._name

    @property
    def model(self) -> str:
        """Configured model identifier."""
        return self._model

    @property
    def call_count(self) -> int:
        """Number of generate() calls made."""
        return len(self.calls)

    def last_system(self) -> str | None:
        """System prompt from the most recent call."""
        return self.calls[-1]["system"]

    def last_user_prompt(self) -> str:
        """Content of the last message in the most recent call."""
        return self.calls[-1]["messages"][-1]["content"]

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Return the next canned response, mimicking provider schema retries.

        Like real providers, schema misses are retried internally (up to 3
        transport attempts, each recorded in ``calls``) before raising
        ``LLMSchemaError``.
        """
        attempts = 3 if response_schema is not None else 1
        last_error: ValueError | None = None
        for _ in range(attempts):
            self.calls.append(
                {
                    "messages": messages,
                    "system": system,
                    "response_schema": response_schema,
                    "timeout": timeout,
                }
            )
            idx = min(len(self.calls) - 1, len(self._responses) - 1)
            entry = self._responses[idx]
            if isinstance(entry, Exception):
                raise entry
            response = LLMResponse(text=entry, model=self._model, provider=self._name)
            if response_schema is None:
                return response
            try:
                response.structured = parse_structured_text(entry, response_schema)
                return response
            except ValueError as e:
                last_error = e
        raise LLMSchemaError(f"fake provider schema miss: {last_error}")
