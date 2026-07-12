"""Resilience and accounting wrapper around any LLMProvider.

Uniform retry/backoff, a circuit breaker, an optional request rate limit,
and per-provider token/call accounting — so individual providers stay thin
transport adapters. Emits one structured log line per call (the Phase-1
token/cost metric surface; Prometheus export lands in Phase 5).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from warpt.daemon.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMProvider,
    LLMResponse,
    LLMSchemaError,
    ToolDef,
)
from warpt.utils.logger import Logger

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0
DEFAULT_BREAKER_THRESHOLD = 5
DEFAULT_BREAKER_COOLDOWN = 60.0


class ResilientProvider:
    """LLMProvider wrapper adding retry, circuit breaker, and accounting.

    Parameters
    ----------
    provider
        The wrapped transport provider.
    retries
        Max attempts per call for retryable errors.
    backoff
        Base backoff in seconds (doubles each attempt).
    breaker_threshold
        Consecutive failures before the circuit opens.
    breaker_cooldown
        Seconds the circuit stays open before a half-open probe.
    max_requests_per_min
        Optional sliding-window request cap; ``None`` disables.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
        breaker_cooldown: float = DEFAULT_BREAKER_COOLDOWN,
        max_requests_per_min: int | None = None,
    ) -> None:
        self._provider = provider
        self._retries = retries
        self._backoff = backoff
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = breaker_cooldown
        self._max_rpm = max_requests_per_min
        self._log = Logger.get("daemon.llm.budget")

        self._consecutive_failures = 0
        self._breaker_opened_at: float | None = None
        self._request_times: deque[float] = deque()

        # Lifetime accounting
        self.total_calls = 0
        self.total_failures = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def name(self) -> str:
        """Wrapped provider name."""
        return self._provider.name

    @property
    def model(self) -> str:
        """Wrapped provider model."""
        return self._provider.model

    def get_stats(self) -> dict[str, Any]:
        """Return lifetime call/token counters and breaker state."""
        return {
            "provider": self.name,
            "model": self.model,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "breaker_open": self._breaker_opened_at is not None,
        }

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        tools: list[ToolDef] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Call the wrapped provider with retry, breaker, and accounting.

        Permanent and schema errors are never retried here (providers
        already bound their own schema retries).
        """
        self._check_breaker()
        self._check_rate_limit()

        last_err: LLMError | None = None
        for attempt in range(self._retries):
            start = time.monotonic()
            self.total_calls += 1
            self._request_times.append(time.monotonic())
            try:
                response = self._provider.generate(
                    messages,
                    system=system,
                    response_schema=response_schema,
                    tools=tools,
                    timeout=timeout,
                )
            except (LLMPermanentError, LLMSchemaError):
                self._record_failure()
                raise
            except LLMError as e:
                self._record_failure()
                last_err = e
                if attempt < self._retries - 1:
                    delay = self._backoff * (2**attempt)
                    self._log.warning(
                        "LLM retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        self._retries,
                        self.name,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                continue

            self._record_success(response, time.monotonic() - start, attempt)
            return response

        raise last_err  # type: ignore[misc]

    def _check_breaker(self) -> None:
        """Raise while the circuit is open; allow a half-open probe after cooldown."""
        if self._breaker_opened_at is None:
            return
        elapsed = time.monotonic() - self._breaker_opened_at
        if elapsed < self._breaker_cooldown:
            raise LLMError(
                f"Circuit breaker open for provider {self.name!r} "
                f"({self._breaker_cooldown - elapsed:.0f}s until half-open probe)"
            )
        # Half-open: let this call through as the probe.
        self._log.info("Circuit breaker half-open for %s, probing", self.name)

    def _check_rate_limit(self) -> None:
        """Enforce the optional sliding-window requests-per-minute cap."""
        if self._max_rpm is None:
            return
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] > 60.0:
            self._request_times.popleft()
        if len(self._request_times) >= self._max_rpm:
            raise LLMError(
                f"Rate limit reached for provider {self.name!r} "
                f"({self._max_rpm} requests/min)"
            )

    def _record_failure(self) -> None:
        self.total_failures += 1
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= self._breaker_threshold
            and self._breaker_opened_at is None
        ):
            self._breaker_opened_at = time.monotonic()
            self._log.error(
                "Circuit breaker OPEN for %s after %d consecutive failures",
                self.name,
                self._consecutive_failures,
            )
        elif self._breaker_opened_at is not None:
            # Failed half-open probe — restart the cooldown window.
            self._breaker_opened_at = time.monotonic()

    def _record_success(
        self, response: LLMResponse, latency: float, attempt: int
    ) -> None:
        self._consecutive_failures = 0
        if self._breaker_opened_at is not None:
            self._log.info("Circuit breaker CLOSED for %s", self.name)
            self._breaker_opened_at = None
        if response.input_tokens:
            self.total_input_tokens += response.input_tokens
        if response.output_tokens:
            self.total_output_tokens += response.output_tokens
        self._log.info(
            "llm_call provider=%s model=%s latency_s=%.2f retries=%d "
            "input_tokens=%s output_tokens=%s total_input=%d total_output=%d",
            self.name,
            self.model,
            latency,
            attempt,
            response.input_tokens,
            response.output_tokens,
            self.total_input_tokens,
            self.total_output_tokens,
        )
