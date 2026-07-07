"""LLM provider abstraction for the warpt daemon intelligence layer.

Exposes a model-agnostic ``LLMProvider`` protocol so agents (Chart Nurse,
Attending) run unchanged against Claude, a local Ollama model, or an
OpenAI-compatible inference cluster — selected via config.
"""

from warpt.daemon.llm.base import (
    LLMError,
    LLMPermanentError,
    LLMProvider,
    LLMResponse,
    LLMSchemaError,
)

__all__ = [
    "LLMError",
    "LLMPermanentError",
    "LLMProvider",
    "LLMResponse",
    "LLMSchemaError",
]
