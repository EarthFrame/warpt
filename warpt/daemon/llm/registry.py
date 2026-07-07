"""Build LLM providers from daemon config, with per-agent selection.

Config shape (new style)::

    llm:
      default: claude
      providers:
        claude:  { type: anthropic, model: claude-sonnet-5,
                   api_key_env: ANTHROPIC_API_KEY }
        local:   { type: ollama, url: http://localhost:11434, model: llama3:8b }
        cluster: { type: openai_compat, url: http://inference.internal/v1,
                   model: llama3:70b }
      agents:
        chart_nurse: { provider: local }
        attending:   { provider: claude }
        escalate_to: cluster   # reserved; wired in Phase 2

Legacy configs (top-level ``ollama_url`` + ``models``) are synthesized into
an equivalent ``llm`` block so existing installs keep working unchanged.
"""

from __future__ import annotations

from typing import Any

from warpt.daemon.llm.base import LLMPermanentError, LLMProvider
from warpt.daemon.llm.budget import ResilientProvider
from warpt.daemon.llm.providers.claude import DEFAULT_CLAUDE_MODEL, ClaudeProvider
from warpt.daemon.llm.providers.ollama import DEFAULT_OLLAMA_URL, OllamaProvider
from warpt.daemon.llm.providers.openai_compat import OpenAICompatProvider
from warpt.daemon.llm.secrets import resolve_api_key

_LEGACY_CHART_MODEL = "llama3:8b"
_LEGACY_ATTENDING_MODEL = "llama3:70b"


def effective_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``llm`` block, synthesizing one from legacy keys if absent.

    Parameters
    ----------
    config
        Full daemon config dict (as returned by ``load_config``).
    """
    llm = config.get("llm")
    if llm:
        return llm

    url = config.get("ollama_url", DEFAULT_OLLAMA_URL)
    models = config.get("models", {})
    chart_model = models.get("chart_nurse", _LEGACY_CHART_MODEL)
    attending_model = models.get("attending", _LEGACY_ATTENDING_MODEL)
    return {
        "default": "local",
        "providers": {
            "local": {"type": "ollama", "url": url, "model": chart_model},
        },
        "agents": {
            "chart_nurse": {"provider": "local", "model": chart_model},
            "attending": {"provider": "local", "model": attending_model},
        },
    }


def build_provider(
    name: str, provider_cfg: dict[str, Any], *, model: str | None = None
) -> LLMProvider:
    """Construct a provider instance from its config entry.

    Parameters
    ----------
    name
        Provider name from the config (used in logs/metrics).
    provider_cfg
        The provider's config dict (must contain ``type``).
    model
        Optional per-agent model override.
    """
    ptype = provider_cfg.get("type")
    resolved_model = model or provider_cfg.get("model")

    if ptype == "ollama":
        if not resolved_model:
            raise LLMPermanentError(f"LLM provider {name!r} has no model configured")
        return OllamaProvider(
            model=resolved_model,
            url=provider_cfg.get("url", DEFAULT_OLLAMA_URL),
            name=name,
        )

    if ptype == "anthropic":
        return ClaudeProvider(
            model=resolved_model or DEFAULT_CLAUDE_MODEL,
            api_key=resolve_api_key(name, provider_cfg),
            name=name,
            max_tokens=provider_cfg.get("max_tokens", 4096),
        )

    if ptype == "openai_compat":
        url = provider_cfg.get("url")
        if not url:
            raise LLMPermanentError(f"LLM provider {name!r} has no url configured")
        if not resolved_model:
            raise LLMPermanentError(f"LLM provider {name!r} has no model configured")
        return OpenAICompatProvider(
            model=resolved_model,
            url=url,
            api_key=resolve_api_key(name, provider_cfg),
            name=name,
        )

    raise LLMPermanentError(
        f"Unknown LLM provider type {ptype!r} for provider {name!r} "
        "(expected one of: ollama, anthropic, openai_compat)"
    )


def provider_for_agent(config: dict[str, Any], agent: str) -> LLMProvider:
    """Build the provider configured for *agent* (e.g. ``"chart_nurse"``).

    Falls back to the ``llm.default`` provider when the agent has no explicit
    assignment.
    """
    llm = effective_llm_config(config)
    agents = llm.get("agents", {})
    agent_cfg = agents.get(agent)
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}

    provider_name = agent_cfg.get("provider") or llm.get("default")
    if not provider_name:
        raise LLMPermanentError(
            f"No LLM provider configured for agent {agent!r} and no default set"
        )

    providers = llm.get("providers", {})
    provider_cfg = providers.get(provider_name)
    if provider_cfg is None:
        raise LLMPermanentError(
            f"LLM provider {provider_name!r} (for agent {agent!r}) "
            "is not defined under llm.providers"
        )

    raw = build_provider(provider_name, provider_cfg, model=agent_cfg.get("model"))
    resilience = provider_cfg.get("resilience", {})
    return ResilientProvider(
        raw,
        retries=resilience.get("retries", 3),
        backoff=resilience.get("backoff", 2.0),
        breaker_threshold=resilience.get("breaker_threshold", 5),
        breaker_cooldown=resilience.get("breaker_cooldown", 60.0),
        max_requests_per_min=resilience.get("max_requests_per_min"),
    )
