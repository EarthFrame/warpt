"""Tests for warpt.daemon.llm.registry — config to providers."""

import pytest

from warpt.daemon.llm.base import LLMPermanentError
from warpt.daemon.llm.budget import ResilientProvider
from warpt.daemon.llm.registry import (
    build_provider,
    effective_llm_config,
    provider_for_agent,
)

_NEW_STYLE_CONFIG = {
    "llm": {
        "default": "local",
        "providers": {
            "local": {
                "type": "ollama",
                "url": "http://ollama.internal:11434",
                "model": "llama3:8b",
            },
        },
        "agents": {
            "chart_nurse": {"provider": "local"},
            "attending": {"provider": "local", "model": "llama3:70b"},
        },
    }
}

_LEGACY_CONFIG = {
    "intelligence_enabled": True,
    "ollama_url": "http://legacy-host:11434",
    "models": {"chart_nurse": "mistral:7b", "attending": "llama3:70b"},
}


def test_effective_llm_config_passthrough():
    """An explicit llm block is returned as-is."""
    assert effective_llm_config(_NEW_STYLE_CONFIG) is _NEW_STYLE_CONFIG["llm"]


def test_effective_llm_config_synthesizes_from_legacy():
    """Legacy ollama_url + models keys synthesize an equivalent llm block."""
    llm = effective_llm_config(_LEGACY_CONFIG)

    assert llm["default"] == "local"
    assert llm["providers"]["local"]["type"] == "ollama"
    assert llm["providers"]["local"]["url"] == "http://legacy-host:11434"
    assert llm["agents"]["chart_nurse"]["model"] == "mistral:7b"
    assert llm["agents"]["attending"]["model"] == "llama3:70b"


def test_effective_llm_config_defaults_when_empty():
    """A bare config still yields a usable local-ollama llm block."""
    llm = effective_llm_config({})
    assert llm["providers"]["local"]["url"] == "http://localhost:11434"
    assert llm["agents"]["chart_nurse"]["model"] == "llama3:8b"
    assert llm["agents"]["attending"]["model"] == "llama3:70b"


def test_provider_for_agent_new_style():
    """Provider for agent new style (wrapped in ResilientProvider)."""
    provider = provider_for_agent(_NEW_STYLE_CONFIG, "attending")
    assert isinstance(provider, ResilientProvider)
    assert provider.name == "local"
    # Per-agent model override wins over the provider's model
    assert provider.model == "llama3:70b"


def test_provider_for_agent_legacy_config():
    """Provider for agent legacy config."""
    provider = provider_for_agent(_LEGACY_CONFIG, "chart_nurse")
    assert isinstance(provider, ResilientProvider)
    assert provider.model == "mistral:7b"


def test_provider_for_agent_falls_back_to_default():
    """An agent with no explicit assignment uses llm.default."""
    provider = provider_for_agent(_NEW_STYLE_CONFIG, "future_agent")
    assert provider.name == "local"
    assert provider.model == "llama3:8b"


def test_provider_for_agent_unknown_provider_name():
    """Provider for agent unknown provider name."""
    config = {
        "llm": {
            "default": "missing",
            "providers": {},
            "agents": {},
        }
    }
    with pytest.raises(LLMPermanentError, match="not defined"):
        provider_for_agent(config, "chart_nurse")


def test_build_provider_unknown_type():
    """Build provider unknown type."""
    with pytest.raises(LLMPermanentError, match="Unknown LLM provider type"):
        build_provider("weird", {"type": "quantum", "model": "q1"})


def test_build_provider_requires_model():
    """Build provider requires model."""
    with pytest.raises(LLMPermanentError, match="no model"):
        build_provider("local", {"type": "ollama"})
