"""Tests for warpt.daemon.config — load, save, merge."""

from warpt.daemon.config import DEFAULTS, load_config, save_config


def test_load_defaults_when_no_file(tmp_path):
    """Loading from a dir with no config.yaml returns DEFAULTS."""
    config = load_config(str(tmp_path))
    assert config == DEFAULTS
    assert config["intelligence_enabled"] is False
    assert config["ollama_url"] == "http://localhost:11434"


def test_save_and_load_round_trip(tmp_path):
    """Saving then loading preserves all values."""
    original = {
        "intelligence_enabled": True,
        "ollama_url": "http://myhost:11434",
        "models": {"chart_nurse": "mistral:7b", "attending": "llama3:70b"},
        "triage_order": ["compute", "memory"],
    }
    save_config(str(tmp_path), original)
    loaded = load_config(str(tmp_path))
    assert loaded["intelligence_enabled"] is True
    assert loaded["ollama_url"] == "http://myhost:11434"
    assert loaded["models"]["chart_nurse"] == "mistral:7b"
    assert loaded["models"]["attending"] == "llama3:70b"
    assert loaded["triage_order"] == ["compute", "memory"]


def test_partial_config_merges_defaults(tmp_path):
    """A partial config fills in missing keys from DEFAULTS."""
    partial = {"intelligence_enabled": True, "models": {"chart_nurse": "phi3:mini"}}
    save_config(str(tmp_path), partial)
    loaded = load_config(str(tmp_path))
    # Overridden values
    assert loaded["intelligence_enabled"] is True
    assert loaded["models"]["chart_nurse"] == "phi3:mini"
    # Merged from defaults
    assert loaded["models"]["attending"] == "llama3:70b"
    assert loaded["ollama_url"] == "http://localhost:11434"
    assert loaded["triage_order"] == DEFAULTS["triage_order"]


def test_load_preserves_unknown_keys(tmp_path):
    """Unknown keys in config.yaml are kept (future-proofing)."""
    custom = {
        "intelligence_enabled": True,
        "custom_future_key": {"nested": True},
    }
    save_config(str(tmp_path), custom)
    loaded = load_config(str(tmp_path))
    assert loaded["custom_future_key"] == {"nested": True}
    # Defaults still filled in
    assert "models" in loaded
