"""Configuration management for warpt daemon intelligence layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from warpt.utils.logger import Logger

DEFAULTS: dict[str, Any] = {
    "intelligence_enabled": False,
    "ollama_url": "http://localhost:11434",
    "models": {
        "chart_nurse": "llama3:8b",
        "attending": "llama3:70b",
    },
    "triage_order": ["thermal_power", "memory", "compute", "storage_io"],
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into a copy of *base*.

    Keys in *overlay* take precedence. Nested dicts are merged rather than
    replaced so that partial overrides (e.g. only ``models.chart_nurse``)
    work as expected.
    """
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(warpt_dir: str) -> dict[str, Any]:
    """Load config from ``{warpt_dir}/config.yaml``, merged with defaults.

    Parameters
    ----------
    warpt_dir
        Path to the warpt data directory (e.g. ``~/.warpt``).

    Returns
    -------
        Fully-merged configuration dict.
    """
    log = Logger.get("daemon.config")
    config_path = Path(warpt_dir) / "config.yaml"
    if not config_path.exists():
        log.debug("No config.yaml found, using defaults")
        return DEFAULTS.copy()
    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}
    log.debug("Loaded config from %s", config_path)
    return _deep_merge(DEFAULTS, user_config)


def save_config(warpt_dir: str, config: dict[str, Any]) -> None:
    """Write config dict to ``{warpt_dir}/config.yaml``.

    Parameters
    ----------
    warpt_dir
        Path to the warpt data directory.
    config
        Configuration dict to persist.
    """
    dir_path = Path(warpt_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    config_path = dir_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    Logger.get("daemon.config").info("Config saved to %s", config_path)
