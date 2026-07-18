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
    "attending": {
        "max_iterations": 5,
        "max_wall_clock_s": 120,
    },
    "remediation": {
        "probes": {
            # Diagnostic stress probes are load-generating; off by default.
            "enabled": False,
            "idle_threshold_pct": 20.0,
            "max_duration_s": 30,
        },
    },
    "fleet": {
        # Node -> central reporting. Additive: the node never depends on it.
        "enabled": False,
        "central_url": "http://127.0.0.1:8787",
        "push_interval_s": 30,
        "token_env": "WARPT_FLEET_TOKEN",
        "max_buffer_mb": 64,
    },
    "daemon_http": {
        # In-process health/status endpoint (loopback only by default).
        "enabled": False,
        "host": "127.0.0.1",
        "port": 8788,
    },
    "retention": {
        # Node-local data lifecycle (see warpt/daemon/janitor.py).
        "vitals_days": 14,
        "closed_cases_days": 90,
        "interval_h": 6,
    },
    "carbon": {
        # Lifetime energy odometer (see warpt/carbon/continuous.py).
        "region": "US",
        "cost_per_kwh": 0.12,
        "poll_interval_s": 60,
    },
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

    Any ``api_key`` fields are stripped before writing — keys must live in
    the environment or a secret file (``api_key_env`` / ``api_key_file``),
    never on disk in the config.

    Parameters
    ----------
    warpt_dir
        Path to the warpt data directory.
    config
        Configuration dict to persist.
    """
    log = Logger.get("daemon.config")
    stripped = _strip_api_keys(config)
    if stripped:
        log.warning(
            "Removed inline api_key field(s) before saving config — "
            "use api_key_env or api_key_file instead"
        )
    dir_path = Path(warpt_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    config_path = dir_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    log.info("Config saved to %s", config_path)


def _strip_api_keys(node: Any) -> int:
    """Recursively delete ``api_key`` keys in-place. Returns count removed."""
    removed = 0
    if isinstance(node, dict):
        if "api_key" in node:
            del node["api_key"]
            removed += 1
        for value in node.values():
            removed += _strip_api_keys(value)
    elif isinstance(node, list):
        for item in node:
            removed += _strip_api_keys(item)
    return removed
