"""Persistent carbon configuration (region / custom intensity)."""

from __future__ import annotations

import json
from pathlib import Path

from warpt.carbon.grid_intensity import GRID_INTENSITY, get_grid_intensity

CONFIG_FILE = Path.home() / ".warpt" / "config.json"

_DEFAULT_REGION = "US"
DEFAULT_KWH_PRICE = 0.12  # USD per kWh — US national avg residential


def load_carbon_config(config_file: Path | None = None) -> dict:
    """Read the carbon section from the config file.

    Parameters
    ----------
    config_file : Path | None
        Override path for testing. Defaults to ``~/.warpt/config.json``.

    Returns
    -------
    dict
        The ``carbon`` section, or ``{}`` if no config exists.
    """
    path = config_file or CONFIG_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        result: dict = data.get("carbon", {})
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def save_carbon_config(config: dict, config_file: Path | None = None) -> None:
    """Write the carbon config to disk.

    Parameters
    ----------
    config : dict
        The ``carbon`` section to persist.  Only one of ``region`` or
        ``intensity`` should be present — the caller is responsible for
        enforcing mutual exclusion.
    config_file : Path | None
        Override path for testing.
    """
    path = config_file or CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"carbon": config}, indent=2) + "\n")


def get_effective_region_and_intensity(
    config_file: Path | None = None,
) -> tuple[str, float]:
    """Resolve the active region code and gCO2/kWh value.

    Priority: config intensity > config region > default US.

    Returns
    -------
    tuple[str, float]
        ``(region_code, intensity_value)``.
    """
    cfg = load_carbon_config(config_file)

    if "intensity" in cfg:
        return ("CUSTOM", float(cfg["intensity"]))

    region = cfg.get("region", _DEFAULT_REGION)
    return (region, get_grid_intensity(region))


def get_effective_kwh_price(config_file: Path | None = None) -> float:
    """Return the configured electricity price, or the default.

    Returns
    -------
    float
        Price in USD per kWh.
    """
    cfg = load_carbon_config(config_file)
    return float(cfg.get("kwh_price", DEFAULT_KWH_PRICE))


def validate_region(code: str) -> bool:
    """Return True if *code* is a known grid region."""
    return code.upper() in GRID_INTENSITY
