"""Grid carbon intensity data by region.

Values represent grams of CO2 emitted per kilowatt-hour of electricity
generated (gCO2/kWh). Different regions have different energy mixes
(coal vs nuclear vs hydro vs renewables), so the same kWh produces
vastly different amounts of CO2.

Sources: IEA, Ember, electricityMap (2023-2024 averages).
"""

from __future__ import annotations

# gCO2 per kWh by region
GRID_INTENSITY: dict[str, float] = {
    # United States
    "US": 390,
    "US-CA": 210,
    "US-TX": 380,
    "US-NY": 230,
    "US-WA": 80,
    # Europe
    "EU": 230,
    "EU-FR": 60,
    "EU-DE": 350,
    "EU-NO": 20,
    # Other regions
    "UK": 200,
    "CN": 550,
    "IN": 700,
    "JP": 450,
    "AU": 600,
    "BR": 75,
    "CA": 120,
    # Global fallback
    "WORLD": 440,
}


def get_grid_intensity(region: str) -> float:
    """Get gCO2/kWh for a region.

    Parameters
    ----------
    region : str
        Region code (e.g. "US", "EU-FR", "WORLD").

    Returns
    -------
    float
        Grid carbon intensity in gCO2/kWh. Falls back to WORLD average
        if region is not found.
    """
    return GRID_INTENSITY.get(region.upper(), GRID_INTENSITY["WORLD"])


def list_regions() -> dict[str, float]:
    """Return all region to intensity mappings.

    Returns
    -------
    dict[str, float]
        Mapping of region codes to gCO2/kWh values.
    """
    return dict(GRID_INTENSITY)
