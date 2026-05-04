"""Grid carbon intensity data by region.

Values represent grams of CO2 emitted per kilowatt-hour of electricity
generated (gCO2/kWh). Different regions have different energy mixes
(coal vs nuclear vs hydro vs renewables), so the same kWh produces
vastly different amounts of CO2.

Sources (2023-2025 data):
  - IEA Emissions Factors 2025
    https://www.iea.org/data-and-statistics/data-product/emissions-factors-2025
  - EPA eGRID 2023 (US sub-regions)
    https://www.epa.gov/egrid
  - Ember Global Electricity Review 2025
    https://ember-energy.org/latest-insights/global-electricity-review-2025/
  - IGES List of Grid Emission Factors v11.6
    https://www.iges.or.jp/en/pub/list-grid-emission-factor/en
  - EEA Greenhouse Gas Emission Intensity of Electricity (EU countries)
    https://www.eea.europa.eu/en/analysis/indicators/greenhouse-gas-emission-intensity-of-1
  - Low Carbon Power (country-level, derived from generation mix)
    https://lowcarbonpower.org/

Last audited: 2026-04-30
"""

from __future__ import annotations

# gCO2 per kWh by region
GRID_INTENSITY: dict[str, float] = {
    # United States
    # US national avg: Ember 2025 (384 gCO2/kWh, 2024 data)
    "US": 385,
    # US-CA: eGRID CAMX subregion + Low Carbon Power (224, 2024)
    "US-CA": 225,
    # US-TX: eGRID ERCT subregion (~335, 2023)
    "US-TX": 335,
    # US-NY: eGRID blended state avg (~200, 2023)
    "US-NY": 200,
    # US-WA: eGRID NWPP + Low Carbon Power (~100, 2023)
    "US-WA": 100,
    # US-FL: eGRID FRCC subregion (~365, 2023)
    "US-FL": 365,
    # US-IL: eGRID RFCW subregion (~285, 2023)
    "US-IL": 285,
    # US-VA: eGRID SRVC subregion (~280, 2023)
    "US-VA": 280,
    # Europe
    # EU avg: Ember 2025 (213 gCO2/kWh, 2024 data)
    "EU": 210,
    # EU-FR: EEA 2024 (43 gCO2/kWh)
    "EU-FR": 43,
    # EU-DE: Ember 2024 (~340, 2024)
    "EU-DE": 340,
    # EU-NO: multiple sources (~20, nearly 100% hydro)
    "EU-NO": 20,
    # EU-IE: EEA 2024 (238 gCO2/kWh)
    "EU-IE": 238,
    # EU-PL: EEA 2024 (554 gCO2/kWh)
    "EU-PL": 554,
    # EU-ES: EEA 2024 (129 gCO2/kWh)
    "EU-ES": 129,
    # United Kingdom
    # UK: Carbon Brief (124 gCO2/kWh, 2024)
    "UK": 125,
    # Asia
    # CN: Ember 2025 (560 gCO2/kWh, 2024 data)
    "CN": 560,
    # IN: Ember 2025 (708 gCO2/kWh, 2024 data)
    "IN": 710,
    # JP: Ember 2025 (482 gCO2/kWh, 2024 data)
    "JP": 480,
    # KR: Low Carbon Power (398, 2025) + Ember
    "KR": 400,
    # TW: Taiwan MOEA (474 gCO2/kWh, 2024)
    "TW": 475,
    # SG: Low Carbon Power (480, 2024) + NCCS
    "SG": 480,
    # Oceania
    # AU: Low Carbon Power (466, 2024)
    "AU": 465,
    # Middle East
    # AE: Low Carbon Power (359, 2023)
    "AE": 360,
    # SA: Low Carbon Power / IGES (543, 2023)
    "SA": 545,
    # Africa
    # ZA: Low Carbon Power (692, 2024)
    "ZA": 690,
    # KE: Low Carbon Power (77-112, 2023-2024; ~85% renewables)
    "KE": 80,
    # NG: Low Carbon Power (383, 2024)
    "NG": 385,
    # Americas
    # CA: Low Carbon Power (138, 2025)
    "CA": 140,
    # BR: Ember 2025 (103, 2024)
    "BR": 100,
    # MX: Low Carbon Power (412, 2025)
    "MX": 410,
    # CL: Low Carbon Power (253, 2025)
    "CL": 255,
    # Global fallback
    # WORLD: Ember 2025 (473 gCO2/kWh, 2024 data)
    "WORLD": 475,
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
