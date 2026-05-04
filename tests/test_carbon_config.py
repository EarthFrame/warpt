"""Tests for carbon config persistence and subcommands."""

from __future__ import annotations

import json

from warpt.carbon.config import (
    DEFAULT_KWH_PRICE,
    get_effective_kwh_price,
    get_effective_region_and_intensity,
    load_carbon_config,
    save_carbon_config,
    validate_region,
)
from warpt.carbon.grid_intensity import GRID_INTENSITY


class TestConfigPersistence:
    """Round-trip save/load for region and intensity."""

    def test_save_and_load_region(self, tmp_path):
        """Save region, reload, verify."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config({"region": "EU-FR"}, config_file=cfg_file)
        loaded = load_carbon_config(config_file=cfg_file)
        assert loaded == {"region": "EU-FR"}

    def test_save_and_load_intensity(self, tmp_path):
        """Save intensity, reload, verify."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config({"intensity": 450}, config_file=cfg_file)
        loaded = load_carbon_config(config_file=cfg_file)
        assert loaded == {"intensity": 450}

    def test_intensity_overwrites_region(self, tmp_path):
        """Setting intensity removes previous region."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config({"region": "EU-FR"}, config_file=cfg_file)
        save_carbon_config({"intensity": 500}, config_file=cfg_file)
        loaded = load_carbon_config(config_file=cfg_file)
        assert "region" not in loaded
        assert loaded["intensity"] == 500

    def test_region_overwrites_intensity(self, tmp_path):
        """Setting region removes previous intensity."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config({"intensity": 500}, config_file=cfg_file)
        save_carbon_config({"region": "US"}, config_file=cfg_file)
        loaded = load_carbon_config(config_file=cfg_file)
        assert "intensity" not in loaded
        assert loaded["region"] == "US"

    def test_no_config_returns_empty(self, tmp_path):
        """Missing file returns empty dict."""
        cfg_file = tmp_path / "does_not_exist.json"
        loaded = load_carbon_config(config_file=cfg_file)
        assert loaded == {}

    def test_creates_parent_directory(self, tmp_path):
        """Save creates intermediate directories."""
        cfg_file = tmp_path / "sub" / "dir" / "config.json"
        save_carbon_config({"region": "UK"}, config_file=cfg_file)
        assert cfg_file.exists()
        loaded = load_carbon_config(config_file=cfg_file)
        assert loaded["region"] == "UK"


class TestEffectiveRegionAndIntensity:
    """Test get_effective_region_and_intensity resolution."""

    def test_no_config_returns_default(self, tmp_path):
        """No config file falls back to US."""
        cfg_file = tmp_path / "config.json"
        region, intensity = get_effective_region_and_intensity(
            config_file=cfg_file,
        )
        assert region == "US"
        assert intensity == GRID_INTENSITY["US"]

    def test_configured_region(self, tmp_path):
        """Configured region returns matching intensity."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config({"region": "EU-FR"}, config_file=cfg_file)
        region, intensity = get_effective_region_and_intensity(
            config_file=cfg_file,
        )
        assert region == "EU-FR"
        assert intensity == GRID_INTENSITY["EU-FR"]

    def test_configured_intensity(self, tmp_path):
        """Configured intensity returns CUSTOM region."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config({"intensity": 450}, config_file=cfg_file)
        region, intensity = get_effective_region_and_intensity(
            config_file=cfg_file,
        )
        assert region == "CUSTOM"
        assert intensity == 450

    def test_intensity_takes_precedence_over_region(self, tmp_path):
        """If both keys somehow exist, intensity wins."""
        cfg_file = tmp_path / "config.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"carbon": {"region": "EU-FR", "intensity": 999}}
        cfg_file.write_text(json.dumps(data))
        region, intensity = get_effective_region_and_intensity(
            config_file=cfg_file,
        )
        assert region == "CUSTOM"
        assert intensity == 999


class TestValidation:
    """Test region validation."""

    def test_known_region_valid(self):
        """Known codes are accepted."""
        assert validate_region("US") is True
        assert validate_region("EU-FR") is True

    def test_unknown_region_invalid(self):
        """Unknown codes are rejected."""
        assert validate_region("FAKE") is False
        assert validate_region("NOPE") is False

    def test_case_insensitive_validation(self):
        """Region validation is case-insensitive."""
        assert validate_region("us") is True
        assert validate_region("eu-fr") is True

    def test_set_region_validates(self):
        """Unknown region code is rejected."""
        assert validate_region("ATLANTIS") is False

    def test_intensity_must_be_positive(self):
        """Zero and negative values are invalid."""
        for bad_value in [0, -1, -100.5]:
            assert bad_value <= 0


class TestKwhPrice:
    """Test kwh-price config persistence and defaults."""

    def test_no_config_returns_default(self, tmp_path):
        """Missing config returns default price."""
        cfg_file = tmp_path / "config.json"
        price = get_effective_kwh_price(config_file=cfg_file)
        assert price == DEFAULT_KWH_PRICE

    def test_save_and_load_kwh_price(self, tmp_path):
        """Saved price is returned by getter."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config(
            {"region": "US", "kwh_price": 0.25},
            config_file=cfg_file,
        )
        price = get_effective_kwh_price(config_file=cfg_file)
        assert price == 0.25

    def test_kwh_price_persists_alongside_region(self, tmp_path):
        """kwh_price is independent of region/intensity."""
        cfg_file = tmp_path / "config.json"
        save_carbon_config(
            {"region": "EU-FR", "kwh_price": 0.30},
            config_file=cfg_file,
        )
        loaded = load_carbon_config(config_file=cfg_file)
        assert loaded["region"] == "EU-FR"
        assert loaded["kwh_price"] == 0.30
