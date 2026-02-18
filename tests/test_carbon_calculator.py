"""Tests for carbon calculator module."""

from warpt.carbon.calculator import CarbonCalculator
from warpt.carbon.grid_intensity import GRID_INTENSITY, get_grid_intensity, list_regions


class TestGridIntensity:
    """Tests for grid intensity lookup."""

    def test_known_region(self):
        """Look up a known region."""
        assert get_grid_intensity("US") == 390

    def test_unknown_region_falls_back_to_world(self):
        """Unknown region returns WORLD average."""
        assert get_grid_intensity("XX") == GRID_INTENSITY["WORLD"]

    def test_case_insensitive(self):
        """Region lookup is case-insensitive."""
        assert get_grid_intensity("us") == 390
        assert get_grid_intensity("eu-fr") == 60

    def test_list_regions_returns_all(self):
        """List all regions."""
        regions = list_regions()
        assert "US" in regions
        assert "WORLD" in regions
        assert len(regions) == len(GRID_INTENSITY)


class TestCarbonCalculator:
    """Tests for CarbonCalculator energy/CO2/cost math."""

    def test_energy_from_samples_empty(self):
        """Empty samples return 0."""
        calc = CarbonCalculator(region="US")
        assert calc.energy_from_samples([]) == 0.0

    def test_energy_from_samples_single(self):
        """Single sample returns 0 (need at least 2 for integration)."""
        calc = CarbonCalculator(region="US")
        assert calc.energy_from_samples([(0.0, 100.0)]) == 0.0

    def test_energy_from_samples_constant_power(self):
        """100W for 3600s = 0.1 kWh."""
        calc = CarbonCalculator(region="US")
        samples = [(0.0, 100.0), (3600.0, 100.0)]
        energy = calc.energy_from_samples(samples)
        assert abs(energy - 0.1) < 1e-9

    def test_energy_from_samples_trapezoidal(self):
        """Linear ramp from 0W to 200W over 3600s = 0.1 kWh."""
        calc = CarbonCalculator(region="US")
        samples = [(0.0, 0.0), (3600.0, 200.0)]
        energy = calc.energy_from_samples(samples)
        assert abs(energy - 0.1) < 1e-9

    def test_energy_from_samples_multiple_points(self):
        """Test multi-segment trapezoidal integration."""
        calc = CarbonCalculator(region="US")
        samples = [
            (0.0, 100.0),
            (1800.0, 100.0),
            (1800.0, 200.0),
            (3600.0, 200.0),
        ]
        energy = calc.energy_from_samples(samples)
        # Seg 1: (100+100)/2 * 1800 = 180000 J
        # Seg 2: (100+200)/2 * 0 = 0 J
        # Seg 3: (200+200)/2 * 1800 = 360000 J
        # Total: 540000 J = 0.15 kWh
        assert abs(energy - 0.15) < 1e-9

    def test_energy_negative_time_gap_ignored(self):
        """Negative time deltas contribute 0 energy."""
        calc = CarbonCalculator(region="US")
        samples = [(100.0, 50.0), (50.0, 50.0)]
        energy = calc.energy_from_samples(samples)
        assert energy == 0.0

    def test_co2_from_energy(self):
        """US grid: 390 gCO2/kWh * 1 kWh = 390g."""
        calc = CarbonCalculator(region="US")
        assert abs(calc.co2_from_energy(1.0) - 390.0) < 1e-9

    def test_co2_from_energy_france(self):
        """France: 60 gCO2/kWh * 1 kWh = 60g."""
        calc = CarbonCalculator(region="EU-FR")
        assert abs(calc.co2_from_energy(1.0) - 60.0) < 1e-9

    def test_cost_from_energy_default_rate(self):
        """1 kWh * $0.12 = $0.12."""
        calc = CarbonCalculator(region="US")
        assert abs(calc.cost_from_energy(1.0) - 0.12) < 1e-9

    def test_cost_from_energy_custom_rate(self):
        """1 kWh * $0.25 = $0.25."""
        calc = CarbonCalculator(region="US")
        assert abs(calc.cost_from_energy(1.0, rate=0.25) - 0.25) < 1e-9

    def test_humanize_tiny(self):
        """Sub-gram CO2 returns breathing comparison."""
        calc = CarbonCalculator(region="US")
        assert "breathing" in calc.humanize(0.5)

    def test_humanize_phone(self):
        """Small CO2 returns phone charging comparison."""
        calc = CarbonCalculator(region="US")
        result = calc.humanize(20.0)
        assert "phone" in result

    def test_humanize_driving(self):
        """Medium CO2 returns driving comparison."""
        calc = CarbonCalculator(region="US")
        result = calc.humanize(200.0)
        assert "driving" in result or "miles" in result

    def test_humanize_ac(self):
        """Large CO2 returns air conditioning comparison."""
        calc = CarbonCalculator(region="US")
        result = calc.humanize(3000.0)
        assert "air conditioning" in result
