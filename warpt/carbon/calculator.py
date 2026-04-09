"""Energy, CO2, and cost calculation from power samples."""

from __future__ import annotations

from warpt.carbon.grid_intensity import get_grid_intensity


class CarbonCalculator:
    """Convert power samples into energy, CO2 emissions, and cost.

    Parameters
    ----------
    region : str
        Grid region code for CO2 intensity lookup.
    """

    def __init__(self, region: str = "US") -> None:
        self.region = region
        self.intensity = get_grid_intensity(region)  # gCO2/kWh

    def energy_from_samples(self, samples: list[tuple[float, float]]) -> float:
        """Compute energy via trapezoidal integration.

        Parameters
        ----------
        samples : list[tuple[float, float]]
            List of (timestamp_seconds, watts) pairs.

        Returns
        -------
        float
            Energy in kilowatt-hours. Returns 0.0 if fewer than 2 samples.
        """
        if len(samples) < 2:
            return 0.0

        energy_joules = 0.0
        for i in range(1, len(samples)):
            t0, w0 = samples[i - 1]
            t1, w1 = samples[i]
            dt = t1 - t0
            if dt > 0:
                # Trapezoidal rule: average of two power readings x time
                energy_joules += (w0 + w1) / 2.0 * dt

        # Convert joules to kWh: 1 kWh = 3,600,000 J
        return energy_joules / 3_600_000.0

    def energy_from_counter(self, joules: float) -> float:
        """Convert a direct energy counter reading to kilowatt-hours.

        Hardware energy counters (NVML on Volta+, RAPL) provide cumulative
        energy in joules. This is more accurate than integrating polled
        power samples since the counter tracks energy continuously at
        the hardware level.

        Parameters
        ----------
        joules : float
            Energy in joules (e.g. delta between two counter reads).

        Returns
        -------
        float
            Energy in kilowatt-hours.
        """
        return joules / 3_600_000.0

    def co2_from_energy(self, energy_kwh: float) -> float:
        """Estimate CO2 emissions from energy consumption.

        Parameters
        ----------
        energy_kwh : float
            Energy in kilowatt-hours.

        Returns
        -------
        float
            CO2 emissions in grams.
        """
        return energy_kwh * self.intensity

    def cost_from_energy(self, energy_kwh: float, rate: float = 0.12) -> float:
        """Estimate electricity cost.

        Parameters
        ----------
        energy_kwh : float
            Energy in kilowatt-hours.
        rate : float
            Electricity rate in USD per kWh. Default $0.12 (US avg residential).

        Returns
        -------
        float
            Estimated cost in USD.
        """
        return energy_kwh * rate

    def humanize(self, co2_grams: float) -> str:
        """Create a human-relatable comparison for CO2 emissions.

        Parameters
        ----------
        co2_grams : float
            CO2 emissions in grams.

        Returns
        -------
        str
            A comparison string (e.g. "like charging your phone 2 times").
        """
        if co2_grams < 1.0:
            return "less than breathing for a minute"

        if co2_grams < 50.0:
            # Average phone charge ≈ 8g CO2
            charges = co2_grams / 8.0
            if charges < 1.5:
                return "like charging your phone once (~8g CO2 per charge)"
            return f"like charging your phone {charges:.0f} times (~8g CO2 per charge)"

        if co2_grams < 500.0:
            # Average car emits ~400g CO2/mile
            miles = co2_grams / 400.0
            if miles < 0.15:
                return "like driving a few hundred feet (~400g CO2/mile, avg car)"
            return f"like driving {miles:.1f} miles (~400g CO2/mile, avg car)"

        # Average AC unit ≈ 1500g CO2/hour
        hours = co2_grams / 1500.0
        if hours < 1.0:
            minutes = hours * 60
            return f"like {minutes:.0f} minutes of air conditioning (~1.5kg CO2/hr)"
        return f"like {hours:.1f} hours of air conditioning (~1.5kg CO2/hr)"
