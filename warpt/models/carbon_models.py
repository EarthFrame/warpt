"""Data models for carbon/energy tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CarbonSession:
    """A single energy tracking session.

    Attributes
    ----------
    id : str
        Unique session identifier (UUID).
    label : str
        Human-readable label (e.g. "warpt stress", "manual").
    start_time : float
        Unix timestamp when the session started.
    end_time : float | None
        Unix timestamp when the session ended (None while running).
    duration_s : float | None
        Duration in seconds.
    energy_kwh : float | None
        Total energy consumed in kilowatt-hours.
    co2_grams : float | None
        Estimated CO2 emissions in grams.
    cost_usd : float | None
        Estimated electricity cost in USD.
    region : str
        Grid region used for CO2 calculation.
    platform : str
        Operating system platform.
    sources : list[str]
        Power measurement sources used (e.g. ["rapl", "nvml"]).
    metadata : dict[str, Any]
        Additional info (avg_power_w, peak_power_w, sample_count, etc.).
    samples : list[dict[str, Any]]
        Raw power samples collected during the session.
    """

    id: str
    label: str
    start_time: float
    end_time: float | None = None
    duration_s: float | None = None
    energy_kwh: float | None = None
    co2_grams: float | None = None
    cost_usd: float | None = None
    region: str = "US"
    platform: str = ""
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": (
                round(self.duration_s, 2) if self.duration_s is not None else None
            ),
            "energy_kwh": (
                round(self.energy_kwh, 8) if self.energy_kwh is not None else None
            ),
            "co2_grams": (
                round(self.co2_grams, 4) if self.co2_grams is not None else None
            ),
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "region": self.region,
            "platform": self.platform,
            "sources": self.sources,
            "metadata": self.metadata,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CarbonSession:
        """Create a CarbonSession from a dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            duration_s=data.get("duration_s"),
            energy_kwh=data.get("energy_kwh"),
            co2_grams=data.get("co2_grams"),
            cost_usd=data.get("cost_usd"),
            region=data.get("region", "US"),
            platform=data.get("platform", ""),
            sources=data.get("sources", []),
            metadata=data.get("metadata", {}),
            samples=data.get("samples", []),
        )


@dataclass
class CarbonSummary:
    """Aggregated summary across multiple sessions.

    Attributes
    ----------
    total_sessions : int
        Number of sessions included.
    total_energy_kwh : float
        Total energy consumed across all sessions.
    total_co2_grams : float
        Total CO2 emissions across all sessions.
    total_cost_usd : float
        Total estimated cost across all sessions.
    avg_power_watts : float
        Average power draw across all sessions.
    period_days : float
        Time span covered in days.
    humanized : str
        Human-relatable comparison string for the CO2 amount.
    """

    total_sessions: int = 0
    total_energy_kwh: float = 0.0
    total_co2_grams: float = 0.0
    total_cost_usd: float = 0.0
    avg_power_watts: float = 0.0
    period_days: float = 0.0
    humanized: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_sessions": self.total_sessions,
            "total_energy_kwh": round(self.total_energy_kwh, 8),
            "total_co2_grams": round(self.total_co2_grams, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_power_watts": round(self.avg_power_watts, 2),
            "period_days": round(self.period_days, 2),
            "humanized": self.humanized,
        }
