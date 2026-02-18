"""JSON file-based storage for energy tracking sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from warpt.carbon.calculator import CarbonCalculator
from warpt.models.carbon_models import CarbonSession, CarbonSummary


class EnergyStore:
    """Persist energy tracking sessions as JSON files.

    Each session is stored as a separate JSON file in the sessions directory.
    Default location: ``~/.warpt/sessions/``.

    Parameters
    ----------
    base_dir : Path | None
        Directory for session files. Defaults to ``~/.warpt/sessions/``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or (Path.home() / ".warpt" / "sessions")

    def _ensure_dir(self) -> None:
        """Create the sessions directory if it doesn't exist."""
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Return the file path for a session."""
        return self._base_dir / f"{session_id}.json"

    def create_session(self, session: CarbonSession) -> None:
        """Write an initial session JSON file.

        Parameters
        ----------
        session : CarbonSession
            The session to persist (typically with end_time=None).
        """
        self._ensure_dir()
        path = self._session_path(session.id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def update_session(self, session: CarbonSession) -> None:
        """Overwrite a session JSON file with updated data.

        Parameters
        ----------
        session : CarbonSession
            The session with finalized data.
        """
        self._ensure_dir()
        path = self._session_path(session.id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def get_session(self, session_id: str) -> CarbonSession | None:
        """Read a single session from disk.

        Parameters
        ----------
        session_id : str
            The session UUID.

        Returns
        -------
        CarbonSession | None
            The session, or None if not found.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return CarbonSession.from_dict(data)

    def get_sessions(
        self, limit: int = 50, since: float | None = None
    ) -> list[CarbonSession]:
        """List sessions, sorted by start time (newest first).

        Parameters
        ----------
        limit : int
            Maximum number of sessions to return.
        since : float | None
            Only include sessions started after this Unix timestamp.

        Returns
        -------
        list[CarbonSession]
            Sessions matching the criteria.
        """
        if not self._base_dir.exists():
            return []

        sessions: list[CarbonSession] = []
        for path in self._base_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                session = CarbonSession.from_dict(data)
                if since is not None and session.start_time < since:
                    continue
                sessions.append(session)
            except (json.JSONDecodeError, KeyError):
                continue

        sessions.sort(key=lambda s: s.start_time, reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> None:
        """Delete a session file.

        Parameters
        ----------
        session_id : str
            The session UUID to delete.
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()

    def get_totals(self, since: float | None = None) -> CarbonSummary:
        """Aggregate totals across all qualifying sessions.

        Parameters
        ----------
        since : float | None
            Only include sessions started after this Unix timestamp.

        Returns
        -------
        CarbonSummary
            Aggregated summary with totals and averages.
        """
        sessions = self.get_sessions(limit=10000, since=since)

        if not sessions:
            return CarbonSummary(humanized="No sessions recorded")

        total_energy = 0.0
        total_co2 = 0.0
        total_cost = 0.0
        total_duration = 0.0
        total_power_sum = 0.0
        power_count = 0

        for s in sessions:
            total_energy += s.energy_kwh or 0.0
            total_co2 += s.co2_grams or 0.0
            total_cost += s.cost_usd or 0.0
            total_duration += s.duration_s or 0.0
            avg_w = s.metadata.get("avg_power_w")
            if avg_w is not None:
                total_power_sum += avg_w
                power_count += 1

        # Time span from oldest to newest session
        oldest = min(s.start_time for s in sessions)
        newest = max(s.start_time for s in sessions)
        period_days = max((newest - oldest) / 86400.0, 0.0)

        # If all sessions are from the same moment, use duration as the period
        if period_days == 0.0 and total_duration > 0:
            period_days = total_duration / 86400.0

        avg_power = total_power_sum / power_count if power_count > 0 else 0.0

        calc = CarbonCalculator(region=sessions[0].region if sessions else "US")
        humanized = calc.humanize(total_co2)

        return CarbonSummary(
            total_sessions=len(sessions),
            total_energy_kwh=total_energy,
            total_co2_grams=total_co2,
            total_cost_usd=total_cost,
            avg_power_watts=avg_power,
            period_days=period_days,
            humanized=humanized,
        )
