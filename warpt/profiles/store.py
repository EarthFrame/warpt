"""JSON file-based storage for device behavioral profiles."""

from __future__ import annotations

import json
from pathlib import Path

from warpt.models.profile_models import (
    DeviceProfile,
    HardwareCategory,
    Observation,
)


class ProfileStore:
    """Persist device profiles as JSON files.

    Each profile is stored as a separate JSON file organised by hardware
    category.  Default location: ``~/.warpt/profiles/``.

    Storage layout::

        ~/.warpt/profiles/
        └── gpu/
            └── {fingerprint_id}.json

    Parameters
    ----------
    base_dir : Path | None
        Root directory for profile files.  Defaults to
        ``~/.warpt/profiles/``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or (Path.home() / ".warpt" / "profiles")

    def _ensure_dir(self, category: HardwareCategory) -> None:
        """Create the category subdirectory if it doesn't exist."""
        (self._base_dir / category.value).mkdir(parents=True, exist_ok=True)

    def _profile_path(self, category: HardwareCategory, fingerprint_id: str) -> Path:
        """Return the file path for a profile."""
        return self._base_dir / category.value / f"{fingerprint_id}.json"

    def save_profile(self, profile: DeviceProfile) -> None:
        """Write or overwrite a profile JSON file.

        Parameters
        ----------
        profile : DeviceProfile
            The profile to persist.
        """
        category = profile.fingerprint.category
        self._ensure_dir(category)
        path = self._profile_path(category, profile.fingerprint.fingerprint_id)
        with open(path, "w") as f:
            json.dump(profile.model_dump(), f, indent=2)

    def get_profile(
        self, category: HardwareCategory, fingerprint_id: str
    ) -> DeviceProfile | None:
        """Read a single profile from disk.

        Parameters
        ----------
        category : HardwareCategory
            Hardware category to look in.
        fingerprint_id : str
            The device fingerprint ID.

        Returns
        -------
        DeviceProfile | None
            The profile, or None if not found.
        """
        path = self._profile_path(category, fingerprint_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return DeviceProfile.model_validate(data)

    def get_profiles(self, category: HardwareCategory) -> list[DeviceProfile]:
        """List all profiles in a category.

        Parameters
        ----------
        category : HardwareCategory
            Hardware category to list.

        Returns
        -------
        list[DeviceProfile]
            All profiles found in the category directory.
        """
        cat_dir = self._base_dir / category.value
        if not cat_dir.exists():
            return []

        profiles: list[DeviceProfile] = []
        for path in cat_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                profiles.append(DeviceProfile.model_validate(data))
            except (json.JSONDecodeError, ValueError):
                continue
        return profiles

    def delete_profile(self, category: HardwareCategory, fingerprint_id: str) -> None:
        """Remove a profile file.

        Parameters
        ----------
        category : HardwareCategory
            Hardware category of the profile.
        fingerprint_id : str
            The device fingerprint ID.
        """
        path = self._profile_path(category, fingerprint_id)
        if path.exists():
            path.unlink()

    def add_observation(
        self,
        category: HardwareCategory,
        fingerprint_id: str,
        observation: Observation,
    ) -> None:
        """Append an observation to an existing profile.

        Loads the profile, appends the observation, updates ``last_seen``
        and ``observation_count``, and writes it back.

        Parameters
        ----------
        category : HardwareCategory
            Hardware category of the profile.
        fingerprint_id : str
            The device fingerprint ID.
        observation : Observation
            The observation to append.

        Raises
        ------
        FileNotFoundError
            If no profile exists for the given category/fingerprint.
        """
        profile = self.get_profile(category, fingerprint_id)
        if profile is None:
            msg = f"No profile found for {category.value}/{fingerprint_id}"
            raise FileNotFoundError(msg)

        profile.observations.append(observation)
        profile.last_seen = observation.timestamp
        profile.observation_count = len(profile.observations)
        self.save_profile(profile)
