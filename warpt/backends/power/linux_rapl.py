"""Linux RAPL (Running Average Power Limit) power monitoring backend.

RAPL provides power measurements for Intel and AMD CPUs through the
powercap sysfs interface at /sys/class/powercap/intel-rapl/.

Power domains typically available:
- package-0: Total CPU package power (cores + uncore)
- core: CPU cores only
- uncore: Integrated GPU, memory controller, cache
- dram: Memory subsystem
- psys: Entire platform (on some systems)

Note: Reading RAPL requires either:
1. Root access
2. Read permissions on /sys/class/powercap/intel-rapl/
3. CAP_SYS_RAWIO capability
"""

from __future__ import annotations

import time
from pathlib import Path

from warpt.backends.power.base import PowerBackend
from warpt.models.power_models import (
    DomainPower,
    PowerDomain,
    PowerSource,
    RAPLDomain,
)

# Mapping from RAPL domain names to our PowerDomain enum
RAPL_DOMAIN_MAP: dict[str, PowerDomain] = {
    "package": PowerDomain.PACKAGE,
    "core": PowerDomain.CORE,
    "uncore": PowerDomain.UNCORE,
    "dram": PowerDomain.DRAM,
    "psys": PowerDomain.PSYS,
}


class LinuxRAPLBackend(PowerBackend):
    """Backend for reading power via Linux RAPL interface.

    Uses the powercap sysfs interface to read energy counters and
    calculate power consumption.
    """

    POWERCAP_PATH = Path("/sys/class/powercap")
    RAPL_PREFIX = "intel-rapl"  # Used by both Intel and AMD on modern kernels

    def __init__(self) -> None:
        """Initialize the RAPL backend."""
        self._domains: list[RAPLDomain] = []
        self._initialized = False
        # name -> (energy_uj, timestamp)
        self._last_readings: dict[str, tuple[int, float]] = {}

    def is_available(self) -> bool:
        """Check if RAPL is available on this system.

        Returns:
            True if RAPL powercap interface exists and is readable.
        """
        rapl_path = self.POWERCAP_PATH / self.RAPL_PREFIX
        if not rapl_path.exists():
            # Also check for AMD-specific path
            rapl_path = self.POWERCAP_PATH / "amd-rapl"
            if not rapl_path.exists():
                return False

        # Check if we can read energy values
        try:
            for entry in rapl_path.iterdir():
                if entry.is_dir() and entry.name.startswith(self.RAPL_PREFIX):
                    energy_file = entry / "energy_uj"
                    if energy_file.exists():
                        with open(energy_file) as f:
                            f.read()
                        return True
        except (PermissionError, OSError):
            return False

        return False

    def get_source(self) -> PowerSource:
        """Get the power source type.

        Returns:
            PowerSource.RAPL
        """
        return PowerSource.RAPL

    def initialize(self) -> bool:
        """Discover and initialize RAPL domains.

        Returns:
            True if at least one domain was found.
        """
        if self._initialized:
            return bool(self._domains)

        self._domains = []
        self._last_readings = {}

        # Check both intel-rapl and amd-rapl paths
        for prefix in ["intel-rapl", "amd-rapl"]:
            rapl_base = self.POWERCAP_PATH / prefix
            if not rapl_base.exists():
                continue

            self._discover_domains(rapl_base, prefix)

        self._initialized = True
        return bool(self._domains)

    def _discover_domains(self, base_path: Path, prefix: str) -> None:
        """Discover RAPL domains under a base path.

        Args:
            base_path: Path to the RAPL root (e.g., /sys/class/powercap/intel-rapl)
            prefix: Domain prefix (intel-rapl or amd-rapl)
        """
        try:
            for entry in sorted(base_path.iterdir()):
                if not entry.is_dir():
                    continue

                # Top-level domains (e.g., intel-rapl:0 for package-0)
                if entry.name.startswith(f"{prefix}:"):
                    self._add_domain(entry)

                    # Sub-domains (intel-rapl:0:0 for core, intel-rapl:0:1 for uncore)
                    for subentry in sorted(entry.iterdir()):
                        if subentry.is_dir() and subentry.name.startswith(f"{prefix}:"):
                            self._add_domain(subentry)

        except (PermissionError, OSError):
            pass

    def _add_domain(self, domain_path: Path) -> None:
        """Add a RAPL domain if it has readable energy file.

        Args:
            domain_path: Path to the domain directory.
        """
        energy_file = domain_path / "energy_uj"
        name_file = domain_path / "name"
        max_energy_file = domain_path / "max_energy_range_uj"

        if not energy_file.exists():
            return

        try:
            # Read domain name
            if name_file.exists():
                with open(name_file) as f:
                    name = f.read().strip()
            else:
                name = domain_path.name

            # Read max energy (for wraparound detection)
            max_energy = 0
            if max_energy_file.exists():
                with open(max_energy_file) as f:
                    max_energy = int(f.read().strip())

            # Test that we can read energy
            with open(energy_file) as f:
                initial_energy = int(f.read().strip())

            domain = RAPLDomain(
                name=name,
                path=str(domain_path),
                energy_path=str(energy_file),
                max_energy_uj=max_energy,
                last_energy_uj=initial_energy,
                last_timestamp=time.time(),
            )
            self._domains.append(domain)

            # Store initial reading
            self._last_readings[name] = (initial_energy, time.time())

        except (PermissionError, OSError, ValueError):
            pass

    def get_power_readings(self) -> list[DomainPower]:
        """Read current power from all RAPL domains.

        Power is calculated from the difference in energy counters
        divided by the time elapsed since the last reading.

        Returns:
            List of DomainPower objects with current power in watts.
        """
        if not self._initialized:
            self.initialize()

        readings: list[DomainPower] = []
        current_time = time.time()

        for domain in self._domains:
            try:
                with open(domain.energy_path) as f:
                    current_energy = int(f.read().strip())

                # Get last reading
                last_energy, last_time = self._last_readings.get(
                    domain.name, (current_energy, current_time)
                )

                # Calculate time delta
                time_delta = current_time - last_time
                if time_delta < 0.001:  # Less than 1ms, skip
                    continue

                # Calculate energy delta (handle wraparound)
                energy_delta = current_energy - last_energy
                if energy_delta < 0 and domain.max_energy_uj > 0:
                    # Counter wrapped around
                    energy_delta = (domain.max_energy_uj - last_energy) + current_energy

                # Convert microjoules to watts: (uj / 1e6) / seconds
                power_watts = (energy_delta / 1_000_000) / time_delta

                # Update stored reading
                self._last_readings[domain.name] = (current_energy, current_time)

                # Map domain name to PowerDomain enum
                power_domain = self._map_domain_name(domain.name)

                readings.append(
                    DomainPower(
                        domain=power_domain,
                        power_watts=max(0.0, power_watts),  # Sanity check
                        energy_joules=current_energy / 1_000_000,
                        source=PowerSource.RAPL,
                        metadata={
                            "rapl_name": domain.name,
                            "rapl_path": domain.path,
                        },
                    )
                )

            except (PermissionError, OSError, ValueError):
                continue

        return readings

    def _map_domain_name(self, name: str) -> PowerDomain:
        """Map RAPL domain name to PowerDomain enum.

        Args:
            name: RAPL domain name (e.g., "package-0", "core", "dram")

        Returns:
            Corresponding PowerDomain enum value.
        """
        # Normalize name
        name_lower = name.lower()

        # Direct mappings
        for key, domain in RAPL_DOMAIN_MAP.items():
            if key in name_lower:
                return domain

        # Default to package for unknown domains
        return PowerDomain.PACKAGE

    def get_domains(self) -> list[RAPLDomain]:
        """Get list of discovered RAPL domains.

        Returns:
            List of RAPLDomain objects.
        """
        if not self._initialized:
            self.initialize()
        return self._domains.copy()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._domains = []
        self._last_readings = {}
        self._initialized = False


def get_rapl_domains_info() -> list[dict]:
    """Return information about available RAPL domains.

    Returns:
        List of dictionaries with domain information.
    """
    backend = LinuxRAPLBackend()
    if not backend.is_available():
        return []

    backend.initialize()
    return [d.to_dict() for d in backend.get_domains()]
