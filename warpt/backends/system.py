"""CPU backend - provides CPU information using psutil and system calls."""

import platform
import subprocess
from enum import Enum

import psutil
from pydantic import BaseModel, ConfigDict, Field


class FrequencyScope(str, Enum):
    """Scope at which CPU frequency is measured."""

    SYSTEM = "system"
    PER_CORE = "per_core"
    PER_SOCKET = "per_socket"


class CPUFrequency(BaseModel):
    """CPU frequency information in MHz."""

    current: float = Field(..., description="Current CPU frequency in MHz")
    min: float = Field(..., description="Minimum CPU frequency in MHz")
    max: float = Field(..., description="Maximum CPU frequency in MHz")

    model_config = ConfigDict(frozen=True)


class SocketCPUInfo(BaseModel):
    """CPU information for a single socket."""

    socket_id: int = Field(..., description="Physical socket ID")
    make: str = Field(..., description="CPU manufacturer (e.g., 'Intel', 'AMD')")
    model: str = Field(..., description="CPU model string")
    physical_cores: int = Field(..., description="Physical cores in this socket")
    logical_cores: int = Field(
        ..., description="Logical cores (threads) in this socket"
    )
    base_frequency: float | None = Field(None, description="Base frequency in MHz")
    boost_frequency_single_core: float | None = Field(
        None, description="Max single-core boost frequency in MHz"
    )
    boost_frequency_multi_core: float | None = Field(
        None, description="Multi-core boost frequency in MHz"
    )

    model_config = ConfigDict(frozen=True)


class CPUInfo(BaseModel):
    """Complete CPU information for the system.

    For simple homogeneous systems, the top-level fields provide all needed info.
    For heterogeneous systems, detailed per-socket information is in socket_info.
    """

    # Identification (primary/most common CPU in system)
    make: str = Field(
        ...,
        description="Primary CPU manufacturer (e.g., 'Apple', 'Intel', 'AMD')",
    )
    model: str = Field(..., description="Primary CPU model")
    architecture: str = Field(..., description="CPU architecture (e.g., x86_64, arm64)")

    # Topology (system totals)
    total_sockets: int = Field(..., description="Number of physical CPU sockets")
    total_physical_cores: int = Field(
        ..., description="Total physical cores across all sockets"
    )
    total_logical_cores: int = Field(
        ...,
        description="Total logical cores (threads) across all sockets",
    )

    # Frequencies (system-level summary or primary socket)
    base_frequency: float | None = Field(
        None, description="Base/minimum frequency in MHz"
    )
    boost_frequency_single_core: float | None = Field(
        None, description="Maximum single-core boost frequency in MHz"
    )
    boost_frequency_multi_core: float | None = Field(
        None, description="Multi-core boost frequency in MHz (all-core turbo)"
    )
    current_frequency: float | None = Field(
        None, description="Current frequency in MHz"
    )
    current_frequency_scope: FrequencyScope | None = Field(
        None, description="Scope of current frequency measurement"
    )

    # Detailed breakdown for heterogeneous systems
    socket_info: list[SocketCPUInfo] | None = Field(
        None, description="Per-socket CPU details (for heterogeneous systems)"
    )

    model_config = ConfigDict(frozen=True)


class CPU:
    """CPU information backend using psutil."""

    def __init__(self) -> None:
        """Initialize CPU backend and cache information."""
        self._info: CPUInfo | None = None

    def get_cpu_info(self) -> CPUInfo:
        """Get comprehensive CPU information.

        Returns
        -------
            CPUInfo object with system-level summary and optional per-socket
            details.
        """
        if self._info is not None:
            return self._info

        # Get brand string and parse it
        brand_string = self._get_cpu_brand()
        make, model = self._parse_brand(brand_string)

        # Get core/thread counts
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1

        # Estimate number of sockets (simplified)
        sockets = max(1, logical_cores // physical_cores) if physical_cores > 0 else 1

        # Get frequency information
        base_freq = None
        single_core_boost = None
        multi_core_boost = None
        current_freq = None
        current_freq_scope = None

        try:
            freq = psutil.cpu_freq()
            if freq:
                base_freq = freq.min
                # For now, treat max as single-core boost
                # Multi-core boost requires platform-specific detection
                single_core_boost = freq.max
                current_freq = freq.current
                current_freq_scope = FrequencyScope.SYSTEM
        except (AttributeError, OSError):
            pass

        # Get architecture
        architecture = platform.machine()

        # Create socket info list (for potential heterogeneous systems)
        # For now, we create entries assuming homogeneous system
        # Platform-specific code could enhance this
        socket_list = []
        if sockets > 1:
            for socket_id in range(sockets):
                socket_info = SocketCPUInfo(
                    socket_id=socket_id,
                    make=make,
                    model=model,
                    physical_cores=physical_cores,
                    logical_cores=logical_cores,
                    base_frequency=base_freq,
                    boost_frequency_single_core=single_core_boost,
                    boost_frequency_multi_core=multi_core_boost,
                )
                socket_list.append(socket_info)

        self._info = CPUInfo(
            make=make,
            model=model,
            architecture=architecture,
            total_sockets=sockets,
            total_physical_cores=physical_cores,
            total_logical_cores=logical_cores,
            base_frequency=base_freq,
            boost_frequency_single_core=single_core_boost,
            boost_frequency_multi_core=multi_core_boost,
            current_frequency=current_freq,
            current_frequency_scope=current_freq_scope,
            socket_info=socket_list if socket_list else None,
        )

        return self._info

    @staticmethod
    def _get_cpu_brand() -> str:
        """Get CPU brand string from system.

        Tries multiple methods to get CPU information:
        1. macOS: sysctl machdep.cpu.brand_string
        2. Linux: /proc/cpuinfo
        3. Fallback: platform.processor()

        Returns
        -------
            CPU brand string
        """
        # Try macOS method
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        # Try Linux method
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except (FileNotFoundError, OSError):
            pass

        # Fallback to platform.processor()
        processor = platform.processor()
        if processor:
            return processor

        return "Unknown CPU"

    @staticmethod
    def _parse_brand(brand_string: str) -> tuple[str, str]:
        """Parse CPU brand string into make and model.

        Examples
        --------
            "Apple M2" -> ("Apple", "M2")
            "Intel Core i9-13900K" -> ("Intel", "Core i9-13900K")
            "AMD Ryzen 9 7950X" -> ("AMD", "Ryzen 9 7950X")
            "arm" -> ("arm", "")

        Args:
            brand_string: Full CPU brand string

        Returns
        -------
            Tuple of (make, model)
        """
        parts = brand_string.strip().split(None, 1)

        if len(parts) == 0:
            return ("Unknown", "")
        elif len(parts) == 1:
            return (parts[0], "")
        else:
            return (parts[0], parts[1])


# Convenience alias for backwards compatibility
System = CPU
