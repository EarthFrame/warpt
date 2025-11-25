"""System backend - provides CPU and RAM information using psutil and system calls."""

import platform
import subprocess

import psutil
from pydantic import BaseModel, Field


class SwapMemoryInfo(BaseModel):
    """Swap memory information."""

    total: int = Field(..., description="Total swap memory in bytes")
    used: int = Field(..., description="Used swap memory in bytes")
    free: int = Field(..., description="Free swap memory in bytes")
    percent: float = Field(..., description="Percentage of swap memory used")
    sin: int = Field(..., description="Bytes swapped in from disk (cumulative)")
    sout: int = Field(..., description="Bytes swapped out to disk (cumulative)")

    class Config:
        """Pydantic config."""

        frozen = True

    @property
    def total_gb(self) -> float:
        """Get total swap memory in GB."""
        return self.total / (1024**3)

    @property
    def used_gb(self) -> float:
        """Get used swap memory in GB."""
        return self.used / (1024**3)

    @property
    def free_gb(self) -> float:
        """Get free swap memory in GB."""
        return self.free / (1024**3)

    def format_bytes(self, bytes_value: int | float) -> str:
        """Format bytes into human-readable string.

        Args:
            bytes_value: Number of bytes

        Returns:
            Human-readable string (e.g., "8.0 GB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"Swap: {self.format_bytes(self.used)} / "
            f"{self.format_bytes(self.total)} ({self.percent:.1f}%)"
        )


class RAMInfo(BaseModel):
    """Complete RAM information for the system."""

    # Physical RAM
    total: int = Field(..., description="Total physical RAM in bytes")
    available: int = Field(
        ...,
        description="Available RAM in bytes (estimate of memory available for "
        "new processes without swapping)",
    )
    used: int = Field(
        ...,
        description="Used RAM in bytes (may not equal total - available)",
    )
    free: int = Field(..., description="Free RAM in bytes (completely unused)")
    percent: float = Field(..., description="Percentage of RAM used")

    # Platform-specific metrics
    active: int | None = Field(
        None,
        description="Memory currently in use or recently used (macOS/BSD)",
    )
    inactive: int | None = Field(
        None, description="Memory marked as not used (macOS/BSD)"
    )
    buffers: int | None = Field(
        None, description="Cache for filesystem metadata (Linux)"
    )
    cached: int | None = Field(None, description="Cache for various things (Linux)")
    shared: int | None = Field(
        None, description="Memory shared between processes (Linux)"
    )
    wired: int | None = Field(
        None, description="Memory marked to always stay in RAM (macOS)"
    )

    # Swap information
    swap: SwapMemoryInfo | None = Field(None, description="Swap memory information")

    class Config:
        """Pydantic config."""

        frozen = True

    @property
    def total_gb(self) -> float:
        """Get total RAM in GB."""
        return self.total / (1024**3)

    @property
    def available_gb(self) -> float:
        """Get available RAM in GB."""
        return self.available / (1024**3)

    @property
    def used_gb(self) -> float:
        """Get used RAM in GB."""
        return self.used / (1024**3)

    @property
    def free_gb(self) -> float:
        """Get free RAM in GB."""
        return self.free / (1024**3)

    def format_bytes(self, bytes_value: int | float) -> str:
        """
        Format bytes into human-readable string.

        Args:
            bytes_value: Number of bytes

        Returns:
            Human-readable string (e.g., "8.0 GB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"RAM: {self.format_bytes(self.used)} / "
            f"{self.format_bytes(self.total)} ({self.percent:.1f}% used, "
            f"{self.format_bytes(self.available)} available)"
        )

    def summary(self) -> str:
        """
        Return a detailed summary of RAM information.

        Returns:
            Multi-line string with comprehensive RAM details.
        """
        lines = [
            "Physical RAM:",
            f"  Total:     {self.format_bytes(self.total)}",
            f"  Available: {self.format_bytes(self.available)}",
            f"  Used:      {self.format_bytes(self.used)} ({self.percent:.1f}%)",
            f"  Free:      {self.format_bytes(self.free)}",
        ]

        # Add platform-specific metrics if present
        platform_metrics = []
        if self.active is not None:
            platform_metrics.append(f"  Active:    {self.format_bytes(self.active)}")
        if self.inactive is not None:
            platform_metrics.append(f"  Inactive:  {self.format_bytes(self.inactive)}")
        if self.wired is not None:
            platform_metrics.append(f"  Wired:     {self.format_bytes(self.wired)}")
        if self.buffers is not None:
            platform_metrics.append(f"  Buffers:   {self.format_bytes(self.buffers)}")
        if self.cached is not None:
            platform_metrics.append(f"  Cached:    {self.format_bytes(self.cached)}")
        if self.shared is not None:
            platform_metrics.append(f"  Shared:    {self.format_bytes(self.shared)}")

        if platform_metrics:
            lines.append("")
            lines.append("Platform-Specific:")
            lines.extend(platform_metrics)

        # Add swap information if present
        if self.swap is not None:
            lines.append("")
            lines.append(str(self.swap))

        return "\n".join(lines)


class RAM:
    """
    RAM information backend using psutil.

    Example:
        >>> ram = RAM()
        >>> info = ram.get_ram_info()
        >>> print(info)  # Human-readable output
        >>> print(f"Available: {info.available_gb:.1f} GB")
        >>> ram.print_summary()  # Quick detailed view
    """

    def __init__(self) -> None:
        """Initialize RAM backend."""
        self._info: RAMInfo | None = None

    def _detect_ddr_info(self) -> tuple[str | None, int | None]:
        """
        Detect DDR type and speed.

        Returns:
            Tuple of (ddr_type, speed_mhz) or (None, None) if detection fails.
        """
        # Try to detect from dmidecode (Linux)
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["sudo", "dmidecode", "-t", "memory"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    ddr_type = None
                    speed = None

                    for line in lines:
                        if "Type:" in line and "DDR" in line:
                            # Extract DDR type (e.g., "DDR4", "DDR5")
                            for part in line.split():
                                if "DDR" in part:
                                    ddr_type = part.strip()
                                    break

                        if "Configured Memory Speed:" in line or "Speed:" in line:
                            # Extract speed (e.g., "3200 MT/s")
                            parts = line.split()
                            for j, part in enumerate(parts):
                                if part.isdigit() and j + 1 < len(parts):
                                    if "MT/s" in parts[j + 1] or "MHz" in parts[j + 1]:
                                        speed = int(part)
                                        break

                    return (ddr_type, speed)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        # Try macOS system_profiler
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPMemoryDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Type:" in line:
                            for part in line.split():
                                if "DDR" in part or "LPDDR" in part:
                                    return (part.strip(), None)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        return (None, None)

    def _detect_memory_channels(self) -> int | None:
        """
        Detect number of memory channels.

        Returns:
            Number of memory channels or None if detection fails.
        """
        # Try to detect from dmidecode (Linux)
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["sudo", "dmidecode", "-t", "memory"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # Count unique "Handle" lines to estimate channels
                    handles = result.stdout.count("Handle")
                    if handles > 0:
                        return handles
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        return None

    def get_ram_info(self) -> RAMInfo:
        """
        Get comprehensive RAM information.

        Returns:
            RAMInfo object with memory statistics.
        """
        if self._info is not None:
            return self._info

        # Get virtual memory info
        vmem = psutil.virtual_memory()

        # Get swap memory info
        swap_info = None
        try:
            swap = psutil.swap_memory()
            swap_info = SwapMemoryInfo(
                total=swap.total,
                used=swap.used,
                free=swap.free,
                percent=swap.percent,
                sin=swap.sin,
                sout=swap.sout,
            )
        except (AttributeError, OSError):
            pass

        # Build platform-specific fields
        platform_fields = {}
        for field in ["active", "inactive", "buffers", "cached", "shared", "wired"]:
            if hasattr(vmem, field):
                platform_fields[field] = getattr(vmem, field)
            else:
                platform_fields[field] = None

        self._info = RAMInfo(
            total=vmem.total,
            available=vmem.available,
            used=vmem.used,
            free=vmem.free,
            percent=vmem.percent,
            swap=swap_info,
            **platform_fields,
        )

        return self._info

    def refresh(self) -> RAMInfo:
        """
        Refresh and return current RAM information.

        Unlike CPU info which is mostly static, RAM usage changes frequently.
        This method forces a refresh of the cached data.

        Returns:
            Updated RAMInfo object with current memory statistics.
        """
        self._info = None
        return self.get_ram_info()

    def print_summary(self) -> None:
        """Print a detailed summary of RAM information to stdout."""
        info = self.get_ram_info()
        print(info.summary())

    def is_memory_pressure(self, threshold: float = 80.0) -> bool:
        """
        Check if system is under memory pressure.

        Args:
            threshold: Memory usage percentage threshold (default: 80%)

        Returns:
            True if memory usage exceeds threshold
        """
        info = self.refresh()
        return info.percent >= threshold

    def get_available_gb(self) -> float:
        """
        Get available RAM in GB.

        Returns:
            Available RAM in gigabytes
        """
        info = self.get_ram_info()
        return info.available_gb
