"""Constants for warpt models and commands."""

from enum import StrEnum, auto


class Target(StrEnum):
    """Hardware and software targets for monitoring and stress testing."""

    # Hardware targets
    CPU = auto()
    GPU = auto()
    MEMORY = auto()
    STORAGE = auto()
    NETWORK = auto()

    # Software targets
    PYTORCH = auto()
    TENSORFLOW = auto()
    JAX = auto()
    CUDA = auto()
    DRIVERS = auto()


class Status(StrEnum):
    """Status values for tests and operations."""

    PASS = auto()
    FAIL = auto()
    WARNING = auto()
    STOPPED = auto()
    RUNNING = auto()


class StorageType(StrEnum):
    """Storage device types."""

    NVME_SSD = "nvme_ssd"
    SATA_SSD = "sata_ssd"
    SATA_HDD = "sata_hdd"


class MemoryType(StrEnum):
    """Memory types."""

    DDR3 = "ddr3"
    DDR4 = "ddr4"
    DDR5 = "ddr5"
    HBM2 = "hbm2"


# Stress test timing defaults (in seconds)
DEFAULT_STRESS_DURATION = 30  # Duration when --duration not specified
DEFAULT_BURNIN_DURATION = 5   # Warmup period before measurements
