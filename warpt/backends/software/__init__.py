"""Software detection backends."""

from warpt.backends.software.base import SoftwareDetector
from warpt.backends.software.docker import DockerDetector
from warpt.backends.software.frameworks import (
    FrameworkDetector,
    PyTorchDetector,
    detect_all_frameworks,
    detect_framework,
)
from warpt.models.list_models import DockerInfo

__all__ = [
    "DockerDetector",
    "DockerInfo",
    "FrameworkDetector",
    "PyTorchDetector",
    "SoftwareDetector",
    "detect_all_frameworks",
    "detect_framework",
]
