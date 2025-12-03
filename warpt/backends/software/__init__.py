"""Software detection backends."""

from warpt.backends.software.base import SoftwareDetector
from warpt.backends.software.docker import DockerDetector
from warpt.backends.software.frameworks import (
    FrameworkDetector,
    PyTorchDetector,
    detect_all_frameworks,
    detect_framework,
)
from warpt.backends.software.nvidia_toolkit import (
    NvidiaContainerToolkitDetectionResult,
    NvidiaContainerToolkitDetector,
)
from warpt.models.list_models import DockerInfo

__all__ = [
    "DockerDetector",
    "DockerInfo",
    "FrameworkDetector",
    "NvidiaContainerToolkitDetectionResult",
    "NvidiaContainerToolkitDetector",
    "PyTorchDetector",
    "SoftwareDetector",
    "detect_all_frameworks",
    "detect_framework",
]
