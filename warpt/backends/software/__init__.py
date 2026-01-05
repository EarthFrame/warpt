"""Software detection backends."""

from warpt.backends.software.base import SoftwareDetector
from warpt.backends.software.docker import DockerDetector
from warpt.backends.software.frameworks import (
    FrameworkDetector,
    PyTorchDetector,
    detect_all_frameworks,
    detect_framework,
)
from warpt.backends.software.libraries import (
    LibraryDetector,
    detect_all_libraries,
)
from warpt.backends.software.nvidia_toolkit import (
    NvidiaContainerToolkitDetector,
    NvidiaContainerToolkitInfo,
)
from warpt.models.list_models import DockerInfo, LibraryInfo

__all__ = [
    "DockerDetector",
    "DockerInfo",
    "FrameworkDetector",
    "LibraryDetector",
    "LibraryInfo",
    "NvidiaContainerToolkitDetector",
    "NvidiaContainerToolkitInfo",
    "PyTorchDetector",
    "SoftwareDetector",
    "detect_all_frameworks",
    "detect_all_libraries",
    "detect_framework",
]
