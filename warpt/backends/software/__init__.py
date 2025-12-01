"""Software detection backends."""

from warpt.backends.software.docker import DockerDetectionResult, DockerDetector
from warpt.backends.software.frameworks import (
    FrameworkDetector,
    PyTorchDetector,
    detect_all_frameworks,
    detect_framework,
)

__all__ = [
    "DockerDetectionResult",
    "DockerDetector",
    "FrameworkDetector",
    "PyTorchDetector",
    "detect_all_frameworks",
    "detect_framework",
]
