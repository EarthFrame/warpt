"""Software detection backends."""

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

__all__ = [
    "FrameworkDetector",
    "PyTorchDetector",
    "detect_all_frameworks",
    "detect_framework",
    "NvidiaContainerToolkitDetectionResult",
    "NvidiaContainerToolkitDetector",
]
