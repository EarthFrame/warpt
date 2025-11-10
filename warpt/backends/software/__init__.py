"""Software detection backends."""

from warpt.backends.software.frameworks import (
    FrameworkDetector,
    PyTorchDetector,
    detect_all_frameworks,
    detect_framework,
)

__all__ = [
    "FrameworkDetector",
    "PyTorchDetector",
    "detect_all_frameworks",
    "detect_framework",
]
