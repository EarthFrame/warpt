"""Framework detection utilities."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.backends.software.frameworks.jax import JAXDetector
from warpt.backends.software.frameworks.pytorch import PyTorchDetector
from warpt.models.list_models import FrameworkInfo

__all__ = [
    "FrameworkDetector",
    "JAXDetector",
    "PyTorchDetector",
    "detect_all_frameworks",
    "detect_framework",
]


# Registry of all available framework detectors
_FRAMEWORK_DETECTORS = [
    PyTorchDetector(),
    JAXDetector(),
]


def detect_all_frameworks() -> dict[str, FrameworkInfo]:
    """Detect all available ML frameworks.

    Returns
    -------
        Dictionary mapping framework names to their FrameworkInfo objects.
        Only includes frameworks that are actually installed.
    """
    detected = {}
    for detector in _FRAMEWORK_DETECTORS:
        info = detector.detect()
        if info is not None:
            detected[detector.framework_name] = info
    return detected


def detect_framework(framework_name: str) -> FrameworkInfo | None:
    """Detect a specific framework by name.

    Args:
        framework_name: Name of the framework to detect (e.g., 'pytorch')

    Returns
    -------
        FrameworkInfo object if framework is installed, None otherwise.
    """
    for detector in _FRAMEWORK_DETECTORS:
        if detector.framework_name == framework_name:
            return detector.detect()
    return None
