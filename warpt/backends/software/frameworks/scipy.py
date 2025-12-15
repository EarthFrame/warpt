"""SciPy framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class SciPyDetector(FrameworkDetector):
    """Detector for SciPy installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "scipy"

    def detect(self) -> FrameworkInfo | None:
        """Detect SciPy installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        scipy = self._safe_import("scipy")
        if scipy is None:
            return None

        # Get version
        try:
            version = scipy.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # SciPy is CPU-only library
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
