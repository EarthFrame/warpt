"""NumPy framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class NumPyDetector(FrameworkDetector):
    """Detector for NumPy installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "numpy"

    def detect(self) -> FrameworkInfo | None:
        """Detect NumPy installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        numpy = self._safe_import("numpy")
        if numpy is None:
            return None

        # Get version
        try:
            version = numpy.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # NumPy is CPU-only library
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
