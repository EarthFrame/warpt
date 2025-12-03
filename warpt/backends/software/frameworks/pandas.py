"""Pandas framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class PandasDetector(FrameworkDetector):
    """Detector for Pandas installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "pandas"

    def detect(self) -> FrameworkInfo | None:
        """Detect Pandas installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        pandas = self._safe_import("pandas")
        if pandas is None:
            return None

        # Get version
        try:
            version = pandas.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Pandas is CPU-only data analysis library
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
