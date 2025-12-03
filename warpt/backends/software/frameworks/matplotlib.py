"""Matplotlib framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class MatplotlibDetector(FrameworkDetector):
    """Detector for Matplotlib installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "matplotlib"

    def detect(self) -> FrameworkInfo | None:
        """Detect Matplotlib installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        matplotlib = self._safe_import("matplotlib")
        if matplotlib is None:
            return None

        # Get version
        try:
            version = matplotlib.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Matplotlib is a visualization library, CPU-only
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
