"""Polars framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class PolarsDetector(FrameworkDetector):
    """Detector for Polars installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "polars"

    def detect(self) -> FrameworkInfo | None:
        """Detect Polars installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        polars = self._safe_import("polars")
        if polars is None:
            return None

        # Get version
        try:
            version = polars.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Polars is CPU-only data frame library (though GPU support exists
        # in some specialized builds)
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
