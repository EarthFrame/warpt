"""Einops framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class EinopsDetector(FrameworkDetector):
    """Detector for Einops installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "einops"

    def detect(self) -> FrameworkInfo | None:
        """Detect Einops installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        einops = self._safe_import("einops")
        if einops is None:
            return None

        # Get version
        try:
            version = einops.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Einops is a tensor manipulation library, CPU-only
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
