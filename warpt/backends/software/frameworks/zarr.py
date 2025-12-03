"""Zarr framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class ZarrDetector(FrameworkDetector):
    """Detector for Zarr installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "zarr"

    def detect(self) -> FrameworkInfo | None:
        """Detect Zarr installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        zarr = self._safe_import("zarr")
        if zarr is None:
            return None

        # Get version
        try:
            version = zarr.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Zarr is a storage format library, CPU-only
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
