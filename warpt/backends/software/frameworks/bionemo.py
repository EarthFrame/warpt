"""NVIDIA BioNeMo framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class BioNeMoDetector(FrameworkDetector):
    """Detector for NVIDIA BioNeMo installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "bionemo"

    def detect(self) -> FrameworkInfo | None:
        """Detect BioNeMo installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        bionemo = self._safe_import("bionemo")
        if bionemo is None:
            return None

        # Get version
        try:
            version = bionemo.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # BioNeMo is GPU-optimized and requires CUDA
        return FrameworkInfo(
            version=version,
            cuda_support=True,
        )
