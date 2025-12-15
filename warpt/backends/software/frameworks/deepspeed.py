"""DeepSpeed framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class DeepSpeedDetector(FrameworkDetector):
    """Detector for DeepSpeed installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "deepspeed"

    def detect(self) -> FrameworkInfo | None:
        """Detect DeepSpeed installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        deepspeed = self._safe_import("deepspeed")
        if deepspeed is None:
            return None

        # Get version
        try:
            version = deepspeed.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # DeepSpeed is GPU-optimized for distributed training
        # It supports CUDA for GPU operations
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=True,
        )
