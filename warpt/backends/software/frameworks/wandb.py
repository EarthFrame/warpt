"""Weights & Biases (wandb) framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class WandBDetector(FrameworkDetector):
    """Detector for Weights & Biases (wandb) installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "wandb"

    def detect(self) -> FrameworkInfo | None:
        """Detect wandb installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        wandb = self._safe_import("wandb")
        if wandb is None:
            return None

        # Get version
        try:
            version = wandb.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # wandb is a CPU-only experiment tracking and logging platform
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
