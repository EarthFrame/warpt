"""Einx framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class EinxDetector(FrameworkDetector):
    """Detector for Einx installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "einx"

    def detect(self) -> FrameworkInfo | None:
        """Detect Einx installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        einx = self._safe_import("einx")
        if einx is None:
            return None

        # Get version
        try:
            version = einx.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Einx is a flexible neural network library
        # GPU support depends on the backend (PyTorch, TensorFlow, JAX)
        cuda_support = False
        try:
            torch = self._safe_import("torch")
            if torch is not None:
                cuda_support = torch.cuda.is_available()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass

        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=cuda_support,
        )
