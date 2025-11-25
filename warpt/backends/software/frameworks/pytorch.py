"""PyTorch framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class PyTorchDetector(FrameworkDetector):
    """Detector for PyTorch installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "pytorch"

    def detect(self) -> FrameworkInfo | None:
        """Detect PyTorch installation and gather version information.

        Returns
        -------
            FrameworkInfo with version and CUDA support status if installed,
            None if PyTorch is not installed.
        """
        torch = self._safe_import("torch")
        if torch is None:
            return None

        # Get version
        try:
            version = torch.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Check for CUDA support
        cuda_support = False
        try:
            cuda_support = torch.cuda.is_available()  # type: ignore[attr-defined]
        except AttributeError:
            # If cuda module doesn't exist, CUDA is not supported
            pass

        return FrameworkInfo(
            version=version,
            cuda_support=cuda_support,
        )
