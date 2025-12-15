"""PyTorch Lightning framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class PyTorchLightningDetector(FrameworkDetector):
    """Detector for PyTorch Lightning installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "pytorch_lightning"

    def detect(self) -> FrameworkInfo | None:
        """Detect PyTorch Lightning installation and gather version information.

        Returns
        -------
            FrameworkInfo with version and CUDA support if installed,
            None if PyTorch Lightning is not installed.
        """
        lightning = self._safe_import("pytorch_lightning")
        if lightning is None:
            return None

        # Get version
        try:
            version = lightning.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # PyTorch Lightning inherits GPU support from PyTorch backend
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
