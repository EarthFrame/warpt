"""ONNX framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class ONNXDetector(FrameworkDetector):
    """Detector for ONNX installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "onnx"

    def detect(self) -> FrameworkInfo | None:
        """Detect ONNX installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        onnx = self._safe_import("onnx")
        if onnx is None:
            return None

        # Get version
        try:
            version = onnx.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,  # ONNX is a model format, not a compute framework
        )
