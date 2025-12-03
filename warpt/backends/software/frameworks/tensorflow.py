"""TensorFlow framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class TensorFlowDetector(FrameworkDetector):
    """Detector for TensorFlow installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "tensorflow"

    def detect(self) -> FrameworkInfo | None:
        """Detect TensorFlow installation and gather version information.

        Returns
        -------
            FrameworkInfo with version and CUDA support status if installed,
            None if TensorFlow is not installed.
        """
        tf = self._safe_import("tensorflow")
        if tf is None:
            return None

        # Get version
        try:
            version = tf.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Check for GPU support
        cuda_support = False
        try:
            # Try to check if GPUs are available
            gpus = tf.config.list_physical_devices("GPU")  # type: ignore[attr-defined]
            cuda_support = len(gpus) > 0
        except (AttributeError, RuntimeError):
            # If we can't check, assume no GPU support
            pass

        return FrameworkInfo(
            version=version,
            cuda_support=cuda_support,
        )
