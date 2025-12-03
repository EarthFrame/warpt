"""Keras framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class KerasDetector(FrameworkDetector):
    """Detector for Keras installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "keras"

    def detect(self) -> FrameworkInfo | None:
        """Detect Keras installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        keras = self._safe_import("keras")
        if keras is None:
            return None

        # Get version
        try:
            version = keras.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Keras can run on various backends (TensorFlow, etc.) with GPU support
        # For simplicity, assume CUDA support is available if Keras is installed
        # with a proper GPU-capable backend
        cuda_support = False
        try:
            # Check if TensorFlow backend has GPU support
            tf = self._safe_import("tensorflow")
            if tf is not None:
                gpus = tf.config.list_physical_devices("GPU")  # type: ignore[attr-defined]
                cuda_support = len(gpus) > 0
        except (AttributeError, RuntimeError):
            pass

        return FrameworkInfo(
            version=version,
            cuda_support=cuda_support,
        )
