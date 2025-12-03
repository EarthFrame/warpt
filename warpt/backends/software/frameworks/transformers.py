"""Hugging Face Transformers framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class TransformersDetector(FrameworkDetector):
    """Detector for Hugging Face Transformers installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "transformers"

    def detect(self) -> FrameworkInfo | None:
        """Detect Hugging Face Transformers installation and gather version info.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        transformers = self._safe_import("transformers")
        if transformers is None:
            return None

        # Get version
        try:
            version = transformers.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Transformers supports GPU through PyTorch/TensorFlow backends
        # For simplicity, check if PyTorch or TensorFlow with GPU is available
        cuda_support = False
        try:
            torch = self._safe_import("torch")
            if torch is not None:
                cuda_support = torch.cuda.is_available()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass

        if not cuda_support:
            try:
                tf = self._safe_import("tensorflow")
                if tf is not None:
                    gpus = tf.config.list_physical_devices("GPU")  # type: ignore[attr-defined]
                    cuda_support = len(gpus) > 0
            except (AttributeError, RuntimeError):
                pass

        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=cuda_support,
        )
