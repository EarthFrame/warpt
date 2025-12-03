"""JAX framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class JAXDetector(FrameworkDetector):
    """Detector for JAX installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "jax"

    def detect(self) -> FrameworkInfo | None:
        """Detect JAX installation and gather version information.

        Returns
        -------
            FrameworkInfo with version and CUDA support status if installed,
            None if JAX is not installed.
        """
        jax = self._safe_import("jax")
        if jax is None:
            return None

        # Get version
        try:
            version = jax.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Check for CUDA support
        cuda_support = False
        try:
            # JAX devices include GPU information
            devices = jax.devices()  # type: ignore[attr-defined]
            # Check if any device is a GPU
            cuda_support = any("gpu" in str(d).lower() for d in devices)
        except (AttributeError, RuntimeError):
            # If we can't get devices, try checking for GPU support directly
            pass

        return FrameworkInfo(
            version=version,
            cuda_support=cuda_support,
        )
