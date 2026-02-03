"""Apache MXNet framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class MXNetDetector(FrameworkDetector):
    """Detector for Apache MXNet installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "mxnet"

    def detect(self) -> FrameworkInfo | None:
        """Detect MXNet installation and gather version information.

        Returns
        -------
            FrameworkInfo with version and CUDA support status if installed,
            None if MXNet is not installed.
        """
        mxnet = self._safe_import("mxnet")
        if mxnet is None:
            return None

        # Get version
        try:
            version = mxnet.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Check for CUDA support
        cuda_support = False
        try:
            # Check if MXNet is built with GPU support
            num_gpus = mxnet.device.num_gpus()  # type: ignore[attr-defined]
            cuda_support = num_gpus > 0
        except (AttributeError, RuntimeError):
            pass

        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=cuda_support,
        )
