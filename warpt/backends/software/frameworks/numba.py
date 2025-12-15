"""Numba framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class NumbaDetector(FrameworkDetector):
    """Detector for Numba installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "numba"

    def detect(self) -> FrameworkInfo | None:
        """Detect Numba installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        numba = self._safe_import("numba")
        if numba is None:
            return None

        # Get version
        try:
            version = numba.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # Numba supports CUDA compilation for GPU acceleration
        cuda_support = False
        try:
            from numba.cuda import is_available

            cuda_support = is_available()
        except (ImportError, RuntimeError):
            pass

        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=cuda_support,
        )
