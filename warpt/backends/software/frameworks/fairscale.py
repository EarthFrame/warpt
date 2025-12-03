"""FairScale framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class FairScaleDetector(FrameworkDetector):
    """Detector for FairScale installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "fairscale"

    def detect(self) -> FrameworkInfo | None:
        """Detect FairScale installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        fairscale = self._safe_import("fairscale")
        if fairscale is None:
            return None

        # Get version
        try:
            version = fairscale.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # FairScale is PyTorch-based distributed training library
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
