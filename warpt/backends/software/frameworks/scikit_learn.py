"""scikit-learn framework detection."""

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class ScikitLearnDetector(FrameworkDetector):
    """Detector for scikit-learn installation."""

    @property
    def framework_name(self) -> str:
        """Return the canonical name of the framework."""
        return "scikit_learn"

    def detect(self) -> FrameworkInfo | None:
        """Detect scikit-learn installation and gather version information.

        Returns
        -------
            FrameworkInfo with version if installed, None otherwise.
        """
        sklearn = self._safe_import("sklearn")
        if sklearn is None:
            return None

        # Get version
        try:
            version = sklearn.__version__  # type: ignore[attr-defined]
        except AttributeError:
            version = "unknown"

        # scikit-learn is CPU-only machine learning library
        return FrameworkInfo(
            installed=True,
            version=version,
            cuda_support=False,
        )
