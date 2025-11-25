"""Base class for framework detection."""

import json
from abc import ABC, abstractmethod

from warpt.models.list_models import FrameworkInfo


class FrameworkDetector(ABC):
    """Base class for detecting ML framework installations.

    Subclasses should implement the detect() method to check for framework
    installation and return a FrameworkInfo object if found.
    """

    @abstractmethod
    def detect(self) -> FrameworkInfo | None:
        """Detect if the framework is installed and gather its information.

        Returns
        -------
            FrameworkInfo object if framework is installed, None otherwise.
        """
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the canonical name of the framework.

        Examples: 'pytorch', 'tensorflow', 'jax'
        """
        pass

    def to_dict(self) -> dict[str, str | bool] | None:
        """Convert framework info to dictionary format.

        Returns
        -------
            Dictionary with framework info, or None if not installed.
            Default implementation returns the FrameworkInfo as a dict.
            Override this method to customize the output structure.
        """
        info = self.detect()
        if info is None:
            return None
        return info.model_dump()

    def to_json(self, indent: int | None = 2) -> str | None:
        """Convert framework info to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact)

        Returns
        -------
            JSON string with framework info, or None if not installed.
            Override this method to customize the JSON output.
        """
        data = self.to_dict()
        if data is None:
            return None
        return json.dumps(data, indent=indent)

    def to_yaml(self) -> str | None:
        """Convert framework info to YAML string.

        Returns
        -------
            YAML string with framework info, or None if not installed.
            Override this method to customize the YAML output.

        Raises
        ------
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore[import-untyped, unused-ignore]
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML output. "
                "Install it with: pip install pyyaml"
            ) from e

        data = self.to_dict()
        if data is None:
            return None
        return yaml.dump(data, default_flow_style=False, sort_keys=False)  # type: ignore[no-any-return, unused-ignore]

    def to_toml(self) -> str | None:
        """Convert framework info to TOML string.

        Returns
        -------
            TOML string with framework info, or None if not installed.
            Override this method to customize the TOML output.

        Raises
        ------
            ImportError: If tomli_w is not installed.
        """
        try:
            import tomli_w
        except ImportError as e:
            raise ImportError(
                "tomli_w is required for TOML output. "
                "Install it with: pip install tomli_w"
            ) from e

        data = self.to_dict()
        if data is None:
            return None
        return tomli_w.dumps(data)  # type: ignore[no-any-return]

    def to_huml(self) -> str | None:
        """Convert framework info to HUML (Human Markup Language) string.

        Returns
        -------
            HUML string with framework info, or None if not installed.
            Override this method to customize the HUML output.

        Raises
        ------
            ImportError: If pyhuml is not installed.
        """
        try:
            import pyhuml
        except ImportError as e:
            raise ImportError(
                "pyhuml is required for HUML output. "
                "Install it with: pip install pyhuml"
            ) from e

        data = self.to_dict()
        if data is None:
            return None
        return pyhuml.dumps(data)  # type: ignore[no-any-return]

    @staticmethod
    def _safe_import(module_name: str) -> object | None:
        """Safely attempt to import a module without raising exceptions.

        Args:
            module_name: Name of the module to import

        Returns
        -------
            The imported module if successful, None otherwise.
        """
        try:
            return __import__(module_name)
        except (ImportError, ModuleNotFoundError):
            return None
