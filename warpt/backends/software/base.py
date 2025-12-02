"""Base class for software detection."""

import json
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

# Type variable for the info model returned by detect()
T = TypeVar("T", bound=BaseModel)


class SoftwareDetector(ABC):
    """Base class for detecting software installations.

    Subclasses should implement the detect() method to check for software
    installation and return an appropriate Pydantic model if found.

    This follows the same pattern as FrameworkDetector but is more general,
    supporting any software type (Docker, compilers, CUDA toolkit, etc.).
    """

    @abstractmethod
    def detect(self) -> BaseModel | None:
        """Detect if the software is installed and gather its information.

        Returns
        -------
            Pydantic model with software info if installed, None otherwise.
        """
        pass

    @property
    @abstractmethod
    def software_name(self) -> str:
        """Return the canonical name of the software.

        Examples: 'docker', 'python', 'cuda', 'gcc'
        """
        pass

    def is_installed(self) -> bool:
        """Check if the software is installed.

        Returns
        -------
            True if software is detected, False otherwise.
        """
        return self.detect() is not None

    def to_dict(self) -> dict | None:
        """Convert software info to dictionary format.

        Returns
        -------
            Dictionary with software info, or None if not installed.
        """
        info = self.detect()
        if info is None:
            return None
        return info.model_dump()

    def to_json(self, indent: int | None = 2) -> str | None:
        """Convert software info to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact)

        Returns
        -------
            JSON string with software info, or None if not installed.
        """
        data = self.to_dict()
        if data is None:
            return None
        return json.dumps(data, indent=indent)

    def to_yaml(self) -> str | None:
        """Convert software info to YAML string.

        Returns
        -------
            YAML string with software info, or None if not installed.

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
        """Convert software info to TOML string.

        Returns
        -------
            TOML string with software info, or None if not installed.

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
        """Convert software info to HUML (Human Markup Language) string.

        Returns
        -------
            HUML string with software info, or None if not installed.

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
    def _run_command(
        args: list[str],
        timeout: float = 5.0,
    ) -> tuple[int, str, str] | None:
        """Run a command and return its output.

        This is a convenience method for detectors that need to run external
        commands (like `docker --version` or `gcc --version`).

        Args:
            args: Command and arguments to run
            timeout: Maximum time to wait for command (seconds)

        Returns
        -------
            Tuple of (return_code, stdout, stderr) if command ran,
            None if command failed to execute.
        """
        import subprocess

        try:
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (result.returncode, result.stdout or "", result.stderr or "")
        except (OSError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _which(executable: str) -> str | None:
        """Find executable in PATH.

        Args:
            executable: Name of executable to find

        Returns
        -------
            Full path to executable if found, None otherwise.
        """
        import shutil

        return shutil.which(executable)
