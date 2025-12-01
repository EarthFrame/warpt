"""Docker detection utilities implemented with the Python standard library."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DockerDetectionResult:
    """Structured result describing a detected Docker installation."""

    path: str
    version: str | None


class DockerDetector:
    """Detect the presence of the Docker CLI."""

    def __init__(self, executable: str = "docker") -> None:
        """Initialize the detector.

        Args:
            executable: Name or absolute path of the docker executable.
        """
        self._executable = executable

    def is_installed(self) -> bool:
        """Return True when Docker is available.

        Returns
        -------
            True if Docker can be located and responds to ``--version``.
        """
        return self.detect() is not None

    def detect(self) -> DockerDetectionResult | None:
        """Detect Docker and collect path and version details.

        Returns
        -------
            DockerDetectionResult when Docker is installed, None otherwise.
        """
        path = shutil.which(self._executable)
        if path is None:
            return None

        version = self._read_version(path)
        return DockerDetectionResult(path=path, version=version)

    @staticmethod
    def _read_version(executable_path: str) -> str | None:
        """Invoke ``docker --version`` and parse the version string.

        Args
        ----
            executable_path: Resolved path to the docker executable.

        Returns
        -------
            Parsed version string if available, otherwise None.
        """
        try:
            result = subprocess.run(
                [executable_path, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if result.returncode != 0:
            return None

        output = (result.stdout or result.stderr or "").strip()
        if not output:
            return None

        match = re.search(r"Docker version ([^,\s]+)", output)
        if match:
            return match.group(1)

        return output


__all__ = ["DockerDetectionResult", "DockerDetector"]
