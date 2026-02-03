"""Docker detection utilities."""

from __future__ import annotations

import re

from warpt.backends.software.base import SoftwareDetector
from warpt.models.constants import DOCKER_NAME
from warpt.models.list_models import DockerInfo


class DockerDetector(SoftwareDetector):
    """Detect the presence of the Docker CLI.

    Inherits serialization methods from SoftwareDetector base class.
    """

    def __init__(self, executable: str = "docker") -> None:
        """Initialize the detector.

        Args:
            executable: Name or absolute path of the docker executable.
        """
        self._executable = executable

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return DOCKER_NAME

    def detect(self) -> DockerInfo | None:
        """Detect Docker and collect path and version details.

        Returns
        -------
            DockerInfo when Docker is installed, None otherwise.
        """
        path = self._which(self._executable)
        if path is None:
            return None

        version = self._read_version(path)
        return DockerInfo(
            installed=True,
            version=version,
            path=path,
        )

    def _read_version(self, executable_path: str) -> str | None:
        """Invoke ``docker --version`` and parse the version string.

        Args:
            executable_path: Resolved path to the docker executable.

        Returns
        -------
            Parsed version string if available, otherwise None.
        """
        result = self._run_command([executable_path, "--version"])
        if result is None:
            return None

        returncode, stdout, stderr = result
        if returncode != 0:
            return None

        output = (stdout or stderr).strip()
        if not output:
            return None

        match = re.search(r"Docker version ([^,\s]+)", output)
        if match:
            return match.group(1)

        return output


__all__ = ["DockerDetector", "DockerInfo"]
