"""Detection utilities for the NVIDIA Container Toolkit."""

from __future__ import annotations

import json
import re

from warpt.backends.software.base import SoftwareDetector
from warpt.models.list_models import NvidiaContainerToolkitInfo

_VERSION_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)+)")


class NvidiaContainerToolkitDetector(SoftwareDetector):
    """Detect NVIDIA Container Toolkit components.

    Inherits serialization methods from SoftwareDetector base class.
    """

    _CLI_CANDIDATES = ("nvidia-container-cli", "nvidia-container-toolkit")
    _RUNTIME_CANDIDATES = ("nvidia-container-runtime",)

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "nvidia_container_toolkit"

    def detect(self) -> NvidiaContainerToolkitInfo | None:
        """Detect toolkit binaries and gather metadata.

        Returns
        -------
            NvidiaContainerToolkitInfo if any component is present,
            otherwise None.
        """
        cli_path = self._find_any(self._CLI_CANDIDATES)
        runtime_path = self._find_any(self._RUNTIME_CANDIDATES)
        if cli_path is None and runtime_path is None:
            return None

        cli_version = self._read_cli_version(cli_path) if cli_path else None
        docker_runtime = self._docker_runtime_available()

        return NvidiaContainerToolkitInfo(
            installed=True,
            cli_version=cli_version,
            cli_path=cli_path,
            runtime_path=runtime_path,
            docker_runtime_ready=docker_runtime,
        )

    def _find_any(self, candidates: tuple[str, ...]) -> str | None:
        """Return the first executable found in PATH from the candidates."""
        for name in candidates:
            path = self._which(name)
            if path:
                return path
        return None

    def _read_cli_version(self, cli_path: str) -> str | None:
        """Return the parsed version from ``nvidia-container-cli --version``."""
        result = self._run_command([cli_path, "--version"])
        if result is None:
            return None

        returncode, stdout, stderr = result
        if returncode != 0:
            return None

        output = (stdout or stderr).strip()
        if not output:
            return None

        match = _VERSION_PATTERN.search(output)
        if match:
            return match.group(1)
        return output

    def _docker_runtime_available(self) -> bool | None:
        """Check whether Docker exposes the 'nvidia' runtime."""
        docker_path = self._which("docker")
        if docker_path is None:
            return None

        result = self._run_command(
            [docker_path, "info", "--format", "{{json .Runtimes}}"]
        )
        if result is None:
            return None

        returncode, stdout, stderr = result
        if returncode != 0:
            return None

        payload = (stdout or stderr).strip()
        if not payload:
            return None

        try:
            runtimes = json.loads(payload)
        except json.JSONDecodeError:
            return None

        entry = runtimes.get("nvidia")
        if entry is None:
            return False
        if isinstance(entry, dict):
            return True
        return bool(entry)


__all__ = [
    "NvidiaContainerToolkitDetector",
    "NvidiaContainerToolkitInfo",
]
