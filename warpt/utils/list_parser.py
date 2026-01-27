"""Parser for warpt list output files."""

from pathlib import Path
from typing import Any

from warpt.models.list_models import ListOutput


class ListParser:
    """Utility for parsing and interacting with warpt list outputs."""

    @staticmethod
    def parse_file(path: str | Path) -> ListOutput:
        """Parse a warpt list JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            ListOutput model instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file content is not valid JSON or doesn't match
                the model.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"List output file not found: {path}")

        try:
            return ListOutput.model_validate_json(path.read_text())
        except Exception as e:
            raise ValueError(f"Failed to parse list output file {path}: {e}") from e

    @staticmethod
    def parse_dict(data: dict[str, Any]) -> ListOutput:
        """Parse a dictionary into a ListOutput model.

        Args:
            data: Dictionary containing list output data.

        Returns:
            ListOutput model instance.
        """
        return ListOutput.model_validate(data)

    @staticmethod
    def get_gpu_count(output: ListOutput) -> int:
        """Get the number of GPUs detected.

        Args:
            output: The parsed list output.

        Returns:
            Number of GPUs (0 if none or not detected).
        """
        if output.hardware:
            if output.hardware.gpu_count is not None:
                return output.hardware.gpu_count
            if output.hardware.gpu:
                return len(output.hardware.gpu)
        return 0

    @staticmethod
    def is_cuda_available(output: ListOutput) -> bool:
        """Check if CUDA is available.

        Args:
            output: The parsed list output.

        Returns:
            True if CUDA version is present.
        """
        return (
            output.software is not None
            and output.software.cuda is not None
            and output.software.cuda.version is not None
        )

    @staticmethod
    def get_cuda_version(output: ListOutput) -> str | None:
        """Get the CUDA version string.

        Args:
            output: The parsed list output.

        Returns:
            CUDA version string or None if not available.
        """
        if output.software and output.software.cuda:
            return output.software.cuda.version
        return None

    @staticmethod
    def get_container_tool(output: ListOutput) -> str | None:
        """Identify available container tool (docker, etc.).

        Args:
            output: The parsed list output.

        Returns:
            Name of the tool or None if none found.
        """
        if (
            output.software
            and output.software.docker
            and output.software.docker.installed
        ):
            return "docker"
        return None

    @staticmethod
    def has_nvidia_container_toolkit(output: ListOutput) -> bool:
        """Check if NVIDIA Container Toolkit is installed.

        Args:
            output: The parsed list output.

        Returns:
            True if installed.
        """
        if output.software and output.software.nvidia_container_toolkit:
            return output.software.nvidia_container_toolkit.installed
        return False

    @staticmethod
    def get_cpu_arch(output: ListOutput) -> str | None:
        """Get the CPU architecture.

        Args:
            output: The parsed list output.

        Returns:
            Architecture string (e.g., 'x86_64', 'arm64') or None.
        """
        if output.hardware and output.hardware.cpu:
            return output.hardware.cpu.architecture
        return None

    @staticmethod
    def get_framework_version(output: ListOutput, name: str) -> str | None:
        """Get the version of a specific ML framework.

        Args:
            output: The parsed list output.
            name: Framework name (e.g., 'pytorch', 'tensorflow', 'jax').

        Returns:
            Version string or None if not installed.
        """
        if output.software and output.software.frameworks:
            framework = output.software.frameworks.get(name.lower())
            if framework and framework.installed:
                return framework.version
        return None
