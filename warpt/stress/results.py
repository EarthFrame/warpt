"""Test results collection and emission.

Supports multiple output formats: JSON, YAML, and stdout.

Usage:
    from warpt.stress.results import TestResults, OutputFormat

    results = TestResults()
    results.add_result("GPUMatMulTest", test_result_dict)
    results.add_result("CPUMatMulTest", test_result_dict)

    # Emit to different formats
    results.emit("results.json", OutputFormat.JSON)
    results.emit("results.yaml", OutputFormat.YAML)
    results.emit(sys.stdout, OutputFormat.TEXT)
"""

import json
import sys
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

if sys.version_info >= (3, 11):  # noqa: UP036
    from datetime import UTC
else:
    UTC = UTC


class OutputFormat(Enum):
    """Supported output formats for test results."""

    JSON = "json"
    YAML = "yaml"
    TEXT = "text"  # Human-readable text for stdout


class TestResults:
    """Collection of test results with flexible emission.

    Stores results from multiple tests and can emit them in various formats.

    Example:
        >>> results = TestResults()
        >>> results.add_result("GPUMatMulTest", {"tflops": 12.5, "duration": 30})
        >>> results.emit("results.json", OutputFormat.JSON)
        >>> results.emit(sys.stdout, OutputFormat.TEXT)
    """

    __test__ = False  # Tell pytest not to collect this as a test class

    def __init__(self) -> None:
        """Initialize empty results collection."""
        self._results: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {
            "timestamp_start": datetime.now(UTC).isoformat(),
            "timestamp_end": None,
            "warpt_version": self._get_version(),
        }
        self._errors: dict[str, str] = {}

    def _get_version(self) -> str:
        """Get warpt version string."""
        try:
            from warpt import __version__

            return str(__version__)
        except (ImportError, AttributeError):
            return "unknown"

    def add_result(self, test_name: str, result: dict[str, Any]) -> None:
        """Add a test result.

        Args:
            test_name: Name of the test (class name).
            result: Dictionary of test results.
        """
        self._results[test_name] = result

    def add_error(self, test_name: str, error: str) -> None:
        """Record an error for a test.

        Args:
            test_name: Name of the test.
            error: Error message.
        """
        self._errors[test_name] = error

    def finalize(self) -> None:
        """Mark results as complete, setting end timestamp."""
        self._metadata["timestamp_end"] = datetime.now(UTC).isoformat()

    @property
    def results(self) -> dict[str, Any]:
        """Get raw results dictionary."""
        return self._results.copy()

    @property
    def errors(self) -> dict[str, str]:
        """Get error dictionary."""
        return self._errors.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a dictionary for serialization.

        Returns:
            Dictionary with metadata, results, and errors.
        """
        return {
            "metadata": self._metadata,
            "results": self._results,
            "errors": self._errors if self._errors else None,
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> dict[str, Any]:
        """Generate a summary of results.

        Returns:
            Summary dictionary.
        """
        total = len(self._results) + len(self._errors)
        passed = len(self._results)
        failed = len(self._errors)

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0.0,
        }

    # -------------------------------------------------------------------------
    # Emission Methods
    # -------------------------------------------------------------------------

    def emit(
        self,
        output: str | Path | TextIO,
        format: OutputFormat = OutputFormat.JSON,
        indent: int = 2,
    ) -> None:
        """Emit results to a file or stream.

        Args:
            output: File path or file-like object (e.g., sys.stdout).
            format: Output format (JSON, YAML, TEXT).
            indent: Indentation level for JSON/YAML.

        Raises:
            ValueError: If format is YAML and pyyaml is not installed.
        """
        self.finalize()

        if format == OutputFormat.JSON:
            content = self._to_json(indent)
        elif format == OutputFormat.YAML:
            content = self._to_yaml(indent)
        elif format == OutputFormat.TEXT:
            content = self._to_text()
        else:
            raise ValueError(f"Unknown format: {format}")

        self._write_output(output, content)

    def _to_json(self, indent: int = 2) -> str:
        """Convert results to JSON string.

        Args:
            indent: Indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def _to_yaml(self, indent: int = 2) -> str:
        """Convert results to YAML string.

        Args:
            indent: Indentation level.

        Returns:
            YAML string.

        Raises:
            ValueError: If pyyaml is not installed.
        """
        try:
            import yaml  # type: ignore[import-untyped, unused-ignore]
        except ImportError:
            raise ValueError(
                "YAML output requires pyyaml. Install with: pip install pyyaml"
            ) from None

        result: str = yaml.dump(self.to_dict(), indent=indent, default_flow_style=False)
        return result

    def _to_text(self) -> str:
        """Convert results to human-readable text.

        Returns:
            Formatted text string.
        """
        output = StringIO()
        data = self.to_dict()

        output.write("\n" + "=" * 60 + "\n")
        output.write("  WARPT STRESS TEST RESULTS\n")
        output.write("=" * 60 + "\n\n")

        # Metadata
        meta = data["metadata"]
        output.write(f"Started:  {meta['timestamp_start']}\n")
        output.write(f"Finished: {meta['timestamp_end']}\n")
        output.write(f"Version:  {meta['warpt_version']}\n\n")

        # Summary
        summary = data["summary"]
        output.write("-" * 40 + "\n")
        output.write(f"Tests Run:    {summary['total_tests']}\n")
        output.write(f"Passed:       {summary['passed']}\n")
        output.write(f"Failed:       {summary['failed']}\n")
        output.write(f"Success Rate: {summary['success_rate']:.1%}\n")
        output.write("-" * 40 + "\n\n")

        # Individual results
        for test_name, result in data["results"].items():
            output.write(f"📊 {test_name}\n")
            output.write("-" * 40 + "\n")
            self._format_result_text(output, result)
            output.write("\n")

        # Errors
        if data["errors"]:
            output.write("❌ ERRORS\n")
            output.write("-" * 40 + "\n")
            for test_name, error in data["errors"].items():
                output.write(f"  {test_name}: {error}\n")
            output.write("\n")

        output.write("=" * 60 + "\n")
        return output.getvalue()

    def _format_result_text(self, output: StringIO, result: dict[str, Any]) -> None:
        """Format a single result for text output.

        Args:
            output: StringIO to write to.
            result: Result dictionary.
        """
        # Highlight key metrics
        key_metrics = ["tflops", "bandwidth_gbps", "duration", "iterations"]
        for key in key_metrics:
            if key in result:
                value = result[key]
                if isinstance(value, float):
                    output.write(f"  {key}: {value:.2f}\n")
                else:
                    output.write(f"  {key}: {value}\n")

        # Other fields
        for key, value in result.items():
            if key not in key_metrics and key != "test_name":
                if isinstance(value, float):
                    output.write(f"  {key}: {value:.2f}\n")
                else:
                    output.write(f"  {key}: {value}\n")

    def _write_output(self, output: str | Path | TextIO, content: str) -> None:
        """Write content to file or stream.

        Args:
            output: File path or file-like object.
            content: String content to write.
        """
        if isinstance(output, str | Path):
            Path(output).write_text(content)
        else:
            output.write(content)
            if output is not sys.stdout and output is not sys.stderr:
                output.flush()

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def emit_json(self, path: str | Path) -> None:
        """Emit results to a JSON file.

        Args:
            path: Output file path.
        """
        self.emit(path, OutputFormat.JSON)

    def emit_yaml(self, path: str | Path) -> None:
        """Emit results to a YAML file.

        Args:
            path: Output file path.
        """
        self.emit(path, OutputFormat.YAML)

    def emit_stdout(self) -> None:
        """Emit human-readable results to stdout."""
        self.emit(sys.stdout, OutputFormat.TEXT)

    def __len__(self) -> int:
        """Return number of results."""
        return len(self._results)

    def __contains__(self, test_name: str) -> bool:
        """Check if a test has results."""
        return test_name in self._results
