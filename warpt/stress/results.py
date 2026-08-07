"""Test results collection and emission.

Supports multiple output formats: JSON, YAML, and stdout.

Usage:
    from warpt.stress.results import TestResults, OutputFormat

    results = TestResults()
    results.add_result("GPUFP32ComputeTest", test_result_dict)
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
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017


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
        >>> results.add_result("GPUFP32ComputeTest", {"tflops": 12.5, "duration": 30})
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
            mode = result.get("mode")
            label = f"{test_name} ({mode})" if mode else test_name
            output.write(f"📊 {label}\n")
            output.write("-" * 40 + "\n")
            self._format_result_text(output, result, test_name)
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

    def _format_result_text(
        self, output: StringIO, result: dict[str, Any], test_name: str = ""
    ) -> None:
        """Format a single result for text output.

        Args:
            output: StringIO to write to.
            result: Result dictionary.
            test_name: Name of the test class for custom formatting dispatch.
        """
        if test_name == "RAMBandwidthTest":
            self._format_ram_bandwidth_text(output, result)
            return
        if test_name == "RAMSwapPressureTest":
            self._format_ram_swap_text(output, result)
            return
        if test_name == "GPUFP32ComputeTest":
            self._format_gpu_fp32_text(output, result)
            return
        if test_name == "GPUFP64ComputeTest":
            self._format_gpu_fp64_text(output, result)
            return
        if test_name == "GPUMemoryBandwidthTest":
            self._format_gpu_memory_bw_text(output, result)
            return
        if test_name == "GPUPrecisionTest":
            self._format_gpu_precision_text(output, result)
            return
        if test_name == "GPUCFDSimulationTest":
            self._format_gpu_cfd_text(output, result)
            return

        # Highlight key metrics
        key_metrics = ["tflops", "bandwidth_gbps", "duration", "iterations"]
        for key in key_metrics:
            if key in result and result[key] is not None:
                value = result[key]
                if isinstance(value, float):
                    output.write(f"  {key}: {value:.2f}\n")
                else:
                    output.write(f"  {key}: {value}\n")

        # Other fields
        for key, value in result.items():
            if key not in key_metrics and key != "test_name":
                if value is None:
                    continue
                if self._is_two_col_table(value):
                    self._format_table_with_bars(output, key, value)
                elif isinstance(value, float):
                    output.write(f"  {key}: {value:.2f}\n")
                else:
                    output.write(f"  {key}: {value}\n")

    @staticmethod
    def _format_ram_system_line(output: StringIO, result: dict[str, Any]) -> None:
        """Write the shared system context line for RAM tests."""
        total = result.get("total_ram_gb", 0.0)
        mem_type = result.get("memory_type")
        alloc = result.get("allocated_memory_gb", 0.0)
        sys_parts = [f"{total:.2f} GB"]
        if mem_type:
            sys_parts.append(mem_type)
        output.write(f"  system: {' '.join(sys_parts)}, tested with {alloc:.2f} GB\n")

    @staticmethod
    def _format_ram_bandwidth_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format RAMBandwidthTest result with measurements first, context second."""
        read = result.get("baseline_read_gbps", 0.0)
        write = result.get("baseline_write_gbps", 0.0)

        output.write(f"  read:  {read:.2f} GB/s\n")
        output.write(f"  write: {write:.2f} GB/s\n")
        output.write("\n")

        TestResults._format_ram_system_line(output, result)

        dur = result.get("duration", 0.0)
        warmup = result.get("burnin_seconds", 0)
        output.write(f"  duration: {dur:.0f}s, warmup: {warmup}s\n")

    @staticmethod
    def _format_ram_swap_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format RAMSwapPressureTest result."""
        read_slow = result.get("read_slowdown_factor")
        write_slow = result.get("write_slowdown_factor")

        if read_slow is not None:
            output.write(f"  read slowdown:  {read_slow:.2f}x\n")
        if write_slow is not None:
            output.write(f"  write slowdown: {write_slow:.2f}x\n")
        output.write("\n")

        bl_read = result.get("baseline_read_gbps", 0.0)
        bl_write = result.get("baseline_write_gbps", 0.0)
        pr_read = result.get("pressure_read_gbps")
        pr_write = result.get("pressure_write_gbps")

        output.write(f"  baseline read:   {bl_read:.2f} GB/s\n")
        output.write(f"  baseline write:  {bl_write:.2f} GB/s\n")
        if pr_read is not None:
            output.write(f"  pressure read:   {pr_read:.2f} GB/s\n")
        if pr_write is not None:
            output.write(f"  pressure write:  {pr_write:.2f} GB/s\n")
        output.write("\n")

        swap_occurred = result.get("swap_occurred")
        if swap_occurred is not None:
            output.write(f"  swap occurred: {'yes' if swap_occurred else 'no'}\n")
        peak_swap = result.get("peak_swap_usage_mb")
        if peak_swap is not None:
            output.write(f"  peak swap usage: {peak_swap:.0f} MB\n")
        output.write("\n")

        TestResults._format_ram_system_line(output, result)

        dur = result.get("duration", 0.0)
        warmup = result.get("burnin_seconds", 0)
        output.write(f"  duration: {dur:.0f}s, warmup: {warmup}s\n")

    # -------------------------------------------------------------------------
    # GPU Shared Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_gpu_device_line(output: StringIO, result: dict[str, Any]) -> None:
        """Write GPU identity line: 'gpu: NVIDIA RTX 5000 Ada (gpu_0)'."""
        name = result.get("gpu_name", "unknown")
        dev = result.get("device_id", 0)
        dev_str = (
            dev if isinstance(dev, str) and dev.startswith("gpu_") else f"gpu_{dev}"
        )
        output.write(f"  gpu: {name} ({dev_str})\n")

    @staticmethod
    def _format_duration_line(output: StringIO, result: dict[str, Any]) -> None:
        """Write duration/warmup line."""
        dur = result.get("duration", 0.0)
        warmup = result.get("burnin_seconds", 0)
        output.write(f"  duration: {dur:.0f}s, warmup: {warmup}s\n")

    # -------------------------------------------------------------------------
    # GPU Formatters
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_gpu_fp32_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format GPUFP32ComputeTest result."""
        tflops = result.get("tflops", 0.0)
        output.write(f"  tflops: {tflops:.2f}\n")
        output.write("\n")

        TestResults._format_gpu_device_line(output, result)

        size = result.get("matrix_size", 0)
        precision = result.get("precision", "fp32").upper()
        tf32 = result.get("tf32_enabled", False)
        tf32_str = "TF32 enabled" if tf32 else "TF32 disabled"
        output.write(f"  matrix: {size}x{size} {precision}, {tf32_str}\n")

        mem_used = result.get("memory_used_gb", 0.0)
        mem_total = result.get("memory_total_gb", 0.0)
        output.write(f"  memory: {mem_used:.2f} / {mem_total:.2f} GB\n")

        iters = result.get("iterations", 0)
        output.write(f"  iterations: {iters}\n")

        TestResults._format_duration_line(output, result)

    @staticmethod
    def _format_gpu_fp64_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format GPUFP64ComputeTest result."""
        avg = result.get("avg_fp64_tflops", 0.0)
        output.write(f"  avg tflops: {avg:.2f}\n")

        peak = result.get("peak_fp64_tflops")
        if peak is not None:
            output.write(f"  peak tflops: {peak:.2f}\n")
        output.write("\n")

        avg_ms = result.get("avg_iteration_time_ms")
        if avg_ms is not None:
            min_ms = result.get("min_iteration_time_ms", 0.0)
            max_ms = result.get("max_iteration_time_ms", 0.0)
            output.write(
                f"  iteration: avg {avg_ms:.2f}ms,"
                f" min {min_ms:.2f}ms, max {max_ms:.2f}ms\n"
            )
            p50 = result.get("p50_iteration_time_ms", 0.0)
            p95 = result.get("p95_iteration_time_ms", 0.0)
            p99 = result.get("p99_iteration_time_ms", 0.0)
            output.write(
                f"  percentiles: p50 {p50:.2f}ms, p95 {p95:.2f}ms, p99 {p99:.2f}ms\n"
            )
            output.write("\n")

        TestResults._format_gpu_device_line(output, result)

        size = result.get("matrix_size", 0)
        output.write(f"  matrix: {size}x{size} FP64\n")

        iters = result.get("matmul_count", 0)
        output.write(f"  iterations: {iters}\n")

        TestResults._format_duration_line(output, result)

    @staticmethod
    def _format_gpu_memory_bw_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format GPUMemoryBandwidthTest result."""
        d2d = result.get("d2d_bandwidth_gbps", 0.0)
        d2d_it = result.get("d2d_iterations", 0)
        h2d = result.get("h2d_bandwidth_gbps")
        h2d_it = result.get("h2d_iterations")
        d2h = result.get("d2h_bandwidth_gbps")
        d2h_it = result.get("d2h_iterations")

        # Find widths for alignment
        bw_vals = [f"{d2d:.2f}"]
        if h2d is not None:
            bw_vals.append(f"{h2d:.2f}")
        if d2h is not None:
            bw_vals.append(f"{d2h:.2f}")
        bw_width = max(len(v) for v in bw_vals)

        output.write(f"  d2d: {d2d:>{bw_width}.2f} GB/s  ({d2d_it} iters)\n")
        if h2d is not None:
            output.write(f"  h2d: {h2d:>{bw_width}.2f} GB/s   ({h2d_it} iters)\n")
        if d2h is not None:
            output.write(f"  d2h: {d2h:>{bw_width}.2f} GB/s   ({d2h_it} iters)\n")
        output.write("\n")

        TestResults._format_gpu_device_line(output, result)

        data_sz = result.get("data_size_gb", 0.0)
        pinned = result.get("used_pinned_memory", False)
        pinned_str = "yes" if pinned else "no"
        output.write(f"  data size: {data_sz:.2f} GB, pinned memory: {pinned_str}\n")

        TestResults._format_duration_line(output, result)

    @staticmethod
    def _format_gpu_precision_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format GPUPrecisionTest result."""
        fp32 = result.get("fp32", {})
        fp16 = result.get("fp16")
        bf16 = result.get("bf16")

        fp32_tflops = fp32.get("tflops", 0.0) if fp32 else 0.0

        # Headline: per-precision TFLOPS with speedup
        label_w = 4  # "BF16" is widest label
        if fp32_tflops:
            output.write(
                f"  {'FP32':>{label_w}}: {fp32_tflops:>7.2f} TFLOPS (baseline)\n"
            )

        fp16_speedup = result.get("fp16_speedup")
        if fp16 and fp16.get("supported") and fp16.get("tflops") is not None:
            output.write(
                f"  {'FP16':>{label_w}}: {fp16['tflops']:>7.2f} TFLOPS"
                f" ({fp16_speedup:.2f}x)\n"
                if fp16_speedup
                else f"  {'FP16':>{label_w}}: {fp16['tflops']:>7.2f} TFLOPS\n"
            )

        bf16_speedup = result.get("bf16_speedup")
        if bf16 and bf16.get("supported") and bf16.get("tflops") is not None:
            output.write(
                f"  {'BF16':>{label_w}}: {bf16['tflops']:>7.2f} TFLOPS"
                f" ({bf16_speedup:.2f}x)\n"
                if bf16_speedup
                else f"  {'BF16':>{label_w}}: {bf16['tflops']:>7.2f} TFLOPS\n"
            )
        output.write("\n")

        # Mixed precision readiness
        mixed_ready = result.get("mixed_precision_ready", False)
        ready_str = "yes" if mixed_ready else "no"
        detail = ""
        if mixed_ready and bf16_speedup:
            detail = f" (BF16 {bf16_speedup:.2f}x speedup)"
        output.write(f"  mixed precision ready: {ready_str}{detail}\n")

        tf32 = result.get("tf32_enabled", False)
        output.write(f"  TF32 enabled: {'yes' if tf32 else 'no'}\n")
        output.write("\n")

        # Matrix size from FP32 baseline
        matrix_size = fp32.get("matrix_size", 0) if fp32 else 0
        if matrix_size:
            output.write(f"  matrix: {matrix_size}x{matrix_size}\n")

    @staticmethod
    def _format_gpu_cfd_text(output: StringIO, result: dict[str, Any]) -> None:
        """Format GPUCFDSimulationTest result."""
        # Operation summary rows: label, rate, unit, avg, p95
        rows = []

        solves_sec = result.get("solves_per_sec", 0.0)
        avg_solver = result.get("avg_solver_time_ms", 0.0)
        p95_solver = result.get("p95_solver_time_ms", 0.0)
        rows.append(
            ("solver", f"{solves_sec:.2f}", "solves/sec", avg_solver, p95_solver)
        )

        grad_sec = result.get("gradient_ops_per_sec", 0.0)
        avg_grad = result.get("avg_gradient_time_ms", 0.0)
        p95_grad = result.get("p95_gradient_time_ms", 0.0)
        rows.append(("gradients", f"{grad_sec:.2f}", "ops/sec", avg_grad, p95_grad))

        flux_sec = result.get("flux_ops_per_sec", 0.0)
        avg_flux = result.get("avg_flux_time_ms", 0.0)
        p95_flux = result.get("p95_flux_time_ms", 0.0)
        rows.append(("flux", f"{flux_sec:.2f}", "ops/sec", avg_flux, p95_flux))

        # Calculate column widths for alignment
        label_w = max(len(r[0]) for r in rows)
        rate_w = max(len(r[1]) for r in rows)
        unit_w = max(len(r[2]) for r in rows)

        for label, rate, unit, avg_ms, p95_ms in rows:
            output.write(
                f"  {label + ':':.<{label_w + 1}} {rate:>{rate_w}} {unit:<{unit_w}}"
                f"    avg {avg_ms:.2f}ms  p95 {p95_ms:.2f}ms\n"
            )
        output.write("\n")

        mem_bw = result.get("memory_bandwidth_gbps")
        if mem_bw is not None:
            output.write(f"  memory bandwidth: {mem_bw:.2f} GB/s\n")
            output.write("\n")

        TestResults._format_gpu_device_line(output, result)

        mesh = result.get("mesh_size", 0)
        # Format mesh size: 30000000 → "30M"
        if mesh >= 1_000_000:
            mesh_str = f"{mesh / 1_000_000:.0f}M"
        elif mesh >= 1_000:
            mesh_str = f"{mesh / 1_000:.0f}K"
        else:
            mesh_str = str(mesh)
        solver_iters = result.get("solver_iterations", 0)
        output.write(f"  mesh: {mesh_str} cells, {solver_iters} solver iterations\n")

        TestResults._format_duration_line(output, result)

    @staticmethod
    def _is_two_col_table(value: Any) -> bool:
        """Check if value is a non-empty list of dicts each with exactly 2 keys."""
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(isinstance(item, dict) and len(item) == 2 for item in value)
        )

    @staticmethod
    def _format_table_with_bars(
        output: StringIO, field_name: str, rows: list[dict[str, Any]]
    ) -> None:
        """Render a list of 2-key dicts as an aligned table with bar chart.

        The first key in each dict is the label column, the second is the
        numeric value column.  Bars are scaled to 32 characters wide.
        """
        bar_width = 32
        keys = list(rows[0].keys())
        label_key, value_key = keys[0], keys[1]

        labels = [str(row[label_key]) for row in rows]
        values = [row[value_key] for row in rows]

        # Column headers — title-case with underscores replaced
        label_header = label_key.replace("_", " ").title()
        value_header = value_key.replace("_", " ").title()

        label_width = max(len(label_header), *(len(lb) for lb in labels))
        max_val = max(abs(v) for v in values if isinstance(v, int | float)) or 1

        # Format numeric values to find the widest string
        formatted_values: list[str] = []
        for v in values:
            if isinstance(v, float):
                formatted_values.append(f"{v:.2f}")
            else:
                formatted_values.append(str(v))
        value_width = max(len(value_header), *(len(fv) for fv in formatted_values))

        output.write(f"  {field_name}:\n")
        # Header
        output.write(
            f"    {label_header:>{label_width}}  {value_header:>{value_width}}\n"
        )
        output.write(f"    {'─' * label_width}  {'─' * value_width}\n")
        # Rows
        for label, value, fval in zip(labels, values, formatted_values, strict=True):
            if isinstance(value, int | float):
                filled = round(abs(value) / max_val * bar_width)
                bar = "█" * filled
                output.write(
                    f"    {label:>{label_width}}  "
                    f"│{bar:<{bar_width}}│ "
                    f"{fval:>{value_width}}\n"
                )
            else:
                output.write(f"    {label:>{label_width}}  {fval:>{value_width}}\n")

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
