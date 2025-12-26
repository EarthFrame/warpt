"""Storage sequential write test for measuring disk write bandwidth.

This test measures sequential write performance by writing large files:
- Writes file sequentially with configurable block sizes
- Measures write bandwidth (MB/s)
- Uses direct I/O and sync operations for accurate measurement
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any

from warpt.backends.storage import Storage
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class StorageSequentialWriteTest(StressTest):
    """Sequential write test for measuring storage write bandwidth.

    Tests sequential write performance by writing large amounts of data to disk.
    Useful for understanding sustained write throughput.

    Measures:
    - Sequential write bandwidth (MB/s)
    - Different block sizes (4KB, 64KB, 1MB)
    - Sustained write performance over time

    Use cases:
    - Video recording performance
    - Large file downloads
    - Database write operations
    - SSD vs HDD comparison
    """

    _PARAM_FIELDS = (
        "file_size_mb",
        "block_size_kb",
        "test_path",
        "burnin_seconds",
    )

    def __init__(
        self,
        file_size_mb: int = 1024,  # 1GB default
        block_size_kb: int = 1024,  # 1MB blocks
        test_path: str | None = None,  # None = use temp directory
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize storage sequential write test.

        Args:
            file_size_mb: Size of test file in megabytes per iteration.
            block_size_kb: Block size for writes in kilobytes.
            test_path: Directory for test files (None = temp dir).
            burnin_seconds: Warmup duration before measurement.
        """
        self.file_size_mb = file_size_mb
        self.block_size_kb = block_size_kb
        self.test_path = test_path
        self.burnin_seconds = burnin_seconds

        # Runtime state
        self._test_file: str | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._write_buffer: bytes | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return "storage_sequential_write"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Storage Sequential Write Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures sequential write bandwidth to storage device"

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.STORAGE

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if storage testing is available.

        Storage hardware is always present.
        Runtime checks (space, permissions) happen in setup().
        """
        return True

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        # Validate file_size_mb
        if self.file_size_mb <= 0:
            raise ValueError("file_size_mb must be greater than 0")

        # Validate block_size_kb
        if self.block_size_kb <= 0:
            raise ValueError("block_size_kb must be greater than 0")

        # Validate burnin_seconds
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Validate test_path if provided
        if self.test_path is not None:
            if not os.path.exists(self.test_path):
                raise ValueError(f"test_path does not exist: {self.test_path}")
            if not os.path.isdir(self.test_path):
                raise ValueError(f"test_path is not a directory: {self.test_path}")

        self.logger.info(f"Test file size: {self.file_size_mb} MB")
        self.logger.info(f"Block size: {self.block_size_kb} KB")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize test environment."""
        # Determine test directory
        if self.test_path is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            test_dir = self._temp_dir.name
        else:
            test_dir = self.test_path

        # Check available disk space (need space for multiple iterations)
        stat = shutil.disk_usage(test_dir)
        required_bytes = self.file_size_mb * 1024 * 1024 * 2  # 2x for safety
        if stat.free < required_bytes * 1.1:  # 10% buffer
            required_mb = required_bytes / (1024 * 1024)
            available_mb = stat.free / (1024 * 1024)
            raise RuntimeError(
                f"Insufficient disk space. "
                f"Required: {required_mb:.2f} MB, Available: {available_mb:.2f} MB"
            )

        # Create test file path
        self._test_file = os.path.join(test_dir, "warpt_write_test.dat")

        # Check write permissions
        try:
            with open(self._test_file, "wb") as f:
                f.write(b"\x00")
            os.remove(self._test_file)
        except PermissionError as e:
            raise RuntimeError(f"No write permission in {test_dir}") from e

        # Pre-allocate write buffer (filled with zeros for fast writes)
        block_size_bytes = self.block_size_kb * 1024
        self._write_buffer = b"\x00" * block_size_bytes

        self.logger.info(f"Test directory: {test_dir}")

    def teardown(self) -> None:
        """Clean up test files."""
        # Remove test file
        if self._test_file and os.path.exists(self._test_file):
            try:
                os.remove(self._test_file)
                self.logger.debug(f"Removed test file: {self._test_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove test file: {e!s}")

        # Clean up temp directory
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e!s}")

        self._test_file = None
        self._temp_dir = None
        self._write_buffer = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup iterations.

        Args:
            duration_seconds: Warmup duration in seconds.
            iterations: Unused, kept for compatibility with base class.
        """
        _ = iterations  # Unused, kept for compatibility
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[Any, Any]:
        """Execute the sequential write test.

        Args:
            duration: Test duration in seconds.
            iterations: Number of iterations (unused for this test).

        Returns:
            Dictionary containing test results.
        """
        _ = iterations  # Unused for this test

        self.logger.info(f"Running sequential write test for {duration}s...")

        if not self._test_file:
            raise RuntimeError("Test file path not set. Did setup() run?")
        if not self._write_buffer:
            raise RuntimeError("Write buffer not allocated. Did setup() run?")

        file_size_bytes = self.file_size_mb * 1024 * 1024
        block_size_bytes = self.block_size_kb * 1024
        total_bytes_written = 0
        iteration_count = 0
        iteration_speeds = []  # Track speed per iteration

        # Check if direct I/O is available
        test_f, direct_io_enabled = Storage.open_direct_io(self._test_file, "wb")
        if test_f:
            test_f.close()
            # Clean up test file
            try:
                os.remove(self._test_file)
            except Exception:
                pass

        if direct_io_enabled:
            self.logger.info("Using direct I/O (bypassing OS cache)")
        else:
            self.logger.warning(
                "Direct I/O unavailable - using sync operations for accuracy"
            )

        start_time = time.perf_counter()
        end_time = start_time + duration

        # Write files sequentially until duration expires
        while time.perf_counter() < end_time:
            iter_start = time.perf_counter()
            iter_bytes = 0

            # Open with direct I/O if available
            f, _ = Storage.open_direct_io(self._test_file, "wb")
            if not f:
                # Fallback to standard open
                f = open(self._test_file, "wb")

            # Hint to kernel about sequential access pattern
            if hasattr(os, "posix_fadvise"):
                try:
                    posix_fadvise = getattr(os, "posix_fadvise")  # noqa: B009
                    fadv_sequential = getattr(os, "POSIX_FADV_SEQUENTIAL")  # noqa: B009
                    posix_fadvise(f.fileno(), 0, 0, fadv_sequential)
                except (OSError, AttributeError):
                    pass  # Advisory hint failed; non-critical, continue without it

            try:
                bytes_written = 0
                while bytes_written < file_size_bytes:
                    # Check if duration expired during write
                    if time.perf_counter() >= end_time:
                        break

                    chunk_size = min(block_size_bytes, file_size_bytes - bytes_written)
                    f.write(self._write_buffer[:chunk_size])
                    bytes_written += chunk_size
                    iter_bytes += chunk_size
                    total_bytes_written += chunk_size

                # Flush to ensure data is written
                f.flush()
                os.fsync(f.fileno())  # Force sync to disk

            finally:
                f.close()

            # Remove file for next iteration
            try:
                os.remove(self._test_file)
            except Exception:
                pass

            iter_elapsed = time.perf_counter() - iter_start
            if iter_elapsed > 0 and iter_bytes > 0:
                iter_speed_mbps = (iter_bytes / (1024 * 1024)) / iter_elapsed
                iteration_speeds.append(iter_speed_mbps)
                iteration_count += 1

            # Check if duration expired after iteration
            if time.perf_counter() >= end_time:
                break

        elapsed = time.perf_counter() - start_time

        # Calculate overall bandwidth
        total_mb = total_bytes_written / (1024 * 1024)
        overall_bandwidth_mbps = total_mb / elapsed if elapsed > 0 else 0

        # Calculate steady-state bandwidth (last 5 iterations)
        steady_state_count = min(5, len(iteration_speeds))
        if steady_state_count > 0:
            steady_state_bandwidth_mbps = (
                sum(iteration_speeds[-steady_state_count:]) / steady_state_count
            )
        else:
            steady_state_bandwidth_mbps = 0

        self.logger.info(
            f"Wrote {total_mb:.2f} MB in {elapsed:.2f}s "
            f"({iteration_count} iterations)"
        )
        self.logger.info(f"Overall bandwidth: {overall_bandwidth_mbps:.2f} MB/s")
        self.logger.info(
            f"Steady-state bandwidth (last {steady_state_count}): "
            f"{steady_state_bandwidth_mbps:.2f} MB/s"
        )

        return {
            "test_name": self.get_name(),
            "file_size_mb": self.file_size_mb,
            "block_size_kb": self.block_size_kb,
            "duration": elapsed,
            "total_bytes_written": total_bytes_written,
            "total_mb_written": total_mb,
            "iteration_count": iteration_count,
            "overall_bandwidth_mbps": overall_bandwidth_mbps,
            "steady_state_bandwidth_mbps": steady_state_bandwidth_mbps,
            "direct_io_enabled": direct_io_enabled,
        }
