"""Storage sequential read test for measuring disk read bandwidth.

This test measures sequential read performance by reading large files:
- Creates test file during setup
- Reads file sequentially with configurable block sizes
- Measures read bandwidth (MB/s)
- Uses direct I/O to bypass OS cache where possible
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


class StorageSequentialReadTest(StressTest):
    """Sequential read test for measuring storage read bandwidth.

    Tests sequential read performance by reading a large file from disk.
    Useful for understanding sustained read throughput.

    Measures:
    - Sequential read bandwidth (MB/s)
    - Different block sizes (4KB, 64KB, 1MB)
    - Sustained read performance over time

    Use cases:
    - Video streaming performance
    - Large file transfers
    - Backup/restore operations
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
        """Initialize storage sequential read test.

        Args:
            file_size_mb: Size of test file in megabytes.
            block_size_kb: Block size for reads in kilobytes.
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

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return "storage_sequential_read"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Storage Sequential Read Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures sequential read bandwidth from storage device"

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
        """Create test file for reading."""
        # Determine test directory
        if self.test_path is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            test_dir = self._temp_dir.name
        else:
            test_dir = self.test_path

        # Check available disk space
        stat = shutil.disk_usage(test_dir)
        required_bytes = self.file_size_mb * 1024 * 1024
        if stat.free < required_bytes * 1.1:  # 10% buffer
            raise RuntimeError(
                f"Insufficient disk space. Required: {self.file_size_mb} MB, "
                f"Available: {stat.free / (1024 * 1024):.2f} MB"
            )

        # Create test file path
        self._test_file = os.path.join(test_dir, "warpt_read_test.dat")

        # Check write permissions
        try:
            with open(self._test_file, "wb") as f:
                f.write(b"\x00")
        except PermissionError as e:
            raise RuntimeError(f"No write permission in {test_dir}") from e

        # Create test file with random data
        self.logger.info(f"Creating test file: {self._test_file}")
        file_size_bytes = self.file_size_mb * 1024 * 1024
        block_size_bytes = 1024 * 1024  # 1MB write blocks

        with open(self._test_file, "wb") as f:
            bytes_written = 0
            while bytes_written < file_size_bytes:
                # Write 1MB blocks of zeros (fast)
                chunk_size = min(block_size_bytes, file_size_bytes - bytes_written)
                f.write(b"\x00" * chunk_size)
                bytes_written += chunk_size

        self.logger.info(f"Test file created: {self.file_size_mb} MB")

    def teardown(self) -> None:
        """Clean up test file."""
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

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup by performing small disk writes.

        Writes small amounts of data to prime the disk and OS cache:
        - Wakes up disk if in power-saving mode (HDDs)
        - Primes disk controller and OS cache layers
        - Establishes steady-state before measurement

        Args:
            duration_seconds: Warmup duration in seconds.
            iterations: Unused, kept for compatibility with base class.
        """
        _ = iterations  # Unused, kept for compatibility
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            self._warmup_disk(duration_seconds)

    def _warmup_disk(self, duration_seconds: int) -> None:
        """Perform warmup disk writes.

        Writes small 4KB payloads to a warmup file to prime the disk.

        Args:
            duration_seconds: Duration to perform warmup writes.
        """
        if not self._test_file:
            # If no test file yet, just sleep
            time.sleep(duration_seconds)
            return

        # Create warmup file in same directory as test file
        test_dir = os.path.dirname(self._test_file)
        warmup_file = os.path.join(test_dir, "warpt_warmup.dat")

        # Small 4KB payload for warmup writes
        warmup_payload = b"\x00" * 4096

        try:
            start = time.time()
            while (time.time() - start) < duration_seconds:
                # Write small amount to disk
                with open(warmup_file, "wb") as f:
                    f.write(warmup_payload)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data reaches disk
                time.sleep(0.1)  # Small delay between writes

            # Clean up warmup file
            if os.path.exists(warmup_file):
                os.remove(warmup_file)

        except Exception as e:
            self.logger.debug(f"Warmup write failed: {e}")

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[Any, Any]:
        """Execute the sequential read test.

        Args:
            duration: Test duration in seconds.
            iterations: Number of iterations (unused for this test).

        Returns:
            Dictionary containing test results.
        """
        _ = iterations  # Unused for this test

        self.logger.info(f"Running sequential read test for {duration}s...")

        if not self._test_file or not os.path.exists(self._test_file):
            raise RuntimeError("Test file not found. Did setup() run?")

        block_size_bytes = self.block_size_kb * 1024
        total_bytes_read = 0
        iteration_count = 0
        iteration_speeds = []  # Track speed per iteration

        start_time = time.perf_counter()
        end_time = start_time + duration

        # Check if direct I/O is available (only need to check once)
        test_f, direct_io_enabled = Storage.open_direct_io(self._test_file, "rb")
        if test_f:
            test_f.close()

        if direct_io_enabled:
            self.logger.info("Using direct I/O (bypassing OS cache)")
        else:
            self.logger.warning(
                "Direct I/O unavailable - results may include RAM cache effects"
            )

        # Read file sequentially until duration expires
        while time.perf_counter() < end_time:
            iter_start = time.perf_counter()
            iter_bytes = 0

            # Open with direct I/O if available
            f, _ = Storage.open_direct_io(self._test_file, "rb")
            if not f:
                # Fallback to standard open if direct I/O fails
                f = open(self._test_file, "rb")

            try:
                while True:
                    chunk = f.read(block_size_bytes)
                    if not chunk:
                        break
                    iter_bytes += len(chunk)
                    total_bytes_read += len(chunk)

                    # Check if duration expired
                    if time.perf_counter() >= end_time:
                        break
            finally:
                f.close()

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
        total_mb = total_bytes_read / (1024 * 1024)
        overall_bandwidth_mbps = total_mb / elapsed if elapsed > 0 else 0

        # Calculate steady-state bandwidth (last 5 iterations)
        steady_state_count = min(5, len(iteration_speeds))
        if steady_state_count > 0:
            steady_state_bandwidth_mbps = (
                sum(iteration_speeds[-steady_state_count:]) / steady_state_count
            )
        else:
            steady_state_bandwidth_mbps = 0

        # Calculate min/max bandwidth across all iterations
        min_bandwidth_mbps = min(iteration_speeds) if iteration_speeds else 0
        max_bandwidth_mbps = max(iteration_speeds) if iteration_speeds else 0

        self.logger.info(
            f"Read {total_mb:.2f} MB in {elapsed:.2f}s "
            f"({iteration_count} iterations)"
        )
        self.logger.info(f"Overall bandwidth: {overall_bandwidth_mbps:.2f} MB/s")
        self.logger.info(
            f"Steady-state bandwidth (last {steady_state_count}): "
            f"{steady_state_bandwidth_mbps:.2f} MB/s"
        )
        self.logger.info(
            f"Bandwidth range: min={min_bandwidth_mbps:.2f} MB/s, "
            f"max={max_bandwidth_mbps:.2f} MB/s"
        )

        return {
            "test_name": self.get_name(),
            "file_size_mb": self.file_size_mb,
            "block_size_kb": self.block_size_kb,
            "duration": elapsed,
            "total_bytes_read": total_bytes_read,
            "total_mb_read": total_mb,
            "iteration_count": iteration_count,
            "overall_bandwidth_mbps": overall_bandwidth_mbps,
            "min_bandwidth_mbps": min_bandwidth_mbps,
            "max_bandwidth_mbps": max_bandwidth_mbps,
            "steady_state_bandwidth_mbps": steady_state_bandwidth_mbps,
            "direct_io_enabled": direct_io_enabled,
        }
