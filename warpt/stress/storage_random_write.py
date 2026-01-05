"""Storage random write test for measuring random write performance (IOPS).

This test measures random write performance with small block sizes:
- Creates test file during setup
- Writes 4KB blocks at random offsets
- Measures IOPS (I/O Operations Per Second)
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
from typing import Any

from warpt.backends.storage import Storage
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class StorageRandomWriteTest(StressTest):
    """Random write test for measuring storage IOPS.

    Tests random write performance by writing small blocks at random offsets.
    This simulates real-world usage patterns like database updates, log writes,
    and metadata operations.

    Measures:
    - Random write IOPS (I/O Operations Per Second)
    - Average latency per operation
    - Sustained random write performance

    Use cases:
    - Database update performance
    - Log file writing
    - Metadata operations
    - General write responsiveness
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
        block_size_kb: int = 4,  # 4KB blocks (typical for random I/O)
        test_path: str | None = None,  # None = use temp directory
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize storage random write test.

        Args:
            file_size_mb: Size of test file in megabytes.
            block_size_kb: Block size for writes in kilobytes (typically 4KB).
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
        return "storage_random_write"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Storage Random Write Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures random write IOPS with 4KB blocks"

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
        """Create test file for writing."""
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
        self._test_file = os.path.join(test_dir, "warpt_random_write_test.dat")

        # Check write permissions
        try:
            with open(self._test_file, "wb") as f:
                f.write(b"\x00")
        except PermissionError as e:
            raise RuntimeError(f"No write permission in {test_dir}") from e

        # Create test file with zeros (to allocate space)
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

        # Pre-allocate write buffer
        write_block_size = self.block_size_kb * 1024
        self._write_buffer = b"\xff" * write_block_size  # Use 0xFF instead of 0x00

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
        """Execute the random write test.

        Args:
            duration: Test duration in seconds.
            iterations: Number of iterations (unused for this test).

        Returns:
            Dictionary containing test results.
        """
        _ = iterations  # Unused for this test

        self.logger.info(f"Running random write test for {duration}s...")

        if not self._test_file or not os.path.exists(self._test_file):
            raise RuntimeError("Test file not found. Did setup() run?")
        if not self._write_buffer:
            raise RuntimeError("Write buffer not allocated. Did setup() run?")

        block_size_bytes = self.block_size_kb * 1024
        file_size_bytes = self.file_size_mb * 1024 * 1024
        total_operations = 0
        total_latency_ms = 0.0
        latencies = []  # Track individual operation latencies

        # Calculate number of blocks in file
        num_blocks = file_size_bytes // block_size_bytes

        # Check if direct I/O is available
        test_f, direct_io_enabled = Storage.open_direct_io(self._test_file, "r+b")
        if test_f:
            test_f.close()

        if direct_io_enabled:
            self.logger.info("Using direct I/O (bypassing OS cache)")
        else:
            self.logger.warning("Direct I/O unavailable - using fsync for accuracy")

        # Open file for random writing (r+b = read+write without truncating)
        f, _ = Storage.open_direct_io(self._test_file, "r+b")
        if not f:
            # Fallback to standard open
            f = open(self._test_file, "r+b")

        try:
            start_time = time.perf_counter()
            end_time = start_time + duration

            # Perform random writes until duration expires
            while time.perf_counter() < end_time:
                # Generate random block offset
                block_offset = random.randint(0, num_blocks - 1)
                byte_offset = block_offset * block_size_bytes

                # Time the seek + write + sync operation
                op_start = time.perf_counter()

                # Seek to random position
                f.seek(byte_offset)

                # Write block
                f.write(self._write_buffer)

                # Flush and sync to ensure data hits disk
                f.flush()
                os.fsync(f.fileno())

                op_elapsed = time.perf_counter() - op_start
                latency_ms = op_elapsed * 1000  # Convert to ms

                total_operations += 1
                total_latency_ms += latency_ms
                latencies.append(latency_ms)

        finally:
            f.close()

        elapsed = time.perf_counter() - start_time

        # Calculate IOPS
        iops = total_operations / elapsed if elapsed > 0 else 0

        # Calculate latency statistics
        avg_latency_ms = (
            total_latency_ms / total_operations if total_operations > 0 else 0
        )

        # Calculate percentiles
        if latencies:
            latencies_sorted = sorted(latencies)
            p50_latency_ms = latencies_sorted[len(latencies_sorted) // 2]
            p95_latency_ms = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99_latency_ms = latencies_sorted[int(len(latencies_sorted) * 0.99)]
        else:
            p50_latency_ms = 0
            p95_latency_ms = 0
            p99_latency_ms = 0

        self.logger.info(
            f"Completed {total_operations} random writes in {elapsed:.2f}s"
        )
        self.logger.info(f"IOPS: {iops:.2f}")
        self.logger.info(f"Average latency: {avg_latency_ms:.3f} ms")
        self.logger.info(
            f"Latency percentiles: p50={p50_latency_ms:.3f}ms "
            f"p95={p95_latency_ms:.3f}ms p99={p99_latency_ms:.3f}ms"
        )

        return {
            "test_name": self.get_name(),
            "file_size_mb": self.file_size_mb,
            "block_size_kb": self.block_size_kb,
            "duration": elapsed,
            "total_operations": total_operations,
            "iops": iops,
            "avg_latency_ms": avg_latency_ms,
            "p50_latency_ms": p50_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "direct_io_enabled": direct_io_enabled,
        }
