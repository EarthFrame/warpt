"""Storage mixed read/write test for measuring realistic workload performance.

This test measures performance under mixed read/write operations:
- Configurable read/write ratio (e.g., 70/30, 50/50)
- Tracks separate metrics for reads and writes
- Measures latency percentiles (p50, p95, p99) and min/max
- Simulates real-world workloads like databases and application servers
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
import tempfile
import time
from typing import Any

from warpt.backends.storage import Storage
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class StorageMixedTest(StressTest):
    """Mixed read/write test for realistic storage performance measurement.

    Tests storage performance under mixed read/write operations with
    configurable ratios. This better represents real-world workloads.

    Measures:
    - Combined IOPS (reads + writes)
    - Separate read/write IOPS and bandwidth
    - Latency percentiles (p50, p95, p99) and min/max for both operations
    - Performance under interference between reads and writes

    Use cases:
    - Database servers (mixed queries and updates)
    - Application servers (serving files while logging)
    - Virtual machines (guest OS doing mixed I/O)
    - General server workloads
    """

    _PARAM_FIELDS = (
        "file_size_mb",
        "block_size_kb",
        "read_ratio",
        "test_path",
        "burnin_seconds",
    )

    def __init__(
        self,
        file_size_mb: int = 1024,  # 1GB default
        block_size_kb: int = 4,  # 4KB blocks (typical for mixed I/O)
        read_ratio: float = 0.7,  # 70% reads, 30% writes
        test_path: str | None = None,  # None = use temp directory
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize storage mixed read/write test.

        Args:
            file_size_mb: Size of test file in megabytes.
            block_size_kb: Block size for I/O operations in kilobytes.
            read_ratio: Ratio of read operations (0.0-1.0). 0.7 = 70% reads.
            test_path: Directory for test files (None = temp dir).
            burnin_seconds: Warmup duration before measurement.
        """
        self.file_size_mb = file_size_mb
        self.block_size_kb = block_size_kb
        self.read_ratio = read_ratio
        self.test_path = test_path
        self.burnin_seconds = burnin_seconds

        # Runtime state
        self._read_file: str | None = None
        self._write_file: str | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return "storage_mixed"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Storage Mixed Read/Write Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures performance under mixed read/write workloads "
            "(configurable ratio via --read-ratio)"
        )

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

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set test parameters from dictionary.

        Override to handle read_ratio as float and test_path as string.

        Args:
            params: Dictionary of parameter names to values.
        """
        if "file_size_mb" in params:
            self.file_size_mb = int(params["file_size_mb"])
        if "block_size_kb" in params:
            self.block_size_kb = int(params["block_size_kb"])
        if "read_ratio" in params:
            self.read_ratio = float(params["read_ratio"])
        if "test_path" in params:
            self.test_path = params["test_path"]
        if "burnin_seconds" in params:
            self.burnin_seconds = int(params["burnin_seconds"])

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        # Validate file_size_mb
        if self.file_size_mb <= 0:
            raise ValueError("file_size_mb must be greater than 0")

        # Validate block_size_kb
        if self.block_size_kb <= 0:
            raise ValueError("block_size_kb must be greater than 0")

        # Validate read_ratio
        if not (0.0 <= self.read_ratio <= 1.0):
            raise ValueError("read_ratio must be between 0.0 and 1.0")

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
        self.logger.info(
            f"Read/Write ratio: {self.read_ratio * 100:.0f}% / "
            f"{(1 - self.read_ratio) * 100:.0f}%"
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Create test files for mixed read/write operations."""
        # Determine test directory
        if self.test_path is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            test_dir = self._temp_dir.name
        else:
            test_dir = self.test_path

        # Check available disk space (need 2x file size for read + write files)
        stat = shutil.disk_usage(test_dir)
        required_bytes = self.file_size_mb * 1024 * 1024 * 2  # 2 files
        if stat.free < required_bytes * 1.1:  # 10% buffer
            raise RuntimeError(
                f"Insufficient disk space. Required: {self.file_size_mb * 2} MB, "
                f"Available: {stat.free / (1024 * 1024):.2f} MB"
            )

        # Check write permissions
        test_permission_file = os.path.join(test_dir, "warpt_permission_test.tmp")
        try:
            with open(test_permission_file, "wb") as f:
                f.write(b"\x00")
            os.remove(test_permission_file)
        except PermissionError as e:
            raise RuntimeError(f"No write permission in {test_dir}") from e

        # Create read file (pre-populated with data)
        self._read_file = os.path.join(test_dir, "warpt_mixed_read.dat")
        self.logger.info(f"Creating read file: {self._read_file}")
        file_size_bytes = self.file_size_mb * 1024 * 1024
        block_size_bytes = 1024 * 1024  # 1MB write blocks for setup

        with open(self._read_file, "wb") as f:
            bytes_written = 0
            while bytes_written < file_size_bytes:
                chunk_size = min(block_size_bytes, file_size_bytes - bytes_written)
                f.write(b"\x00" * chunk_size)
                bytes_written += chunk_size

        # Create write file (empty initially, will be written during test)
        self._write_file = os.path.join(test_dir, "warpt_mixed_write.dat")
        self.logger.info(f"Creating write file: {self._write_file}")

        # Pre-allocate write file
        with open(self._write_file, "wb") as f:
            bytes_written = 0
            while bytes_written < file_size_bytes:
                chunk_size = min(block_size_bytes, file_size_bytes - bytes_written)
                f.write(b"\x00" * chunk_size)
                bytes_written += chunk_size

        self.logger.info(f"Test files created: {self.file_size_mb} MB each")

    def teardown(self) -> None:
        """Clean up test files."""
        # Remove read file
        if self._read_file and os.path.exists(self._read_file):
            try:
                os.remove(self._read_file)
                self.logger.debug(f"Removed read file: {self._read_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove read file: {e!s}")

        # Remove write file
        if self._write_file and os.path.exists(self._write_file):
            try:
                os.remove(self._write_file)
                self.logger.debug(f"Removed write file: {self._write_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove write file: {e!s}")

        # Clean up temp directory
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e!s}")

        self._read_file = None
        self._write_file = None
        self._temp_dir = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup by performing mixed disk I/O operations.

        Performs actual mixed read/write operations using the same block size
        and ratio as the test to prime the disk subsystem.

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
        """Perform warmup disk I/O operations.

        Performs mixed read/write operations matching the test workload
        to prime the disk, OS cache, and storage subsystem.

        Args:
            duration_seconds: Duration to perform warmup operations.
        """
        if not self._read_file or not self._write_file:
            # If no test files yet, just sleep
            time.sleep(duration_seconds)
            return

        # Create warmup file in same directory
        test_dir = os.path.dirname(self._read_file)
        warmup_file = os.path.join(test_dir, "warpt_warmup.dat")

        # Use same block size as actual test (not hardcoded 4KB)
        warmup_payload = b"\x00" * (self.block_size_kb * 1024)

        try:
            start = time.time()
            while (time.time() - start) < duration_seconds:
                # Use same read/write ratio as actual test
                if random.random() < self.read_ratio:
                    # Warmup read
                    try:
                        with open(self._read_file, "rb") as f:
                            f.read(len(warmup_payload))
                    except Exception:
                        pass
                else:
                    # Warmup write
                    try:
                        with open(warmup_file, "wb") as f:
                            f.write(warmup_payload)
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception:
                        pass

                time.sleep(0.1)

            # Clean up warmup file
            if os.path.exists(warmup_file):
                os.remove(warmup_file)

        except Exception as e:
            self.logger.debug(f"Warmup failed: {e}")

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[Any, Any]:
        """Execute the mixed read/write test.

        Args:
            duration: Test duration in seconds.
            iterations: Unused for this test.

        Returns:
            Dictionary containing test results.
        """
        _ = iterations  # Unused for this test

        self.logger.info(f"Running mixed read/write test for {duration}s...")

        if not self._read_file or not os.path.exists(self._read_file):
            raise RuntimeError("Read file not found. Did setup() run?")
        if not self._write_file or not os.path.exists(self._write_file):
            raise RuntimeError("Write file not found. Did setup() run?")

        block_size_bytes = self.block_size_kb * 1024
        file_size_bytes = self.file_size_mb * 1024 * 1024
        max_offset = file_size_bytes - block_size_bytes

        # Tracking variables
        read_count = 0
        write_count = 0
        read_bytes = 0
        write_bytes = 0
        read_latencies: list[float] = []
        write_latencies: list[float] = []

        # Check if direct I/O is available
        test_f, direct_io_enabled = Storage.open_direct_io(self._read_file, "rb")
        if test_f:
            test_f.close()

        if direct_io_enabled:
            self.logger.info("Using direct I/O (bypassing OS cache)")
        else:
            self.logger.warning(
                "Direct I/O unavailable - results may include RAM cache effects"
            )

        # Prepare write buffer
        write_buffer = b"\x00" * block_size_bytes

        # Run mixed operations
        start_time = time.perf_counter()
        end_time = start_time + duration

        while time.perf_counter() < end_time:
            # Decide read or write based on ratio
            is_read = random.random() < self.read_ratio

            if is_read:
                # Perform read operation
                offset = random.randint(0, max_offset)
                op_start = time.perf_counter()

                try:
                    f, _ = Storage.open_direct_io(self._read_file, "rb")
                    if not f:
                        f = open(self._read_file, "rb")

                    f.seek(offset)
                    data = f.read(block_size_bytes)
                    f.close()

                    if data:
                        read_bytes += len(data)
                        read_count += 1
                        op_latency = (time.perf_counter() - op_start) * 1000  # ms
                        read_latencies.append(op_latency)

                except Exception as e:
                    self.logger.debug(f"Read operation failed: {e}")

            else:
                # Perform write operation
                offset = random.randint(0, max_offset)
                op_start = time.perf_counter()

                try:
                    # Use append mode to avoid truncating
                    f, _ = Storage.open_direct_io(self._write_file, "r+b")
                    if not f:
                        f = open(self._write_file, "r+b")

                    f.seek(offset)
                    f.write(write_buffer)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure write reaches disk
                    f.close()

                    write_bytes += len(write_buffer)
                    write_count += 1
                    op_latency = (time.perf_counter() - op_start) * 1000  # ms
                    write_latencies.append(op_latency)

                except Exception as e:
                    self.logger.debug(f"Write operation failed: {e}")

        elapsed = time.perf_counter() - start_time

        # Calculate overall metrics
        total_operations = read_count + write_count
        overall_iops = total_operations / elapsed if elapsed > 0 else 0
        total_bytes = read_bytes + write_bytes
        overall_bandwidth_mbps = (
            (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        )

        # Calculate read metrics
        read_iops = read_count / elapsed if elapsed > 0 else 0
        read_bandwidth_mbps = (
            (read_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        )

        if read_latencies:
            avg_read_latency = statistics.mean(read_latencies)
            min_read_latency = min(read_latencies)
            max_read_latency = max(read_latencies)
            p50_read_latency = statistics.median(read_latencies)
            p95_read_latency = statistics.quantiles(read_latencies, n=20)[18]  # 95th
            p99_read_latency = statistics.quantiles(read_latencies, n=100)[98]  # 99th
        else:
            avg_read_latency = 0
            min_read_latency = 0
            max_read_latency = 0
            p50_read_latency = 0
            p95_read_latency = 0
            p99_read_latency = 0

        # Calculate write metrics
        write_iops = write_count / elapsed if elapsed > 0 else 0
        write_bandwidth_mbps = (
            (write_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        )

        if write_latencies:
            avg_write_latency = statistics.mean(write_latencies)
            min_write_latency = min(write_latencies)
            max_write_latency = max(write_latencies)
            p50_write_latency = statistics.median(write_latencies)
            p95_write_latency = statistics.quantiles(write_latencies, n=20)[18]
            p99_write_latency = statistics.quantiles(write_latencies, n=100)[98]
        else:
            avg_write_latency = 0
            min_write_latency = 0
            max_write_latency = 0
            p50_write_latency = 0
            p95_write_latency = 0
            p99_write_latency = 0

        # Calculate actual ratio achieved
        actual_read_ratio = read_count / total_operations if total_operations > 0 else 0

        self.logger.info(
            f"Completed {total_operations} operations in {elapsed:.2f}s "
            f"({read_count} reads, {write_count} writes)"
        )
        self.logger.info(f"Overall IOPS: {overall_iops:.2f}")
        self.logger.info(
            f"Read: {read_iops:.2f} IOPS, "
            f"latency: avg={avg_read_latency:.2f}ms p95={p95_read_latency:.2f}ms"
        )
        self.logger.info(
            f"Write: {write_iops:.2f} IOPS, "
            f"latency: avg={avg_write_latency:.2f}ms p95={p95_write_latency:.2f}ms"
        )
        self.logger.info(
            f"Actual read/write ratio: {actual_read_ratio * 100:.1f}% / "
            f"{(1 - actual_read_ratio) * 100:.1f}%"
        )

        return {
            "test_name": self.get_name(),
            "file_size_mb": self.file_size_mb,
            "block_size_kb": self.block_size_kb,
            "configured_read_ratio": self.read_ratio,
            "actual_read_ratio": actual_read_ratio,
            "duration": elapsed,
            # Overall metrics
            "total_operations": total_operations,
            "overall_iops": overall_iops,
            "overall_bandwidth_mbps": overall_bandwidth_mbps,
            # Read metrics
            "read_operations": read_count,
            "read_iops": read_iops,
            "read_bandwidth_mbps": read_bandwidth_mbps,
            "read_bytes_total": read_bytes,
            "avg_read_latency_ms": avg_read_latency,
            "min_read_latency_ms": min_read_latency,
            "max_read_latency_ms": max_read_latency,
            "p50_read_latency_ms": p50_read_latency,
            "p95_read_latency_ms": p95_read_latency,
            "p99_read_latency_ms": p99_read_latency,
            # Write metrics
            "write_operations": write_count,
            "write_iops": write_iops,
            "write_bandwidth_mbps": write_bandwidth_mbps,
            "write_bytes_total": write_bytes,
            "avg_write_latency_ms": avg_write_latency,
            "min_write_latency_ms": min_write_latency,
            "max_write_latency_ms": max_write_latency,
            "p50_write_latency_ms": p50_write_latency,
            "p95_write_latency_ms": p95_write_latency,
            "p99_write_latency_ms": p99_write_latency,
            # Config
            "direct_io_enabled": direct_io_enabled,
        }
