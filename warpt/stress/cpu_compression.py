"""CPU zlib compression stress test.

This test measures single-core CPU performance using zlib compression,
which represents real-world workloads like backups, log processing,
and data transfers.
"""

import os
import time
import zlib
from typing import Any

from warpt.backends.system import CPU
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class CPUZlibTest(StressTest):
    """CPU zlib compression stress test.

    Measures single-core CPU performance using zlib compression/decompression.
    This represents real-world workloads like:
    - Log file compression
    - Backup operations
    - Data transfer with compression
    - Archive creation

    Unlike multi-threaded NumPy tests, this runs on a single thread and
    provides practical metrics (MB/s) directly applicable to real tasks.

    Note: This tests zlib specifically. Other compression algorithms
    (lz4, zstd, brotli, etc.) will have separate test classes.
    """

    _PARAM_FIELDS = ("data_size_mb", "compression_level", "burnin_seconds")

    def __init__(
        self,
        data_size_mb: int = 10,
        compression_level: int = 6,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize single-threaded CPU compression test.

        Args:
            data_size_mb: Size of data to compress in MB. Default 10MB.
            compression_level: zlib compression level (1-9). 6 is default.
                Higher = slower but better compression.
            burnin_seconds: Warmup duration before measurement.
        """
        self.data_size_mb = data_size_mb
        self.compression_level = compression_level
        self.burnin_seconds = burnin_seconds
        self._cpu: CPU | None = None
        self._cpu_info: Any = None
        self._test_data: bytes | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "CPU Zlib Compression Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures single-core CPU performance via zlib compression"

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.CPU

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Compression is always available (stdlib)."""
        return True

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        if self.data_size_mb < 1:
            raise ValueError("data_size_mb must be >= 1")
        if self.data_size_mb > 1000:
            raise ValueError("data_size_mb must be <= 1000 (avoid excessive memory)")
        if not (1 <= self.compression_level <= 9):
            raise ValueError("compression_level must be between 1 and 9")
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize CPU info and test data."""
        self._cpu = CPU()
        self._cpu_info = self._cpu.get_cpu_info()

        # Generate random test data (incompressible for consistent results)
        data_size_bytes = self.data_size_mb * 1024 * 1024
        self.logger.info(f"Generating {self.data_size_mb} MB of test data...")
        self._test_data = os.urandom(data_size_bytes)
        self.logger.info(f"Test data generated: {len(self._test_data)} bytes")

    def teardown(self) -> None:
        """Clean up resources."""
        self._cpu = None
        self._cpu_info = None
        self._test_data = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup iterations to let CPU warm up.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
        if self._test_data is None:
            return

        # Use burnin_seconds if no duration specified
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                # Warmup with compression/decompression
                compressed = zlib.compress(self._test_data, self.compression_level)
                _ = zlib.decompress(compressed)
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                compressed = zlib.compress(self._test_data, self.compression_level)
                _ = zlib.decompress(compressed)

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the single-threaded compression test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored for this test (runs for duration).

        Returns:
            Dictionary containing test results.
        """
        del iterations  # Unused; test runs for duration

        if self._test_data is None:
            raise RuntimeError("Test data not initialized. Did setup() run?")

        self.logger.info(f"Running single-threaded compression test for {duration}s...")
        self.logger.info(
            f"Testing with {self.data_size_mb} MB data, "
            f"compression level {self.compression_level}"
        )

        start_time = time.time()
        compress_count = 0
        decompress_count = 0
        total_bytes_compressed = 0
        total_bytes_decompressed = 0
        compression_speeds = []  # MB/s per iteration
        decompression_speeds = []  # MB/s per iteration
        compressed_sizes = []

        while (time.time() - start_time) < duration:
            # Compression
            comp_start = time.perf_counter()
            compressed = zlib.compress(self._test_data, self.compression_level)
            comp_elapsed = time.perf_counter() - comp_start

            compress_count += 1
            total_bytes_compressed += len(self._test_data)
            compressed_sizes.append(len(compressed))

            if comp_elapsed > 0:
                comp_speed_mbps = (len(self._test_data) / (1024 * 1024)) / comp_elapsed
                compression_speeds.append(comp_speed_mbps)

            # Decompression
            decomp_start = time.perf_counter()
            decompressed = zlib.decompress(compressed)
            decomp_elapsed = time.perf_counter() - decomp_start

            decompress_count += 1
            total_bytes_decompressed += len(decompressed)

            if decomp_elapsed > 0:
                decomp_speed_mbps = (len(decompressed) / (1024 * 1024)) / decomp_elapsed
                decompression_speeds.append(decomp_speed_mbps)

            # Sanity check
            assert decompressed == self._test_data, "Decompression mismatch!"

        elapsed = time.time() - start_time

        # Calculate overall metrics
        total_mb_compressed = total_bytes_compressed / (1024 * 1024)
        total_mb_decompressed = total_bytes_decompressed / (1024 * 1024)
        avg_compress_mbps = total_mb_compressed / elapsed if elapsed > 0 else 0
        avg_decompress_mbps = total_mb_decompressed / elapsed if elapsed > 0 else 0

        # Calculate compression ratio (original / compressed)
        avg_compressed_size = sum(compressed_sizes) / len(compressed_sizes)
        compression_ratio = len(self._test_data) / avg_compressed_size

        # Calculate min/max/p95 for compression
        if compression_speeds:
            min_compress = min(compression_speeds)
            max_compress = max(compression_speeds)
            comp_sorted = sorted(compression_speeds)
            p95_compress = comp_sorted[int(len(comp_sorted) * 0.95)]
        else:
            min_compress = max_compress = p95_compress = 0

        # Calculate min/max/p95 for decompression
        if decompression_speeds:
            min_decompress = min(decompression_speeds)
            max_decompress = max(decompression_speeds)
            decomp_sorted = sorted(decompression_speeds)
            p95_decompress = decomp_sorted[int(len(decomp_sorted) * 0.95)]
        else:
            min_decompress = max_decompress = p95_decompress = 0

        self.logger.info(
            f"Completed {compress_count} compression cycles in {elapsed:.2f}s"
        )
        self.logger.info(f"Compression: {avg_compress_mbps:.2f} MB/s (average)")
        self.logger.info(
            f"  Range: min={min_compress:.2f} MB/s, "
            f"max={max_compress:.2f} MB/s, p95={p95_compress:.2f} MB/s"
        )
        self.logger.info(f"Decompression: {avg_decompress_mbps:.2f} MB/s (average)")
        self.logger.info(
            f"  Range: min={min_decompress:.2f} MB/s, "
            f"max={max_decompress:.2f} MB/s, p95={p95_decompress:.2f} MB/s"
        )
        self.logger.info(f"Compression ratio: {compression_ratio:.2f}x")

        return {
            "test_name": self.get_name(),
            "duration": elapsed,
            "data_size_mb": self.data_size_mb,
            "compression_level": self.compression_level,
            "compression_cycles": compress_count,
            "decompression_cycles": decompress_count,
            "total_mb_compressed": total_mb_compressed,
            "total_mb_decompressed": total_mb_decompressed,
            "avg_compression_mbps": avg_compress_mbps,
            "min_compression_mbps": min_compress,
            "max_compression_mbps": max_compress,
            "p95_compression_mbps": p95_compress,
            "avg_decompression_mbps": avg_decompress_mbps,
            "min_decompression_mbps": min_decompress,
            "max_decompression_mbps": max_decompress,
            "p95_decompression_mbps": p95_decompress,
            "compression_ratio": compression_ratio,
            "burnin_seconds": self.burnin_seconds,
            "cpu_physical_cores": self._cpu_info.total_physical_cores,
            "cpu_logical_cores": self._cpu_info.total_logical_cores,
        }
