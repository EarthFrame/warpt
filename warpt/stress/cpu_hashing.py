"""CPU hashing stress test.

This test measures CPU performance for cryptographic hashing operations,
which are critical for security applications, authentication,
and data integrity verification.
"""

import hashlib
import os
import time
from typing import Any

from warpt.backends.system import CPU
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class CPUHashingTest(StressTest):
    """CPU hashing stress test.

    Measures CPU performance using cryptographic hashing operations.
    This represents real-world security workloads like:
    - Password verification (login systems)
    - File integrity checking (checksums)
    - Digital signatures
    - Blockchain operations
    - SSL/TLS handshakes

    Tests both fast hashing (SHA-256) and slow hashing (PBKDF2).
    Provides practical metrics directly applicable to security operations.
    """

    _PARAM_FIELDS = ("data_size_mb", "pbkdf2_iterations", "burnin_seconds")

    def __init__(
        self,
        data_size_mb: int = 10,
        pbkdf2_iterations: int = 100000,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize CPU crypto test.

        Args:
            data_size_mb: Size of data for SHA-256 hashing. Default 10MB.
            pbkdf2_iterations: Number of PBKDF2 iterations (password hashing).
                100,000 is typical for password storage.
            burnin_seconds: Warmup duration before measurement.
        """
        self.data_size_mb = data_size_mb
        self.pbkdf2_iterations = pbkdf2_iterations
        self.burnin_seconds = burnin_seconds
        self._cpu: CPU | None = None
        self._cpu_info: Any = None
        self._test_data: bytes | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "CPU Hashing Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures CPU hashing performance for security workloads (SHA-256, PBKDF2)"
        )

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.CPU

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Cryptographic hashing is always available (stdlib)."""
        return True

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        if self.data_size_mb < 1:
            raise ValueError("data_size_mb must be >= 1")
        if self.data_size_mb > 1000:
            raise ValueError("data_size_mb must be <= 1000 (avoid excessive memory)")
        if self.pbkdf2_iterations < 10000:
            raise ValueError("pbkdf2_iterations must be >= 10,000 (security minimum)")
        if self.pbkdf2_iterations > 1000000:
            raise ValueError(
                "pbkdf2_iterations must be <= 1,000,000 (avoid excessive runtime)"
            )
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize CPU info and test data."""
        self._cpu = CPU()
        self._cpu_info = self._cpu.get_cpu_info()

        # Generate random test data
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
                # Warmup with SHA-256 hashing
                _ = hashlib.sha256(self._test_data).hexdigest()
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                _ = hashlib.sha256(self._test_data).hexdigest()

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the cryptographic hashing test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored for this test (runs for duration).

        Returns:
            Dictionary containing test results.
        """
        del iterations  # Unused; test runs for duration

        if self._test_data is None:
            raise RuntimeError("Test data not initialized. Did setup() run?")

        self.logger.info(f"Running cryptographic hashing test for {duration}s...")
        self.logger.info(
            f"SHA-256 data size: {self.data_size_mb} MB, "
            f"PBKDF2 iterations: {self.pbkdf2_iterations:,}"
        )

        start_time = time.time()

        # Track SHA-256 metrics
        sha256_count = 0
        sha256_bytes_hashed = 0
        sha256_speeds = []  # MB/s per iteration

        # Track PBKDF2 metrics
        pbkdf2_count = 0
        pbkdf2_speeds = []  # verifications/sec

        # Allocate roughly equal time to each test type
        sha256_duration = duration / 2
        pbkdf2_duration = duration / 2

        # =====================================================================
        # Part 1: SHA-256 Hashing (Fast hashing - file integrity, checksums)
        # =====================================================================
        self.logger.info("Testing SHA-256 hashing performance...")
        sha256_start = time.time()

        while (time.time() - sha256_start) < sha256_duration:
            iter_start = time.perf_counter()
            _ = hashlib.sha256(self._test_data).hexdigest()
            iter_elapsed = time.perf_counter() - iter_start

            sha256_count += 1
            sha256_bytes_hashed += len(self._test_data)

            if iter_elapsed > 0:
                iter_speed = (len(self._test_data) / (1024 * 1024)) / iter_elapsed
                sha256_speeds.append(iter_speed)

        sha256_elapsed = time.time() - sha256_start

        # =====================================================================
        # Part 2: PBKDF2 Password Hashing (Slow hashing - password verification)
        # =====================================================================
        self.logger.info("Testing PBKDF2 password hashing performance...")
        pbkdf2_start = time.time()

        # Simulate password hashing (typical: hash password with salt)
        password = b"test_password_12345"
        salt = b"random_salt_value"

        while (time.time() - pbkdf2_start) < pbkdf2_duration:
            iter_start = time.perf_counter()
            _ = hashlib.pbkdf2_hmac("sha256", password, salt, self.pbkdf2_iterations)
            iter_elapsed = time.perf_counter() - iter_start

            pbkdf2_count += 1

            if iter_elapsed > 0:
                # Calculate verifications per second
                iter_speed = 1.0 / iter_elapsed
                pbkdf2_speeds.append(iter_speed)

        pbkdf2_elapsed = time.time() - pbkdf2_start

        # =====================================================================
        # Calculate metrics
        # =====================================================================
        total_elapsed = time.time() - start_time

        # SHA-256 metrics
        sha256_mb = sha256_bytes_hashed / (1024 * 1024)
        avg_sha256_mbps = sha256_mb / sha256_elapsed if sha256_elapsed > 0 else 0

        if sha256_speeds:
            min_sha256 = min(sha256_speeds)
            max_sha256 = max(sha256_speeds)
            sha256_sorted = sorted(sha256_speeds)
            p95_sha256 = sha256_sorted[int(len(sha256_sorted) * 0.95)]
        else:
            min_sha256 = max_sha256 = p95_sha256 = 0

        # PBKDF2 metrics
        avg_pbkdf2_per_sec = pbkdf2_count / pbkdf2_elapsed if pbkdf2_elapsed > 0 else 0

        if pbkdf2_speeds:
            min_pbkdf2 = min(pbkdf2_speeds)
            max_pbkdf2 = max(pbkdf2_speeds)
            pbkdf2_sorted = sorted(pbkdf2_speeds)
            p95_pbkdf2 = pbkdf2_sorted[int(len(pbkdf2_sorted) * 0.95)]
        else:
            min_pbkdf2 = max_pbkdf2 = p95_pbkdf2 = 0

        # Logging
        self.logger.info(
            f"SHA-256: Hashed {sha256_mb:.2f} MB in {sha256_elapsed:.2f}s "
            f"({sha256_count} iterations)"
        )
        self.logger.info(f"  Average: {avg_sha256_mbps:.2f} MB/s")
        self.logger.info(
            f"  Range: min={min_sha256:.2f} MB/s, "
            f"max={max_sha256:.2f} MB/s, p95={p95_sha256:.2f} MB/s"
        )

        self.logger.info(
            f"PBKDF2: {pbkdf2_count} password hashes in {pbkdf2_elapsed:.2f}s"
        )
        self.logger.info(f"  Average: {avg_pbkdf2_per_sec:.2f} verifications/sec")
        self.logger.info(
            f"  Range: min={min_pbkdf2:.2f}/s, "
            f"max={max_pbkdf2:.2f}/s, p95={p95_pbkdf2:.2f}/s"
        )

        return {
            "test_name": self.get_name(),
            "duration": total_elapsed,
            "data_size_mb": self.data_size_mb,
            "pbkdf2_iterations": self.pbkdf2_iterations,
            # SHA-256 metrics
            "sha256_iterations": sha256_count,
            "sha256_mb_hashed": sha256_mb,
            "sha256_avg_mbps": avg_sha256_mbps,
            "sha256_min_mbps": min_sha256,
            "sha256_max_mbps": max_sha256,
            "sha256_p95_mbps": p95_sha256,
            # PBKDF2 metrics
            "pbkdf2_hashes": pbkdf2_count,
            "pbkdf2_avg_per_sec": avg_pbkdf2_per_sec,
            "pbkdf2_min_per_sec": min_pbkdf2,
            "pbkdf2_max_per_sec": max_pbkdf2,
            "pbkdf2_p95_per_sec": p95_pbkdf2,
            # System info
            "burnin_seconds": self.burnin_seconds,
            "cpu_physical_cores": self._cpu_info.total_physical_cores,
            "cpu_logical_cores": self._cpu_info.total_logical_cores,
        }
