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

    Tests configurable fast hashing algorithms and slow hashing (PBKDF2).
    Provides practical metrics directly applicable to security operations.
    """

    # Supported hash functions from hashlib (always available)
    SUPPORTED_HASH_FUNCTIONS: tuple[str, ...] = (
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "sha3_224",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
        "blake3",  # Optional, requires blake3 package
    )

    _PARAM_FIELDS = (
        "hash_functions",
        "num_iterations",
        "data_size_mb",
        "pbkdf2_iterations",
        "burnin_seconds",
    )

    def __init__(
        self,
        hash_functions: list[str] | None = None,
        num_iterations: int | None = None,
        data_size_mb: int = 10,
        pbkdf2_iterations: int = 100000,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize CPU crypto test.

        Args:
            hash_functions: List of hash functions to test. Defaults to ["sha256"].
                Supported: md5, sha1, sha224, sha256, sha384, sha512,
                sha3_224, sha3_256, sha3_384, sha3_512, blake2b, blake2s,
                blake3 (requires optional blake3 package).
            num_iterations: Number of iterations per hash function. If None,
                runs for duration-based testing.
            data_size_mb: Size of data for hashing. Default 10MB.
            pbkdf2_iterations: Number of PBKDF2 iterations (password hashing).
                100,000 is typical for password storage.
            burnin_seconds: Warmup duration before measurement.
        """
        self.hash_functions = hash_functions if hash_functions else ["sha256"]
        self.num_iterations = num_iterations
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
        # Validate hash_functions
        if not self.hash_functions:
            raise ValueError("hash_functions must not be empty")
        for hf in self.hash_functions:
            if hf not in self.SUPPORTED_HASH_FUNCTIONS:
                raise ValueError(
                    f"Unsupported hash function: {hf}. "
                    f"Supported: {', '.join(self.SUPPORTED_HASH_FUNCTIONS)}"
                )
            # Check blake3 availability
            if hf == "blake3":
                try:
                    import blake3 as _  # noqa: F401
                except ImportError as e:
                    raise ValueError(
                        "blake3 hash function requires the 'blake3' package. "
                        "Install with: pip install blake3"
                    ) from e

        # Validate num_iterations
        if self.num_iterations is not None and self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")

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

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set test parameters from config, handling list and optional types."""
        # Handle hash_functions (list type)
        if "hash_functions" in params:
            value = params["hash_functions"]
            if isinstance(value, list):
                self.hash_functions = value
            elif isinstance(value, str):
                self.hash_functions = [h.strip() for h in value.split(",")]

        # Handle num_iterations (optional int)
        if "num_iterations" in params:
            value = params["num_iterations"]
            self.num_iterations = None if value is None else int(value)

        # Handle remaining int fields via parent
        int_params = {
            k: v
            for k, v in params.items()
            if k in {"data_size_mb", "pbkdf2_iterations", "burnin_seconds"}
        }
        if int_params:
            super().set_parameters(int_params)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_hasher(self, name: str, data: bytes) -> bytes:
        """Hash data using the specified hash function.

        Args:
            name: Name of the hash function (e.g., "sha256", "blake3").
            data: Data to hash.

        Returns:
            The hash digest as bytes.
        """
        if name == "blake3":
            import blake3

            result: bytes = blake3.blake3(data).digest()
            return result
        else:
            return hashlib.new(name, data).digest()

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

        # Use first configured hash function for warmup
        warmup_hash = self.hash_functions[0]

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                _ = self._get_hasher(warmup_hash, self._test_data)
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                _ = self._get_hasher(warmup_hash, self._test_data)

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the cryptographic hashing test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored for this test (uses num_iterations or duration).

        Returns:
            Dictionary containing test results with per-hash function stats.
        """
        del iterations  # Unused; test uses self.num_iterations or duration

        if self._test_data is None:
            raise RuntimeError("Test data not initialized. Did setup() run?")

        self.logger.info(f"Running cryptographic hashing test for {duration}s...")
        self.logger.info(
            f"Hash functions: {', '.join(self.hash_functions)}, "
            f"Data size: {self.data_size_mb} MB"
        )
        if self.num_iterations:
            self.logger.info(f"Iterations per hash: {self.num_iterations}")

        start_time = time.time()

        # Number of test phases: each hash function + PBKDF2
        num_phases = len(self.hash_functions) + 1
        phase_duration = duration / num_phases

        # =====================================================================
        # Part 1: Fast Hashing (configurable hash functions)
        # =====================================================================
        hash_results: dict[str, dict[str, Any]] = {}

        for hash_name in self.hash_functions:
            self.logger.info(f"Testing {hash_name.upper()} hashing performance...")

            hash_count = 0
            hash_bytes = 0
            hash_speeds: list[float] = []

            hash_start = time.time()

            # Run for duration or num_iterations
            while True:
                # Check termination condition
                if self.num_iterations:
                    if hash_count >= self.num_iterations:
                        break
                else:
                    if (time.time() - hash_start) >= phase_duration:
                        break

                iter_start = time.perf_counter()
                _ = self._get_hasher(hash_name, self._test_data)
                iter_elapsed = time.perf_counter() - iter_start

                hash_count += 1
                hash_bytes += len(self._test_data)

                if iter_elapsed > 0:
                    iter_speed = (len(self._test_data) / (1024 * 1024)) / iter_elapsed
                    hash_speeds.append(iter_speed)

            hash_elapsed = time.time() - hash_start

            # Calculate metrics for this hash function
            hash_mb = hash_bytes / (1024 * 1024)
            avg_mbps = hash_mb / hash_elapsed if hash_elapsed > 0 else 0

            if hash_speeds:
                min_mbps = min(hash_speeds)
                max_mbps = max(hash_speeds)
                sorted_speeds = sorted(hash_speeds)
                p95_mbps = sorted_speeds[int(len(sorted_speeds) * 0.95)]
            else:
                min_mbps = max_mbps = p95_mbps = 0.0

            hash_results[hash_name] = {
                "iterations": hash_count,
                "mb_hashed": hash_mb,
                "elapsed_seconds": hash_elapsed,
                "avg_mbps": avg_mbps,
                "min_mbps": min_mbps,
                "max_mbps": max_mbps,
                "p95_mbps": p95_mbps,
            }

            self.logger.info(
                f"  {hash_name.upper()}: Hashed {hash_mb:.2f} MB "
                f"in {hash_elapsed:.2f}s ({hash_count} iterations)"
            )
            self.logger.info(f"    Average: {avg_mbps:.2f} MB/s")
            self.logger.info(
                f"    Range: min={min_mbps:.2f} MB/s, "
                f"max={max_mbps:.2f} MB/s, p95={p95_mbps:.2f} MB/s"
            )

        # =====================================================================
        # Part 2: PBKDF2 Password Hashing (Slow hashing - password verification)
        # =====================================================================
        self.logger.info("Testing PBKDF2 password hashing performance...")

        password = b"test_password_12345"
        salt = b"random_salt_value"

        pbkdf2_count = 0
        pbkdf2_speeds: list[float] = []

        pbkdf2_start = time.time()

        while (time.time() - pbkdf2_start) < phase_duration:
            iter_start = time.perf_counter()
            _ = hashlib.pbkdf2_hmac("sha256", password, salt, self.pbkdf2_iterations)
            iter_elapsed = time.perf_counter() - iter_start

            pbkdf2_count += 1

            if iter_elapsed > 0:
                iter_speed = 1.0 / iter_elapsed
                pbkdf2_speeds.append(iter_speed)

        pbkdf2_elapsed = time.time() - pbkdf2_start

        # PBKDF2 metrics
        avg_pbkdf2_per_sec = pbkdf2_count / pbkdf2_elapsed if pbkdf2_elapsed > 0 else 0

        if pbkdf2_speeds:
            min_pbkdf2 = min(pbkdf2_speeds)
            max_pbkdf2 = max(pbkdf2_speeds)
            pbkdf2_sorted = sorted(pbkdf2_speeds)
            p95_pbkdf2 = pbkdf2_sorted[int(len(pbkdf2_sorted) * 0.95)]
        else:
            min_pbkdf2 = max_pbkdf2 = p95_pbkdf2 = 0.0

        self.logger.info(
            f"  PBKDF2: {pbkdf2_count} password hashes in {pbkdf2_elapsed:.2f}s"
        )
        self.logger.info(f"    Average: {avg_pbkdf2_per_sec:.2f} verifications/sec")
        self.logger.info(
            f"    Range: min={min_pbkdf2:.2f}/s, "
            f"max={max_pbkdf2:.2f}/s, p95={p95_pbkdf2:.2f}/s"
        )

        # =====================================================================
        # Build results
        # =====================================================================
        total_elapsed = time.time() - start_time

        return {
            "test_name": self.get_name(),
            "duration": total_elapsed,
            "data_size_mb": self.data_size_mb,
            "hash_functions": self.hash_functions,
            "num_iterations": self.num_iterations,
            # Per-hash function results
            "hash_results": hash_results,
            # PBKDF2 metrics
            "pbkdf2_iterations_config": self.pbkdf2_iterations,
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
