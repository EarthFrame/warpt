"""Network bidirectional stress test for upload and download measurement.

This test measures both upload and download performance simultaneously
or sequentially between two machines.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from warpt.backends.network import Network
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class NetworkBidirectionalTest(StressTest):
    """Bidirectional network test for full-duplex performance measurement.

    Tests network performance in both directions (upload and download)
    either sequentially or simultaneously.

    Measures:
    - Upload bandwidth (client → server)
    - Download bandwidth (server → client)
    - Aggregate bandwidth (upload + download combined)

    Use cases:
    - Testing full-duplex network capability
    - Detecting bandwidth asymmetry
    - Validating bidirectional workloads (databases, web servers, etc.)
    """

    _PARAM_FIELDS = (
        "target_ip",
        "test_mode",
        "duration",
        "chunk_size",
        "port",
        "burnin_seconds",
        "timeout_seconds",
    )

    def __init__(
        self,
        target_ip: str = "127.0.0.1",
        test_mode: str = "simultaneous",  # "sequential", "simultaneous"
        duration: int = 60,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
        port: int = 5201,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
        timeout_seconds: float = 10.0,
    ):
        """Initialize network bidirectional test.

        Args:
            target_ip: Target IP address to test.
            test_mode: Test mode - "sequential" or "simultaneous".
                - "sequential": Runs upload first, then download (one at a time).
                  Total test time = 2x duration. Tests half-duplex performance.
                - "simultaneous": Runs upload and download concurrently in
                  separate threads. Total test time = duration. Tests full-duplex
                  performance and how the link handles contention.
            duration: Test duration in seconds (per direction for sequential,
                total for simultaneous).
            chunk_size: Data chunk size in bytes.
            port: Port number for connection.
            burnin_seconds: Warmup duration before measurement.
            timeout_seconds: Socket timeout for connections.
        """
        self.target_ip = target_ip
        self.test_mode = test_mode
        self.duration = duration
        self.chunk_size = chunk_size
        self.port = port
        self.burnin_seconds = burnin_seconds
        self.timeout_seconds = timeout_seconds

        # Runtime state
        self._upload_results: dict[str, Any] = {}
        self._download_results: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return "network_bidirectional"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Network Bidirectional Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures bidirectional network performance "
            "(upload and download simultaneously)"
        )

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.NETWORK

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_parameters(self, params: dict[Any, Any]) -> None:
        """Override to handle non-integer parameters.

        The base class converts all values to int, but this test has:
        - target_ip: str
        - test_mode: str
        - timeout_seconds: float

        Without this override, int("192.168.1.1") would raise ValueError.
        """
        for field in self._PARAM_FIELDS:
            if field in params:
                setattr(self, field, params[field])

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if network testing is available."""
        return True

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        # Validate test_mode
        valid_modes = ["sequential", "simultaneous"]
        if self.test_mode not in valid_modes:
            raise ValueError(
                f"test_mode must be one of {valid_modes}, got '{self.test_mode}'"
            )

        # Validate target_ip
        resolved_ip = Network.resolve_hostname(self.target_ip)
        if resolved_ip is None:
            raise ValueError(f"Cannot resolve target IP: {self.target_ip}")

        # Reject localhost - no utility in network test to self
        localhost_addresses = {"127.0.0.1", "::1", "localhost"}
        if resolved_ip in localhost_addresses or resolved_ip.startswith("127."):
            raise ValueError(
                f"Cannot test to localhost ({self.target_ip}). "
                "Bidirectional test requires a remote target."
            )
        self.target_ip = resolved_ip

        # Validate duration
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")

        # Validate chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        # Validate port
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be between 1 and 65535")

        # Validate burnin_seconds
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Validate timeout
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.logger.info(f"Target: {self.target_ip}:{self.port}")
        self.logger.info(f"Mode: {self.test_mode}")
        self.logger.info(f"Duration: {self.duration}s")
        self.logger.info(f"Chunk size: {self.chunk_size / 1024:.1f} KB")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize test resources."""
        self.logger.info(
            f"Testing bidirectional network performance to {self.target_ip}"
        )

    def teardown(self) -> None:
        """Clean up test resources."""
        del self._upload_results
        del self._download_results

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup to prime TCP connection.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Unused, kept for compatibility.
        """
        _ = iterations
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            self._warmup_connection()

    def _warmup_connection(self) -> None:
        """Send warmup packets to prime TCP connection."""
        warmup_payload = b"\x00" * 4096
        num_warmup_packets = 10

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            sock.connect((self.target_ip, self.port))

            for _ in range(num_warmup_packets):
                try:
                    sock.sendall(warmup_payload)
                    sock.recv(len(warmup_payload))
                except (TimeoutError, OSError):
                    break

            sock.close()
            self.logger.debug("Warmup complete")

        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            self.logger.debug(f"Warmup failed: {e}")

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[Any, Any]:
        """Execute the bidirectional network test.

        Args:
            duration: Test duration in seconds (overrides self.duration).
            iterations: Unused for network tests.

        Returns:
            Dictionary containing test results.

        Raises:
            RuntimeError: If connection to target fails.
        """
        _ = iterations
        test_duration = duration if duration > 0 else self.duration

        self.logger.info(
            f"Running {self.test_mode} bidirectional test for {test_duration}s..."
        )

        results: dict[str, Any] = {
            "test_name": self.get_name(),
            "target_ip": self.target_ip,
            "test_mode": self.test_mode,
            "duration": test_duration,
            "chunk_size": self.chunk_size,
        }

        if self.test_mode == "sequential":
            seq_results = self._run_sequential(test_duration)
            results.update(seq_results)

        elif self.test_mode == "simultaneous":
            sim_results = self._run_simultaneous(test_duration)
            results.update(sim_results)

        return results

    # -------------------------------------------------------------------------
    # Test Mode Implementations
    # -------------------------------------------------------------------------

    def _run_sequential(self, duration: int) -> dict[str, Any]:
        """Run sequential test: upload then download.

        Args:
            duration: Duration for each direction (total = 2x duration).

        Returns:
            Dictionary with upload and download results.
        """
        self.logger.info("Running sequential mode (upload, then download)...")

        # Upload test
        self.logger.info(f"Upload test ({duration}s)...")
        upload_results = self._test_upload(duration)

        # Download test
        self.logger.info(f"Download test ({duration}s)...")
        download_results = self._test_download(duration)

        upload_mbps = upload_results.get("bandwidth_mbps", 0)
        download_mbps = download_results.get("bandwidth_mbps", 0)

        return {
            "upload_bandwidth_mbps": upload_mbps,
            "upload_bytes": upload_results.get("total_bytes", 0),
            "download_bandwidth_mbps": download_mbps,
            "download_bytes": download_results.get("total_bytes", 0),
            "aggregate_bandwidth_mbps": upload_mbps + download_mbps,
        }

    def _run_simultaneous(self, duration: int) -> dict[str, Any]:
        """Run simultaneous test: upload and download at same time.

        Args:
            duration: Duration for both directions (running concurrently).

        Returns:
            Dictionary with combined results.
        """
        self.logger.info("Running simultaneous mode (upload + download)...")

        # Shared results dict (thread-safe for simple assignments)
        results: dict[str, Any] = {}

        # Upload thread
        def upload_worker() -> None:
            results["upload"] = self._test_upload(duration)

        # Download thread
        def download_worker() -> None:
            results["download"] = self._test_download(duration)

        # Start both threads
        upload_thread = threading.Thread(target=upload_worker)
        download_thread = threading.Thread(target=download_worker)

        upload_thread.start()
        download_thread.start()

        # Wait for both to complete
        upload_thread.join()
        download_thread.join()

        # Extract results
        upload_mbps = results.get("upload", {}).get("bandwidth_mbps", 0)
        download_mbps = results.get("download", {}).get("bandwidth_mbps", 0)

        return {
            "upload_bandwidth_mbps": upload_mbps,
            "upload_bytes": results.get("upload", {}).get("total_bytes", 0),
            "download_bandwidth_mbps": download_mbps,
            "download_bytes": results.get("download", {}).get("total_bytes", 0),
            "aggregate_bandwidth_mbps": upload_mbps + download_mbps,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _test_upload(self, duration: int) -> dict[str, Any]:
        """Test upload bandwidth (client → server).

        Args:
            duration: Test duration in seconds.

        Returns:
            Dictionary with upload metrics.
        """
        chunk = bytes(range(256)) * (self.chunk_size // 256)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            sock.connect((self.target_ip, self.port))

            start_time = time.perf_counter()
            total_bytes_sent = 0

            while (time.perf_counter() - start_time) < duration:
                try:
                    sock.sendall(chunk)
                    total_bytes_sent += len(chunk)
                except (TimeoutError, BrokenPipeError, OSError):
                    break

            elapsed = time.perf_counter() - start_time
            sock.close()

            bandwidth_mbps = (total_bytes_sent * 8) / (elapsed * 1_000_000)

            self.logger.info(f"Upload: {bandwidth_mbps:.2f} Mbps")

            return {
                "bandwidth_mbps": bandwidth_mbps,
                "total_bytes": total_bytes_sent,
                "duration": elapsed,
            }

        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            self.logger.error(f"Upload test failed: {e}")
            return {"bandwidth_mbps": 0, "total_bytes": 0, "error": str(e)}

    def _test_download(self, duration: int) -> dict[str, Any]:
        """Test download bandwidth (server → client).

        Sends special header to server requesting bidirectional mode,
        then receives data from server.

        Args:
            duration: Test duration in seconds.

        Returns:
            Dictionary with download metrics.
        """
        # TODO: Add packet loss measurement

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            sock.connect((self.target_ip, self.port))

            # Send BIDIR header to tell server to send data back
            header = f"BIDIR:{duration}:{self.chunk_size}".encode()
            sock.sendall(header)

            # Receive data from server
            start_time = time.perf_counter()
            total_bytes_received = 0

            while (time.perf_counter() - start_time) < duration:
                try:
                    data = sock.recv(1024 * 1024)  # 1MB chunks
                    if not data:
                        break
                    total_bytes_received += len(data)
                except (TimeoutError, OSError):
                    break

            elapsed = time.perf_counter() - start_time
            sock.close()

            bandwidth_mbps = (total_bytes_received * 8) / (elapsed * 1_000_000)

            self.logger.info(f"Download: {bandwidth_mbps:.2f} Mbps")

            return {
                "bandwidth_mbps": bandwidth_mbps,
                "total_bytes": total_bytes_received,
                "duration": elapsed,
            }

        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            self.logger.error(f"Download test failed: {e}")
            return {"bandwidth_mbps": 0, "total_bytes": 0, "error": str(e)}
