"""Real network point-to-point stress test for latency and bandwidth measurement.

This test measures actual network performance between two machines:
- Source machine (running this test) → Target machine (running server)
"""

from __future__ import annotations

import socket
import statistics
import time
from typing import Any

from warpt.backends.network import Network
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class NetworkPointToPointTest(StressTest):
    """Point-to-point network test for real network latency and bandwidth.

    Tests actual network performance between two machines on a network.
    Requires a simple server running on the target machine for latency tests.

    Measures:
    - Real network latency (ping-pong round-trip time)
    - Network bandwidth (upload/download speeds)
    - Multiple target IPs in one test run

    Use cases:
    - Datacenter node-to-node communication
    - Cloud instance network profiling
    - LAN performance testing
    - Internet connectivity testing
    """

    _PARAM_FIELDS = (
        "target_ips",
        "test_mode",
        "payload_size",
        "port",
        "burnin_seconds",
        "timeout_seconds",
    )

    def __init__(
        self,
        target_ips: list[str] | str = "127.0.0.1",
        test_mode: str = "both",  # "latency", "bandwidth", or "both"
        payload_size: int = 4096,  # 4KB for latency, auto-adjusted for bandwidth
        port: int = 5201,  # Default port
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
        timeout_seconds: float = 5.0,  # Connection timeout
    ):
        """Initialize network point-to-point test.

        Args:
            target_ips: Target IP(s) to test. Can be single IP string or list of IPs.
            test_mode: What to measure - "latency", "bandwidth", or "both".
            payload_size: Data payload size in bytes.
            port: Port number for connection.
            burnin_seconds: Warmup duration before measurement.
            timeout_seconds: Socket timeout for connections.
        """
        # Convert single IP to list
        if isinstance(target_ips, str):
            self.target_ips = [target_ips]
        else:
            self.target_ips = target_ips

        self.test_mode = test_mode
        self.payload_size = payload_size
        self.port = port
        self.burnin_seconds = burnin_seconds
        self.timeout_seconds = timeout_seconds

        # Runtime state
        self._payload: bytes | None = None
        self._results: dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return "network_point_to_point"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Network Point-to-Point Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures real network latency and bandwidth between network endpoints"

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.NETWORK

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_parameters(self, params: dict[Any, Any]) -> None:
        """Override to handle non-integer parameters."""
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
        valid_modes = ["latency", "bandwidth", "both"]
        if self.test_mode not in valid_modes:
            raise ValueError(
                f"test_mode must be one of {valid_modes}, got '{self.test_mode}'"
            )

        # Validate target_ips
        if not self.target_ips:
            raise ValueError("target_ips cannot be empty")

        resolved_ips = []
        for ip in self.target_ips:
            resolved = Network.resolve_hostname(ip)
            if resolved is None:
                raise ValueError(f"Cannot resolve target IP: {ip}")
            resolved_ips.append(resolved)

        self.target_ips = resolved_ips

        # Validate payload_size
        if self.payload_size <= 0:
            raise ValueError("payload_size must be greater than 0")

        # Validate port
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be between 1 and 65535")

        # Validate burnin_seconds
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Validate timeout
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.logger.info(f"Targets: {', '.join(self.target_ips)}")
        self.logger.info(f"Mode: {self.test_mode}")
        self.logger.info(
            f"Payload: {self.payload_size} bytes ({self.payload_size / 1024:.1f} KB)"
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize network resources."""
        # Generate payload (random data)
        import random

        self._payload = bytes(random.getrandbits(8) for _ in range(self.payload_size))

        self.logger.info(f"Testing {len(self.target_ips)} target(s)")

    def teardown(self) -> None:
        """Clean up network resources."""
        self._payload = None
        self._results = {}

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
        """Execute the network test for all targets.

        Args:
            duration: Test duration in seconds (for bandwidth tests).
            iterations: Number of iterations (for latency tests).

        Returns:
            Dictionary containing results for all targets.
        """
        # TODO: Consider Pydantic models for network results
        # (NetworkTestResults in stress_models.py)
        results = {}

        for target_ip in self.target_ips:
            self.logger.info(f"Testing {target_ip}...")

            if self.test_mode == "latency":
                results[target_ip] = self._test_latency(target_ip, iterations)
            elif self.test_mode == "bandwidth":
                results[target_ip] = self._test_bandwidth(target_ip, duration)
            elif self.test_mode == "both":
                lat_results = self._test_latency(target_ip, iterations)
                bw_results = self._test_bandwidth(target_ip, duration)
                results[target_ip] = {**lat_results, **bw_results}

        return {
            "test_name": self.get_name(),
            "test_mode": self.test_mode,
            "targets": self.target_ips,
            "results": results,
        }

    def _test_latency(self, target_ip: str, iterations: int = 10) -> dict[str, Any]:
        """Measure network latency via TCP ping-pong.

        Attempts to connect and measure round-trip time.
        If connection fails, returns error information.

        Args:
            target_ip: Target IP address.
            iterations: Number of ping iterations.

        Returns:
            Dictionary with latency measurements or error info.
        """
        self.logger.info(
            f"Measuring latency to {target_ip} ({iterations} iterations)..."
        )

        latencies = []
        failed_attempts = 0

        for i in range(iterations):
            try:
                start = time.perf_counter()

                # Create socket and attempt connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_seconds)

                try:
                    # Attempt connection - this measures RTT
                    sock.connect((target_ip, self.port))

                    # Send small payload and wait for response
                    # (requires echo server on target)
                    assert self._payload is not None
                    sock.sendall(self._payload)
                    _ = sock.recv(len(self._payload))  # Receive echo

                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed * 1000)  # Convert to ms

                finally:
                    sock.close()

            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                failed_attempts += 1
                self.logger.debug(f"Attempt {i+1} failed: {e}")

        if not latencies:
            # All attempts failed
            self.logger.warning(
                f"Could not connect to {target_ip}:{self.port}. "
                "Is a server running on the target?"
            )
            return {
                "target_ip": target_ip,
                "iterations": iterations,
                "failed_attempts": failed_attempts,
                "avg_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "std_dev_ms": None,
                "error": f"Connection failed - no server at {target_ip}:{self.port}",
            }

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        self.logger.info(
            f"Latency to {target_ip}: "
            f"avg={avg_latency:.2f}ms min={min_latency:.2f}ms max={max_latency:.2f}ms"
        )

        return {
            "target_ip": target_ip,
            "iterations": iterations,
            "successful_attempts": len(latencies),
            "failed_attempts": failed_attempts,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "std_dev_ms": std_dev,
        }

    def _test_bandwidth(self, target_ip: str, duration: int) -> dict[str, Any]:
        """Measure network bandwidth via sustained data transfer.

        Sends large chunks of data continuously and measures throughput.

        Args:
            target_ip: Target IP address.
            duration: Test duration in seconds.

        Returns:
            Dictionary with bandwidth measurements.
        """
        self.logger.info(f"Measuring bandwidth to {target_ip} ({duration}s)...")

        # Use larger chunks for bandwidth testing
        chunk_size = 1024 * 1024  # 1MB chunks
        chunk = bytes(range(256)) * (chunk_size // 256)  # Repeating pattern

        try:
            # Connect to target
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            sock.connect((target_ip, self.port))

            # Send data continuously for duration
            start_time = time.perf_counter()
            total_bytes_sent = 0
            chunks_sent = 0

            while (time.perf_counter() - start_time) < duration:
                try:
                    sock.sendall(chunk)
                    total_bytes_sent += len(chunk)
                    chunks_sent += 1
                except (socket.timeout, BrokenPipeError, OSError):
                    break

            elapsed = time.perf_counter() - start_time
            sock.close()

            # Calculate bandwidth
            # Mbps = (bytes * 8 bits/byte) / (time_seconds * 1_000_000 bits/Mbps)
            bandwidth_mbps = (total_bytes_sent * 8) / (elapsed * 1_000_000)
            total_mb = total_bytes_sent / (1024 * 1024)

            self.logger.info(
                f"Bandwidth to {target_ip}: {bandwidth_mbps:.2f} Mbps "
                f"({total_mb:.2f} MB transferred)"
            )

            return {
                "target_ip": target_ip,
                "duration": elapsed,
                "bandwidth_mbps": bandwidth_mbps,
                "total_bytes": total_bytes_sent,
                "total_mb": total_mb,
                "chunks_sent": chunks_sent,
            }

        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            self.logger.warning(
                f"Bandwidth test to {target_ip}:{self.port} failed: {e!s}"
            )
            return {
                "target_ip": target_ip,
                "duration": duration,
                "bandwidth_mbps": None,
                "total_bytes": 0,
                "error": f"Connection failed - {e!s}",
            }
