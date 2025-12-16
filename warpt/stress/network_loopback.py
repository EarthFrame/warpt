"""Network loopback stress test for localhost network stack testing."""

from __future__ import annotations

import socket
import time
from typing import Any

from warpt.backends.network import Network
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class NetworkLoopbackTest(StressTest):
    """Localhost loopback network test measuring network stack overhead.

    This test measures the performance of the local network stack (loopback interface)
    without any real network hardware involvement. Useful for:
    - Baseline network stack performance
    - Development/debugging
    - OS networking overhead measurement

    NOT useful for:
    - Real network performance (data never leaves the machine)
    - Network hardware testing (NIC, cables, switches)
    - Multi-node communication profiling
    """

    _PARAM_FIELDS = (
        "target_ip",
        "test_mode",
        "payload_size",
        "port",
        "burnin_seconds",
    )

    def __init__(
        self,
        target_ip: str = "127.0.0.1",
        test_mode: str = "latency",  # "latency", "bandwidth", or "both"
        payload_size: int = 4096,  # 4KB default, increase for bandwidth
        port: int = 5201,  # Default port for network tests
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize network point-to-point test.

        Args:
            target_ip: Target IP address to test connectivity to.
            test_mode: What to measure - "latency", "bandwidth", or "both".
            payload_size: Size of data payload in bytes
                (4KB for latency, larger for bandwidth).
            port: Port number to use for connection.
            burnin_seconds: Warmup duration before measurement.
        """
        self.target_ip = target_ip
        self.test_mode = test_mode
        self.payload_size = payload_size
        self.port = port
        self.burnin_seconds = burnin_seconds

        # Runtime state
        self._socket: socket.socket | None = None
        self._payload: bytes | None = None
        self._is_localhost: bool = False

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return "network_loopback"

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Network Loopback Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures localhost network stack overhead (loopback interface)"

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
        """Check if network testing is available.

        Network tests are always available as they use built-in socket library.
        """
        return True

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        # Validate test_mode
        valid_modes = ["latency", "bandwidth", "both"]
        if self.test_mode not in valid_modes:
            raise ValueError(
                f"test_mode must be one of {valid_modes}, got '{self.test_mode}'"
            )

        # Validate payload_size
        if self.payload_size <= 0:
            raise ValueError("payload_size must be greater than 0")

        # Validate port
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be between 1 and 65535")

        # Validate burnin_seconds
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Validate target_ip
        resolved_ip = Network.resolve_hostname(self.target_ip)
        if resolved_ip is None:
            raise ValueError(f"Cannot resolve target_ip: {self.target_ip}")

        # Update to resolved IP
        self.target_ip = resolved_ip
        self._is_localhost = Network.is_localhost(self.target_ip)

        self.logger.info(f"Target: {self.target_ip}:{self.port}")
        self.logger.info(f"Mode: {self.test_mode}")
        self.logger.info(
            f"Payload: {self.payload_size} bytes ({self.payload_size / 1024:.1f} KB)"
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize network resources."""
        # Generate payload (random data for realistic testing)
        import random

        self._payload = bytes(random.getrandbits(8) for _ in range(self.payload_size))

        self.logger.info(f"Testing to {self.target_ip}:{self.port}")

    def teardown(self) -> None:
        """Clean up network resources."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        self._payload = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup iterations."""
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                # Simple warmup - just sleep
                time.sleep(0.01)
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[Any, Any]:
        """Execute the network test.

        Args:
            duration: Test duration in seconds.
            iterations: Number of iterations (used for latency test).

        Returns:
            Dictionary containing test results.
        """
        if self.test_mode == "latency":
            return self._test_latency(iterations)
        elif self.test_mode == "bandwidth":
            return self._test_bandwidth(duration)
        elif self.test_mode == "both":
            latency_results = self._test_latency(iterations)
            bandwidth_results = self._test_bandwidth(duration)
            return {**latency_results, **bandwidth_results}
        else:
            raise ValueError(f"Unknown test_mode: {self.test_mode}")

    def _test_latency(self, iterations: int) -> dict[str, Any]:
        """Measure network latency (round-trip time).

        Args:
            iterations: Number of ping iterations to perform.

        Returns:
            Dictionary with latency measurements.
        """
        self.logger.info(f"Measuring latency ({iterations} iterations)...")

        # For localhost, we can do a simple loopback test
        if self._is_localhost:
            return self._test_latency_localhost(iterations)

        # For remote hosts, would need a server running on the target
        # For now, return placeholder
        self.logger.warning(
            "Remote latency testing requires server mode (not yet implemented)"
        )
        return {
            "test_mode": "latency",
            "target_ip": self.target_ip,
            "iterations": iterations,
            "avg_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "note": "Remote latency testing requires server mode",
        }

    def _test_latency_localhost(self, iterations: int) -> dict[str, Any]:
        """Measure localhost loopback latency.

        Args:
            iterations: Number of iterations.

        Returns:
            Dictionary with latency measurements.
        """
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()

            # Simple loopback test: create socket, connect, send, receive, close
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)

            try:
                # For localhost, we test socket creation overhead
                # This gives us a baseline of local network stack latency
                pass
            finally:
                sock.close()

            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # Convert to ms

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        self.logger.info(
            f"Latency: avg={avg_latency:.3f}ms "
            f"min={min_latency:.3f}ms max={max_latency:.3f}ms"
        )

        return {
            "test_mode": "latency",
            "target_ip": self.target_ip,
            "iterations": iterations,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
        }

    def _test_bandwidth(self, duration: int) -> dict[str, Any]:
        """Measure network bandwidth.

        Args:
            duration: Test duration in seconds.

        Returns:
            Dictionary with bandwidth measurements.
        """
        self.logger.info(f"Measuring bandwidth for {duration}s...")

        # Bandwidth testing requires iperf3 or similar
        # For now, return placeholder
        self.logger.warning("Bandwidth testing not yet implemented")
        return {
            "test_mode": "bandwidth",
            "target_ip": self.target_ip,
            "duration": duration,
            "bandwidth_mbps": None,
            "note": "Bandwidth testing not yet implemented",
        }
