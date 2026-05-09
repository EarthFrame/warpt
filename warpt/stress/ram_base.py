"""Shared base class for RAM stress tests."""

from typing import Any

from warpt.stress.base import StressTest, TestCategory


class RAMBaseTest(StressTest):
    """Intermediate base for RAM tests.

    Provides shared state, availability check, category, benchmark methods,
    and memory cleanup. Leaves six abstract methods from StressTest unimplemented
    so this class stays abstract and the registry skips it.
    """

    def __init__(self) -> None:
        self._array: Any | None = None
        self._source_array_for_writes: Any | None = None
        self._total_ram_gb: float = 0.0
        self._available_ram_gb: float = 0.0
        self._memory_type: str | None = None

    # ------------------------------------------------------------------
    # Identity & Metadata (concrete)
    # ------------------------------------------------------------------

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.RAM

    # ------------------------------------------------------------------
    # Hardware & Availability (concrete)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if NumPy and psutil are available."""
        try:
            import numpy  # noqa: F401
            import psutil  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _detect_memory_type(self) -> None:
        """Populate ``_memory_type`` via the RAM backend."""
        from warpt.backends.ram import RAM

        ram = RAM()
        self._memory_type, _ = ram._detect_ddr_info()

    def _free_arrays(self) -> None:
        """Free ``_array`` and ``_source_array_for_writes``."""
        if self._array is not None:
            del self._array
            self._array = None
        if self._source_array_for_writes is not None:
            del self._source_array_for_writes
            self._source_array_for_writes = None

    # ------------------------------------------------------------------
    # Benchmark methods
    # ------------------------------------------------------------------

    def _benchmark_sequential_read(self, duration: int) -> float:
        """Benchmark sequential read bandwidth.

        Args:
            duration: Test duration in seconds.

        Returns:
            Read bandwidth in GB/s.
        """
        import time

        import numpy as np

        assert self._array is not None, "Array not allocated in setup()"

        start_time = time.time()
        bytes_read = 0
        iter_count = 0
        next_log = start_time + 5  # Log progress every 5 seconds

        while (time.time() - start_time) < duration:
            _ = np.sum(self._array)
            bytes_read += self._array.nbytes
            iter_count += 1

            now = time.time()
            if now >= next_log:
                elapsed_so_far = now - start_time
                gb_so_far = bytes_read / (1024**3)
                bw = gb_so_far / elapsed_so_far
                self.logger.info(
                    f"  Read progress: {elapsed_so_far:.0f}/{duration}s "
                    f"({bw:.2f} GB/s, {iter_count} iters)"
                )
                next_log = now + 5

        elapsed = time.time() - start_time
        gb_read = bytes_read / (1024**3)
        bandwidth_gbps = gb_read / elapsed

        self.logger.debug(
            f"Read: {iter_count} iterations, {gb_read:.2f} GB in {elapsed:.2f}s"
        )

        return bandwidth_gbps

    def _benchmark_sequential_write(self, duration: int) -> float:
        """Benchmark sequential write bandwidth.

        Args:
            duration: Test duration in seconds.

        Returns:
            Write bandwidth in GB/s.
        """
        import time

        assert self._array is not None, "Array not allocated in setup()"
        assert (
            self._source_array_for_writes is not None
        ), "Source array not allocated in setup()"

        start_time = time.time()
        bytes_written = 0
        iter_count = 0
        next_log = start_time + 5  # Log progress every 5 seconds

        while (time.time() - start_time) < duration:
            self._array[:] = self._source_array_for_writes
            bytes_written += self._array.nbytes
            iter_count += 1

            now = time.time()
            if now >= next_log:
                elapsed_so_far = now - start_time
                gb_so_far = bytes_written / (1024**3)
                bw = gb_so_far / elapsed_so_far
                self.logger.info(
                    f"  Write progress: {elapsed_so_far:.0f}/{duration}s "
                    f"({bw:.2f} GB/s, {iter_count} iters)"
                )
                next_log = now + 5

        elapsed = time.time() - start_time
        gb_written = bytes_written / (1024**3)
        bandwidth_gbps = gb_written / elapsed

        self.logger.debug(
            f"Write: {iter_count} iterations, {gb_written:.2f} GB in {elapsed:.2f}s"
        )

        return bandwidth_gbps
