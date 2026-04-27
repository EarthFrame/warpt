"""Utility functions for stress tests."""


def calculate_tflops(operations: int, time_seconds: float) -> float:
    """Calculate TFLOPS (trillion floating point operations per second).

    Args:
        operations: Total number of floating point operations
        time_seconds: Time taken in seconds

    Returns
    -------
        Performance in TFLOPS

    Raises
    ------
        ValueError: If time_seconds is zero or negative
    """
    if time_seconds <= 0:
        raise ValueError(f"time_seconds must be positive, got {time_seconds}")

    return (operations / time_seconds) / 1e12


def measure_loop(
    duration: int,
    work_fn,
    sync_fn=None,
    device=None,
) -> tuple[float, int]:
    """Run a timed measurement loop with optional CUDA event timing.

    Primary path: CUDA events for GPU-side timing (when device.type == "cuda").
    Fallback: time.time() + sync_fn for vendor-agnostic backends.

    Args:
        duration: Target duration in seconds.
        work_fn: Zero-arg callable to execute each iteration
            (e.g., lambda: torch.matmul(a, b)).
        sync_fn: Optional callable to synchronize the device after the loop.
            Used in fallback path only (e.g., torch.cuda.synchronize).
            None = no synchronization.
        device: torch.device (or similar). If device.type == "cuda",
            uses CUDA events. Otherwise falls back to time.time().

    Returns:
        Tuple of (elapsed_seconds, iterations).
    """
    import time

    use_cuda_events = (
        device is not None and hasattr(device, "type") and device.type == "cuda"
    )

    if use_cuda_events:
        import torch

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        iterations = 0
        start_event.record()
        loop_start = time.time()

        while (time.time() - loop_start) < duration:
            work_fn()
            iterations += 1
            torch.cuda.synchronize()  # keep loop in lockstep with GPU

        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event) / 1000.0
        return elapsed, iterations

    # Fallback: time.time() + per-iteration sync
    iterations = 0
    start_time = time.time()

    while (time.time() - start_time) < duration:
        work_fn()
        if sync_fn is not None:
            sync_fn()
        iterations += 1

    elapsed = time.time() - start_time
    return elapsed, iterations


def format_results(results: dict) -> str:
    """Format test results for console display.

    Args:
        results: Dictionary containing test results

    Returns
    -------
        Formatted string for display
    """
    # TODO: Implement result formatting
    _ = results  # Suppress unused argument warning
    return ""
