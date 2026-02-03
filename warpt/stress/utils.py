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
