"""GPU backend factory for automatic vendor detection.

This factory auto-detects which GPU vendor is present on the system and
returns the appropriate backend implementation.

Priority order: NVIDIA → AMD → Intel
"""

from warpt.backends.base import GPUBackend


def get_gpu_backend() -> GPUBackend:
    """Auto-detect GPU vendor and return appropriate backend.

    Returns:
        GPUBackend: The appropriate backend for detected GPU vendor

    Raises:
        RuntimeError: If no GPUs are detected on the system
    """
    backend: GPUBackend

    try:
        from warpt.backends.nvidia import NvidiaBackend

        backend = NvidiaBackend()
        if backend.is_available():
            return backend
    except Exception:
        # pass
        raise RuntimeError(
            "Failed to detect NVIDIA GPUs on this system. "
            "AMD/Intel GPUs not currently supported."
        ) from None

    # TODO: Add support for AMD and Intel
    try:
        from warpt.backends.amd import AMDBackend

        backend = AMDBackend()
        if backend.is_available():
            return backend
    except Exception:
        pass

    try:
        from warpt.backends.intel import IntelBackend

        backend = IntelBackend()
        if backend.is_available():
            return backend
    except Exception:
        pass

    # No GPUs detected
    raise RuntimeError(
        "No GPUs detected on this system. "
        "Please ensure GPU drivers are installed and GPUs are properly configured."
    )
