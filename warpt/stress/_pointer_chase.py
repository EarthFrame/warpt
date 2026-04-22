"""C++ pointer-chase backend with Python fallback.

Imports a pre-compiled C extension (built at ``pip install`` time) for
nanosecond-accurate pointer-chase latency measurement.  Falls back to a
pure-Python implementation when the extension was not compiled.

The ``_`` prefix keeps the registry scanner from treating this as
a StressTest module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_cached_fn: Callable[[NDArray[np.int64], int], int] | None = None
_fn_loaded: bool = False


def get_pointer_chase() -> Callable[[NDArray[np.int64], int], int] | None:
    """Return the C++ pointer-chase function, or ``None`` if unavailable.

    The returned callable has signature ``(arr: NDArray[int64], iters: int) -> int``.
    """
    global _cached_fn, _fn_loaded

    if _fn_loaded:
        return _cached_fn

    _fn_loaded = True

    try:
        from warpt.stress._pointer_chase_ext import pointer_chase as _cpp_chase

        def _chase(arr: NDArray[np.int64], iters: int) -> int:
            return int(_cpp_chase(arr.ctypes.data, iters))

        _cached_fn = _chase
    except ImportError:
        logger.debug("C++ extension not available; will use Python fallback")
        _cached_fn = None

    return _cached_fn


def pointer_chase_python(arr: NDArray[np.int64], iters: int) -> int:
    """Pure-Python fallback pointer chase (much slower than C++)."""
    idx = 0
    for _ in range(iters):
        idx = int(arr[idx])
    return idx


def reset_cache() -> None:
    """Reset the cached function state (for testing)."""
    global _cached_fn, _fn_loaded
    _cached_fn = None
    _fn_loaded = False
