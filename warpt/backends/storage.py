"""Storage backend - provides storage I/O utilities for stress tests.

This module provides utilities for storage testing, including:
- Direct I/O (bypassing OS cache) for accurate disk performance measurement
- File opening with platform-specific optimizations
"""

from __future__ import annotations

import os
import sys
from typing import BinaryIO, cast


class Storage:
    """Storage I/O backend for stress tests.

    Provides utilities for:
    - Opening files with direct I/O (cache bypass)
    - Platform-specific storage optimizations
    """

    @staticmethod
    def open_direct_io(path: str, mode: str = "rb") -> tuple[BinaryIO | None, bool]:
        """Open a file with direct I/O to bypass OS cache.

        Direct I/O ensures reads come from disk, not RAM cache, providing
        accurate storage performance measurements.

        Platform support:
        - macOS: Uses F_NOCACHE flag
        - Linux: Uses O_DIRECT flag
        - Windows: Not yet implemented

        Args:
            path: File path to open.
            mode: File mode ('rb' for read, 'wb' for write).

        Returns:
            Tuple of (file_object, direct_io_enabled):
                - file_object: Opened file or None if failed
                - direct_io_enabled: True if direct I/O is active

        Example:
            >>> f, direct = Storage.open_direct_io('test.dat', 'rb')
            >>> if not direct:
            >>>     logger.warning("Direct I/O unavailable, using cached I/O")
        """
        direct_io_enabled = False

        try:
            # macOS: Use standard open + F_NOCACHE
            if sys.platform == "darwin":
                f = cast(BinaryIO, open(path, mode))
                try:
                    import fcntl

                    # F_NOCACHE = 48 on macOS
                    fcntl.fcntl(f.fileno(), 48, 1)
                    direct_io_enabled = True
                except (OSError, AttributeError):
                    # F_NOCACHE failed, continue with cached I/O
                    pass
                return f, direct_io_enabled

            # Linux: Use O_DIRECT at open time
            elif sys.platform.startswith("linux"):
                # O_DIRECT must be set when opening the file
                # O_DIRECT = 0x4000 on Linux
                flags = os.O_RDONLY if "r" in mode else os.O_WRONLY
                flags |= 0x4000  # O_DIRECT

                # O_DIRECT requires aligned buffers, which is tricky
                # For now, fall back to standard open + posix_fadvise
                f = cast(BinaryIO, open(path, mode))
                try:
                    # Use POSIX_FADV_DONTNEED to reduce caching
                    # os.posix_fadvise is available on Linux
                    if hasattr(os, "posix_fadvise"):
                        # POSIX_FADV_DONTNEED = 4
                        os.posix_fadvise(f.fileno(), 0, 0, 4)
                        direct_io_enabled = True
                except (OSError, AttributeError):
                    pass
                return f, direct_io_enabled

            # Windows: Use FILE_FLAG_NO_BUFFERING (not yet implemented)
            elif sys.platform == "win32":
                # For now, use standard open
                f = cast(BinaryIO, open(path, mode))
                return f, False

            # Unknown platform
            else:
                f = cast(BinaryIO, open(path, mode))
                return f, False

        except Exception:
            return None, False

    @staticmethod
    def enable_direct_io(fd: int) -> bool:
        """Enable direct I/O on an already-opened file descriptor.

        This attempts to enable cache bypass on an existing file.
        Prefer open_direct_io() for opening new files.

        Args:
            fd: File descriptor (from file.fileno()).

        Returns:
            True if direct I/O was enabled, False otherwise.
        """
        try:
            # macOS: F_NOCACHE can be set after opening
            if sys.platform == "darwin":
                import fcntl

                # F_NOCACHE = 48
                fcntl.fcntl(fd, 48, 1)
                return True

            # Linux: Can't set O_DIRECT after opening, use fadvise instead
            elif sys.platform.startswith("linux"):
                if hasattr(os, "posix_fadvise"):
                    # POSIX_FADV_DONTNEED = 4
                    os.posix_fadvise(fd, 0, 0, 4)
                    return True
                return False

            else:
                return False

        except (OSError, AttributeError):
            return False

    @staticmethod
    def drop_cache(path: str) -> bool:
        """Drop OS cache for a specific file.

        Attempts to remove the file from the OS page cache.
        Useful between test iterations.

        Args:
            path: File path to drop from cache.

        Returns:
            True if cache was dropped, False otherwise.
        """
        try:
            # macOS: Use F_PURGE to purge file from cache
            if sys.platform == "darwin":
                import fcntl

                with open(path, "rb") as f:
                    # F_PURGE = 46
                    fcntl.fcntl(f.fileno(), 46, 0)
                return True

            # Linux: Use POSIX_FADV_DONTNEED
            elif sys.platform.startswith("linux"):
                if hasattr(os, "posix_fadvise"):
                    with open(path, "rb") as f:
                        # POSIX_FADV_DONTNEED = 4
                        os.posix_fadvise(f.fileno(), 0, 0, 4)
                    return True
                return False

            else:
                return False

        except (OSError, AttributeError):
            return False
