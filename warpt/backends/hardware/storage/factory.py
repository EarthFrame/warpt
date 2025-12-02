"""Storage backend factory for automatic type detection.

Provides unified access to all storage backends and auto-detection
of available storage types on the system.
"""

from warpt.backends.hardware.storage.base import StorageBackend, StorageDeviceInfo
from warpt.backends.hardware.storage.local import LocalStorageBackend


class StorageManager:
    """Unified manager for all storage backends.

    Handles initialization and aggregation of multiple storage backends.
    Use this class to get a complete view of all storage on a system.
    """

    def __init__(self) -> None:
        """Initialize storage manager with available backends."""
        self._backends: list[StorageBackend] = []
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize all available storage backends."""
        # Local storage (always try)
        try:
            local = LocalStorageBackend()
            if local.is_available():
                self._backends.append(local)
        except Exception:
            pass

        # Future backends will be added here:
        # - NFSBackend
        # - LustreBackend
        # - S3Backend
        # etc.

    def list_all_devices(self) -> list[StorageDeviceInfo]:
        """List all storage devices across all backends.

        Returns:
            List of all detected storage devices
        """
        all_devices = []
        for backend in self._backends:
            try:
                devices = backend.list_devices()
                all_devices.extend(devices)
            except Exception:
                # Skip backends that fail, continue with others
                pass
        return all_devices

    def list_local_devices(self) -> list[StorageDeviceInfo]:
        """List only local block devices.

        Returns:
            List of local storage devices (NVMe, SATA, HDD)
        """
        for backend in self._backends:
            if backend.storage_type == "local":
                return backend.list_devices()
        return []

    def get_backend(self, storage_type: str) -> StorageBackend | None:
        """Get a specific backend by type.

        Args:
            storage_type: Backend type ('local', 'nfs', 's3', etc.)

        Returns:
            StorageBackend if available, None otherwise
        """
        for backend in self._backends:
            if backend.storage_type == storage_type:
                return backend
        return None

    def get_total_capacity_gb(self) -> int:
        """Get total capacity across all storage.

        Returns:
            Total capacity in GB
        """
        return sum(d.capacity_gb for d in self.list_all_devices())

    def shutdown(self) -> None:
        """Cleanup all backends."""
        for backend in self._backends:
            try:
                backend.shutdown()
            except Exception:
                pass


def get_storage_backend(storage_type: str = "local") -> StorageBackend:
    """Get a specific storage backend by type.

    Args:
        storage_type: Type of storage backend ('local', 'nfs', 's3', etc.)

    Returns:
        StorageBackend instance

    Raises:
        RuntimeError: If requested backend is not available
    """
    if storage_type == "local":
        backend = LocalStorageBackend()
        if backend.is_available():
            return backend
        raise RuntimeError("Local storage detection not available on this platform")

    # Future backends:
    # elif storage_type == "nfs":
    #     return NFSBackend()
    # elif storage_type == "s3":
    #     return S3Backend()

    raise RuntimeError(f"Unknown storage backend type: {storage_type}")


def get_storage_manager() -> StorageManager:
    """Get a storage manager with all available backends.

    Returns:
        StorageManager instance
    """
    return StorageManager()
