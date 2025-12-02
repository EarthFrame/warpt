"""Abstract base class for storage backends.

Defines the interface that all storage backends must implement.

Storage types supported:
- Local block devices (SSDs, HDDs)
- Network storage (NFS, iSCSI, CIFS/SMB)
- Distributed/parallel filesystems (Lustre, GPFS, BeeGFS, Ceph)
- Object storage (S3-compatible, Azure Blob, GCS)

Platform support:
- Bare metal: Full hardware details (PCIe gen, SMART data, etc.)
- Cloud VMs (AWS EC2, GCP, Azure): Device enumeration works, but hardware
  details are limited due to virtualization. Use performance benchmarking
  to measure actual IOPS, throughput, and latency.
- Containers: Sees mounted devices/volumes

Design philosophy:
- Base StorageDeviceInfo: Fields common to ALL storage
- LocalBlockDeviceInfo: Adds physical hardware details (bus, health, SMART)
- NetworkStorageInfo: Adds network/mount details
- ObjectStorageInfo: Adds bucket/endpoint details

This separation ensures cloud VMs work correctly (they see block devices
but not physical PCIe buses).
"""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class StorageType(str, Enum):
    """High-level storage device type classification.

    Use this to answer questions like:
    - "Is this an SSD?" -> device_type == StorageType.SSD
    - "Is this network storage?" -> device_type in [NFS, CIFS, ISCSI, ...]

    For detailed interface info (NVMe vs SATA), use bus_type and protocol fields.
    For specific vendors/providers, use the manufacturer field.
    """

    # Local block devices
    SSD = "ssd"  # Solid-state drive
    HDD = "hdd"  # Hard disk drive (spinning)
    NVME_SSD = "nvme_ssd"  # Bare-metal NVMe SSD
    SATA_SSD = "sata_ssd"  # SATA/SAS SSD
    USB = "usb"  # USB-attached block device
    UNKNOWN_BLOCK = "unknown_block"  # Local block device with unknown type

    # Virtual/logical volumes
    LOGICAL_VOLUME = "logical_volume"  # LVM, RAID, ZFS, etc.

    # Network block storage
    ISCSI = "iscsi"  # iSCSI block device
    NVME_OF = "nvme_of"  # NVMe over Fabrics

    # Network file storage
    NFS = "nfs"  # Network File System
    CIFS = "cifs"  # CIFS/SMB

    # Distributed/parallel filesystems
    PARALLEL_FS = "parallel_fs"  # Lustre, GPFS, BeeGFS, etc.

    # Object storage
    S3 = "s3"  # S3-compatible object storage
    OBJECT_STORAGE = "object_storage"  # Other object storage

    # Unknown
    UNKNOWN = "unknown"


class BusType(str, Enum):
    """Bus/interconnect type for storage devices."""

    # Local buses
    PCIE = "pcie"  # PCIe
    SATA = "sata"  # SATA
    SAS = "sas"  # SAS
    USB = "usb"  # USB
    THUNDERBOLT = "thunderbolt"  # Thunderbolt

    # Network
    NETWORK = "network"  # Network-attached
    FIBRE_CHANNEL = "fibre_channel"  # Fibre Channel
    INFINIBAND = "infiniband"  # InfiniBand

    # Virtual
    VIRTUAL = "virtual"  # Virtual/logical

    # Unknown
    UNKNOWN = "unknown"


class StorageProtocol(str, Enum):
    """Storage access protocol."""

    # Block protocols
    NVME = "nvme"  # NVMe
    SCSI = "scsi"  # SCSI
    ATA = "ata"  # ATA/SATA

    # Network block
    ISCSI = "iscsi"  # iSCSI
    NVME_OF = "nvme_of"  # NVMe over Fabrics

    # File protocols
    NFS = "nfs"  # NFS
    CIFS = "cifs"  # CIFS/SMB

    # Object protocols
    S3 = "s3"  # S3 API
    HTTP = "http"  # HTTP/HTTPS

    # Unknown
    UNKNOWN = "unknown"


class LatencyMetrics(BaseModel):
    """Latency measurements in microseconds (μs) or milliseconds (ms)."""

    average_us: float | None = Field(
        None, description="Average latency in microseconds"
    )
    p50_us: float | None = Field(None, description="50th percentile (median) in μs")
    p95_us: float | None = Field(None, description="95th percentile in μs")
    p99_us: float | None = Field(None, description="99th percentile in μs")
    p999_us: float | None = Field(None, description="99.9th percentile in μs")
    min_us: float | None = Field(None, description="Minimum latency in μs")
    max_us: float | None = Field(None, description="Maximum latency in μs")

    class Config:
        """Pydantic config."""

        frozen = True


class IOPSMetrics(BaseModel):
    """IOPS (I/O operations per second) measurements."""

    # Read IOPS
    random_read_iops: int | None = Field(
        None, description="Random read IOPS (4K blocks typical)"
    )
    sequential_read_iops: int | None = Field(None, description="Sequential read IOPS")

    # Write IOPS
    random_write_iops: int | None = Field(
        None, description="Random write IOPS (4K blocks typical)"
    )
    sequential_write_iops: int | None = Field(None, description="Sequential write IOPS")

    # Mixed workloads
    mixed_iops: int | None = Field(
        None, description="Mixed read/write IOPS (70/30 typical)"
    )

    # Queue depth info
    queue_depth: int | None = Field(
        None, description="Queue depth used for measurement"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class ThroughputMetrics(BaseModel):
    """Throughput measurements in MB/s and GB/s."""

    # Sequential throughput (large blocks, typically 128K-1MB)
    sequential_read_mbps: float | None = Field(
        None, description="Sequential read throughput in MB/s"
    )
    sequential_write_mbps: float | None = Field(
        None, description="Sequential write throughput in MB/s"
    )

    # Random throughput (small blocks, typically 4K)
    random_read_mbps: float | None = Field(
        None, description="Random read throughput in MB/s"
    )
    random_write_mbps: float | None = Field(
        None, description="Random write throughput in MB/s"
    )

    # Convenience fields in GB/s for high-performance devices
    sequential_read_gbps: float | None = Field(
        None, description="Sequential read throughput in GB/s"
    )
    sequential_write_gbps: float | None = Field(
        None, description="Sequential write throughput in GB/s"
    )

    # Block size used for measurement
    block_size_kb: int | None = Field(
        None, description="Block size in KB used for measurement"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class PerformanceMetrics(BaseModel):
    """Comprehensive storage performance metrics.

    All fields are optional as they may require benchmarking or may not be
    available for certain storage types. Metrics can come from:
    - Device specifications (manufacturer ratings)
    - Live benchmarking (fio, dd, etc.)
    - Historical measurements
    - Vendor-provided data
    """

    # Throughput measurements
    throughput: ThroughputMetrics | None = Field(
        None, description="Throughput metrics (MB/s, GB/s)"
    )

    # IOPS measurements
    iops: IOPSMetrics | None = Field(None, description="IOPS metrics")

    # Latency measurements
    read_latency: LatencyMetrics | None = Field(
        None, description="Read latency distribution"
    )
    write_latency: LatencyMetrics | None = Field(
        None, description="Write latency distribution"
    )

    # Sustained performance (vs burst)
    sustained_write_mbps: float | None = Field(
        None, description="Sustained write performance (after cache exhaustion)"
    )

    # For network/object storage
    metadata_ops_per_sec: int | None = Field(
        None, description="Metadata operations per second (file create, stat, etc.)"
    )

    # Measurement methodology
    measured: bool = Field(False, description="True if measured, False if from specs")
    measurement_time: str | None = Field(
        None, description="ISO timestamp of measurement"
    )
    benchmark_tool: str | None = Field(
        None, description="Tool used for measurement (fio, dd, custom)"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class StorageDeviceInfo(BaseModel):
    """Base storage device information.

    This is the base class for all storage devices. Use specific subclasses
    for different storage types:
    - LocalBlockDeviceInfo: SSDs, HDDs with physical bus details
    - NetworkStorageInfo: NFS, CIFS, iSCSI mounts
    - ObjectStorageInfo: S3-compatible object storage

    This base class contains only fields common to ALL storage types.
    Works on bare metal, VMs, and cloud instances (AWS EC2, GCP, etc.).
    """

    # Identification
    device_path: str = Field(..., description="Device path (e.g., /dev/nvme0n1)")
    model: str = Field(..., description="Device model name")
    manufacturer: str | None = Field(
        None, description="Manufacturer (Samsung, AWS, etc.)"
    )

    # Classification
    device_type: StorageType = Field(..., description="Storage device type")
    protocol: StorageProtocol | None = Field(
        None, description="Access protocol (NVMe, SCSI, S3, NFS, etc.)"
    )

    # For partitions and logical volumes
    parent_device: str | None = Field(
        None, description="Parent device path (e.g., /dev/nvme0n1 for /dev/nvme0n1p1)"
    )
    partition_number: int | None = Field(
        None, description="Partition number if this is a partition"
    )

    # Capacity
    capacity_bytes: int = Field(..., description="Raw capacity in bytes")
    capacity_gb: int = Field(..., description="Capacity in GB (base 10)")
    capacity_gib: int | None = Field(
        None, description="Capacity in GiB (base 2, 1024^3)"
    )
    usable_bytes: int | None = Field(
        None, description="Usable capacity after formatting/provisioning"
    )

    # Performance metrics (measured or from specs)
    performance: PerformanceMetrics | None = Field(
        None, description="Detailed performance metrics"
    )

    # Mount/filesystem info
    mount_points: list[str] = Field(
        default_factory=list, description="Active mount points"
    )
    filesystem: str | None = Field(None, description="Filesystem type if mounted")
    is_system_disk: bool = Field(
        False, description="True if this is the boot/system disk"
    )
    is_read_only: bool = Field(False, description="True if device is read-only")

    # Platform-specific extras
    extra_info: dict | None = Field(
        None, description="Platform-specific additional info"
    )

    @property
    def is_ssd(self) -> bool:
        """Check if this is an SSD.

        Returns:
            True if device is solid-state
        """
        return self.device_type in (
            StorageType.SSD,
            StorageType.NVME_SSD,
            StorageType.SATA_SSD,
        )

    @property
    def is_hdd(self) -> bool:
        """Check if this is an HDD.

        Returns:
            True if device is spinning disk
        """
        return self.device_type == StorageType.HDD

    @property
    def is_network_storage(self) -> bool:
        """Check if this is network-attached storage.

        Returns:
            True if device is accessed over network
        """
        return self.device_type in (
            StorageType.NFS,
            StorageType.CIFS,
            StorageType.ISCSI,
            StorageType.NVME_OF,
            StorageType.PARALLEL_FS,
            StorageType.S3,
            StorageType.OBJECT_STORAGE,
        )

    @property
    def is_local_block(self) -> bool:
        """Check if this is local block storage.

        Returns:
            True if device is directly attached block storage
        """
        return self.device_type in (
            StorageType.SSD,
            StorageType.HDD,
            StorageType.NVME_SSD,
            StorageType.SATA_SSD,
            StorageType.USB,
            StorageType.UNKNOWN_BLOCK,
        )

    def is_partition(self) -> bool:
        """Check if this device is a partition.

        Returns:
            True if this is a partition of a larger device
        """
        return self.parent_device is not None

    def same_physical_disk(self, other: "StorageDeviceInfo") -> bool:
        """Check if this device is on the same physical disk as another.

        Args:
            other: Another StorageDeviceInfo to compare with

        Returns:
            True if both devices are on the same physical disk

        Examples:
            /dev/nvme0n1p1 and /dev/nvme0n1p2 -> True (same disk)
            /dev/sda1 and /dev/sda2 -> True (same disk)
            /dev/sda1 and /dev/sdb1 -> False (different disks)
        """
        # If both have parent devices, compare parents
        if self.parent_device and other.parent_device:
            return self.parent_device == other.parent_device

        # If one is parent of the other
        if self.parent_device == other.device_path:
            return True
        if other.parent_device == self.device_path:
            return True

        # If both are whole disks (no parent), compare device paths
        if not self.parent_device and not other.parent_device:
            return self.device_path == other.device_path

        return False

    class Config:
        """Pydantic config."""

        frozen = True


class LocalBlockDeviceInfo(StorageDeviceInfo):
    """Local block storage device (SSD, HDD).

    Adds physical hardware details like bus type, interface speed, and health.
    Works on:
    - Bare metal: Full hardware details available
    - Cloud VMs (EC2, GCP): Limited details (virtualized hardware)

    On cloud VMs, you'll see the device but not PCIe/SATA details since
    the storage is virtualized. Use performance benchmarking instead.
    """

    # Serial/firmware (more relevant for physical drives)
    serial: str | None = Field(None, description="Serial number")
    firmware: str | None = Field(None, description="Firmware version")

    # Bus/interface (only available on bare metal)
    bus_type: BusType = Field(
        BusType.UNKNOWN,
        description="Bus type (only on bare metal, UNKNOWN on cloud VMs)",
    )

    # PCIe details (only for NVMe on bare metal)
    pcie_generation: int | None = Field(
        None, description="PCIe gen (3,4,5) - bare metal NVMe only"
    )
    pcie_lanes: int | None = Field(
        None, description="PCIe lanes - bare metal NVMe only"
    )

    # SATA/SAS link speed
    link_speed_gbps: float | None = Field(
        None, description="Link speed (SATA: 6.0, SAS: 12.0, etc.) - bare metal only"
    )

    # Physical characteristics
    form_factor: str | None = Field(
        None, description='Form factor (M.2, 2.5", U.2, 3.5") - bare metal only'
    )
    rpm: int | None = Field(None, description="Spindle speed for HDDs")
    is_removable: bool = Field(False, description="True if device is removable")

    # Health/SMART data (may not be available on cloud VMs)
    health_status: str | None = Field(
        None, description="Health: healthy, warning, critical"
    )
    temperature_celsius: int | None = Field(None, description="Current temperature")
    wear_level_percent: int | None = Field(
        None, description="SSD wear (0-100, 100=end of life)"
    )
    power_on_hours: int | None = Field(None, description="Total power-on hours")
    total_bytes_written: int | None = Field(
        None, description="Lifetime bytes written (SSDs)"
    )
    total_bytes_read: int | None = Field(None, description="Lifetime bytes read (SSDs)")

    class Config:
        """Pydantic config."""

        frozen = True


class NetworkStorageInfo(StorageDeviceInfo):
    """Network-attached storage (NFS, CIFS, iSCSI).

    For network mounts and distributed filesystems.
    Works everywhere - bare metal, VMs, containers.
    """

    # Network details
    endpoint: str | None = Field(
        None, description="Network endpoint (hostname:port, IP address)"
    )
    mount_options: list[str] = Field(default_factory=list, description="Mount options")

    # Server info
    server_name: str | None = Field(None, description="NFS/CIFS server hostname")
    share_path: str | None = Field(None, description="Share path on server")

    # Protocol version
    protocol_version: str | None = Field(
        None, description="Protocol version (NFSv4, SMB3, etc.)"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class ObjectStorageInfo(StorageDeviceInfo):
    """Object storage (S3-compatible, Azure Blob, GCS).

    For cloud object storage accessed via API.
    Works everywhere - any system with network access.
    """

    # Endpoint and access
    endpoint: str | None = Field(None, description="API endpoint URL")
    provider: str | None = Field(
        None, description="Provider (AWS, Cloudflare R2, MinIO, Azure, GCS, etc.)"
    )
    region: str | None = Field(
        None, description="Cloud region (us-east-1, eu-west-1, etc.)"
    )

    # Bucket/container
    bucket_name: str | None = Field(None, description="Bucket or container name")
    storage_class: str | None = Field(
        None, description="Storage tier (STANDARD, GLACIER, INTELLIGENT_TIERING, etc.)"
    )

    # API details
    api_version: str | None = Field(None, description="API version")
    ssl_enabled: bool = Field(True, description="True if using HTTPS")

    class Config:
        """Pydantic config."""

        frozen = True


class BenchmarkConfig(BaseModel):
    """Configuration for storage benchmarking."""

    duration_seconds: int = Field(30, description="Duration of benchmark in seconds")
    block_size_kb: int = Field(128, description="Block size in KB for sequential tests")
    random_block_size_kb: int = Field(
        4, description="Block size in KB for random tests"
    )
    queue_depth: int = Field(32, description="I/O queue depth")
    num_jobs: int = Field(1, description="Number of parallel jobs")
    direct_io: bool = Field(True, description="Use direct I/O (bypass cache)")
    sequential: bool = Field(True, description="Run sequential tests")
    random: bool = Field(True, description="Run random tests")
    read_write_ratio: int = Field(
        70, description="Read percentage for mixed workload (0-100)"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Implementations should handle:
    - Graceful degradation when tools/permissions unavailable
    - Platform-specific detection (Linux, macOS, Windows, cloud VMs)
    - Both physical and virtual devices
    - Optional performance benchmarking

    Important for cloud environments (AWS EC2, GCP, Azure):
    - Block devices appear as /dev/nvme*, /dev/xvda, etc.
    - Hardware details (PCIe gen, bus speed) not available (virtualized)
    - Use performance benchmarking to measure actual throughput/IOPS/latency
    """

    @property
    @abstractmethod
    def storage_type(self) -> str:
        """Return the type of storage this backend handles.

        Examples: 'local', 'nfs', 's3', 'lustre'
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this storage type is accessible on the system.

        Returns:
            bool: True if at least one device of this type is detected
        """
        pass

    @abstractmethod
    def list_devices(self) -> list[StorageDeviceInfo]:
        """List all storage devices of this type.

        Returns appropriate subclass of StorageDeviceInfo:
        - LocalBlockDeviceInfo for SSDs/HDDs
        - NetworkStorageInfo for NFS/CIFS
        - ObjectStorageInfo for S3

        Returns:
            List of StorageDeviceInfo objects with available information.
            Returns empty list if no devices found (not an error).
        """
        pass

    @abstractmethod
    def get_device_info(self, device_path: str) -> StorageDeviceInfo | None:
        """Get detailed information for a specific device.

        Returns appropriate subclass of StorageDeviceInfo based on device type.

        Args:
            device_path: Device path or identifier

        Returns:
            StorageDeviceInfo (or subclass) if device exists, None otherwise
        """
        pass

    def benchmark_device(
        self, _device_path: str, _config: BenchmarkConfig | None = None
    ) -> PerformanceMetrics | None:
        """Run performance benchmark on a specific device.

        This is an optional method - not all backends need to implement it.
        Benchmarking requires appropriate tools (fio, dd) and permissions.

        Args:
            _device_path: Device path or identifier to benchmark
            _config: Benchmark configuration (uses defaults if None)

        Returns:
            PerformanceMetrics with measured results, or None if not supported

        Note:
            This may take significant time (_config.duration_seconds per test).
            For production systems, use with caution as it generates load.
        """
        return None  # Default: not implemented

    def get_total_capacity_gb(self) -> int:
        """Get total capacity across all devices of this type.

        Returns:
            Total capacity in GB
        """
        devices = self.list_devices()
        return sum(d.capacity_gb for d in devices)

    def shutdown(self) -> None:
        """Cleanup any resources.

        Override if backend needs cleanup (e.g., closing connections).
        Default implementation does nothing.
        """
        return None


# Type alias for any storage device info
AnyStorageDevice = (
    StorageDeviceInfo | LocalBlockDeviceInfo | NetworkStorageInfo | ObjectStorageInfo
)
