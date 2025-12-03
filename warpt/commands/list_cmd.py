"""List command - displays CPU and GPU information."""

import random
import string
from datetime import datetime
from pathlib import Path

import pynvml

from warpt.backends.factory import get_gpu_backend
from warpt.backends.hardware.storage.base import (
    BusType,
    StorageDeviceInfo,
    StorageType,
)
from warpt.backends.hardware.storage.factory import get_storage_manager
from warpt.backends.ram import RAM
from warpt.backends.software import DockerDetector, NvidiaContainerToolkitDetector
from warpt.backends.system import CPU
from warpt.models.list_models import (
    CPUInfo as CPUInfoModel,
)
from warpt.models.list_models import (
    CUDAInfo,
    DockerInfo,
    HardwareInfo,
    ListOutput,
    MemoryInfo,
    NvidiaContainerToolkitInfo,
    SoftwareInfo,
)
from warpt.models.list_models import (
    StorageDevice as StorageDeviceModel,
)


def random_string(length: int) -> str:
    """Generate random uppercase string for unique filenames."""
    chars = string.ascii_uppercase
    return "".join(random.choice(chars) for _ in range(length))


_STORAGE_TYPE_LABELS: dict[StorageType, str] = {
    StorageType.NVME_SSD: "NVMe SSD",
    StorageType.SATA_SSD: "SATA SSD",
    StorageType.SSD: "SSD",
    StorageType.HDD: "HDD",
    StorageType.USB: "USB storage",
    StorageType.UNKNOWN_BLOCK: "Unknown block device",
}


def _storage_type_label(device: StorageDeviceInfo) -> str:
    """Return a human-readable label for a storage device type.

    Args:
        device: StorageDeviceInfo describing the device.

    Returns:
        Human-readable label for the detected storage type.
    """
    label = _STORAGE_TYPE_LABELS.get(device.device_type)
    if label:
        return label
    return device.device_type.value.replace("_", " ").title()


def _storage_bus_type(device: StorageDeviceInfo) -> str | None:
    """Render the bus type for a detected device."""
    bus_value = getattr(device, "bus_type", None)
    if isinstance(bus_value, BusType):
        return bus_value.value
    if isinstance(bus_value, str):
        return bus_value
    return None


def _storage_link_speed(device: StorageDeviceInfo) -> float | None:
    """Return the reported link speed for the provided device."""
    link_speed = getattr(device, "link_speed_gbps", None)
    if isinstance(link_speed, int | float):
        return float(link_speed)
    return None


def _collect_storage_devices() -> list[StorageDeviceModel]:
    """Collect local storage devices for list command reporting.

    Returns:
        List of StorageDeviceModel items derived from detected local disks.
    """
    try:
        manager = get_storage_manager()
    except RuntimeError:
        return []

    devices: list[StorageDeviceModel] = []
    try:
        local_devices = manager.list_local_devices()
    except Exception:
        return []

    for device in local_devices:
        devices.append(
            StorageDeviceModel(
                device_path=device.device_path,
                capacity_gb=device.capacity_gb,
                type=_storage_type_label(device),
                model=device.model,
                manufacturer=device.manufacturer,
                serial=getattr(device, "serial", None),
                bus_type=_storage_bus_type(device),
                link_speed_gbps=_storage_link_speed(device),
            )
        )
    return devices


def run_list(export_format=None, export_filename=None) -> None:
    """Display comprehensive CPU and GPU information."""
    if export_format:
        raise NotImplementedError("Export format not implemented")

    cpu = CPU()
    info = cpu.get_cpu_info()

    print("CPU Information:")
    print(f"  Make:               {info.make}")
    print(f"  Model:              {info.model}")
    print(f"  Architecture:       {info.architecture}")

    print("\nTopology:")
    print(f"  Total Sockets:      {info.total_sockets}")
    print(f"  Total Phys Cores:   {info.total_physical_cores}")
    print(f"  Total Logic Cores:  {info.total_logical_cores}")

    print("\nFrequencies:")
    if info.base_frequency is not None:
        print(f"  Base Frequency:     {info.base_frequency:.0f} MHz")

    if info.boost_frequency_single_core is not None:
        print(f"  Single-Core Boost:  {info.boost_frequency_single_core:.0f} MHz")

    if info.boost_frequency_multi_core is not None:
        boost = info.boost_frequency_multi_core
        print(f"  Multi-Core Boost:   {boost:.0f} MHz")
    else:
        # If only single-core boost is available, show it as the main boost
        if info.boost_frequency_single_core is not None:
            print(f"  Boost Frequency:    {info.boost_frequency_single_core:.0f} MHz")

    if info.current_frequency is not None:
        freq_str = f"{info.current_frequency:.0f} MHz"
        if info.current_frequency_scope:
            freq_str += f" ({info.current_frequency_scope.value})"
        print(f"  Current Frequency:  {freq_str}")

    # Show detailed socket information if available
    if info.socket_info:
        print("\nPer-Socket Details:")
        for socket in info.socket_info:
            print(f"\n  Socket {socket.socket_id}:")
            print(f"    Make/Model:       {socket.make} {socket.model}")
            print(f"    Phys Cores:       {socket.physical_cores}")
            print(f"    Logic Cores:      {socket.logical_cores}")

            if socket.base_frequency is not None:
                print(f"    Base Freq:        {socket.base_frequency:.0f} MHz")

            if socket.boost_frequency_single_core is not None:
                boost_single = socket.boost_frequency_single_core
                print(f"    Single-Core Boost: {boost_single:.0f} MHz")

            if socket.boost_frequency_multi_core is not None:
                boost_multi = socket.boost_frequency_multi_core
                print(f"    Multi-Core Boost: {boost_multi:.0f} MHz")

    # GPU Detection
    print("\nGPU Information:")
    try:
        backend = get_gpu_backend()
        gpus = backend.list_devices()

        # backend.list_devices() returns empty list when no GPUs are present
        if not gpus:
            print("  No GPUs detected")
        else:
            for gpu in gpus:
                print(f"  [{gpu.index}] {gpu.model}")
                print(f"      Memory:         {gpu.memory_gb} GB")
                print(f"      CUDA Compute:   {gpu.compute_capability}")
                if gpu.pcie_gen:
                    print(f"      PCIe Gen:       {gpu.pcie_gen}")
                if gpu.driver_version:
                    print(f"      Driver Version: {gpu.driver_version}")

        gpu_list = gpus  # Save for JSON export (empty list if no GPUs)

    except ImportError:
        print("  GPU detection unavailable (nvidia-ml-py not installed)")
        gpu_list = None
    except Exception as e:
        print(f"  GPU detection failed: {e}")
        gpu_list = None

    # CUDA Detection
    # TODO: Add CUDA toolkit version detection (nvcc) - for now just using
    # driver version
    print("\nCUDA Information:")
    cuda_driver_version = None

    # Get CUDA driver version from pynvml (if GPUs available)
    if gpu_list:
        try:
            # returns version like 12010 for CUDA 12.1
            driver_version_int = pynvml.nvmlSystemGetCudaDriverVersion()
            major = driver_version_int // 1000
            minor = (driver_version_int % 1000) // 10
            cuda_driver_version = f"{major}.{minor}"
            print(f"  Driver Version: {cuda_driver_version}")
        except Exception as e:
            print(f"  CUDA detection failed: {e}")
    else:
        print("  No CUDA information (no GPUs detected)")

    cpu_model = CPUInfoModel(
        manufacturer=info.make,
        model=info.model,
        architecture=info.architecture,
        cores=info.total_physical_cores,
        threads=info.total_logical_cores,
        base_frequency_mhz=info.base_frequency,
        boost_frequency_single_core_mhz=info.boost_frequency_single_core,
        boost_frequency_multi_core_mhz=info.boost_frequency_multi_core,
        current_frequency_mhz=info.current_frequency,
        instruction_sets=None,
    )

    gpu_models = None
    gpu_count = None
    if gpu_list:
        gpu_models = gpu_list  # list[GPUInfo]
        gpu_count = len(gpu_list)

    cuda_info = None
    if cuda_driver_version:
        cuda_info = CUDAInfo(version=cuda_driver_version, driver=cuda_driver_version)

    # NVIDIA Container Toolkit Detection
    print("\nNVIDIA Container Toolkit:")
    toolkit_detector = NvidiaContainerToolkitDetector()
    toolkit_result = toolkit_detector.detect()
    toolkit_info = NvidiaContainerToolkitInfo(
        installed=toolkit_result.installed if toolkit_result else False,
        cli_version=toolkit_result.cli_version if toolkit_result else None,
        cli_path=toolkit_result.cli_path if toolkit_result else None,
        runtime_path=toolkit_result.runtime_path if toolkit_result else None,
        docker_runtime_ready=(
            toolkit_result.docker_runtime_ready if toolkit_result else None
        ),
    )

    if toolkit_info.installed:
        print("  Installed:         Yes")
        if toolkit_info.cli_version:
            print(f"  CLI Version:       {toolkit_info.cli_version}")
        if toolkit_info.cli_path:
            print(f"  CLI Path:          {toolkit_info.cli_path}")
        if toolkit_info.runtime_path:
            print(f"  Runtime Path:      {toolkit_info.runtime_path}")
        if toolkit_info.docker_runtime_ready is not None:
            docker_state = "Yes" if toolkit_info.docker_runtime_ready else "No"
            print(f"  Docker Runtime:    {docker_state}")
    else:
        print("  Installed:         No")
    # Docker Detection
    print("\nDocker Information:")
    docker_detector = DockerDetector()
    docker_result = docker_detector.detect()
    docker_info = DockerInfo(
        installed=docker_result is not None,
        version=docker_result.version if docker_result else None,
        path=docker_result.path if docker_result else None,
    )
    if docker_info.installed:
        print("  Installed:         Yes")
        if docker_info.version:
            print(f"  Version:           {docker_info.version}")
        if docker_info.path:
            print(f"  Path:              {docker_info.path}")
    else:
        print("  Docker CLI not found")

    # RAM Detection
    print("\nMemory Information:")
    ram_backend = RAM()
    ram_info = ram_backend.get_ram_info()
    ddr_type, speed_mhz = ram_backend._detect_ddr_info()
    channels = ram_backend._detect_memory_channels()

    total_gb = ram_info.total / (1024**3)
    free_gb = ram_info.free / (1024**3)

    print(f"  Total:              {total_gb:.1f} GB")
    print(f"  Free:               {free_gb:.1f} GB")
    if ddr_type:
        print(f"  Type:               {ddr_type}")
    if speed_mhz:
        print(f"  Speed:              {speed_mhz} MHz")
    if channels:
        print(f"  Channels:           {channels}")

    memory_info = MemoryInfo(
        total_gb=int(total_gb),
        free_gb=free_gb,
        type=ddr_type,
        speed_mhz=speed_mhz,
        channels=channels,
    )

    storage_devices = _collect_storage_devices()
    if storage_devices:
        print("\nStorage:")
        for storage in storage_devices:
            print(f"  {storage.device_path}: {storage.capacity_gb} GB ({storage.type})")
            if storage.manufacturer or storage.model:
                name = storage.model or ""
                if storage.manufacturer:
                    name = f"{storage.manufacturer} {name}".strip()
                print(f"      Model:             {name}")
            if storage.serial:
                print(f"      Serial:            {storage.serial}")
            if storage.bus_type or storage.link_speed_gbps:
                interface = []
                if storage.bus_type:
                    interface.append(storage.bus_type.upper())
                if storage.link_speed_gbps is not None:
                    interface.append(f"{storage.link_speed_gbps:.1f} Gbps")
                print(f"      Interface:         {' @ '.join(interface)}")
    else:
        print("\nStorage: No local block devices detected")

    # Build software info
    software = SoftwareInfo(
        python=None,
        cuda=cuda_info,
        frameworks=None,
        compilers=None,
        nvidia_container_toolkit=toolkit_info,
        docker=docker_info,
    )

    # Build output with CPU data
    hardware = HardwareInfo(
        cpu=cpu_model,
        gpu_count=gpu_count,
        gpu=gpu_models,
        memory=memory_info,
        storage=storage_devices or None,
    )
    output = ListOutput(hardware=hardware, software=software)

    # Generate filename if not provided
    if not export_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_tag = random_string(6)
        export_filename = f"warpt_list_{timestamp}_{random_tag}.json"

    # Write JSON file using Pydantic's built-in serialization
    output_path = Path(export_filename)
    output_path.write_text(output.model_dump_json(indent=2))

    print(f"\n✓ JSON exported to: {export_filename}")
