"""Tests for GPU mismatch warnings in list command."""

import os
from unittest.mock import MagicMock, patch

from warpt.backends.pci import PCIDevice
from warpt.commands.list_cmd import run_list


@patch("warpt.commands.list_cmd.get_gpu_backend")
@patch("warpt.commands.list_cmd.PCIBackend")
@patch("warpt.commands.list_cmd.CPU")
@patch("warpt.commands.list_cmd.RAM")
@patch("warpt.commands.list_cmd._collect_storage_devices")
@patch("warpt.commands.list_cmd.detect_all_frameworks")
@patch("warpt.commands.list_cmd.detect_all_libraries")
@patch("warpt.commands.list_cmd.NvidiaContainerToolkitDetector")
@patch("warpt.commands.list_cmd.DockerDetector")
def test_run_list_gpu_warning(
    mock_docker,
    mock_toolkit,
    mock_libs,
    mock_frameworks,
    mock_storage,
    mock_ram,
    mock_cpu,
    mock_pci_cls,
    mock_gpu_backend_factory,
    capsys,
):
    """Test that run_list warns when PCI GPUs are found but driver GPUs are not."""
    # Mock CPU
    mock_cpu_instance = mock_cpu.return_value
    mock_cpu_info = MagicMock()
    mock_cpu_info.make = "Apple"
    mock_cpu_info.model = "M2"
    mock_cpu_info.architecture = "arm64"
    mock_cpu_info.total_sockets = 1
    mock_cpu_info.total_physical_cores = 8
    mock_cpu_info.total_logical_cores = 8
    mock_cpu_info.base_frequency = 600
    mock_cpu_info.boost_frequency_single_core = 3504
    mock_cpu_info.boost_frequency_multi_core = None
    mock_cpu_info.current_frequency = 3504
    mock_cpu_info.current_frequency_scope = MagicMock(value="system")
    mock_cpu_info.socket_info = None
    mock_cpu_instance.get_cpu_info.return_value = mock_cpu_info

    # Mock RAM
    mock_ram_instance = mock_ram.return_value
    mock_ram_instance.get_ram_info.return_value = MagicMock(
        total=16 * 1024**3, free=8 * 1024**3
    )
    mock_ram_instance._detect_ddr_info.return_value = ("LPDDR5", 6400)
    mock_ram_instance._detect_memory_channels.return_value = 4

    # Mock Storage
    mock_storage.return_value = []

    # Mock Frameworks/Libs
    mock_frameworks.return_value = {}
    mock_libs.return_value = {}

    # Mock Docker/Toolkit
    mock_docker.return_value.detect.return_value = None

    toolkit_result = MagicMock()
    toolkit_result.installed = False
    toolkit_result.cli_version = None
    toolkit_result.cli_path = None
    toolkit_result.runtime_path = None
    toolkit_result.docker_runtime_ready = None
    mock_toolkit.return_value.detect.return_value = toolkit_result

    # --- THE CORE OF THE TEST ---

    # 1. Mock GPU Backend to return NO GPUs
    mock_backend = MagicMock()
    mock_backend.list_devices.return_value = []
    mock_gpu_backend_factory.return_value = mock_backend

    # 2. Mock PCI Backend to return ONE NVIDIA GPU
    mock_pci_instance = mock_pci_cls.return_value
    mock_pci_instance.is_available.return_value = True
    mock_pci_instance.get_gpus.return_value = [
        PCIDevice(
            slot="01:00.0",
            vendor_id="10de",
            device_id="2684",
            class_name="VGA compatible controller",
            vendor_name="NVIDIA",
            device_name="GeForce RTX 4090",
        )
    ]

    # Run the command
    run_list(export_filename="test_output.json")

    captured = capsys.readouterr()

    # Check for the warning
    assert "WARNING: NVIDIA GeForce RTX 4090 detected on PCI bus" in captured.out
    assert "but not accessible via drivers" in captured.out

    # Clean up the json file if it was created
    if os.path.exists("test_output.json"):
        os.remove("test_output.json")


@patch("warpt.commands.list_cmd.get_gpu_backend")
@patch("warpt.commands.list_cmd.PCIBackend")
@patch("warpt.commands.list_cmd.CPU")
@patch("warpt.commands.list_cmd.RAM")
@patch("warpt.commands.list_cmd._collect_storage_devices")
@patch("warpt.commands.list_cmd.detect_all_frameworks")
@patch("warpt.commands.list_cmd.detect_all_libraries")
@patch("warpt.commands.list_cmd.NvidiaContainerToolkitDetector")
@patch("warpt.commands.list_cmd.DockerDetector")
def test_run_list_no_gpu_warning_when_matched(
    mock_docker,
    mock_toolkit,
    mock_libs,
    mock_frameworks,
    mock_storage,
    mock_ram,
    mock_cpu,
    mock_pci_cls,
    mock_gpu_backend_factory,
    capsys,
):
    """Test that run_list DOES NOT warn when PCI GPUs and driver GPUs match."""
    # Mock CPU
    mock_cpu_instance = mock_cpu.return_value
    mock_cpu_info = MagicMock()
    mock_cpu_info.make = "Apple"
    mock_cpu_info.model = "M2"
    mock_cpu_info.architecture = "arm64"
    mock_cpu_info.total_sockets = 1
    mock_cpu_info.total_physical_cores = 8
    mock_cpu_info.total_logical_cores = 8
    mock_cpu_info.base_frequency = 600
    mock_cpu_info.boost_frequency_single_core = 3504
    mock_cpu_info.boost_frequency_multi_core = None
    mock_cpu_info.current_frequency = 3504
    mock_cpu_info.current_frequency_scope = MagicMock(value="system")
    mock_cpu_info.socket_info = None
    mock_cpu_instance.get_cpu_info.return_value = mock_cpu_info

    # Mock RAM
    mock_ram_instance = mock_ram.return_value
    mock_ram_instance.get_ram_info.return_value = MagicMock(
        total=16 * 1024**3, free=8 * 1024**3
    )
    mock_ram_instance._detect_ddr_info.return_value = ("LPDDR5", 6400)
    mock_ram_instance._detect_memory_channels.return_value = 4

    # Mock Storage/Frameworks/Libs
    mock_storage.return_value = []
    mock_frameworks.return_value = {}
    mock_libs.return_value = {}

    # Mock Docker/Toolkit
    mock_docker.return_value.detect.return_value = None
    toolkit_result = MagicMock()
    toolkit_result.installed = False
    toolkit_result.cli_version = None
    toolkit_result.cli_path = None
    toolkit_result.runtime_path = None
    toolkit_result.docker_runtime_ready = None
    mock_toolkit.return_value.detect.return_value = toolkit_result

    # Mock GPU Backend to return ONE GPU
    from warpt.models.list_models import GPUInfo

    gpu_info = GPUInfo(
        index=0,
        model="NVIDIA GeForce RTX 4090",
        memory_gb=24,
        compute_capability="8.9",
        pcie_gen=4,
        driver_version="535.104.05",
    )

    mock_backend = MagicMock()
    mock_backend.list_devices.return_value = [gpu_info]
    mock_backend.get_cuda_driver_version.return_value = "12.2"
    mock_gpu_backend_factory.return_value = mock_backend

    # Mock PCI Backend to return ONE NVIDIA GPU (same count)
    mock_pci_instance = mock_pci_cls.return_value
    mock_pci_instance.is_available.return_value = True
    mock_pci_instance.get_gpus.return_value = [
        PCIDevice(
            slot="01:00.0",
            vendor_id="10de",
            device_id="2684",
            class_name="VGA compatible controller",
            vendor_name="NVIDIA",
            device_name="GeForce RTX 4090",
        )
    ]

    run_list(export_filename="test_output_matched.json")

    captured = capsys.readouterr()

    # Should NOT have the warning
    assert "WARNING" not in captured.out

    if os.path.exists("test_output_matched.json"):
        os.remove("test_output_matched.json")
