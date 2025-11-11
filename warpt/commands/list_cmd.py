"""List command - displays CPU and GPU information."""

import random
import string
import subprocess
from datetime import datetime
from pathlib import Path
import pynvml

from warpt.backends.nvidia import NvidiaBackend
from warpt.backends.system import CPU, System
from warpt.models.list_models import CUDAInfo, GPUInfo, HardwareInfo, ListOutput, SoftwareInfo


def random_string(length: int) -> str:
    """Generate random uppercase string for unique filenames."""
    chars = string.ascii_uppercase
    return "".join(random.choice(chars) for _ in range(length))


def run_list(export_format=None, export_filename=None) -> None:
    """Display comprehensive CPU and GPU information."""
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
        backend = NvidiaBackend()
        gpus = backend.list_devices()

        # backend.list_devices() returns empty list when no GPUs are present
        if not gpus:
            print("  No GPUs detected")
        else:
            for gpu in gpus:
                print(f"  [{gpu['index']}] {gpu['model']}")
                print(f"      Memory:         {gpu['memory_gb']} GB")
                print(f"      CUDA Compute:   {gpu['compute_capability']}")
                if gpu.get('pcie_gen'):
                    print(f"      PCIe Gen:       {gpu['pcie_gen']}")
                if gpu.get('driver_version'):
                    print(f"      Driver Version: {gpu['driver_version']}")

        gpu_list = gpus  # Save for JSON export (empty list of no gpus)

    except ImportError:
        print("  GPU detection unavailable (nvidia-ml-py not installed)")
        gpu_list = None
    except Exception as e:
        print(f"  GPU detection failed: {e}")
        gpu_list = None

    # CUDA Detection
    # TODO: Add CUDA toolkit version detection (nvcc) - for now just using driver version
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

    # Export to JSON if requested
    if export_format == 'json':
        # Build CPUInfo model from backend data
        from warpt.models.list_models import CPUInfo as ExportCPUInfo
        cpu_model = ExportCPUInfo(
            manufacturer=info.make,
            model=info.model,
            architecture=info.architecture,
            cores=info.total_physical_cores,
            threads=info.total_logical_cores,
            base_frequency_mhz=info.base_frequency,
            boost_frequency_single_core_mhz=info.boost_frequency_single_core,
            boost_frequency_multi_core_mhz=info.boost_frequency_multi_core,
            current_frequency_mhz=info.current_frequency,
            instruction_sets=None,  # TODO: Populate from backend when available
        )

        # Build GPUInfo models
        gpu_models = None
        gpu_count = None
        if gpu_list:
            gpu_models = [GPUInfo(**gpu) for gpu in gpu_list]
            gpu_count = len(gpu_list)

        # Build CUDA info if available
        cuda_info = None
        if cuda_driver_version:
            cuda_info = CUDAInfo(version=cuda_driver_version, driver=cuda_driver_version)

        # Build software info
        software = None
        if cuda_info:
            software = SoftwareInfo(cuda=cuda_info)

        # Build output with CPU data
        hardware = HardwareInfo(cpu=cpu_model, gpu_count=gpu_count, gpu=gpu_models)
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
