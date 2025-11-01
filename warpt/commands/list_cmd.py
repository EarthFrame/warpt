"""
List command - displays hardware and software information.

"""

from warpt.models.list_models import GPUInfo, HardwareInfo, ListOutput
from warpt.backends.system import System

# TODO: Implement GPU backend factory once vendor-specific backends are created
# GPU detection is disabled to avoid deprecated pynvml dependency


def run_list():
    """
    Lists CPU and GPU information, formatted with ListOutput model

    Currently returns hardware info with GPU set to None until GPU backends
    are implemented using the new GPUBackend interface.
    """
    # GPU backend temporarily disabled - will be re-enabled with proper
    # vendor backends (NVIDIA using nvidia-ml-py, AMD using amdsmi, etc.)
    gpu_models = None

    hardware = HardwareInfo(gpu=gpu_models)
    output = ListOutput(hardware=hardware)

    # output as JSON
    print(output.model_dump_json(indent=2))


'''
from CLI-Design.md

Hardware:
    CPU: Intel Xeon E5-2686 v4
        Cores: 16, Threads: 32
        Features: AVX, AVX2, SSE4.2, FMA

    GPU:
        [0] NVIDIA RTX 4090 (24GB, CUDA 8.9)
        [1] NVIDIA RTX 4090 (24GB, CUDA 8.9)

    Memory: 64GB DDR4

    Storage:
        /dev/nvme0n1: 2TB NVMe SSD

Software:
    Python: 3.11.4 (/usr/bin/python3.11)
    CUDA: 12.1.1 (driver 530.30.02)

    Frameworks:
        PyTorch: 2.0.1 (CUDA 12.1)
        TensorFlow: 2.13.0 (CUDA 12.1)

    Compilers:
        GCC: 11.4.0
        NVCC: 12.1.105
'''
