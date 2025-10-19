"""
List command - displays CPU information
"""

from warpt.backends.system import System


def run_list():
    """Lists CPU information"""

    backend = System()
    cpu_info = backend.list_devices()

    print(f"Physical Cores: {cpu_info['physical_cores']}")
    print(f"Logical Cores: {cpu_info['logical_cores']}")
    print(f"CPU Usage: {cpu_info['cpu_percent']}%")


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
