"""Device fingerprint generation.

Only GPU fingerprinting (via NVIDIA UUID) is implemented.
CPU/RAM/Storage/Network fingerprinting deferred to future work.
"""

from __future__ import annotations

from warpt.models.list_models import GPUInfo
from warpt.models.profile_models import DeviceFingerprint, HardwareCategory


def generate_gpu_fingerprint(gpu_info: GPUInfo) -> DeviceFingerprint | None:
    """Generate a fingerprint for a GPU using its NVML UUID.

    Parameters
    ----------
    gpu_info : GPUInfo
        GPU information from hardware detection.

    Returns
    -------
    DeviceFingerprint | None
        A fingerprint if the GPU has a UUID, otherwise None.
    """
    if gpu_info.uuid:
        return DeviceFingerprint(
            fingerprint_id=gpu_info.uuid,
            category=HardwareCategory.GPU,
            source="nvml_uuid",
        )
    return None
