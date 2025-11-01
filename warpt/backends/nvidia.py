import pynvml


class NvidiaBackend:
    """Backend for NVIDIA GPU information using pynvml."""

    def __init__(self):
        """Initialize NVIDIA backend and prepare NVIDIA device access."""
        pynvml.nvmlInit()

    def list_devices(self):
        """List all NVIDIA GPU devices and their basic information.

        Returns:
            list: List of dicts containing device name, memory, and temperature.
        """
        device_count = pynvml.nvmlDeviceGetCount()
        device_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            device_info.append(
                {
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "memory": pynvml.nvmlDeviceGetMemoryInfo(
                        handle
                    ).total,  # total memory
                    "temperature": pynvml.nvmlDeviceGetTemperature(handle, 0),
                }
            )
        return device_info
