import pynvml

class NvidiaBackend:
    def __init__(self):
        pynvml.nvmlInit()
        
    def list_devices(self):
        device_count = pynvml.nvmlDeviceGetCount()
        device_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            device_info.append({
                'name': pynvml.nvmlDeviceGetName(handle),
                'memory': pynvml.nvmlDeviceGetMemoryInfo(handle).total, # we can also break into free vs used memory
                'temperature': pynvml.nvmlDeviceGetTemperature(handle, 0)
            })
        return device_info
    

    