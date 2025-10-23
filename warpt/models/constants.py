"""Constants for warpt models and commands.

Going to include constants here in single file for now but will move to separate files later

"""
#for list command
class Target:
    
    #hardware
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    
    #software
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    CUDA = "cuda"
    DRIVERS = "drivers"
    
class Status:
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    STOPPED = "stopped"
    RUNNING = "running"
    
class Limits:
    PERCENT_MIN = 0
    PERCENT_MAX = 100
    
    TEMPERATURE_MIN = 0
    TEMPERATURE_MAX = 100

class StorageType:
    # TODO
    pass

class MemoryType:
    # TODO
    pass