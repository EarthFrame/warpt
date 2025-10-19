"""
Utilizes psutil for system information
"""

import psutil


class System:
    def list_devices(self):
        """
        Lists CPUs available on the machine

        Returns:
            dict with CPU information
        """
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=1),
        }
        return cpu_info
