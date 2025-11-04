"""List command - displays CPU information."""

from warpt.backends.system import CPU

from warpt.models.list_models import GPUInfo, HardwareInfo, ListOutput
from warpt.backends.system import System

# TODO: Implement GPU backend factory once vendor-specific backends are created
# GPU detection is disabled to avoid deprecated pynvml dependency


def run_list() -> None:
    """Display comprehensive CPU information."""
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
