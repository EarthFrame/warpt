"""Network backend - provides network interface information using psutil and socket."""

from __future__ import annotations

import socket

import psutil

from warpt.models.network_models import NetworkInfo, NetworkInterfaceInfo


class Network:
    """Network information backend using psutil and socket.

    Provides information about network interfaces, IP addresses, and connectivity.
    Follows the same pattern as CPU and RAM backends.
    """

    def __init__(self) -> None:
        """Initialize network backend and cache information."""
        self._info: NetworkInfo | None = None

    def get_network_info(self) -> NetworkInfo:
        """Get comprehensive network interface information.

        Main method that returns complete network information for the system,
        including all detected interfaces, their addresses, and status.

        Returns
        -------
            NetworkInfo: Complete network information with all interfaces.
        """
        if self._info is not None:
            return self._info

        hostname = socket.gethostname()
        interfaces = self._detect_interfaces()
        default_interface = self._detect_default_interface()

        self._info = NetworkInfo(
            hostname=hostname,
            interfaces=interfaces,
            default_interface=default_interface,
        )

        return self._info

    def refresh(self) -> NetworkInfo:
        """Force refresh of network information.

        Clears cache and re-detects all network interfaces.
        Useful if network configuration changes during runtime.

        Returns
        -------
            NetworkInfo: Freshly detected network information.
        """
        self._info = None
        return self.get_network_info()

    @staticmethod
    def _detect_interfaces() -> list[NetworkInterfaceInfo]:
        """Detect all network interfaces on the system.

        Returns
        -------
            List of NetworkInterfaceInfo objects for all detected interfaces.
        """
        interfaces = []
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()

        for interface_name, interface_addrs in addrs.items():
            # Get interface stats
            if_stats = stats.get(interface_name)
            is_up = if_stats.isup if if_stats else False
            speed_mbps = if_stats.speed if if_stats else None
            mtu = if_stats.mtu if if_stats else None

            # Extract IP addresses and MAC
            ipv4_addrs = []
            mac_address = None
            for addr in interface_addrs:
                if addr.family == socket.AF_INET:
                    ipv4_addrs.append(addr.address)
                elif addr.family == psutil.AF_LINK:
                    mac_address = addr.address

            # Check if loopback
            is_loopback = interface_name.startswith(("lo", "lo0"))

            interfaces.append(
                NetworkInterfaceInfo(
                    name=interface_name,
                    addresses=ipv4_addrs,
                    is_up=is_up,
                    is_loopback=is_loopback,
                    mac_address=mac_address,
                    mtu=mtu,
                    speed_mbps=speed_mbps,
                )
            )

        return interfaces

    @staticmethod
    def _detect_default_interface() -> str | None:
        """Attempt to detect the default network interface.

        Uses interface stats to find first non-loopback, active interface.

        Returns
        -------
            Interface name or None if not detectable.
        """
        try:
            stats = psutil.net_if_stats()

            # Look for first non-loopback, active interface
            for iface_name, iface_stats in stats.items():
                if iface_stats.isup and not iface_name.startswith(("lo", "lo0")):
                    return iface_name

            return None
        except (AttributeError, OSError):
            return None

    @staticmethod
    def get_interface_by_name(name: str) -> NetworkInterfaceInfo | None:
        """Get information about a specific network interface.

        Args:
            name: Interface name (e.g., 'eth0', 'en0')

        Returns
        -------
            NetworkInterfaceInfo or None if interface not found.
        """
        backend = Network()
        info = backend.get_network_info()

        for interface in info.interfaces:
            if interface.name == name:
                return interface

        return None

    @staticmethod
    def resolve_hostname(hostname: str) -> str | None:
        """Resolve a hostname to an IP address.

        Args:
            hostname: Hostname or IP address to resolve.

        Returns
        -------
            IP address string or None if resolution fails.
        """
        # If already an IP, return it
        if Network.is_valid_ip(hostname):
            return hostname

        # Try to resolve hostname
        try:
            return socket.gethostbyname(hostname)
        except (socket.gaierror, OSError):
            return None

    @staticmethod
    def is_valid_ip(ip_address: str) -> bool:
        """Check if a string is a valid IPv4 address.

        Args:
            ip_address: String to validate.

        Returns
        -------
            True if valid IPv4 address, False otherwise.
        """
        try:
            socket.inet_aton(ip_address)
            return True
        except OSError:
            return False

    @staticmethod
    def is_localhost(ip_address: str) -> bool:
        """Check if an IP address is localhost/loopback.

        Args:
            ip_address: IP address to check.

        Returns
        -------
            True if localhost (127.x.x.x or ::1), False otherwise.
        """
        # Resolve "localhost" to IP if needed
        if ip_address.lower() == "localhost":
            return True

        # Check if 127.x.x.x
        if ip_address.startswith("127."):
            return True

        # Check if ::1 (IPv6 loopback)
        if ip_address == "::1":
            return True

        return False
