"""Pydantic models for network information and test results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NetworkInterfaceInfo(BaseModel):
    """Information about a single network interface."""

    name: str = Field(..., description="Interface name (e.g., 'eth0', 'en0')")
    addresses: list[str] = Field(
        default_factory=list, description="IP addresses bound to this interface"
    )
    is_up: bool = Field(..., description="Whether interface is currently up")
    is_loopback: bool = Field(..., description="Whether this is a loopback interface")
    mac_address: str | None = Field(None, description="MAC address (if available)")
    mtu: int | None = Field(None, description="Maximum transmission unit")
    speed_mbps: int | None = Field(
        None, description="Link speed in Mbps (if available)"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class NetworkInfo(BaseModel):
    """Complete network information for the system."""

    hostname: str = Field(..., description="System hostname")
    interfaces: list[NetworkInterfaceInfo] = Field(
        ..., description="List of network interfaces"
    )
    default_interface: str | None = Field(
        None, description="Default network interface (if detectable)"
    )

    class Config:
        """Pydantic config."""

        frozen = True
