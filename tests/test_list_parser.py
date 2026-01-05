"""Tests for the ListParser utility."""

from warpt.models.list_models import (
    HardwareInfo,
    ListOutput,
    SoftwareInfo,
)
from warpt.utils.list_parser import ListParser


def test_parse_dict():
    """Test parsing a dictionary into a ListOutput model."""
    data = {
        "hardware": {
            "cpu": {
                "manufacturer": "Intel",
                "model": "Core i9",
                "architecture": "x86_64",
                "cores": 8,
                "threads": 16,
            },
            "gpu_count": 1,
        },
        "software": {
            "docker": {
                "installed": True,
                "version": "20.10.7",
                "path": "/usr/bin/docker",
            }
        },
    }
    output = ListParser.parse_dict(data)
    assert isinstance(output, ListOutput)
    assert output.hardware.cpu.manufacturer == "Intel"
    assert ListParser.get_gpu_count(output) == 1
    assert ListParser.get_cpu_arch(output) == "x86_64"
    assert ListParser.get_container_tool(output) == "docker"


def test_get_gpu_count():
    """Test retrieving GPU count from ListOutput."""
    output = ListOutput(hardware=HardwareInfo(gpu_count=2))
    assert ListParser.get_gpu_count(output) == 2

    output = ListOutput(hardware=HardwareInfo(gpu_count=None, gpu=[]))
    assert ListParser.get_gpu_count(output) == 0


def test_is_cuda_available():
    """Test checking for CUDA availability."""
    output = ListOutput(software=SoftwareInfo(cuda=None))
    assert not ListParser.is_cuda_available(output)


def test_get_framework_version():
    """Test retrieving ML framework versions."""
    from warpt.models.list_models import FrameworkInfo

    output = ListOutput(
        software=SoftwareInfo(
            frameworks={"pytorch": FrameworkInfo(installed=True, version="2.0.0")}
        )
    )
    assert ListParser.get_framework_version(output, "pytorch") == "2.0.0"
    assert ListParser.get_framework_version(output, "tensorflow") is None
