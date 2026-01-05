"""Tests for PCI backend hardware discovery."""

import subprocess
import types

from warpt.backends.pci import PCIBackend, PCIVendor


def test_pci_backend_availability(monkeypatch):
    """Test is_available returns True if lspci exists."""
    monkeypatch.setattr(
        "shutil.which", lambda _x: "/usr/bin/lspci" if _x == "lspci" else None
    )
    backend = PCIBackend()
    assert backend.is_available() is True

    monkeypatch.setattr("shutil.which", lambda _x: None)
    backend = PCIBackend()
    assert backend.is_available() is False


def test_list_devices_parsing(monkeypatch):
    """Test parsing of lspci -nn output."""
    mock_output = (
        "00:00.0 Host bridge [0600]: Intel Corporation 11th Gen Core Processor "
        "Host Bridge/DRAM Registers [8086:9a14] (rev 01)\n"
        "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 "
        "[GeForce RTX 4090] [10de:2684] (rev a1)\n"
        "02:00.0 Display controller [0380]: Advanced Micro Devices, Inc. "
        "[AMD/ATI] Navi 21 [Radeon RX 6800/6800 XT / 6900 XT] [1002:73bf] (rev c1)\n"
        "04:00.0 Ethernet controller [0200]: Realtek Semiconductor Co., Ltd. "
        "RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller [10ec:8168] "
        "(rev 15)\n"
    )

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout=mock_output,
            stderr="",
        )

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/lspci")
    monkeypatch.setattr("subprocess.run", fake_run)

    backend = PCIBackend()
    devices = backend.list_devices()

    assert len(devices) == 4

    # Check NVIDIA device
    nv = next(d for d in devices if d.vendor_id == PCIVendor.NVIDIA)
    assert nv.slot == "01:00.0"
    assert nv.device_id == "2684"
    assert "RTX 4090" in nv.device_name
    assert "VGA compatible controller" in nv.class_name

    # Check AMD device
    amd = next(d for d in devices if d.vendor_id == PCIVendor.AMD)
    assert amd.slot == "02:00.0"
    assert "Navi 21" in amd.device_name

    # Check non-GPU device
    eth = next(d for d in devices if d.vendor_id == "10ec")
    assert eth.device_id == "8168"
    assert "Ethernet controller" in eth.class_name


def test_get_gpus_filtering(monkeypatch):
    """Test get_gpus only returns display controllers."""
    mock_output = (
        "00:00.0 Host bridge [0600]: Intel Corporation [8086:9a14]\n"
        "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation "
        "[10de:2684]\n"
        "02:00.0 Display controller [0380]: AMD [1002:73bf]\n"
        "03:00.0 3D controller [0302]: NVIDIA [10de:2235]\n"
        "04:00.0 Ethernet controller [0200]: Realtek [10ec:8168]\n"
    )

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(returncode=0, stdout=mock_output)

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/lspci")
    monkeypatch.setattr("subprocess.run", fake_run)

    backend = PCIBackend()
    gpus = backend.get_gpus()

    # Should find VGA, Display, and 3D controllers
    assert len(gpus) == 3
    classes = [g.class_name for g in gpus]
    assert "VGA compatible controller" in classes
    assert "Display controller" in classes
    assert "3D controller" in classes

    # Ethernet and Host bridge should be excluded
    assert not any("Ethernet" in g.class_name for g in gpus)
    assert not any("Host bridge" in g.class_name for g in gpus)


def test_empty_lspci_output(monkeypatch):
    """Test handling of empty lspci output."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/lspci")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=""),
    )

    backend = PCIBackend()
    assert backend.list_devices() == []
    assert backend.get_gpus() == []


def test_lspci_error(monkeypatch):
    """Test handling of lspci command failure."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/lspci")

    def error_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "lspci")

    monkeypatch.setattr("subprocess.run", error_run)

    backend = PCIBackend()
    assert backend.list_devices() == []
