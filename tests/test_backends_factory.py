"""Tests for the GPU backend factory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from warpt.backends.factory import get_gpu_backend


def test_get_gpu_backend_nvidia():
    """Test factory returns NvidiaBackend when available."""
    mock_nvidia_module = MagicMock()
    mock_nvidia_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.is_available.return_value = True
    mock_nvidia_cls.return_value = mock_instance
    mock_nvidia_module.NvidiaBackend = mock_nvidia_cls

    with patch.dict(sys.modules, {"warpt.backends.nvidia": mock_nvidia_module}):
        backend = get_gpu_backend()
        assert backend == mock_instance
        mock_instance.is_available.assert_called_once()


def test_get_gpu_backend_none_available():
    """Test factory raises RuntimeError when no backends are available."""
    # Mock all backend modules
    mock_nvidia = MagicMock()
    mock_nvidia.NvidiaBackend.return_value.is_available.return_value = False

    mock_amd = MagicMock()
    mock_amd.AMDBackend.return_value.is_available.return_value = False

    mock_intel = MagicMock()
    mock_intel.IntelBackend.return_value.is_available.return_value = False

    modules = {
        "warpt.backends.nvidia": mock_nvidia,
        "warpt.backends.amd": mock_amd,
        "warpt.backends.intel": mock_intel,
    }

    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError) as excinfo:
            get_gpu_backend()
        assert "No GPUs detected on this system" in str(excinfo.value)
