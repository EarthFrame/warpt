"""Test framework detection functionality."""

import sys
from unittest.mock import MagicMock

import pytest

from warpt.backends.software.frameworks import PyTorchDetector


@pytest.fixture
def mock_torch():
    """Fixture to mock torch module."""
    mock = MagicMock()
    mock.__version__ = "2.1.0+cu121"
    mock.cuda.is_available.return_value = True
    sys.modules["torch"] = mock
    yield mock
    # Cleanup
    if "torch" in sys.modules:
        del sys.modules["torch"]


@pytest.fixture
def cleanup_torch():
    """Fixture to ensure torch is not in sys.modules."""
    if "torch" in sys.modules:
        del sys.modules["torch"]
    yield
    if "torch" in sys.modules:
        del sys.modules["torch"]


def test_framework_name():
    """Test that framework name is correct."""
    detector = PyTorchDetector()
    assert detector.framework_name == "pytorch"


def test_pytorch_detection_with_mock(mock_torch):  # noqa: ARG001
    """Test PyTorch detection with a mocked torch module."""
    detector = PyTorchDetector()

    # Test detection
    info = detector.detect()
    assert info is not None, "Detection should succeed with mocked torch"

    # Test version
    assert info.version == "2.1.0+cu121"

    # Test CUDA support
    assert info.cuda_support is True


def test_pytorch_not_installed(cleanup_torch):  # noqa: ARG001
    """Test PyTorch detection when torch is not installed."""
    detector = PyTorchDetector()
    info = detector.detect()

    assert info is None, "Detection should return None when torch is not installed"


def test_pytorch_without_cuda():
    """Test PyTorch detection when CUDA is not available."""
    mock = MagicMock()
    mock.__version__ = "2.1.0"
    mock.cuda.is_available.return_value = False
    sys.modules["torch"] = mock

    try:
        detector = PyTorchDetector()
        info = detector.detect()

        assert info is not None
        assert info.version == "2.1.0"
        assert info.cuda_support is False
    finally:
        if "torch" in sys.modules:
            del sys.modules["torch"]


def test_pytorch_without_cuda_module():
    """Test PyTorch detection when cuda module doesn't exist."""
    mock = MagicMock()
    mock.__version__ = "2.1.0"
    # Remove cuda attribute
    del mock.cuda
    sys.modules["torch"] = mock

    try:
        detector = PyTorchDetector()
        info = detector.detect()

        assert info is not None
        assert info.version == "2.1.0"
        assert info.cuda_support is False
    finally:
        if "torch" in sys.modules:
            del sys.modules["torch"]


def test_model_dump(mock_torch):  # noqa: ARG001
    """Test that FrameworkInfo can be serialized to dict."""
    detector = PyTorchDetector()
    info = detector.detect()

    assert info is not None
    data = info.model_dump()

    assert isinstance(data, dict)
    assert "version" in data
    assert "cuda_support" in data
    assert data["version"] == "2.1.0+cu121"
    assert data["cuda_support"] is True
