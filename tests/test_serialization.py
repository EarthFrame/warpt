"""Test serialization methods of framework detectors."""

import json
import sys
from unittest.mock import MagicMock

import pytest

from warpt.backends.software.frameworks import PyTorchDetector


@pytest.fixture
def mock_torch_installed():
    """Fixture to mock installed torch with CUDA."""
    mock = MagicMock()
    mock.__version__ = "2.1.0"
    mock.cuda.is_available.return_value = True
    sys.modules["torch"] = mock
    yield mock
    if "torch" in sys.modules:
        del sys.modules["torch"]


@pytest.fixture
def mock_torch_no_cuda():
    """Fixture to mock installed torch without CUDA."""
    mock = MagicMock()
    mock.__version__ = "2.1.0"
    mock.cuda.is_available.return_value = False
    sys.modules["torch"] = mock
    yield mock
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


class TestToDict:
    """Tests for to_dict() method."""

    def test_to_dict_with_framework(self, _mock_torch_installed):
        """Test to_dict() when framework is installed."""
        detector = PyTorchDetector()
        data = detector.to_dict()

        assert data is not None
        assert isinstance(data, dict)
        assert "version" in data
        assert "cuda_support" in data
        assert data["version"] == "2.1.0"
        assert data["cuda_support"] is True

    def test_to_dict_without_framework(self, _cleanup_torch):
        """Test to_dict() when framework is not installed."""
        detector = PyTorchDetector()
        data = detector.to_dict()

        assert data is None


class TestToJson:
    """Tests for to_json() method."""

    def test_to_json_with_indentation(self, _mock_torch_no_cuda):
        """Test to_json() with indentation."""
        detector = PyTorchDetector()
        json_str = detector.to_json(indent=2)

        assert json_str is not None
        assert isinstance(json_str, str)

        # Parse and verify
        data = json.loads(json_str)
        assert data["version"] == "2.1.0"
        assert data["cuda_support"] is False

    def test_to_json_compact(self, _mock_torch_installed):
        """Test to_json() without indentation."""
        detector = PyTorchDetector()
        compact = detector.to_json(indent=None)

        assert compact is not None
        assert "\n" not in compact

    def test_to_json_without_framework(self, _cleanup_torch):
        """Test to_json() when framework is not installed."""
        detector = PyTorchDetector()
        json_str = detector.to_json()

        assert json_str is None


class TestToYaml:
    """Tests for to_yaml() method."""

    def test_to_yaml(self, _mock_torch_installed):
        """Test to_yaml() produces valid YAML."""
        pytest.importorskip("yaml")

        import yaml  # type: ignore[import-untyped, unused-ignore]

        detector = PyTorchDetector()
        yaml_str = detector.to_yaml()

        assert yaml_str is not None
        assert isinstance(yaml_str, str)

        # Parse and verify
        data = yaml.safe_load(yaml_str)
        assert data["version"] == "2.1.0"
        assert data["cuda_support"] is True

    def test_to_yaml_import_error(self, _mock_torch_installed, monkeypatch):
        """Test to_yaml() raises ImportError when yaml not available."""
        # Mock yaml import to fail
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        detector = PyTorchDetector()
        with pytest.raises(ImportError, match="PyYAML"):
            detector.to_yaml()


class TestToToml:
    """Tests for to_toml() method."""

    def test_to_toml(self, _mock_torch_installed):
        """Test to_toml() produces valid TOML."""
        pytest.importorskip("tomli")
        pytest.importorskip("tomli_w")

        import tomli

        detector = PyTorchDetector()
        toml_str = detector.to_toml()

        assert toml_str is not None
        assert isinstance(toml_str, str)

        # Parse and verify
        data = tomli.loads(toml_str)
        assert data["version"] == "2.1.0"
        assert data["cuda_support"] is True

    def test_to_toml_import_error(self, _mock_torch_installed, monkeypatch):
        """Test to_toml() raises ImportError when tomli_w not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tomli_w":
                raise ImportError("No module named 'tomli_w'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        detector = PyTorchDetector()
        with pytest.raises(ImportError, match="tomli_w"):
            detector.to_toml()


class TestToHuml:
    """Tests for to_huml() method."""

    def test_to_huml(self, _mock_torch_installed):
        """Test to_huml() produces valid HUML."""
        pyhuml = pytest.importorskip("pyhuml")

        detector = PyTorchDetector()
        huml_str = detector.to_huml()

        assert huml_str is not None
        assert isinstance(huml_str, str)

        # Parse and verify (pyhuml should be able to load what it dumped)
        data = pyhuml.loads(huml_str)
        assert data["version"] == "2.1.0"
        assert data["cuda_support"] is True

    def test_to_huml_import_error(self, _mock_torch_installed, monkeypatch):
        """Test to_huml() raises ImportError when pyhuml not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyhuml":
                raise ImportError("No module named 'pyhuml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        detector = PyTorchDetector()
        with pytest.raises(ImportError, match="pyhuml"):
            detector.to_huml()


class TestCustomOverride:
    """Tests for custom serialization overrides."""

    def test_custom_to_dict(self, _mock_torch_installed):
        """Test that to_dict() can be overridden."""

        class CustomDetector(PyTorchDetector):
            def to_dict(self) -> dict[str, str | bool] | None:
                base = super().to_dict()
                if base is None:
                    return None
                base["custom_field"] = "custom_value"
                return base

        detector = CustomDetector()
        data = detector.to_dict()

        assert data is not None
        assert "custom_field" in data
        assert data["custom_field"] == "custom_value"
        assert "version" in data

    def test_custom_to_dict_affects_json(self, _mock_torch_installed):
        """Test that custom to_dict() is used by to_json()."""

        class CustomDetector(PyTorchDetector):
            def to_dict(self) -> dict[str, str | bool] | None:
                base = super().to_dict()
                if base is None:
                    return None
                base["custom_field"] = "custom_value"
                return base

        detector = CustomDetector()
        json_str = detector.to_json()

        assert json_str is not None
        data = json.loads(json_str)
        assert "custom_field" in data
