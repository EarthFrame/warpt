# Framework Detection

This document describes the framework detection system in `warpt`.

## Overview

The framework detection system provides a structured, extensible way to detect ML frameworks (PyTorch, TensorFlow, JAX, etc.) and gather information about them. The system uses Pydantic models to ensure type safety and enable easy serialization to JSON/YAML.

## Architecture

### Components

1. **Pydantic Models** (`warpt/models/list_models.py`)

   - `FrameworkInfo`: Stores framework metadata (version, CUDA support)
   - `SoftwareInfo`: Container for all software information including frameworks

1. **Base Detector** (`warpt/backends/software/frameworks/base.py`)

   - `FrameworkDetector`: Abstract base class for all framework detectors
   - Provides safe import functionality and enforces interface

1. **Framework-Specific Detectors** (`warpt/backends/software/frameworks/`)

   - `PyTorchDetector`: Detects PyTorch installation
   - Future: TensorFlowDetector, JAXDetector, etc.

1. **Detection Functions** (`warpt/backends/software/frameworks/__init__.py`)

   - `detect_all_frameworks()`: Detect all available frameworks
   - `detect_framework(name)`: Detect specific framework by name

## Usage

### Basic Detection

```python
from warpt.backends.software import detect_all_frameworks

# Detect all installed frameworks
frameworks = detect_all_frameworks()

for name, info in frameworks.items():
    print(f"{name}: {info.version} (CUDA: {info.cuda_support})")
```

### Detect Specific Framework

```python
from warpt.backends.software import detect_framework

# Check if PyTorch is installed
pytorch_info = detect_framework("pytorch")
if pytorch_info:
    print(f"PyTorch {pytorch_info.version} is installed")
    print(f"CUDA support: {pytorch_info.cuda_support}")
else:
    print("PyTorch is not installed")
```

### JSON/YAML Export

```python
import json
from warpt.backends.software import detect_all_frameworks

frameworks = detect_all_frameworks()

# Convert to dict for JSON serialization
frameworks_dict = {
    name: info.model_dump()
    for name, info in frameworks.items()
}

# Export to JSON
print(json.dumps(frameworks_dict, indent=2))
```

Example output:

```json
{
  "pytorch": {
    "version": "2.1.0",
    "cuda_support": true
  }
}
```

### Integration with List Command

The framework detection integrates with the existing `SoftwareInfo` model:

```python
from warpt.backends.software import detect_all_frameworks
from warpt.models.list_models import SoftwareInfo

frameworks = detect_all_frameworks()
software_info = SoftwareInfo(frameworks=frameworks)

# Now software_info can be included in ListOutput
```

### Serialization Methods

Each `FrameworkDetector` provides built-in methods to serialize framework information in multiple formats:

#### Available Methods

```python
from warpt.backends.software import PyTorchDetector

detector = PyTorchDetector()

# Dictionary format
data = detector.to_dict()
# Returns: {'version': '2.1.0', 'cuda_support': True} or None

# JSON format
json_str = detector.to_json(indent=2)
# Returns: '{\n  "version": "2.1.0",\n  "cuda_support": true\n}' or None

# Compact JSON
compact = detector.to_json(indent=None)
# Returns: '{"version":"2.1.0","cuda_support":true}' or None

# YAML format (requires PyYAML)
yaml_str = detector.to_yaml()
# Returns YAML string or raises ImportError if PyYAML not installed

# TOML format (requires tomli_w)
toml_str = detector.to_toml()
# Returns TOML string or raises ImportError if tomli_w not installed

# HUML format (requires pyhuml)
huml_str = detector.to_huml()
# Returns HUML string or raises ImportError if pyhuml not installed
```

#### Custom Serialization

You can override any serialization method to customize the output:

```python
class CustomPyTorchDetector(PyTorchDetector):
    """PyTorch detector with extended metadata."""

    def to_dict(self) -> dict[str, str | bool] | None:
        """Add custom fields to output."""
        base_dict = super().to_dict()
        if base_dict is None:
            return None

        # Add custom metadata
        base_dict["framework_name"] = self.framework_name
        base_dict["supports_distributed"] = True
        base_dict["supports_mobile"] = True

        return base_dict

detector = CustomPyTorchDetector()
data = detector.to_dict()
# Returns extended dictionary with custom fields
# JSON, YAML, and TOML outputs automatically use the overridden to_dict()
```

#### Optional Dependencies

- **YAML support**: Install with `pip install pyyaml`
- **TOML support**: Install with `pip install tomli_w`
- **HUML support**: Install with `pip install pyhuml` ([pyhuml on GitHub](https://github.com/huml-lang/pyhuml))

The serialization methods will raise helpful `ImportError` messages if the required libraries are not installed.

## Adding New Framework Detectors

To add support for a new framework:

1. **Create a detector class** in `warpt/backends/software/frameworks/`:

```python
from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.models.list_models import FrameworkInfo


class TensorFlowDetector(FrameworkDetector):
    """Detector for TensorFlow installation."""

    @property
    def framework_name(self) -> str:
        return "tensorflow"

    def detect(self) -> FrameworkInfo | None:
        tf = self._safe_import("tensorflow")
        if tf is None:
            return None

        version = tf.__version__
        cuda_support = tf.test.is_gpu_available()

        return FrameworkInfo(
            version=version,
            cuda_support=cuda_support,
        )
```

2. **Register the detector** in `__init__.py`:

```python
from warpt.backends.software.frameworks.tensorflow import TensorFlowDetector

_FRAMEWORK_DETECTORS = [
    PyTorchDetector(),
    TensorFlowDetector(),  # Add new detector
]
```

3. **Export the class** (optional):

```python
__all__ = [
    "FrameworkDetector",
    "PyTorchDetector",
    "TensorFlowDetector",  # Add to exports
    "detect_all_frameworks",
    "detect_framework",
]
```

## Demo Script

Run the demo script to see the framework detection in action:

```bash
python examples/framework_detection_demo.py
```

This will:

- Detect all installed frameworks
- Display version and CUDA support information
- Show JSON output for integration with other tools
- Demonstrate individual framework detection

## Design Rationale

### Why Pydantic Models?

- **Type Safety**: Ensures data integrity at runtime
- **Validation**: Automatic validation of data
- **Serialization**: Built-in JSON/YAML export via `model_dump()`
- **Documentation**: Self-documenting with field descriptions

### Why Abstract Base Class?

- **Consistency**: All detectors follow the same interface
- **Extensibility**: Easy to add new framework detectors
- **Safety**: `_safe_import()` prevents import errors from crashing the application

### Why Registry Pattern?

- **Automatic Discovery**: `detect_all_frameworks()` automatically uses all registered detectors
- **Decoupling**: Detectors don't need to know about each other
- **Easy Testing**: Can test detectors individually

## Future Enhancements

1. **Extended Metadata**: Add more fields to `FrameworkInfo`

   - Installation path
   - Build configuration
   - Supported backends (CPU, CUDA, ROCm, etc.)
   - Available extensions/plugins

1. **More Frameworks**:

   - TensorFlow
   - JAX
   - MXNet
   - ONNX Runtime

1. **Version Compatibility Checks**:

   - Check for known compatibility issues
   - Warn about deprecated versions

1. **Performance**: Cache detection results to avoid repeated imports
