# Examples

This directory contains example scripts demonstrating various features of `warpt`.

## Framework Detection

### `framework_detection_demo.py`

A demonstration of the framework detection system. This script:

- Detects all installed ML frameworks (PyTorch, TensorFlow, etc.)
- Displays framework information (version, CUDA support)
- Shows how to export results to JSON

**Usage:**

```bash
python examples/framework_detection_demo.py
```

**Example output when PyTorch is installed:**

```text
============================================================
Framework Detection Demo
============================================================

Detecting all installed frameworks...

Found 1 framework(s):

  pytorch:
    Version: 2.1.0+cu121
    CUDA Support: Yes

============================================================
JSON Output (for integration with other tools):
============================================================
{
  "pytorch": {
    "version": "2.1.0+cu121",
    "cuda_support": true
  }
}

============================================================
Individual Framework Detection Example:
============================================================

PyTorch is installed:
  Version: 2.1.0+cu121
  CUDA Support: Yes
```

### `framework_serialization_demo.py`

Demonstrates the serialization methods available on framework detectors. Shows:

- Converting framework info to dict, JSON, YAML, and TOML formats
- How to override serialization methods for custom output
- Exporting in multiple formats simultaneously

**Usage:**

```bash
python examples/framework_serialization_demo.py
```

**Features demonstrated:**

```python
detector = PyTorchDetector()

# Multiple output formats
data = detector.to_dict()           # Python dict
json = detector.to_json(indent=2)   # Pretty JSON
yaml = detector.to_yaml()            # YAML (requires PyYAML)
toml = detector.to_toml()            # TOML (requires tomli_w)
huml = detector.to_huml()            # HUML (requires pyhuml)

# Custom serialization
class CustomDetector(PyTorchDetector):
    def to_dict(self):
        base = super().to_dict()
        base["custom_field"] = "value"
        return base
```

## Testing

For unit tests, see the `tests/` directory. Run tests with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_framework_detection.py
pytest tests/test_serialization.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "to_json"
```

## See Also

- [Framework Detection Documentation](../docs/FRAMEWORK_DETECTION.md) - Detailed guide on the framework detection system
- [Architecture Documentation](../docs/ARCHITECTURE.md) - Overall project architecture
