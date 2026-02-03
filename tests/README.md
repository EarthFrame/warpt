# Tests

This directory contains pytest-compatible tests for `warpt`.

## Running Tests

First, install the development dependencies (including pytest):

```bash
pip install -e ".[dev]"
```

Or install just pytest:

```bash
pip install pytest
```

Then run the tests:

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

# Run tests with coverage (requires pytest-cov)
pytest --cov=warpt --cov-report=html
```

## Test Files

### `test_framework_detection.py`

Tests for framework detection functionality:

- Framework name property
- Detection with mocked frameworks
- Handling missing frameworks
- CUDA support detection
- Model serialization

### `test_serialization.py`

Tests for serialization methods:

- `to_dict()` - Dictionary output
- `to_json()` - JSON output (with/without indentation)
- `to_yaml()` - YAML output (optional dependency)
- `to_toml()` - TOML output (optional dependency)
- `to_huml()` - HUML output (optional dependency)
- Custom override functionality
- Import error handling

## Test Structure

Tests use pytest fixtures for:

- Mocking torch module
- Cleanup of sys.modules
- Setup and teardown

Tests are organized into classes for better organization and can be run individually:

```bash
# Run all dict tests
pytest tests/test_serialization.py::TestToDict

# Run specific test
pytest tests/test_serialization.py::TestToJson::test_to_json_compact
```

## Optional Dependencies

Some serialization tests require optional packages:

- `pyyaml` for YAML tests
- `tomli` and `tomli_w` for TOML tests
- `pyhuml` for HUML tests

These tests will be skipped if the packages are not installed (using `pytest.importorskip`).

## Writing New Tests

When adding new tests:

1. Use descriptive test names starting with `test_`
1. Use fixtures for setup/teardown
1. Use `pytest.importorskip()` for optional dependencies
1. Group related tests into classes
1. Add docstrings explaining what each test does
