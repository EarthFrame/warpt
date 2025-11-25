# Version System

The warpt package includes a comprehensive versioning system that tracks semantic versioning, package hash, and build date.

## Accessing Version Information

### From the Main Package

```python
import warpt

# Get the version string (e.g., "0.1.0")
print(warpt.__version__)

# Get the full version object
version = warpt.__version_info__
# or
from warpt import WARPT_VERSION
```

### Version Class Methods

The `Version` class provides several convenient methods to access version information:

```python
from warpt import WARPT_VERSION

# Get semantic version string
print(str(WARPT_VERSION))  # "0.1.0"
print(WARPT_VERSION)       # "0.1.0"

# Get full version info
print(WARPT_VERSION.full_version())
# Output: "0.1.0 (hash: a1b2c3d4, date: 2025-10-27)"

# Get semantic version as tuple
major, minor, patch = WARPT_VERSION.semver()

# Access individual components
print(WARPT_VERSION.major)      # 0
print(WARPT_VERSION.minor)      # 1
print(WARPT_VERSION.patch)      # 0
print(WARPT_VERSION.hash)       # Full SHA256 hash
print(WARPT_VERSION.date)       # datetime object

# Get shortened hash (default 8 characters)
print(WARPT_VERSION.hash_short())       # "a1b2c3d4"
print(WARPT_VERSION.hash_short(16))     # "a1b2c3d4e5f6g7h8"

# Format date with custom format
print(WARPT_VERSION.date_string())               # "2025-10-27"
print(WARPT_VERSION.date_string("%Y-%m-%d %H:%M:%S"))  # With time
```

## Updating Version

To update the version, edit the `WARPT_VERSION` constant in `warpt/version/warpt_version.py`:

```python
WARPT_VERSION = Version(
    major=0,
    minor=1,
    patch=1,  # Increment patch for bugfixes
    hash=_compute_package_hash(),
    date=datetime(2025, 10, 28),
)
```

Also update the version in `pyproject.toml`:

```toml
[project]
version = "0.1.1"
```

## Version Hash

The package hash is automatically computed from all Python source files in the warpt package. This provides a consistent way to identify a specific build of the package without relying on external git information.

The hash excludes:

- `__pycache__` directories
- `.pyc`, `.pyo`, and `.pyd` files
- `.git` and `.pytest_cache` directories

## Immutable Version Object

The `Version` class is implemented as a frozen dataclass, making it immutable. This ensures version information cannot be accidentally modified at runtime.
