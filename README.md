# warpt

Performance monitoring and system utilities.

**Note:** This is an internal development build. Not for public release.

## Quick Start

### Installation

```bash
# Clone and navigate to the project
cd warpt

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in development mode
pip install -e .
```

### Running warpt

```bash
# Display CPU information
warpt list

# Show version
warpt version

# Get help
warpt --help
```

## Development Setup

### Prerequisites

- Python 3.11+
- pip and venv

### Initial Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install development dependencies
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. (Optional) Run pre-commit on all files
pre-commit run --all-files
```

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com) for automated code quality checks. Hooks run automatically on `git commit`.

**What gets checked:**

- **ruff**: Fast Python linting (PEP 8, naming, imports, etc.)
- **ruff format**: Code formatting
- **YAML syntax**: Valid YAML files
- **Large files**: Prevents accidentally committing large files
- **Merge conflicts**: Detects unresolved merge conflicts
- **Trailing whitespace**: Removes trailing whitespace

**Manual checks:**

```bash
# Run pre-commit on staged files
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files

# Skip pre-commit for a commit (not recommended)
git commit --no-verify
```

### Linting & Formatting

**Manual linting:**

```bash
# Check code with ruff
ruff check warpt/

# Fix issues automatically
ruff check warpt/ --fix

# Format code
ruff format warpt/
```

**Configuration:**

Linting rules are configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",      # Errors
    "W",      # Warnings
    "F",      # Pyflakes
    "I",      # Import sorting
    "N",      # Naming
    "D",      # Docstrings (Google style)
    # ... more rules
]
```

### Running Tests

```bash
# Run pytest
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_example.py
```

## Project Structure

```
warpt/
├── warpt/                      # Main package
│   ├── __init__.py             # Package exports
│   ├── cli.py                  # CLI entry point
│   ├── backends/               # Hardware backends
│   │   └── system.py           # CPU information
│   ├── commands/               # CLI command handlers
│   │   ├── list_cmd.py         # List CPU info
│   │   └── version_cmd.py      # Display version
│   ├── models/                 # Data models
│   ├── utils/                  # Utilities
│   └── version/                # Version info
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml              # Project configuration
├── .pre-commit-config.yaml     # Pre-commit hooks config
└── README.md                   # This file
```

## Key Modules

### CPU Backend (`warpt/backends/system.py`)

Comprehensive CPU information with Pydantic models:

```python
from warpt.backends.system import CPU

cpu = CPU()
info = cpu.get_cpu_info()

print(f"Processor: {info.make} {info.model}")
print(f"Cores: {info.total_physical_cores}")
print(f"Threads: {info.total_logical_cores}")
print(f"Base Freq: {info.base_frequency} MHz")
print(f"Single-Core Boost: {info.boost_frequency_single_core} MHz")
```

See [docs/CPU.md](docs/CPU.md) for detailed documentation.

### Version Info (`warpt/version/warpt_version.py`)

Access version information easily:

```python
import warpt

print(warpt.__version__)  # "0.1.0"
print(warpt.WARPT_VERSION)  # Full version object
print(warpt.WARPT_VERSION.full_version())  # With hash and date
```

See [docs/VERSION.md](docs/VERSION.md) for detailed documentation.

## Contributing

### Code Quality Guidelines

- Follow [PEP 8](https://pep8.org/) style guide (enforced by ruff)
- Use type hints for all functions
- Write docstrings in Google style
- Keep functions focused and well-named

### Before Committing

```bash
# 1. Make your changes
# 2. Run pre-commit checks
pre-commit run --all-files

# 3. Fix any issues (most are auto-fixable)
# 4. Commit when all checks pass
git add .
git commit -m "Descriptive commit message"
```

## Resources

- [Pre-commit Documentation](https://pre-commit.com)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8](https://pep8.org/)

## License

Internal use only.
