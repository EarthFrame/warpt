# Contributing to Warpt

Thanks for contributing! Warpt is a performance monitoring tool for CPUs, GPUs, and system resources. We welcome bug fixes, features, and documentation improvements.

**Table of Contents**

- [Quick Start](#quick-start)
- [Before You Start](#before-you-start)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Setup Your Environment

```bash
git clone <repo>
cd warpt
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

Hooks run automatically before each commit to catch issues early.

### 3. Make Your Changes

Create a branch for your work:

```bash
git checkout -b feature/short-description
```

### 4. Run Quality Checks Before Committing

```bash
make format    # Format code
make check     # Lint everything
make types     # Type checking
```

Or auto-fix issues:

```bash
make fix       # Auto-format and fix linting issues
```

### 5. Commit & Push

```bash
git add .
git commit -m "Brief: description of what changed"
git push origin feature/short-description
```

Then open a pull request.

## Before You Start

### For Bug Fixes

- Search existing issues to avoid duplicates
- Include reproducible steps in the issue
- Test your fix before submitting

### For Features

- Open an issue first to discuss the idea
- Keep it scoped and focused
- Add tests and documentation

### For Documentation

- Fix typos and improve clarity
- Add examples if helpful
- Keep it concise

## Development Workflow

### Running Tests

```bash
pytest tests/                    # Run all tests
pytest tests/test_file.py        # Run specific test file
pytest tests/ -v                 # Verbose output
```

### Testing Your Changes

```bash
# Test the CLI
warpt --help

# Install locally for testing
pip install -e .

# Test specific commands
warpt list --hardware
warpt list --software
```

### Debugging

```bash
# Enable type checking for debugging
make types

# Check for style issues
ruff check warpt/ --show-fixes

# Test individual module
python -m warpt.backends.system
```

## Code Standards

### Python Style

- **Format**: Black (88 char line length)
- **Lint**: Ruff with Google-style docstrings
- **Types**: Mypy type hints (run `make types`)
- **Syntax**: Use `X | None` not `Optional[X]` (Python 3.10+ style)

### Docstring Format (Google Style)

```python
def calculate_memory_usage(process_id: int) -> dict[str, int] | None:
    """Calculate memory usage for a process.

    Queries the system to get detailed memory information for the
    specified process, including RSS and VMS.

    Args:
        process_id: The process ID to check

    Returns:
        Dictionary with 'rss' and 'vms' keys in bytes, or None if process
        not found.

    Raises:
        ValueError: If process_id is negative
    """
```

### Commit Message Format

```
Brief: what changed (50 chars max)

Optional longer explanation if needed. Explain WHY not just WHAT.
Keep lines under 72 characters.

Fixes #123          # Reference issues
```

### Type Hints

Always add type hints:

```python
# Good
def get_gpu_count() -> int:
    """Return number of GPUs."""

def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse version string to tuple."""

# Avoid
def get_gpu_count():  # Missing return type
    ...
```

## Common Tasks

### Add a New Backend

1. Create `warpt/backends/new_backend.py`
1. Implement required interface
1. Add to `warpt/backends/__init__.py`
1. Add tests in `tests/backends/`
1. Update docs

### Add a New Command

1. Create `warpt/commands/new_cmd.py`
1. Add to `warpt/cli.py`
1. Create corresponding Pydantic model
1. Add tests in `tests/commands/`
1. Update `QUICK_REFERENCE.md`

### Update Models

- Edit `warpt/models/list_models.py`
- Use type unions: `str | None` not `Optional[str]`
- Add descriptive Field descriptions
- Run `make check` to validate

## Troubleshooting

### Pre-commit Hooks Failing

```bash
# See what's failing
pre-commit run --all-files

# Fix most issues automatically
make fix

# If still failing, fix manually then retry
git add .
git commit -m "Fix linting issues"
```

### Type Checking Errors

```bash
# See full mypy output
make types

# Check specific file
mypy warpt/path/to/file.py --show-traceback

# Some errors need manual fixes (type: ignore comments)
# Last resort: # type: ignore
```

### Import Issues After Changes

```bash
# Reinstall in development mode
pip install -e .

# Clear cache
rm -rf warpt/__pycache__ .pytest_cache .mypy_cache
```

### Pre-commit Takes Forever

The first run installs tools. Subsequent runs are fast (~1s). If slow:

```bash
# Check what's running
pre-commit run --all-files --verbose

# Clear cache
rm -rf ~/.cache/pre-commit
```

### Tests Not Finding Modules

```bash
# Make sure dev dependencies are installed
pip install -e ".[dev]"

# Run from repo root
cd /path/to/warpt
pytest tests/
```

## Resources

- **Full Linting Guide**: `docs/LINTING.md`
- **Quick Commands**: `QUICK_REFERENCE.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **CLI Design**: `docs/CLI-Design.md`

## Questions?

Check the docs above first. If stuck:

1. Search existing issues
1. Check Discord/discussions
1. Open an issue with details

______________________________________________________________________

**Thanks for making Warpt better!**
