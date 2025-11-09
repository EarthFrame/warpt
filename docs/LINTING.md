# Linting and Code Quality with Ruff, Black, Mypy, and Markdown Formatter

This project uses a comprehensive code quality toolchain:

- **Ruff**: Fast Python linting
- **Black**: Python code formatter
- **Mypy**: Static type checking
- **mdformat**: Markdown formatter

## Quick Start

### Running All Checks

```bash
# Check all (ruff + Black format check + mypy)
./lint.sh check

# Auto-fix and format all
./lint.sh fix

# Format with Black only
./lint.sh format

# Type checking only
./lint.sh types
```

### Pre-commit Integration

```bash
# Install pre-commit framework
pip install pre-commit

# Install git hooks
pre-commit install

# Run all hooks on all files
pre-commit run --all-files
```

## Installation

All tools are in the dev dependencies:

```bash
pip install -e ".[dev]"
# or individually:
pip install black ruff mypy types-psutil mdformat
```

## Tools Overview

### 1. Black - Python Formatter

Black automatically formats Python code for consistency and readability.

**Configuration** (`pyproject.toml`):

```toml
[tool.black]
line-length = 88
target-version = ["py311"]
```

**Features:**

- Consistent code style across the project
- Function arguments spread across multiple lines for readability
- No configuration options for style (opinionated by design)
- Automatically formats when `./lint.sh fix` is run

**Example**:

```python
# Before
result = some_very_long_function_name(param1, param2, param3, param4, param5)

# After
result = some_very_long_function_name(
    param1, param2, param3, param4, param5
)
```

### 2. Ruff - Fast Linting

Ruff provides fast linting with 11 rule categories.

**Key Rules:**

- **E, W, F**: PEP 8 compliance and basic errors
- **I**: Import sorting
- **N**: Naming conventions
- **UP**: Modern Python syntax
- **B**: Bug detection
- **C4**: Code simplifications
- **D**: Docstring validation (Google style)

\[See earlier sections for full rule details\]

### 3. Mypy - Static Type Checking

Mypy checks Python code for type errors statically, helping catch bugs before runtime.

**Configuration** (`pyproject.toml`):

```toml
[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
ignore_missing_imports = true
```

**Key Features:**

- Checks variable types
- Validates function signatures
- Detects unused variables
- Ensures Optional types are handled

**Example**:

```python
# ❌ Error: Argument has incompatible type
def greet(name: str) -> str:
    return f"Hello, {name}"

greet(42)  # mypy error: Argument 1 has incompatible type "int"; expected "str"

# ✅ Correct
greet("Alice")  # OK
```

### 4. mdformat - Markdown Formatter

Formats Markdown files for consistency and readability.

**Formatting Applied:**

- Consistent spacing and indentation
- Proper list formatting
- Standardized link formatting
- Code block formatting

## Pre-commit Hooks

The `.pre-commit-config.yaml` configures automatic checks before commits:

1. **Black** - Formats Python code
1. **Ruff** - Lints and fixes Python code
1. **Mypy** - Type checks Python code
1. **Pre-commit hooks** - General checks (YAML, file sizes, conflicts)
1. **mdformat** - Formats Markdown files

```bash
# Setup (one-time)
pre-commit install

# Manual run on staged files
pre-commit run

# Manual run on all files
pre-commit run --all-files
```

## Development Workflow

### Before Committing

```bash
# 1. Check everything
./lint.sh check

# 2. Auto-fix issues
./lint.sh fix

# 3. Check types
./lint.sh types

# 4. Commit
git add .
git commit -m "Feature: add new functionality"
```

### In Your IDE

**VSCode:**

- Install "Black Formatter" extension
- Install "Ruff" extension
- Install "mypy" extension
- Settings → Format on Save ✓

**PyCharm:**

- Settings → Tools → Python Integrated Tools
- Configure Black as formatter
- Configure Ruff as linter
- Configure mypy as type checker

## Common Issues and Solutions

### Black vs Ruff Formatting

Both Black and Ruff can format code. Black is used as the primary formatter because it:

- Handles line wrapping more intelligently
- Spreads long function arguments better
- Has more mature formatting decisions

**Resolution:** Let Black run last in the pre-commit pipeline (it does).

### Mypy "No name in module" Error

**Issue**: `error: No module named 'X'` or `No name in module 'X'`

**Solution**: Add type stubs or configure imports:

```python
# mypy: ignore[import]
import some_untyped_module
```

### Black Changes Function Signatures

**Before:**

```python
def long_function_name(param1: str, param2: int, param3: float) -> None:
    pass
```

**After:**

```python
def long_function_name(
    param1: str, param2: int, param3: float
) -> None:
    pass
```

This is intentional! It improves readability.

## Running Individual Tools

### Black Only

```bash
# Check
black warpt/ --check

# Format
black warpt/

# Specific file
black warpt/cli.py
```

### Ruff Only

```bash
# Check
ruff check warpt/

# Fix
ruff check warpt/ --fix

# Show details
ruff check warpt/ --show-fixes
```

### Mypy Only

```bash
# Check all files
mypy warpt/

# Specific file
mypy warpt/cli.py

# Strict checking
mypy warpt/ --strict
```

### mdformat Only

```bash
# Check
mdformat docs/ --check

# Format
mdformat docs/

# Specific file
mdformat docs/LINTING.md
```

## Performance

| Tool | Time | Type |
|------|------|------|
| Black | ~100ms | Formatting |
| Ruff | ~50ms | Linting |
| Mypy | ~500ms | Type checking |
| Pre-commit (all) | ~1s | All tools |

These are typical times for the warpt project. Larger projects may take longer.

## Ignoring Issues

### Line-level Ignore

**Black:**

```python
# fmt: off
some_code = that_should_not_be_formatted
# fmt: on
```

**Ruff:**

```python
import unused  # noqa: F401
```

**Mypy:**

```python
some_value = some_func()  # type: ignore
```

### File-level Ignore

**Ruff:**

```python
# ruff: noqa
```

**Mypy:**

```python
# mypy: ignore-errors
```

### Disable Rule

**Black:**

```python
# Don't format this section
# fmt: off
# ... code ...
# fmt: on
```

## Continuous Integration

### GitHub Actions

```yaml
- name: Install dependencies
  run: pip install -e ".[dev]"

- name: Run linting
  run: ./lint.sh check

- name: Type checking
  run: mypy warpt/
```

### GitLab CI

```yaml
lint:
  image: python:3.11
  script:
    - pip install -e ".[dev]"
    - ./lint.sh check
```

## Further Reading

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [mdformat Documentation](https://mdformat.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
