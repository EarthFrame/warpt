# Warpt Coding Standards

## Code Style

- Follow existing project patterns and conventions
- Use type hints throughout
- Use `X | None` instead of `Optional[X]` for type annotations
- Keep lines under 88 characters (black/ruff defaults)
- No blank lines with whitespace
- Comprehensive NumPy-style docstrings with Parameters/Returns sections

## Naming Conventions

- Classes: PascalCase (e.g., `NvidiaBackend`, `TenstorrentBackend`)
- Functions/methods: snake_case (e.g., `get_device_count`, `is_available`)
- Constants: UPPER_SNAKE_CASE (e.g., `PYNVML_AVAILABLE`)
- Private methods: prefix with `_` (e.g., `_get_device_handle`)

## Import Ordering (isort)

1. Standard library imports
1. Third-party imports
1. Warpt imports (`from warpt.xxx import yyy`)

## Error Handling

- Wrap vendor SDK calls in try/except blocks
- Return None or empty collections on failure
- Never let vendor exceptions propagate to callers
- Use specific exception types where possible

## Type Annotations

- Use `from __future__ import annotations` for forward references
- Use `X | None` union syntax (not `Optional[X]`)
- Use `list[X]` not `List[X]`, `dict[K, V]` not `Dict[K, V]`

## Testing Patterns

- Use `unittest.mock` for all vendor SDK mocking
- Use `patch.dict(sys.modules, {...})` for module-level mocks
- Test happy path, error path, and fallthrough behavior
- Never require real hardware in tests

## Linting

- Tool: ruff (replaces flake8 + isort + pyupgrade)
- Rules: E, W, F, I, N, UP, B, C4, ARG, RUF, D
- Docstring convention: NumPy

## Project Structure

- Backend implementations: `warpt/backends/`
- Power backends: `warpt/backends/power/`
- Models: `warpt/models/` (Pydantic for structured data)
- Tests: `tests/` (flat, prefixed with `test_`)
