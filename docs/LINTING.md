# Linting & Formatting Guide

Keep Python and Markdown tidy with `ruff`, `black`, `mypy`, `mdformat`, and `pre-commit`.

## Setup

Install the project and the dev dependencies that provide enforcement tools:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Key Targets

| Target | Description |
|--------|-------------|
| `make lint` | Run `ruff`, `black`, and `mdformat` checks via pre-commit hooks. |
| `make format` | Apply `black`, `ruff --fix`, and `mdformat docs/ examples/ README.md`. |
| `make mdformat` | Format Markdown files under `docs/`, `examples/`, and `README.md`. |
| `make type-check` | Run `mypy` via pre-commit. |
| `make check` | Run every hook defined in `.pre-commit-config.yaml` (`ruff`, `black`, `mdformat`, `mypy`, and the utility checks). |

Each target honors the configuration in `pyproject.toml`, so they share a single source of truth for styling rules.

## Pre-commit Hooks

`.pre-commit-config.yaml` currently runs:

- `black` – applies the formatting rules from `[tool.black]`.
- `ruff` – enforces lint rules (PEP 8, naming, docstrings, etc.).
- `mypy` – enforces typing rules defined under `[tool.mypy]`.
- `mdformat` – formats Markdown under `docs/`, `examples/`, and `README.md`.
- Utility hooks from `pre-commit-hooks` (`check-yaml`, `check-added-large-files`, `trailing-whitespace`, `end-of-file-fixer`, `check-merge-conflict`, etc.).

Run them manually with:

```bash
pre-commit run --all-files
```

## Continuous Integration

CI installs the dev dependencies, caches pre-commit, and then runs `make check`. That command mirrors the developer workflow.

## Troubleshooting

- If `make check` fails because `mdformat` flags a Markdown file, run `make mdformat` (or `python -m mdformat docs/ examples/ README.md`) and rerun `make check`. Modifications should show up in Git before rerunning hooks.
- If a binary like `mdformat` is missing, ensure the dev dependencies are installed and the correct Python environment is active.
- Use `pre-commit run <hook-id>` to rerun a specific hook when debugging.

## Reference

- `pyproject.toml` – configuration for `black`, `ruff`, and `mypy`.
- `.pre-commit-config.yaml` – hooks executed by `make check`.
- `Makefile` – entry points to run `lint`, `format`, `mdformat`, `type-check`, and `check`.
