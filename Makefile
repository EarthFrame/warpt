.PHONY: check format help lint type-check mdformat

help:
	@echo "Available targets:"
	@echo "  lint        - Run style checks via pre-commit (ruff + Black + mdformat)"
	@echo "  check       - Run lint + type checking"
	@echo "  format      - Apply formatting (Black, Ruff fix, mdformat)"
	@echo "  type-check  - Run mypy for static typing"
	@echo "  mdformat    - Format Markdown files in docs/examples/README"

lint:
	pre-commit run --all-files ruff
	pre-commit run --all-files black
	pre-commit run --all-files mdformat

check:
	pre-commit run --all-files

format:
	black warpt/
	ruff check warpt/ --fix
	mdformat docs/ examples/ README.md

type-check:
	pre-commit run --all-files mypy

mdformat:
	mdformat docs/ examples/ README.md
