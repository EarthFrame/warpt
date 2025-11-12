.PHONY: format check fix types help

help:
	@echo "Available targets:"
	@echo "  format  - Run Black and mdformat (Python & markdown)"
	@echo "  fix     - Run ruff fixes and apply Black + mdformat"
	@echo "  check   - Run ruff, Black format check, and mypy"
	@echo "  types   - Run mypy type checker only"

format:
	./lint.sh format

fix:
	./lint.sh fix

check:
	./lint.sh check

types:
	./lint.sh types
