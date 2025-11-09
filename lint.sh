#!/bin/bash
# Warpt Linting and Formatting Script
# Usage: ./lint.sh [check|fix|format|types]

set -e

COMMAND="${1:-check}"

case "$COMMAND" in
  check)
    echo "Running ruff check..."
    ruff check warpt/
    echo "✓ Ruff checks passed"
    echo
    echo "Running Black format check..."
    black warpt/ --check
    echo "✓ Black format check passed"
    echo
    echo "Running mypy type checking..."
    mypy warpt/ || true
    echo "✓ Type checking complete"
    ;;
  fix)
    echo "Running ruff with automatic fixes..."
    ruff check warpt/ --fix
    echo "✓ Ruff fixes applied"
    echo
    echo "Running Black formatter..."
    black warpt/
    echo "✓ Black formatting applied"
    echo
    echo "Running mdformat on markdown files..."
    mdformat docs/
    echo "✓ Markdown formatting applied"
    ;;
  format)
    echo "Running Black formatter..."
    black warpt/
    echo "✓ Black formatting applied"
    echo
    echo "Running mdformat on markdown files..."
    mdformat docs/
    echo "✓ Markdown formatting applied"
    ;;
  types)
    echo "Running mypy type checking..."
    mypy warpt/
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Usage: ./lint.sh [check|fix|format|types]"
    echo
    echo "Commands:"
    echo "  check   - Run ruff, Black format check, and mypy (default)"
    echo "  fix     - Run ruff fixes and apply Black formatting"
    echo "  format  - Run Black formatter only"
    echo "  types   - Run mypy type checker only"
    exit 1
    ;;
esac

echo
echo "✅ Linting complete!"
