# Pre-commit Hooks Setup Guide

Quick reference for setting up and using pre-commit hooks in warpt.

## ⚡ Quick Setup

```bash
# 1. Install pre-commit
pip install pre-commit

# 2. Install git hooks
pre-commit install

# 3. Test it works
pre-commit run --all-files
```

That's it! Pre-commit will now run automatically on every `git commit`.

## 🔧 What Gets Checked

| Hook | Purpose | Auto-fixes? |
|------|---------|------------|
| **ruff** | Fast linting (PEP 8, imports, naming, etc.) | ✅ Yes |
| **ruff-format** | Code formatting | ✅ Yes |
| **check-yaml** | Valid YAML syntax | ❌ No |
| **trailing-whitespace** | Remove trailing spaces | ✅ Yes |
| **end-of-file-fixer** | Ensure files end with newline | ✅ Yes |
| **check-merge-conflict** | Detect merge conflicts | ❌ No |
| **check-added-large-files** | Prevent large files (>500KB) | ❌ No |

## 📋 Common Commands

### Run on Staged Files (Before Commit)
```bash
pre-commit run
```

### Run on All Files
```bash
pre-commit run --all-files
```

### Run Specific Hook
```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```

### Skip Pre-commit (Not Recommended!)
```bash
git commit --no-verify
```

### Reinstall Hooks
```bash
pre-commit install
```

### Update Hook Versions
```bash
pre-commit autoupdate
```

## 🐛 Troubleshooting

### "pre-commit: command not found"
```bash
pip install pre-commit
```

### Hooks Not Running on Commit
```bash
# Reinstall hooks
pre-commit install

# Verify installation
ls -la .git/hooks/pre-commit
```

### Want to Disable a Specific Hook
Edit `.pre-commit-config.yaml` and set `stages: [manual]` on the hook, then run manually when needed.

### Large File Rejection
Files over 500KB will be rejected. Options:
1. Split into smaller files
2. Use Git LFS for large files
3. Skip with `git commit --no-verify` (not recommended)

## 📝 Workflow Example

```bash
# 1. Make changes to your code
vim warpt/backends/system.py

# 2. Stage changes
git add warpt/backends/system.py

# 3. Try to commit
git commit -m "Fix CPU detection"

# 4. Pre-commit runs hooks
# ✓ Linting passes
# ✓ Formatting passes
# Commit succeeds!

# If there are issues:
# Pre-commit shows errors and auto-fixes what it can
# Stage the fixes
git add warpt/backends/system.py
# Commit again
git commit -m "Fix CPU detection"
```

## 🔗 Configuration

The hooks configuration is in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]  # Auto-fix enabled
        types: [python]

      - id: ruff-format
        types: [python]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      # ... more hooks
```

## 📚 Manual Linting (Without Pre-commit)

If you want to lint without pre-commit:

```bash
# Check code
ruff check warpt/

# Auto-fix issues
ruff check warpt/ --fix

# Format code
ruff format warpt/
```

See [README.md](README.md) for more details.

## 🎯 Best Practices

1. ✅ Install pre-commit hooks in all local repos
2. ✅ Run `pre-commit run --all-files` before pushing
3. ✅ Keep hook configuration in version control
4. ✅ Update hooks regularly: `pre-commit autoupdate`
5. ❌ Avoid `--no-verify` unless absolutely necessary

## 🔗 Resources

- [Pre-commit.com](https://pre-commit.com) - Official docs
- [Ruff](https://docs.astral.sh/ruff/) - Linter documentation
- [PEP 8](https://pep8.org/) - Python style guide
