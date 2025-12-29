"""Stress test command - runs hardware stress tests via registry.

CLI Examples:
    warpt stress --list                  # List available tests
    warpt stress --list -c cpu           # List CPU tests only
    warpt stress                         # Run all available tests
    warpt stress -c cpu                  # Run CPU category
    warpt stress -c accelerator          # Run accelerator tests
    warpt stress -t GPUMatMulTest        # Run specific test
    warpt stress -o results.json         # Save to JSON
    warpt stress -o a.json -o b.yaml     # Multiple outputs
    warpt stress --config tests.yaml     # Use config file
"""

import sys
from pathlib import Path
from typing import Any

import click

from warpt.stress import (
    OutputFormat,
    TestCategory,
    TestRegistry,
    TestRunner,
)

# Map CLI strings to TestCategory enum
CATEGORY_MAP: dict[str, TestCategory | str] = {
    "cpu": TestCategory.CPU,
    "accelerator": TestCategory.ACCELERATOR,
    "ram": TestCategory.RAM,
    "storage": TestCategory.STORAGE,
    "network": TestCategory.NETWORK,
    "all": "all",  # Special case: run all available tests
}


def _resolve_category_enum(name: str) -> TestCategory | None:
    """Return the TestCategory enum for the CLI name if available."""
    value = CATEGORY_MAP.get(name)
    if isinstance(value, TestCategory):
        return value
    return None


def load_config(config_path: str) -> dict[str, Any]:
    """Load test configuration from YAML file.

    Config format:
        defaults:
          duration: 60
          warmup: 10

        tests:
          - name: GPUMatMulTest
            duration: 120
            device_id: 0

          - name: CPUMatMulTest

          - category: accelerator
            duration: 90

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        click.ClickException: If file not found or invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        raise click.ClickException(
            "YAML config requires pyyaml. Install with: pip install pyyaml"
        ) from None

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise click.ClickException(f"Error parsing config: {e}") from e

    if not isinstance(config, dict):
        raise click.ClickException("Config must be a YAML dictionary")

    return config


def get_output_format(output: str | None, fmt: str | None) -> OutputFormat:
    """Determine output format from filename or explicit format."""
    if fmt:
        return OutputFormat(fmt.lower())

    if output:
        suffix = Path(output).suffix.lower()
        if suffix == ".json":
            return OutputFormat.JSON
        elif suffix in (".yaml", ".yml"):
            return OutputFormat.YAML

    return OutputFormat.TEXT


def list_tests(
    registry: TestRegistry,
    category: str | None = None,
    verbose: bool = False,
) -> None:
    """List available stress tests."""
    if category:
        if category not in CATEGORY_MAP:
            valid = ", ".join(sorted(CATEGORY_MAP.keys()))
            click.echo(f"Error: Unknown category '{category}'. Valid: {valid}")
            sys.exit(1)
        if category == "all":
            tests = registry.get_all_tests()
            click.echo("\nStress Tests (ALL)\n")
        else:
            cat_enum = _resolve_category_enum(category)
            if cat_enum is None:
                tests = []
            else:
                tests = registry.get_tests_by_category(cat_enum)
            click.echo(f"\nStress Tests ({category.upper()})\n")
    else:
        tests = registry.get_all_tests()
        click.echo("\nStress Tests\n")

    if not tests:
        click.echo("  No tests registered.")
        return

    # Group tests by category
    from collections import defaultdict

    tests_by_cat: dict[str, list[Any]] = defaultdict(list)
    for test_cls in tests:
        try:
            try:
                instance = test_cls()
            except TypeError:
                instance = test_cls(**{"device_id": 0})

            cat = instance.get_category().value
            tests_by_cat[cat].append((test_cls, instance))
        except Exception as e:
            click.echo(f"Error instantiating {test_cls.__name__}: {e}")

    # Get all categories in defined order
    all_categories = [cat.value for cat in TestCategory]

    click.echo("-" * 60)

    # Display tests grouped by category
    for cat_value in all_categories:
        if cat_value not in tests_by_cat:
            click.echo(f"\n[{cat_value.upper()}]")
            click.echo("  No tests registered")
            continue

        click.echo(f"\n[{cat_value.upper()}]")

        # Sort tests within category by name
        sorted_tests = sorted(tests_by_cat[cat_value], key=lambda x: x[0].__name__)

        for test_cls, instance in sorted_tests:
            try:
                available = instance.is_available()
                status = "Available" if available else "Not Available"

                click.echo(f"  {test_cls.__name__:<35} {status}")
                if verbose:
                    click.echo(f"      {instance.get_pretty_name()}")
                    # Wrap description at 70 chars
                    import textwrap

                    wrapped = textwrap.fill(
                        instance.get_description(),
                        width=70,
                        initial_indent="      ",
                        subsequent_indent="      ",
                    )
                    click.echo(wrapped)
                    click.echo()

            except Exception as e:
                click.echo(f"  {test_cls.__name__:<35} Error: {e}")

    click.echo("\n" + "-" * 60)
    click.echo(f"\nTotal: {len(tests)} tests registered")
    if not verbose:
        click.echo("Use --verbose for detailed info")


def _parse_test_spec_config(
    test_spec: dict[str, Any],
    skip_key: str,
) -> dict[str, Any]:
    """Parse a test spec dict into config, mapping warmup -> burnin_seconds."""
    test_config: dict[str, Any] = {}
    for key, value in test_spec.items():
        if key == skip_key:
            continue
        elif key == "warmup":
            test_config["burnin_seconds"] = value
        else:
            test_config[key] = value
    return test_config


def run_stress(
    categories: tuple[str, ...],
    tests: tuple[str, ...],
    duration: int,
    warmup: int,
    device_id: str | None,
    outputs: tuple[str, ...],
    fmt: str | None,
    config: str | None,
    list_only: bool,
    verbose: bool,
) -> None:
    """Run stress tests based on CLI arguments."""
    registry = TestRegistry()

    # Handle --list
    if list_only:
        cat = categories[0] if categories else None
        list_tests(registry, cat, verbose)
        return

    # Load config file if provided
    config_data: dict[str, Any] = {}
    if config:
        config_data = load_config(config)

    # Get defaults from config
    defaults = config_data.get("defaults", {})
    cfg_duration = defaults.get("duration", duration)
    cfg_warmup = defaults.get("warmup", warmup)

    # Collect tests to run
    tests_to_run: list[type] = []
    configs: dict[str, dict[str, Any]] = {}

    # Parse device IDs from CLI
    device_ids: list[int] | None = None
    if device_id:
        try:
            device_ids = [int(x.strip()) for x in device_id.split(",") if x.strip()]
        except ValueError:
            click.echo(f"Error: Invalid device ID '{device_id}'. Must be integers.")
            sys.exit(1)

    # If config file has tests section, use that
    if config_data.get("tests"):
        for test_spec in config_data["tests"]:
            if not isinstance(test_spec, dict):
                continue

            # Test by name
            if "name" in test_spec:
                test_name = test_spec["name"]
                try:
                    test_cls = registry.get_test(test_name)
                    tests_to_run.append(test_cls)

                    test_config = _parse_test_spec_config(test_spec, "name")
                    if "duration" not in test_config:
                        test_config["duration"] = cfg_duration
                    if "burnin_seconds" not in test_config:
                        test_config["burnin_seconds"] = cfg_warmup

                    configs[test_name] = test_config

                except Exception:
                    click.echo(f"Warning: Test '{test_name}' not found, skipping.")

            # Test by category
            elif "category" in test_spec:
                cat_name = test_spec["category"]
                if cat_name not in CATEGORY_MAP:
                    click.echo(f"Warning: Unknown category '{cat_name}', skipping.")
                    continue

                if cat_name == "all":
                    cat_tests = registry.get_available_tests()
                else:
                    cat_enum = _resolve_category_enum(cat_name)
                    if cat_enum is None:
                        continue
                    cat_tests = registry.get_tests_by_category(cat_enum)
                for test_cls in cat_tests:
                    tests_to_run.append(test_cls)

                    test_config = _parse_test_spec_config(test_spec, "category")
                    if "duration" not in test_config:
                        test_config["duration"] = cfg_duration
                    if "burnin_seconds" not in test_config:
                        test_config["burnin_seconds"] = cfg_warmup

                    configs[test_cls.__name__] = test_config

    # CLI arguments (used if no config tests)
    elif tests:
        for test_name in tests:
            try:
                test_cls = registry.get_test(test_name)
                tests_to_run.append(test_cls)
            except Exception:
                click.echo(f"Error: Test '{test_name}' not found.")
                click.echo("Use 'warpt stress --list' to see available tests.")
                sys.exit(1)

    elif categories:
        for cat in categories:
            if cat not in CATEGORY_MAP:
                valid = ", ".join(sorted(CATEGORY_MAP.keys()))
                click.echo(f"Error: Unknown category '{cat}'. Valid: {valid}")
                sys.exit(1)

            if cat == "all":
                # Special case: get all available tests
                tests_to_run.extend(registry.get_available_tests())
            else:
                cat_enum = _resolve_category_enum(cat)
                if cat_enum:
                    cat_tests = registry.get_tests_by_category(cat_enum)
                    tests_to_run.extend(cat_tests)

    # Default: show helpful message instead of discovering all tests
    else:
        click.echo("No tests or categories specified.")
        click.echo("\nQuick start:")
        click.echo("  warpt stress --list           # List available tests")
        click.echo("  warpt stress -c cpu           # Run CPU tests")
        click.echo("  warpt stress -c accelerator   # Run accelerator tests")
        click.echo("  warpt stress -c all           # Run all tests")
        click.echo("  warpt stress -t TestName      # Run specific test")
        click.echo("\nRun 'warpt stress --help' for more options.")
        sys.exit(0)

    if not tests_to_run:
        click.echo("No tests available to run.")
        click.echo("Use 'warpt stress --list' to see available tests.")
        sys.exit(1)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tests = []
    for t in tests_to_run:
        if t.__name__ not in seen:
            seen.add(t.__name__)
            unique_tests.append(t)
    tests_to_run = unique_tests

    # Apply CLI device_id to tests without config
    if device_ids:
        for test_cls in tests_to_run:
            name = test_cls.__name__
            if name not in configs:
                configs[name] = {}
            if "device_id" not in configs[name]:
                configs[name]["device_id"] = device_ids[0]

    # Apply CLI warmup to tests without config
    for test_cls in tests_to_run:
        name = test_cls.__name__
        if name not in configs:
            configs[name] = {}
        if "burnin_seconds" not in configs[name]:
            configs[name]["burnin_seconds"] = cfg_warmup

    # Display what we're running
    click.echo("\n" + "=" * 60)
    click.echo("  WARPT STRESS TEST")
    click.echo("=" * 60)
    if config:
        click.echo(f"\nConfig: {config}")
    click.echo(f"\nTests to run: {len(tests_to_run)}")
    for t in tests_to_run:
        test_cfg = configs.get(t.__name__, {})
        dur = test_cfg.get("duration", cfg_duration)
        click.echo(f"  • {t.__name__} ({dur}s)")
    click.echo(f"\nDefault duration: {cfg_duration}s")
    click.echo(f"Default warmup: {cfg_warmup}s")
    if device_ids:
        click.echo(f"Device IDs: {', '.join(map(str, device_ids))}")
    click.echo("\n" + "-" * 60 + "\n")

    # Run tests
    runner = TestRunner()
    for test_cls in tests_to_run:
        test_config = configs.get(test_cls.__name__, {})
        runner.add_test(test_cls, test_config)

    results = runner.run(duration=cfg_duration, skip_unavailable=True)

    # Emit results to multiple outputs
    if outputs:
        for out_path in outputs:
            out_format = get_output_format(out_path, None)
            results.emit(out_path, out_format)
            click.echo(f"✓ Results saved to: {out_path}")

    # Emit to stdout if no outputs or explicit format requested
    if not outputs or fmt:
        out_format = OutputFormat(fmt) if fmt else OutputFormat.TEXT
        if out_format == OutputFormat.TEXT:
            results.emit_stdout()
        else:
            import sys as _sys

            results.emit(_sys.stdout, out_format)

    # Summary
    summary = results.to_dict()["summary"]
    click.echo(f"\n✓ Completed: {summary['passed']}/{summary['total_tests']} passed")

    if results.errors:
        click.echo("\n⚠️  Errors:")
        for name, error in results.errors.items():
            click.echo(f"  • {name}: {error}")
