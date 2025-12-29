"""Benchmark command - run and manage performance benchmarks.

CLI Examples:
    warpt benchmark                          # List available benchmarks
    warpt benchmark list                     # List supported benchmarks
    warpt benchmark validate                 # Validate configuration files
    warpt benchmark build -b hpl             # Build benchmark container
    warpt benchmark run -b hpl               # Run HPL benchmark
    warpt benchmark run -b hpl --output results.json  # Run with output file
"""

import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml  # type: ignore[import-untyped, unused-ignore]

from warpt.benchmarks.base import BenchmarkResult
from warpt.benchmarks.registry import BenchmarkRegistry
from warpt.models.benchmark_models import BenchmarksConfig


def _load_config(config_path: str | None) -> dict[str, Any] | None:
    """Load configuration from JSON or YAML file."""
    if not config_path:
        return None

    path = Path(config_path)
    if not path.exists():
        click.echo(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)

        if not isinstance(data, dict):
            click.echo(f"Error: Config file {config_path} must be a dictionary")
            sys.exit(1)

        return data
    except Exception as e:
        click.echo(f"Error: Failed to parse config file {config_path}: {e}")
        sys.exit(1)


def _validate_benchmark_name(
    benchmark: str | None, registry: BenchmarkRegistry
) -> str | None:
    """Validate benchmark name against registered benchmarks."""
    if benchmark is None:
        return None
    # First try exact match
    if benchmark in registry:
        return benchmark

    # Try case-insensitive match
    benchmark_lower = benchmark.lower()
    for registered_name in registry._benchmarks.keys():
        if registered_name.lower() == benchmark_lower:
            return registered_name

    # Try common aliases (hpl -> HPLBenchmark)
    aliases = {
        "hpl": "HPLBenchmark",
    }
    if benchmark_lower in aliases and aliases[benchmark_lower] in registry:
        return aliases[benchmark_lower]

    valid = ", ".join(sorted(registry._benchmarks.keys()))
    click.echo(f"Error: Unknown benchmark '{benchmark}'. Valid: {valid}")
    sys.exit(1)


def list_benchmarks(registry: BenchmarkRegistry) -> None:
    """List all registered benchmarks with availability status."""
    click.echo("Available Benchmarks:")
    click.echo("-" * 50)

    benchmarks = registry.list_benchmarks()
    if not benchmarks:
        click.echo("  No benchmarks registered.")
        return

    for bench in benchmarks:
        status = "Available" if bench["available"] else "Not Available"
        click.echo(f"  {bench['name']:<20} {status}")
        click.echo(f"      {bench['pretty_name']}")
        # Wrap description
        import textwrap

        description = str(bench.get("description", ""))
        wrapped = textwrap.fill(
            description,
            width=70,
            initial_indent="      ",
            subsequent_indent="      ",
        )
        click.echo(wrapped)
        click.echo()

    click.echo("-" * 50)
    click.echo(f"Total: {len(benchmarks)} benchmarks registered")


def validate_configs(
    benchmark: str | None,
    system_config: str | None,
    benchmark_config: str | None,
    cluster_config: str | None,
    overrides: tuple[str, ...],
    registry: BenchmarkRegistry,
) -> None:
    """Validate configuration files without execution."""
    benchmark = _validate_benchmark_name(benchmark, registry)

    click.echo("Configuration Validation")
    click.echo("-" * 40)

    if benchmark:
        click.echo(f"Benchmark: {benchmark}")

    configs = [
        ("System", system_config),
        ("Benchmark", benchmark_config),
        ("Cluster", cluster_config),
    ]

    all_valid = True
    for name, path in configs:
        if path:
            if Path(path).exists():
                click.echo(f"✓ {name} config found: {path}")
            else:
                click.echo(f"✗ {name} config missing: {path}")
                all_valid = False
        else:
            click.echo(f"- {name} config: not specified")

    if overrides:
        click.echo(f"\nOverrides: {', '.join(overrides)}")

    # TODO: Implement actual config validation using system_config,
    # benchmark_config, cluster_config
    click.echo("\nDetailed validation not yet implemented")

    if not all_valid:
        sys.exit(1)


def build_benchmark(
    benchmark: str | None,
    _system_config: str | None,
    benchmark_config: str | None,
    _cluster_config: str | None,
    overrides: tuple[str, ...],
    _force: bool,
    dry_run: bool,
    _debug: bool,
    registry: BenchmarkRegistry,
) -> None:
    """Build benchmark container without running."""
    # Load configuration if provided
    config_data = _load_config(benchmark_config)
    benchmarks_to_build = []

    if config_data:
        try:
            config = BenchmarksConfig.model_validate(config_data)
            for b_cfg in config.benchmarks:
                benchmarks_to_build.append((b_cfg.name, b_cfg.parameters))
        except Exception as e:
            click.echo(f"Error: Invalid benchmark configuration: {e}")
            sys.exit(1)
    elif benchmark:
        # No config file, use command line benchmark name
        benchmark_name = _validate_benchmark_name(benchmark, registry)
        if benchmark_name:
            # Parse overrides into parameters
            params = {}
            for override in overrides:
                if "=" in override:
                    k, v = override.split("=", 1)
                    params[k] = v
            benchmarks_to_build.append((benchmark_name, params))
    else:
        click.echo("Error: Either --benchmark or --benchmark-config required")
        sys.exit(1)

    for bench_name, params in benchmarks_to_build:
        actual_name = _validate_benchmark_name(bench_name, registry)
        if not actual_name:
            continue

        click.echo(f"Building {actual_name} benchmark")
        click.echo("-" * 40)

        if dry_run:
            click.echo(f"DRY RUN - Would build {actual_name} with params: {params}")
            continue

        try:
            benchmark_cls = registry.get_benchmark(actual_name)
            benchmark_instance = benchmark_cls()

            # Apply parameters from config
            if params:
                benchmark_instance.set_parameters(params)

            # Special case: if we are building and no execution_mode is set,
            # or it's set to numpy, we should probably default to docker
            # if the benchmark supports it and we are in a build command.
            if hasattr(benchmark_instance, "execution_mode"):
                current_mode = benchmark_instance.execution_mode
                if current_mode == "numpy" and "execution_mode" not in params:
                    # If building hpl specifically, default to docker
                    if actual_name == "HPLBenchmark":
                        benchmark_instance.execution_mode = "docker"

            benchmark_instance.build()
            click.echo(f"\nSuccessfully built {actual_name}")

        except Exception as e:
            click.echo(f"Error building benchmark {actual_name}: {e}")
            if _debug:
                import traceback

                traceback.print_exc()
            sys.exit(1)


def run_benchmark(
    benchmark: str | None,
    _system_config: str | None,
    benchmark_config: str | None,
    _cluster_config: str | None,
    overrides: tuple[str, ...],
    output: str | None,
    dry_run: bool,
    _debug: bool,
    registry: BenchmarkRegistry,
    no_monitor: bool = False,
) -> None:
    """Build (if needed) and execute a benchmark."""
    # Load configuration if provided
    config_data = _load_config(benchmark_config)
    benchmarks_to_run = []

    if config_data:
        try:
            config = BenchmarksConfig.model_validate(config_data)
            for b_cfg in config.benchmarks:
                benchmarks_to_run.append((b_cfg.name, b_cfg.parameters))
        except Exception as e:
            click.echo(f"Error: Invalid benchmark configuration: {e}")
            sys.exit(1)
    elif benchmark:
        # No config file, use command line benchmark name
        benchmark_name = _validate_benchmark_name(benchmark, registry)
        if benchmark_name:
            # Parse overrides into parameters
            params = {}
            for override in overrides:
                if "=" in override:
                    k, v = override.split("=", 1)
                    params[k] = v
            benchmarks_to_run.append((benchmark_name, params))
    else:
        click.echo("Error: Either --benchmark or --benchmark-config required")
        sys.exit(1)

    for bench_name, params in benchmarks_to_run:
        # Validate benchmark name again if it came from config
        actual_name = _validate_benchmark_name(bench_name, registry)
        if not actual_name:
            continue

        click.echo(f"Running {actual_name} benchmark")
        click.echo("-" * 40)

        if dry_run:
            click.echo(f"DRY RUN - Would execute {actual_name} with params: {params}")
            continue

        try:
            benchmark_cls = registry.get_benchmark(actual_name)
            benchmark_instance = benchmark_cls()

            # Apply parameters from config
            if params:
                benchmark_instance.set_parameters(params)

            if not benchmark_instance.is_available():
                click.echo(f"Error: {actual_name} is not available on this system")
                sys.exit(1)

            # Execute benchmark
            result = benchmark_instance.run(no_monitor=no_monitor)

            # Handle results
            if isinstance(result, BenchmarkResult):
                res_dict = result.to_dict()
                click.echo("\nBenchmark Results:")
                for k, val in res_dict["metrics"].items():
                    if isinstance(val, int | float):
                        click.echo(f"  {k}: {float(val):.3f}")
                    else:
                        click.echo(f"  {k}: {val}")

                # Display power summary if available
                p_res = res_dict.get("power", {})
                if p_res and p_res.get("avg_power_w", 0) > 0:
                    click.echo("\nPower & Energy Summary:")
                    click.echo(f"  Avg Power: {p_res['avg_power_w']:.2f} W")
                    click.echo(f"  Total Energy: {p_res['total_energy_j']:.2f} J")
                    click.echo(f"  Duration: {p_res['duration_s']:.2f} s")

                if output:
                    with open(output, "w") as f:
                        json.dump(res_dict, f, indent=2)
                    click.echo(f"\nResults saved to: {output}")
            else:
                click.echo(f"Benchmark returned raw data: {result}")

        except Exception as e:
            click.echo(f"Error running benchmark {actual_name}: {e}")
            if _debug:
                import traceback

                traceback.print_exc()
            sys.exit(1)
