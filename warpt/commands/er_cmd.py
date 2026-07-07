"""ER setup wizard — interactive configuration for the intelligence layer.

Detects available LLM backends (Claude API key, local Ollama, an optional
OpenAI-compatible cluster), lets the user assign a provider per agent, and
writes the ``llm`` config block. API keys are never prompted for or written
to disk — the wizard only records which environment variable holds them.
"""

from __future__ import annotations

import os
from typing import Any

import click

from warpt.daemon.config import load_config, save_config

ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHART_MODEL = "llama3:8b"
DEFAULT_ATTENDING_MODEL = "llama3:70b"


def er_wizard(warpt_dir: str) -> None:
    """Run the interactive ER intelligence setup wizard.

    Parameters
    ----------
    warpt_dir
        Path to the warpt data directory (e.g. ``~/.warpt``).
    """
    config = load_config(warpt_dir)

    click.echo("--- warpt ER Intelligence Setup ---\n")

    import importlib.util

    if importlib.util.find_spec("requests") is None:
        click.echo(
            "The 'requests' package is required for intelligence features.\n"
            "Install it with: pip install requests"
        )
        return

    providers: dict[str, Any] = {}

    # --- Claude (Anthropic API) ---
    claude_available = _setup_claude(providers)

    # --- Local Ollama ---
    ollama_available = _setup_ollama(config, providers)

    # --- OpenAI-compatible cluster ---
    _setup_cluster(providers)

    if not providers:
        click.echo(
            "\nNo LLM backend configured. Set ANTHROPIC_API_KEY, start Ollama "
            "(ollama serve), or provide a cluster URL, then re-run: "
            "warpt daemon er"
        )
        return

    # --- Per-agent assignment ---
    # Cheap/fast triage on local when present; high-quality diagnosis on
    # Claude when a key is available.
    chart_default = "local" if ollama_available else next(iter(providers))
    attending_default = "claude" if claude_available else next(iter(providers))

    click.echo(f"\nConfigured providers: {', '.join(providers)}")
    chart_provider = _prompt_provider(providers, "Chart Nurse", chart_default)
    attending_provider = _prompt_provider(providers, "Attending", attending_default)

    config["llm"] = {
        "default": attending_provider,
        "providers": providers,
        "agents": {
            "chart_nurse": {"provider": chart_provider},
            "attending": {"provider": attending_provider},
        },
    }
    config["intelligence_enabled"] = True

    save_config(warpt_dir, config)
    click.echo("\nIntelligence enabled. Config saved.")
    click.echo(
        f"  Chart Nurse: {chart_provider} ({providers[chart_provider]['model']})"
    )
    click.echo(
        f"  Attending:   {attending_provider} "
        f"({providers[attending_provider]['model']})"
    )
    if "claude" in (chart_provider, attending_provider):
        click.echo(
            f"\nNote: the Claude API key is read from ${ANTHROPIC_KEY_ENV} at "
            "runtime — it is never stored in config.yaml. Make sure the "
            "daemon's environment has it set."
        )


def _setup_claude(providers: dict[str, Any]) -> bool:
    """Offer the Claude provider when an API key is present in the env."""
    key_present = bool(os.environ.get(ANTHROPIC_KEY_ENV, "").strip())
    if key_present:
        click.echo(f"Found ${ANTHROPIC_KEY_ENV} in the environment.")
        if click.confirm("Use the Claude API as a provider?", default=True):
            model = click.prompt("  Claude model", default=DEFAULT_CLAUDE_MODEL)
            providers["claude"] = {
                "type": "anthropic",
                "model": model,
                "api_key_env": ANTHROPIC_KEY_ENV,
            }
            return True
    else:
        click.echo(
            f"No ${ANTHROPIC_KEY_ENV} found — skipping the Claude API. "
            "(Set it and re-run to enable.)"
        )
    return False


def _setup_ollama(config: dict[str, Any], providers: dict[str, Any]) -> bool:
    """Detect Ollama and configure the local provider."""
    from warpt.daemon.llm.providers.ollama import get_installed_models

    ollama_url = config.get("ollama_url", DEFAULT_OLLAMA_URL)
    click.echo(f"\nChecking Ollama at {ollama_url} ...")

    models = get_installed_models(ollama_url)
    if models is None:
        click.echo(
            f"Could not reach Ollama at {ollama_url} " "(start it with: ollama serve)."
        )
        if not click.confirm(
            "Configure a local Ollama provider anyway?", default=False
        ):
            return False
        models = []

    if models:
        click.echo(f"Found {len(models)} installed model(s):")
        for i, m in enumerate(models, 1):
            click.echo(f"  {i}. {m}")
    else:
        click.echo("No models found. You can pull one later: ollama pull llama3:8b")

    model = _prompt_model_choice(models, DEFAULT_CHART_MODEL)
    providers["local"] = {"type": "ollama", "url": ollama_url, "model": model}
    if models and model not in models:
        click.echo(f"Note: pull the model before starting: ollama pull {model}")
    return True


def _setup_cluster(providers: dict[str, Any]) -> bool:
    """Optionally configure an OpenAI-compatible inference cluster."""
    if not click.confirm(
        "\nConfigure a self-hosted inference cluster (OpenAI-compatible)?",
        default=False,
    ):
        return False

    url = click.prompt("  Cluster base URL (e.g. http://inference.internal/v1)")
    model = click.prompt("  Model name", default=DEFAULT_ATTENDING_MODEL)

    reachable, detail = _test_cluster(url)
    if reachable:
        click.echo(f"  Cluster reachable. {detail}")
    else:
        click.echo(f"  Warning: could not reach the cluster ({detail}).")
        if not click.confirm("  Keep this cluster config anyway?", default=True):
            return False

    entry: dict[str, Any] = {"type": "openai_compat", "url": url, "model": model}
    key_env = click.prompt(
        "  Env var holding the cluster API key (blank if none)",
        default="",
        show_default=False,
    ).strip()
    if key_env:
        entry["api_key_env"] = key_env
    providers["cluster"] = entry
    return True


def _test_cluster(url: str) -> tuple[bool, str]:
    """Probe ``GET {url}/models`` for connectivity."""
    import requests

    try:
        resp = requests.get(f"{url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        names = [m.get("id", "?") for m in data[:5]]
        return True, f"Models: {', '.join(names)}" if names else "No models listed."
    except requests.RequestException as e:
        return False, str(e)


def _prompt_provider(providers: dict[str, Any], agent_label: str, default: str) -> str:
    """Prompt for the provider assigned to an agent."""
    return click.prompt(
        f"  Provider for {agent_label}",
        type=click.Choice(sorted(providers)),
        default=default,
    )


def _prompt_model_choice(available: list[str], default: str) -> str:
    """Prompt the user to pick a model from the available list.

    Parameters
    ----------
    available
        List of installed Ollama model names.
    default
        Current/default model name.

    Returns
    -------
        Selected model name.
    """
    if available:
        choice = click.prompt(
            f"  Enter model name or number (1-{len(available)})",
            default=default,
        )
        # If they typed a number, look it up
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx]
        except ValueError:
            pass
        return choice
    else:
        return click.prompt("  Enter model name", default=default)
