"""ER setup wizard — interactive configuration for the intelligence layer."""

from __future__ import annotations

import click

from warpt.daemon.config import load_config, save_config


def er_wizard(warpt_dir: str) -> None:
    """Run the interactive ER intelligence setup wizard.

    Detects Ollama, lists available models, and lets the user pick
    models for Chart Nurse and Attending agents.

    Parameters
    ----------
    warpt_dir
        Path to the warpt data directory (e.g. ``~/.warpt``).
    """
    config = load_config(warpt_dir)

    click.echo("--- warpt ER Intelligence Setup ---\n")

    # Detect Ollama
    import importlib.util

    if importlib.util.find_spec("requests") is None:
        click.echo(
            "The 'requests' package is required for intelligence features.\n"
            "Install it with: pip install requests"
        )
        return

    ollama_url = config.get("ollama_url", "http://localhost:11434")
    click.echo(f"Checking Ollama at {ollama_url} ...")

    models = _get_installed_models(ollama_url)
    if models is None:
        click.echo(
            f"\nCould not reach Ollama at {ollama_url}.\n"
            "Make sure Ollama is running: ollama serve"
        )
        if not click.confirm("Continue setup anyway?", default=False):
            return
        models = []

    if models:
        click.echo(f"\nFound {len(models)} installed model(s):")
        for i, m in enumerate(models, 1):
            click.echo(f"  {i}. {m}")
    else:
        click.echo("\nNo models found (or Ollama unreachable).")
        click.echo("You can install models later: ollama pull llama3:8b")

    # Chart Nurse model selection
    label = "installed" if models else "default"
    click.echo(
        f"\nChart Nurse model ({label}: {config['models']['chart_nurse']})"
    )
    chart_model = _prompt_model_choice(models, config["models"]["chart_nurse"])
    config["models"]["chart_nurse"] = chart_model

    # Attending model selection
    click.echo(
        f"\nAttending model ({label}: {config['models']['attending']})"
    )
    attending_model = _prompt_model_choice(models, config["models"]["attending"])
    config["models"]["attending"] = attending_model

    # Enable intelligence
    config["intelligence_enabled"] = True

    save_config(warpt_dir, config)
    click.echo("\nIntelligence enabled. Config saved.")
    click.echo(f"  Chart Nurse model: {chart_model}")
    click.echo(f"  Attending model:   {attending_model}")
    if not models:
        click.echo(
            "\nNote: Models are not yet installed. "
            "Pull them with Ollama before starting the daemon:"
        )
        click.echo(f"  ollama pull {chart_model}")
        if attending_model != chart_model:
            click.echo(f"  ollama pull {attending_model}")


def _get_installed_models(ollama_url: str) -> list[str] | None:
    """Query Ollama for installed models.

    Returns
    -------
        List of model names, or ``None`` if Ollama is unreachable.
    """
    import requests

    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.RequestException:
        return None


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
