"""CLI commands for the backend integration pipeline.

Provides the `warpt integrate` command group. Running it with
no subcommand auto-detects whether to init or iterate.
"""

from __future__ import annotations

import re

import click

from warpt.integrate.loaders.base import DocLoader
from warpt.integrate.session import list_sessions


def _resolve_vendor(vendor: str | None) -> str | None:
    """Resolve vendor from flag or active sessions.

    Parameters
    ----------
    vendor : str | None
        Vendor passed via --vendor, or None.

    Returns
    -------
    str | None
        Resolved vendor name, or None if ambiguous.
    """
    if vendor:
        return vendor.lower().strip()

    sessions = list_sessions()
    if len(sessions) == 1:
        return sessions[0]
    return None


@click.group(invoke_without_command=True)
@click.option(
    "--vendor",
    default=None,
    help="Vendor name (e.g., tenstorrent, amd, qualcomm)",
)
@click.option(
    "--sdk-docs",
    default=None,
    help=(
        "Path to SDK docs (local dir, PDF, git URL, "
        "or web URL)"
    ),
)
@click.option(
    "--vendor-context",
    default=None,
    help="Optional notes about the hardware",
)
@click.pass_context
def integrate(
    ctx: click.Context,
    vendor: str | None,
    sdk_docs: str | None,
    vendor_context: str | None,
):
    r"""Integrate new hardware accelerator backends using AI.

    \b
    First run (starts a new integration):
      warpt integrate --vendor tenstorrent --sdk-docs ./docs/

    \b
    Subsequent runs (processes answered questions):
      warpt integrate
    """
    # Store vendor in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["vendor"] = vendor

    # If a subcommand was invoked, let it handle everything
    if ctx.invoked_subcommand is not None:
        return

    # --- Smart dispatch: init or iterate ---

    # Case 1: --sdk-docs provided → init mode
    if sdk_docs:
        if not vendor:
            raise click.ClickException(
                "--vendor is required when starting a new "
                "integration with --sdk-docs."
            )
        _run_init(vendor, sdk_docs, vendor_context)
        return

    # Case 2: No --sdk-docs → iterate mode
    resolved = _resolve_vendor(vendor)

    if resolved is None:
        sessions = list_sessions()
        if not sessions:
            raise click.ClickException(
                "No active integrations found.\n"
                "Start one with: warpt integrate "
                "--vendor <name> --sdk-docs <path>"
            )
        raise click.ClickException(
            "Multiple active integrations found: "
            f"{', '.join(sessions)}\n"
            "Specify which one with: warpt integrate "
            "--vendor <name>"
        )

    # Check if this vendor has a session — if not,
    # they probably forgot --sdk-docs
    from warpt.integrate.session import session_exists

    if not session_exists(resolved):
        raise click.ClickException(
            f"No existing integration for '{resolved}'.\n"
            "To start a new one, provide SDK docs:\n"
            f"  warpt integrate --vendor {resolved} "
            "--sdk-docs <path>"
        )

    _run_iterate(resolved)


# Valid vendor name: lowercase letters and underscores only
_VENDOR_PATTERN = re.compile(r"^[a-z][a-z_]*$")


def _validate_vendor_name(vendor: str) -> str:
    """Validate and normalize a vendor name.

    Parameters
    ----------
    vendor : str
        Raw vendor name input.

    Returns
    -------
    str
        Normalized lowercase vendor name.

    Raises
    ------
    click.ClickException
        If the name contains invalid characters.
    """
    vendor = vendor.lower().strip()

    if not vendor:
        raise click.ClickException("Vendor name cannot be empty.")

    if not _VENDOR_PATTERN.match(vendor):
        raise click.ClickException(
            f"Invalid vendor name: '{vendor}'\n"
            "Vendor name must be lowercase letters and "
            "underscores only (e.g., 'tenstorrent', "
            "'some_vendor').\n"
            "This name is used for file paths, class names, "
            "and Python imports."
        )

    return vendor


def _check_existing_backend(vendor: str) -> None:
    """Check if a backend already exists for this vendor."""
    from warpt.integrate.agent import _REPO_ROOT

    backends_dir = _REPO_ROOT / "warpt" / "backends"

    # Build a case-insensitive map of existing backend files
    existing = {
        f.stem.lower(): f.name
        for f in backends_dir.glob("*.py")
        if f.stem not in ("__init__", "base", "factory")
    }

    match = existing.get(vendor.lower())
    if match:
        raise click.ClickException(
            f"Backend already exists: "
            f"warpt/backends/{match}\n"
            "This vendor has an existing implementation. "
            "If you want to regenerate it, "
            "remove the file first."
        )


def _run_init(
    vendor: str,
    sdk_docs: str,
    vendor_context: str | None,
) -> None:
    """Load docs and start a new integration."""
    from warpt.integrate.agent import run_init

    vendor = _validate_vendor_name(vendor)

    # Check for existing backend before doing anything
    _check_existing_backend(vendor)

    # Confirm vendor name before starting
    click.echo(
        f"\nThis will create the following files:\n"
        f"  warpt/backends/{vendor}.py\n"
        f"  warpt/backends/power/{vendor}_power.py\n"
        f"  tests/test_{vendor}_backend.py\n"
        f"  Class name: "
        f"{vendor.capitalize()}Backend\n"
    )
    confirmed = click.prompt(
        f"Vendor name is '{vendor}'. "
        "Press Enter to confirm or type a new name",
        default=vendor,
        show_default=False,
    ).strip()

    if confirmed != vendor:
        vendor = _validate_vendor_name(confirmed)
        click.echo(f"Updated vendor name to: {vendor}")
        # Re-check with the corrected name
        _check_existing_backend(vendor)

    click.echo(f"Loading SDK documentation from: {sdk_docs}")
    try:
        sdk_docs_text = DocLoader.detect_and_load(sdk_docs)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e

    token_estimate = len(sdk_docs_text) // 4
    click.echo(
        f"Loaded ~{token_estimate:,} tokens of documentation."
    )

    run_init(
        vendor=vendor,
        sdk_docs_text=sdk_docs_text,
        vendor_context=vendor_context,
    )


def _run_iterate(vendor: str) -> None:
    """Run an iteration pass for a vendor."""
    from warpt.integrate.agent import run_iterate

    run_iterate(vendor=vendor)


@integrate.command()
@click.option(
    "--vendor",
    default=None,
    help="Filter by vendor",
)
@click.pass_context
def status(ctx: click.Context, vendor: str | None):
    """Show current state of the questions document."""
    from warpt.integrate.agent import run_status

    vendor = _resolve_vendor(vendor or ctx.obj.get("vendor"))
    run_status(vendor=vendor)


@integrate.command()
@click.option(
    "--vendor",
    default=None,
    help="Vendor to validate",
)
@click.pass_context
def validate(ctx: click.Context, vendor: str | None):
    """Run tests and linting on generated backend code."""
    from warpt.integrate.agent import run_validate

    resolved = _resolve_vendor(
        vendor or ctx.obj.get("vendor")
    )
    if not resolved:
        sessions = list_sessions()
        if not sessions:
            raise click.ClickException(
                "No active integrations found."
            )
        raise click.ClickException(
            "Multiple active integrations. "
            "Specify with --vendor."
        )

    run_validate(vendor=resolved)
