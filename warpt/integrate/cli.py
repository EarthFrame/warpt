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
    help=("Path to SDK docs (local dir, PDF, git URL, " "or web URL)"),
)
@click.option(
    "--vendor-context",
    default=None,
    help="Optional notes about the hardware",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=("Validate setup and show summary without " "starting the agent session"),
)
@click.pass_context
def integrate(
    ctx: click.Context,
    vendor: str | None,
    sdk_docs: str | None,
    vendor_context: str | None,
    dry_run: bool,
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
        _run_init(vendor, sdk_docs, vendor_context, dry_run)
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
    dry_run: bool = False,
) -> None:
    """Load docs, optionally preview, and start integration."""
    from warpt.integrate.agent import run_init
    from warpt.integrate.system_prompt import build_system_prompt

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
        f"Vendor name is '{vendor}'. " "Press Enter to confirm or type a new name",
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
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e

    from warpt.models.constants import MAX_SDK_TOKENS

    sdk_token_estimate = len(sdk_docs_text) // 4

    if sdk_token_estimate > MAX_SDK_TOKENS:
        raise click.ClickException(
            f"SDK docs are ~{sdk_token_estimate:,} tokens, "
            f"which exceeds the {MAX_SDK_TOKENS:,} token "
            "limit.\n\n"
            "To fix this, point --sdk-docs at a smaller "
            "subdirectory — typically the Python bindings "
            "or API reference folder.\n\n"
            "Example:\n"
            "  warpt integrate --vendor myvendor "
            "--sdk-docs ./sdk-repo/python/\n\n"
            "Tip: The agent only needs the SDK's Python "
            "interface (function signatures, enums, types)."
        )

    click.echo(f"Loaded ~{sdk_token_estimate:,} tokens " "of documentation.")

    # Build system prompt early so we can report its size
    system_prompt = build_system_prompt(
        vendor=vendor,
        sdk_docs_text="",
        vendor_context=vendor_context,
    )
    prompt_token_estimate = len(system_prompt) // 4
    total_tokens = sdk_token_estimate + prompt_token_estimate

    # Count SDK doc source files
    sdk_file_count = _count_sdk_files(sdk_docs)

    # --- Dry-run summary ---
    if dry_run:
        click.echo("\n" + "=" * 50)
        click.echo("DRY RUN SUMMARY")
        click.echo("=" * 50)
        click.echo(
            f"  SDK docs:      ~{sdk_token_estimate:,} tokens "
            f"from {sdk_file_count} file(s)"
        )
        click.echo(f"  System prompt: ~{prompt_token_estimate:,} " "tokens")
        click.echo(f"  Total context: ~{total_tokens:,} tokens " "(limit ~200k)")
        click.echo(
            f"\n  Files to generate:\n"
            f"    warpt/backends/{vendor}.py\n"
            f"    warpt/backends/power/"
            f"{vendor}_power.py\n"
            f"    tests/test_{vendor}_backend.py\n"
            f"    questions.yaml"
        )
        click.echo(f"\n  Branch: backend/{vendor}")
        click.echo("=" * 50)

        try:
            click.prompt(
                "\nPress Enter to start the agent, " "or Ctrl+C to abort",
                default="",
                show_default=False,
            )
        except (KeyboardInterrupt, click.Abort):
            click.echo("\nAborted. Nothing was created.")
            return

    run_init(
        vendor=vendor,
        sdk_docs_text=sdk_docs_text,
        vendor_context=vendor_context,
        system_prompt=system_prompt,
    )


def _count_sdk_files(sdk_docs: str) -> int:
    """Count the number of source files in an SDK docs path.

    Parameters
    ----------
    sdk_docs : str
        Path or URL passed to --sdk-docs.

    Returns
    -------
    int
        Number of files, or 1 for non-directory sources.
    """
    from pathlib import Path

    path = Path(sdk_docs)
    if path.is_dir():
        return sum(1 for f in path.rglob("*") if f.is_file())
    if path.is_file():
        return 1
    # URL or other source — can't count files
    return 1


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

    resolved = _resolve_vendor(vendor or ctx.obj.get("vendor"))
    if not resolved:
        sessions = list_sessions()
        if not sessions:
            raise click.ClickException("No active integrations found.")
        raise click.ClickException(
            "Multiple active integrations. " "Specify with --vendor."
        )

    run_validate(vendor=resolved)


@integrate.command()
@click.option(
    "--vendor",
    default=None,
    help="Vendor to reset",
)
@click.pass_context
def reset(ctx: click.Context, vendor: str | None):
    """Reset a vendor integration, deleting all generated files."""
    from pathlib import Path

    from warpt.integrate.agent import _REPO_ROOT, _git_run
    from warpt.integrate.session import (
        delete_session,
        load_session,
        session_exists,
    )

    resolved = _resolve_vendor(vendor or ctx.obj.get("vendor"))
    if not resolved:
        sessions = list_sessions()
        if not sessions:
            raise click.ClickException("No active integrations found.")
        raise click.ClickException(
            "Multiple active integrations. " "Specify with --vendor."
        )

    branch_name = f"backend/{resolved}"

    # Collect files that would be affected
    vendor_files = [
        Path("warpt") / "backends" / f"{resolved}.py",
        Path("warpt") / "backends" / "power" / f"{resolved}_power.py",
        Path("tests") / f"test_{resolved}_backend.py",
        Path("questions.yaml"),
    ]

    existing_files = [f for f in vendor_files if (_REPO_ROOT / f).exists()]

    # Load parent branch from session metadata
    parent_branch = None
    if session_exists(resolved):
        try:
            _, metadata = load_session(resolved)
            parent_branch = metadata.get("parent_branch")
        except FileNotFoundError:
            pass

    # Show what will be affected
    click.echo(f"\nThis will reset the '{resolved}' integration:")
    click.echo(f"  Branch: {branch_name} (will be deleted)")
    if existing_files:
        click.echo("  Files on branch:")
        for f in existing_files:
            click.echo(f"    {f}")
    if session_exists(resolved):
        click.echo(f"  Session: ~/.warpt/integrate/{resolved}/")
    if parent_branch:
        click.echo(f"  Will switch back to: {parent_branch}")
    click.echo()

    confirmation = click.prompt(
        'Type "reset" to confirm',
        default="",
        show_default=False,
    )
    if confirmation.strip().lower() != "reset":
        click.echo("Aborted.")
        return

    # 1. Delete vendor files before switching branches
    #    (uncommitted files would otherwise carry over)
    for f in vendor_files:
        full_path = _REPO_ROOT / f
        if full_path.exists():
            full_path.unlink()
            click.echo(f"Deleted {f}")

    # 2. Switch to parent branch before deleting
    current = _git_run("branch", "--show-current")
    current_branch = current.stdout.strip()

    if current_branch == branch_name:
        if parent_branch:
            target = parent_branch
        else:
            # No saved parent — ask the user
            target = click.prompt(
                "Which branch should we switch to?",
                default="main",
            ).strip()

        result = _git_run("checkout", target)
        if result.returncode != 0:
            raise click.ClickException(
                f"Could not checkout '{target}': "
                f"{result.stderr.strip()}\n"
                "Checkout a different branch manually, "
                "then re-run reset."
            )
        click.echo(f"Switched to branch '{target}'")

    # 3. Delete the vendor branch
    branch_exists = _git_run("branch", "--list", branch_name)
    if branch_name in branch_exists.stdout:
        result = _git_run("branch", "-D", branch_name)
        if result.returncode == 0:
            click.echo(f"Deleted branch '{branch_name}'")
        else:
            click.echo(
                f"Warning: could not delete branch "
                f"'{branch_name}': {result.stderr.strip()}"
            )
    else:
        click.echo(f"Branch '{branch_name}' does not exist, " "skipping.")

    # 4. Delete session data
    if delete_session(resolved):
        click.echo(f"Deleted session data for '{resolved}'")

    click.echo(f"\nReset complete for '{resolved}'.")
