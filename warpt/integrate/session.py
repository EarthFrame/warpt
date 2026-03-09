"""Session persistence for the integration agent.

Stores Agent SDK session IDs so that `warpt integrate iterate`
can resume a previous agent session.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Session data stored under ~/.warpt/integrate/<vendor>/
SESSION_DIR = Path.home() / ".warpt" / "integrate"


def _vendor_dir(vendor: str) -> Path:
    """Get the session directory for a vendor."""
    return SESSION_DIR / vendor.lower()


def save_session(
    vendor: str,
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist an agent session ID and metadata.

    Parameters
    ----------
    vendor : str
        Vendor name (e.g., "tenstorrent").
    session_id : str
        Agent SDK session ID.
    metadata : dict[str, Any] | None
        Extra metadata (creation time, pass count, etc.).

    Returns
    -------
    Path
        Path to the session directory.
    """
    vdir = _vendor_dir(vendor)
    vdir.mkdir(parents=True, exist_ok=True)

    # Write session ID
    (vdir / "session_id.txt").write_text(session_id)

    # Build and write metadata
    meta = metadata or {}
    meta.setdefault("vendor", vendor)
    meta.setdefault(
        "created",
        datetime.now(UTC).isoformat(),
    )
    meta.setdefault("pass_count", 1)
    meta["last_run"] = datetime.now(UTC).isoformat()

    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return vdir


def load_session(vendor: str) -> tuple[str, dict[str, Any]]:
    """Load a saved session ID and metadata.

    Parameters
    ----------
    vendor : str
        Vendor name.

    Returns
    -------
    tuple[str, dict[str, Any]]
        (session_id, metadata_dict)

    Raises
    ------
    FileNotFoundError
        If no session exists for this vendor.
    """
    vdir = _vendor_dir(vendor)
    sid_path = vdir / "session_id.txt"
    meta_path = vdir / "metadata.json"

    if not sid_path.exists():
        raise FileNotFoundError(
            f"No integration session found for vendor "
            f"{vendor!r}. Run 'warpt integrate init' first."
        )

    session_id = sid_path.read_text().strip()
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    return session_id, metadata


def session_exists(vendor: str) -> bool:
    """Check if a session exists for this vendor."""
    return (_vendor_dir(vendor) / "session_id.txt").exists()


def list_sessions() -> list[str]:
    """List all vendors with active sessions.

    Returns
    -------
    list[str]
        Vendor names with existing sessions.
    """
    if not SESSION_DIR.exists():
        return []
    return [
        d.name
        for d in sorted(SESSION_DIR.iterdir())
        if d.is_dir() and (d / "session_id.txt").exists()
    ]


def delete_session(vendor: str) -> bool:
    """Delete all session data for a vendor.

    Parameters
    ----------
    vendor : str
        Vendor name.

    Returns
    -------
    bool
        True if session data was deleted, False if none existed.
    """
    import shutil

    vdir = _vendor_dir(vendor)
    if not vdir.exists():
        return False
    shutil.rmtree(vdir)
    return True


def increment_pass(vendor: str) -> int:
    """Increment the pass counter and update last_run.

    Returns
    -------
    int
        The new pass number.
    """
    _, metadata = load_session(vendor)
    metadata["pass_count"] = metadata.get("pass_count", 1) + 1
    metadata["last_run"] = datetime.now(UTC).isoformat()

    vdir = _vendor_dir(vendor)
    (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return int(metadata["pass_count"])
