"""CLI for the fleet control plane — run central, mint tokens."""

from __future__ import annotations

import os

import click

DEFAULT_DB = "sqlite:///~/.warpt/fleet.db"
DEFAULT_PORT = 8787


@click.group()
def fleet() -> None:
    """Fleet control plane — central ingest API and dashboard."""


@fleet.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=DEFAULT_PORT, show_default=True, type=int)
@click.option(
    "--db",
    default=DEFAULT_DB,
    show_default=True,
    help="Fleet store URL (postgresql://... for production, sqlite:/// for dev)",
)
@click.option(
    "--no-auth",
    is_flag=True,
    default=False,
    help="Disable bearer-token auth (dev only; never in production)",
)
@click.option("--ssl-keyfile", default=None, help="TLS private key (enables HTTPS)")
@click.option("--ssl-certfile", default=None, help="TLS certificate")
@click.option(
    "--ssl-ca-certs",
    default=None,
    help="CA bundle for client-certificate verification (mTLS)",
)
def serve(
    host: str,
    port: int,
    db: str,
    no_auth: bool,
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
    ssl_ca_certs: str | None,
) -> None:
    r"""Run the central control plane (ingest API + dashboard).

    \b
    Auth: set WARPT_FLEET_TOKEN in the environment (generate one with
    'warpt fleet token'); nodes present it as a bearer token.

    \b
    Examples:
      warpt fleet token                          # mint a shared token
      WARPT_FLEET_TOKEN=... warpt fleet serve    # dev (sqlite)
      warpt fleet serve --db postgresql://fleet@db/fleet \
        --ssl-keyfile key.pem --ssl-certfile cert.pem \
        --ssl-ca-certs ca.pem                    # production + mTLS
    """
    try:
        import uvicorn

        from warpt.fleet.central.app import create_app
        from warpt.fleet.central.store import FleetStore
    except ImportError as e:
        raise click.ClickException(
            f"Fleet central requires the 'fleet' extra: pip install "
            f"'warpt[fleet]' ({e})"
        ) from e

    token = os.environ.get("WARPT_FLEET_TOKEN", "").strip() or None
    if token is None and not no_auth:
        raise click.ClickException(
            "WARPT_FLEET_TOKEN is not set. Generate one with 'warpt fleet "
            "token', or pass --no-auth for local development."
        )

    if db.startswith("sqlite:///~"):
        db = "sqlite:///" + os.path.expanduser(db[len("sqlite:///") :])

    store = FleetStore(db)
    app = create_app(store, token=token)

    ssl_kwargs = {}
    if ssl_keyfile and ssl_certfile:
        ssl_kwargs["ssl_keyfile"] = ssl_keyfile
        ssl_kwargs["ssl_certfile"] = ssl_certfile
        if ssl_ca_certs:
            # Require verified client certificates (mTLS).
            ssl_kwargs["ssl_ca_certs"] = ssl_ca_certs
            ssl_kwargs["ssl_cert_reqs"] = 2  # ssl.CERT_REQUIRED

    click.echo(
        f"warpt fleet central on {host}:{port} "
        f"(db={db.split('@')[-1]}, auth={'on' if token else 'OFF'}, "
        f"tls={'on' if ssl_kwargs else 'off'})"
    )
    uvicorn.run(app, host=host, port=port, log_level="info", **ssl_kwargs)


@fleet.command()
def token() -> None:
    """Generate a fleet bearer token (set as WARPT_FLEET_TOKEN on both ends)."""
    try:
        from warpt.fleet.central.auth import generate_token
    except ImportError:
        # Token generation only needs stdlib; keep it dependency-free.
        import secrets

        click.echo(secrets.token_hex(32))
        return
    click.echo(generate_token())
