"""Bearer-token auth for the central API.

Transport encryption (TLS) and mutual TLS are terminated at the deployment
layer — run uvicorn with ``--ssl-keyfile/--ssl-certfile`` (and
``--ssl-ca-certs`` + ``--ssl-cert-reqs 2`` for mTLS) or front with a proxy.
In-app auth is a shared bearer token from the environment; per-node tokens
and SSO/OIDC for the dashboard land with Phase 4/5 hardening.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


def generate_token() -> str:
    """Return a new random fleet token (hex, 32 bytes)."""
    return secrets.token_hex(32)


class BearerAuth:
    """FastAPI dependency enforcing a shared bearer token.

    Parameters
    ----------
    token
        The expected token; ``None`` disables auth (explicit dev opt-in).
    """

    def __init__(self, token: str | None) -> None:
        self._token = token

    def __call__(self, request: Request) -> None:
        """Reject the request unless it carries the expected token."""
        if self._token is None:
            return
        header = request.headers.get("Authorization", "")
        expected = f"Bearer {self._token}"
        if not secrets.compare_digest(header, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")
