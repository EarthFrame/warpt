"""API key resolution for LLM providers.

Keys are never stored in ``config.yaml`` — they are resolved at runtime from
an environment variable (``api_key_env``) or a secret file (``api_key_file``).
An inline ``api_key`` in config is refused outright.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from warpt.daemon.llm.base import LLMPermanentError

REDACTED = "[redacted]"


def resolve_api_key(provider_name: str, provider_cfg: dict[str, Any]) -> str | None:
    """Resolve the API key for a provider config entry.

    Parameters
    ----------
    provider_name
        Provider name from config (for error messages).
    provider_cfg
        The provider's config dict. Reads ``api_key_env`` or ``api_key_file``.

    Returns
    -------
        The key, or ``None`` when the config names no key source.

    Raises
    ------
    LLMPermanentError
        If an inline ``api_key`` is present, or a named source is empty or
        missing.
    """
    if "api_key" in provider_cfg:
        raise LLMPermanentError(
            f"LLM provider {provider_name!r} has an inline api_key in config. "
            "Keys must never be stored in config.yaml — use api_key_env "
            "(environment variable name) or api_key_file (path) instead."
        )

    env_var = provider_cfg.get("api_key_env")
    if env_var:
        key = os.environ.get(env_var, "").strip()
        if not key:
            raise LLMPermanentError(
                f"LLM provider {provider_name!r}: environment variable "
                f"{env_var!r} is not set (or empty)."
            )
        return key

    key_file = provider_cfg.get("api_key_file")
    if key_file:
        path = Path(key_file).expanduser()
        try:
            key = path.read_text().strip()
        except OSError as e:
            raise LLMPermanentError(
                f"LLM provider {provider_name!r}: cannot read api_key_file "
                f"{key_file!r}: {e}"
            ) from e
        if not key:
            raise LLMPermanentError(
                f"LLM provider {provider_name!r}: api_key_file {key_file!r} is empty."
            )
        return key

    return None


def redact(value: str | None) -> str:
    """Return a safe placeholder for *value* in logs and error messages."""
    if not value:
        return "(none)"
    return REDACTED
