"""Environment variable helpers with type coercion, validation, and logging.

Usage:
    from warpt.utils.env import get_env, set_env, require_env

    # Basic usage
    debug = get_env("DEBUG", default=False, as_type=bool)
    port = get_env("PORT", default=8080, as_type=int)

    # Required (raises if not set)
    api_key = require_env("API_KEY")

    # With logging
    gpu_id = get_env("CUDA_VISIBLE_DEVICES", default="0", log=True)

    # Set with validation
    set_env("WARPT_LOG_LEVEL", "DEBUG", log=True)
"""

from __future__ import annotations

import os
from typing import Any, TypeVar, cast, overload

T = TypeVar("T")


class EnvVarError(Exception):
    """Base exception for environment variable errors."""

    pass


class EnvVarNotSetError(EnvVarError):
    """Raised when a required environment variable is not set."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Required environment variable not set: {name}")


class EnvVarTypeError(EnvVarError):
    """Raised when an environment variable cannot be converted to the expected type."""

    def __init__(self, name: str, value: str, expected_type: type) -> None:
        self.name = name
        self.value = value
        self.expected_type = expected_type
        super().__init__(f"Cannot convert {name}='{value}' to {expected_type.__name__}")


class EnvVarSetError(EnvVarError):
    """Raised when setting an environment variable fails."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        super().__init__(f"Failed to set environment variable {name}: {reason}")


def _coerce_type(name: str, value: str, as_type: type) -> Any:
    """Convert a string value to the specified type.

    Args:
        name: Variable name (for error messages).
        value: String value to convert.
        as_type: Target type.

    Returns:
        Converted value.

    Raises:
        EnvVarTypeError: If conversion fails.
    """
    try:
        # Handle bool specially - "false", "0", "" are False
        if as_type is bool:
            return value.lower() not in ("false", "0", "", "no", "off")

        # Handle common types
        if as_type is int:
            return int(value)
        if as_type is float:
            return float(value)
        if as_type is str:
            return value

        # Handle list[str] as comma-separated
        origin = getattr(as_type, "__origin__", None)
        if as_type is list or origin is list:
            return [item.strip() for item in value.split(",") if item.strip()]

        # Try direct conversion for other types
        return as_type(value)

    except (ValueError, TypeError) as e:
        raise EnvVarTypeError(name, value, as_type) from e


def _log_access(
    action: str, name: str, value: str | None, masked: bool = False
) -> None:
    """Log environment variable access if logger is configured.

    Args:
        action: "get" or "set".
        name: Variable name.
        value: Variable value (may be masked).
        masked: If True, mask the value in logs.
    """
    try:
        from warpt.utils.logger import Logger

        if not Logger.is_configured():
            return

        display_value = "***" if masked else value
        if action == "get":
            Logger.get("env").debug(f"ENV GET {name}={display_value}")
        else:
            Logger.get("env").debug(f"ENV SET {name}={display_value}")
    except Exception:
        # Don't fail if logging isn't available
        pass


# Overloads for proper type hints
@overload
def get_env(name: str, *, default: T, as_type: type[T], log: bool = ...) -> T:
    ...


@overload
def get_env(name: str, *, default: T, log: bool = ...) -> T:
    ...


@overload
def get_env(name: str, *, as_type: type[T], log: bool = ...) -> T | None:
    ...


@overload
def get_env(name: str, *, log: bool = ...) -> str | None:
    ...


def get_env(
    name: str,
    *,
    default: T | None = None,
    as_type: type[T] | None = None,
    log: bool = False,
    mask_in_log: bool = False,
) -> T | str | None:
    """Get an environment variable with optional type coercion.

    Args:
        name: Environment variable name.
        default: Default value if not set. If provided, this value is returned
            when the variable is not set (never raises for missing var).
        as_type: Type to convert the value to. Supports:
            - bool: "false", "0", "", "no", "off" → False, else True
            - int, float, str: Direct conversion
            - list: Comma-separated string → list of strings
        log: If True, log the access (uses Logger if configured).
        mask_in_log: If True, mask the value in logs (for secrets).

    Returns:
        The environment variable value, converted to as_type if specified,
        or default if not set.

    Raises:
        EnvVarTypeError: If as_type is specified and conversion fails.

    Examples:
        >>> get_env("DEBUG", default=False, as_type=bool)
        False
        >>> get_env("PORT", default=8080, as_type=int)
        8080
        >>> get_env("HOSTS", default=["localhost"], as_type=list)
        ["localhost"]
        >>> get_env("API_KEY", log=True, mask_in_log=True)
        None
    """
    value = os.environ.get(name)

    if log:
        _log_access("get", name, value, masked=mask_in_log)

    if value is None:
        return default

    if as_type is not None:
        return cast(T, _coerce_type(name, value, as_type))

    return value


def require_env(
    name: str,
    *,
    as_type: type[T] | None = None,
    log: bool = False,
    mask_in_log: bool = False,
) -> T | str:
    """Get a required environment variable (raises if not set).

    Args:
        name: Environment variable name.
        as_type: Type to convert the value to (see get_env for supported types).
        log: If True, log the access.
        mask_in_log: If True, mask the value in logs.

    Returns:
        The environment variable value, converted to as_type if specified.

    Raises:
        EnvVarNotSetError: If the variable is not set.
        EnvVarTypeError: If as_type is specified and conversion fails.

    Examples:
        >>> require_env("API_KEY")
        "sk-..."
        >>> require_env("GPU_COUNT", as_type=int)
        4
    """
    value = os.environ.get(name)

    if value is None:
        raise EnvVarNotSetError(name)

    if log:
        _log_access("get", name, value, masked=mask_in_log)

    if as_type is not None:
        return cast(T, _coerce_type(name, value, as_type))

    return value


def set_env(
    name: str,
    value: str | int | float | bool,
    *,
    log: bool = False,
    mask_in_log: bool = False,
    overwrite: bool = True,
    strict: bool = False,
) -> bool:
    """Set an environment variable.

    Args:
        name: Environment variable name.
        value: Value to set (converted to string).
        log: If True, log the operation.
        mask_in_log: If True, mask the value in logs.
        overwrite: If False, don't overwrite existing values.
        strict: If True, raise on failure instead of returning False.

    Returns:
        True if the variable was set, False if skipped (existing + overwrite=False).

    Raises:
        EnvVarSetError: If strict=True and setting fails.

    Examples:
        >>> set_env("DEBUG", True)  # Sets DEBUG="True"
        True
        >>> set_env("PORT", 8080, log=True)
        True
        >>> set_env("API_KEY", "secret", overwrite=False)  # Won't overwrite
        False
    """
    # Check if already set and overwrite is False
    if not overwrite and name in os.environ:
        return False

    # Convert value to string
    str_value = str(value)

    try:
        os.environ[name] = str_value

        if log:
            _log_access("set", name, str_value, masked=mask_in_log)

        return True

    except Exception as e:
        if strict:
            raise EnvVarSetError(name, str(e)) from e
        return False


def unset_env(name: str, *, log: bool = False, strict: bool = False) -> bool:
    """Unset an environment variable.

    Args:
        name: Environment variable name.
        log: If True, log the operation.
        strict: If True, raise if variable doesn't exist.

    Returns:
        True if unset, False if wasn't set.

    Raises:
        EnvVarNotSetError: If strict=True and variable doesn't exist.
    """
    if name not in os.environ:
        if strict:
            raise EnvVarNotSetError(name)
        return False

    del os.environ[name]

    if log:
        _log_access("set", name, "(unset)")

    return True


def env_is_set(name: str) -> bool:
    """Check if an environment variable is set (not empty).

    Args:
        name: Environment variable name.

    Returns:
        True if set and non-empty, False otherwise.
    """
    value = os.environ.get(name)
    return value is not None and value != ""
