"""Tests for the environment variable utility."""

import os

import pytest

from warpt.utils.env import (
    EnvVarNotSetError,
    EnvVarTypeError,
    get_env,
    require_env,
    set_env,
    unset_env,
)


def test_get_env_basic():
    """Test getting set variables and missing variables with defaults."""
    os.environ["WARPT_TEST_VAR"] = "test_value"
    assert get_env("WARPT_TEST_VAR") == "test_value"
    assert get_env("WARPT_MISSING_VAR", default="default") == "default"
    assert get_env("WARPT_MISSING_VAR") is None
    del os.environ["WARPT_TEST_VAR"]


def test_get_env_coercion():
    """Test type coercion for common types."""
    os.environ["WARPT_BOOL_TRUE"] = "true"
    os.environ["WARPT_BOOL_FALSE"] = "0"
    os.environ["WARPT_INT"] = "123"
    os.environ["WARPT_FLOAT"] = "1.23"
    os.environ["WARPT_LIST"] = "a, b, c "

    assert get_env("WARPT_BOOL_TRUE", as_type=bool) is True
    assert get_env("WARPT_BOOL_FALSE", as_type=bool) is False
    assert get_env("WARPT_INT", as_type=int) == 123
    assert get_env("WARPT_FLOAT", as_type=float) == 1.23
    assert get_env("WARPT_LIST", as_type=list) == ["a", "b", "c"]

    # Test coercion failure
    os.environ["WARPT_INVALID_INT"] = "not_an_int"
    with pytest.raises(EnvVarTypeError):
        get_env("WARPT_INVALID_INT", as_type=int)

    # Cleanup
    for var in [
        "WARPT_BOOL_TRUE",
        "WARPT_BOOL_FALSE",
        "WARPT_INT",
        "WARPT_FLOAT",
        "WARPT_LIST",
        "WARPT_INVALID_INT",
    ]:
        if var in os.environ:
            del os.environ[var]


def test_require_env():
    """Test getting required variables."""
    os.environ["WARPT_REQUIRED"] = "exists"
    assert require_env("WARPT_REQUIRED") == "exists"

    with pytest.raises(EnvVarNotSetError):
        require_env("WARPT_NON_EXISTENT")

    del os.environ["WARPT_REQUIRED"]


def test_set_unset_env():
    """Test setting and unsetting environment variables."""
    # Test setting
    set_env("WARPT_SET_TEST", "new_value")
    assert os.environ["WARPT_SET_TEST"] == "new_value"

    # Test overwrite=False
    set_env("WARPT_SET_TEST", "ignored", overwrite=False)
    assert os.environ["WARPT_SET_TEST"] == "new_value"

    # Test unsetting
    assert unset_env("WARPT_SET_TEST") is True
    assert "WARPT_SET_TEST" not in os.environ
    assert unset_env("WARPT_SET_TEST") is False
