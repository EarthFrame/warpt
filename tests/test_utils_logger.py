"""Tests for the centralized logging utility."""

from io import StringIO

import pytest

from warpt.utils.logger import Logger, LoggerNotConfiguredError


def test_logger_unconfigured():
    """Test that using Logger before configuration raises error."""
    # Reset logger state for test
    Logger._configured = False

    with pytest.raises(LoggerNotConfiguredError):
        Logger.get("test")


def test_logger_configuration():
    """Test logger configuration."""
    output = StringIO()
    Logger.configure(level="DEBUG", output=output, timestamps=False)

    assert Logger.is_configured()

    log = Logger.get("test_config")
    log.debug("Debug message")

    content = output.getvalue()
    assert "DEBUG" in content
    assert "[warpt.test_config]" in content
    assert "Debug message" in content


def test_logger_set_level():
    """Test changing log level."""
    output = StringIO()
    Logger.configure(level="INFO", output=output, timestamps=False)

    log = Logger.get("test_level")
    log.debug("Hidden")
    assert "Hidden" not in output.getvalue()

    Logger.set_level("DEBUG")
    log.debug("Visible")
    assert "Visible" in output.getvalue()
