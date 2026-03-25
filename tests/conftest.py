"""Shared test fixtures."""

from warpt.utils.logger import Logger


def pytest_configure(config):
    """Configure Logger once for the entire test session."""
    if not Logger.is_configured():
        Logger.configure(level="WARNING")
