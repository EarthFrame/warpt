"""Centralized logging for warpt.

Simple, explicit logging that requires configuration before use.

Usage:
    from warpt.utils.logger import Logger

    # Configure once at startup (required before any logging)
    Logger.configure(level="INFO", timestamps=True)

    # Get a logger anywhere in the codebase
    log = Logger.get("stress.GPUMatMulTest")
    log.info("Starting test...")

    # Or log directly
    Logger.info("stress", "Starting test...")
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import TextIO


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert to Python logging level."""
        level: int = getattr(logging, self.value)
        return level


class LoggerNotConfiguredError(Exception):
    """Raised when trying to use Logger before calling Logger.configure()."""

    def __init__(self) -> None:
        super().__init__(
            "Logger not configured. Call Logger.configure() at application startup."
        )


class Logger:
    """Centralized logging for warpt.

    Must be configured once before use. Attempting to log before configuration
    raises LoggerNotConfiguredError.

    Example:
        >>> # At application startup
        >>> Logger.configure(level="INFO")
        >>>
        >>> # Anywhere in the codebase
        >>> log = Logger.get("stress.GPUMatMulTest")
        >>> log.info("Running test...")
    """

    _configured: bool = False
    _root_name: str = "warpt"

    @classmethod
    def configure(
        cls,
        level: str | LogLevel = "INFO",
        output: str | Path | TextIO | None = None,
        timestamps: bool = True,
        include_location: bool = False,
        format_string: str | None = None,
    ) -> None:
        """Configure the logger. Must be called before any logging.

        Args:
            level: Log level - "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
                or a LogLevel enum value.
            output: Where to send logs:
                - None: stdout (default)
                - "stderr": sys.stderr
                - str/Path: File path
                - TextIO: Any file-like object
            timestamps: Include timestamps in messages (default True).
            include_location: Include [filename:lineno] (default False).
            format_string: Custom format string (overrides timestamps/include_location).

        Example:
            >>> Logger.configure(level="DEBUG", timestamps=True)
            >>> Logger.configure(level="INFO", output="warpt.log")
            >>> Logger.configure(level="WARNING", output=sys.stderr)
        """
        # Convert string level to LogLevel if needed
        if isinstance(level, str):
            level = LogLevel(level.upper())

        logger = logging.getLogger(cls._root_name)
        logger.setLevel(level.to_logging_level())

        # Remove existing handlers
        for existing_handler in logger.handlers[:]:
            logger.removeHandler(existing_handler)
            existing_handler.close()

        # Create handler
        new_handler: logging.Handler
        if output is None:
            new_handler = logging.StreamHandler(sys.stdout)
        elif output == "stderr":
            new_handler = logging.StreamHandler(sys.stderr)
        elif isinstance(output, str | Path):
            new_handler = logging.FileHandler(str(output))
        elif hasattr(output, "write"):
            new_handler = logging.StreamHandler(output)
        else:
            raise ValueError(f"Invalid output: {type(output)}")

        new_handler.setLevel(level.to_logging_level())

        # Build format string
        if format_string is None:
            parts = []
            if timestamps:
                parts.append("%(asctime)s")
            parts.append("%(levelname)s")
            parts.append("[%(name)s]")
            if include_location:
                parts.append("[%(filename)s:%(lineno)d]")
            parts.append("%(message)s")
            format_string = " ".join(parts)

        new_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(new_handler)
        logger.propagate = False

        cls._configured = True

    @classmethod
    def get(cls, name: str | None = None) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name (appended to "warpt."). If None, returns root logger.

        Returns:
            Logger instance.

        Raises:
            LoggerNotConfiguredError: If configure() hasn't been called.

        Example:
            >>> log = Logger.get("stress.GPUMatMulTest")
            >>> log.info("Test starting...")
        """
        if not cls._configured:
            raise LoggerNotConfiguredError()

        if name:
            return logging.getLogger(f"{cls._root_name}.{name}")
        return logging.getLogger(cls._root_name)

    @classmethod
    def set_level(cls, level: str | LogLevel) -> None:
        """Change log level without reconfiguring.

        Args:
            level: New log level.

        Raises:
            LoggerNotConfiguredError: If configure() hasn't been called.
        """
        if not cls._configured:
            raise LoggerNotConfiguredError()

        if isinstance(level, str):
            level = LogLevel(level.upper())

        logger = logging.getLogger(cls._root_name)
        logger.setLevel(level.to_logging_level())
        for handler in logger.handlers:
            handler.setLevel(level.to_logging_level())

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logger has been configured."""
        return cls._configured

    # -------------------------------------------------------------------------
    # Direct logging methods (convenience)
    # -------------------------------------------------------------------------

    @classmethod
    def debug(cls, name: str, message: str) -> None:
        """Log debug message."""
        cls.get(name).debug(message)

    @classmethod
    def info(cls, name: str, message: str) -> None:
        """Log info message."""
        cls.get(name).info(message)

    @classmethod
    def warning(cls, name: str, message: str) -> None:
        """Log warning message."""
        cls.get(name).warning(message)

    @classmethod
    def error(cls, name: str, message: str) -> None:
        """Log error message."""
        cls.get(name).error(message)

    @classmethod
    def critical(cls, name: str, message: str) -> None:
        """Log critical message."""
        cls.get(name).critical(message)
