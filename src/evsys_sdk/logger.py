"""Configurable logging for the evsys_sdk SDK.

Mirrors the level-based, env-driven logger from the older `trajectory` SDK (pre-rebrand).
Level is read from EVSYS_LOGGING_LEVEL (DEBUG/INFO/WARNING/ERROR/CRITICAL)
and can be overridden at runtime via ``configure_logger(level=...)``.

Usage::

    from evsys_sdk.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import ClassVar

from .constants import (
    DEFAULT_LOG_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOGGING_LEVEL,
    EVSYS_LOGGING_LEVEL_ENV,
    LOGGER_NAME,
    SUPPORTED_LOGGING_LEVELS,
)

RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
GRAY = "\033[90m"


class ColorFormatter(logging.Formatter):
    """Wrap formatted records in ANSI color based on level (TTY only)."""

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: GRAY,
        logging.INFO: GRAY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record):
        message = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            if color:
                message = f"{color}{message}{RESET}"
        return message


def _resolve_level(level: str | None) -> str:
    candidate = (level or os.getenv(EVSYS_LOGGING_LEVEL_ENV, DEFAULT_LOGGING_LEVEL)).upper()
    if candidate in SUPPORTED_LOGGING_LEVELS:
        return candidate
    print(
        f"Warning: invalid logging level '{candidate}' "
        f"(set {EVSYS_LOGGING_LEVEL_ENV} to one of {SUPPORTED_LOGGING_LEVELS}). "
        f"Using default {DEFAULT_LOGGING_LEVEL}."
    )
    return DEFAULT_LOGGING_LEVEL


def configure_logger(
    level: str | None = None,
    *,
    format_string: str | None = None,
    date_format: str | None = None,
    use_color: bool | None = None,
) -> logging.Logger:
    """(Re)configure the root SDK logger. Returns the configured logger.

    The SDK root logger is named ``evsys_sdk``; per-module loggers
    obtained via :func:`get_logger` propagate to it.
    """
    resolved = _resolve_level(level)
    fmt = format_string or DEFAULT_LOG_FORMAT
    datefmt = date_format or DEFAULT_LOG_DATE_FORMAT
    color = (
        use_color
        if use_color is not None
        else (sys.stdout.isatty() and os.getenv("NO_COLOR") is None)
    )

    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, resolved))
    handler.setFormatter(ColorFormatter(fmt=fmt, datefmt=datefmt, use_color=color))

    logger.setLevel(getattr(logging, resolved))
    logger.addHandler(handler)
    # Don't double-emit through the root logger's handlers.
    logger.propagate = False
    logger.debug("evsys_sdk logger configured | level=%s color=%s", resolved, color)
    return logger


def set_level(level: str) -> None:
    """Convenience: change the SDK log level at runtime."""
    configure_logger(level=level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger under the SDK root (e.g. ``get_logger(__name__)``)."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    # Normalise dotted module names to children of the SDK root logger.
    short = name.split(".")[-1]
    return logging.getLogger(f"{LOGGER_NAME}.{short}")


# Configure once on import from the environment.
_root_logger = configure_logger()


__all__ = ["ColorFormatter", "configure_logger", "get_logger", "set_level"]
