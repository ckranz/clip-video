"""Structured logging for clip-video.

Provides configurable logging with:
- Multiple verbosity levels
- Structured log output
- File and console handlers
- Context-aware logging
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any


class LogLevel(IntEnum):
    """Log verbosity levels."""

    QUIET = 0  # Only errors
    NORMAL = 1  # Errors + warnings + key info
    VERBOSE = 2  # Errors + warnings + info
    DEBUG = 3  # Everything including debug


@dataclass
class LogConfig:
    """Configuration for logging.

    Attributes:
        level: Verbosity level
        log_file: Optional path to log file
        json_format: Use JSON format for logs
        include_timestamp: Include timestamp in logs
        include_context: Include context dict in logs
        color: Use colored output (console only)
    """

    level: LogLevel = LogLevel.NORMAL
    log_file: Path | None = None
    json_format: bool = False
    include_timestamp: bool = True
    include_context: bool = True
    color: bool = True


# ANSI color codes for console output
class Colors:
    """ANSI color codes."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log records.

    Supports both text and JSON formats with optional coloring.
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED,
    }

    def __init__(
        self,
        json_format: bool = False,
        include_timestamp: bool = True,
        include_context: bool = True,
        color: bool = True,
    ):
        super().__init__()
        self.json_format = json_format
        self.include_timestamp = include_timestamp
        self.include_context = include_context
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        if self.json_format:
            return self._format_json(record)
        return self._format_text(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        data: dict[str, Any] = {
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "logger": record.name,
        }

        if self.include_timestamp:
            data["timestamp"] = datetime.now().isoformat()

        # Include extra fields
        if hasattr(record, "__dict__"):
            extra = {}
            for key, value in record.__dict__.items():
                if key not in (
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "exc_info",
                    "exc_text",
                    "thread",
                    "threadName",
                    "message",
                    "taskName",
                ):
                    # Try to serialize the value
                    try:
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)

            if extra and self.include_context:
                data["context"] = extra

        # Include exception info if present
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data)

    def _format_text(self, record: logging.LogRecord) -> str:
        """Format as text."""
        parts = []

        # Timestamp
        if self.include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.color:
                parts.append(f"{Colors.GRAY}{timestamp}{Colors.RESET}")
            else:
                parts.append(timestamp)

        # Level
        level = record.levelname.upper()[:5].ljust(5)
        if self.color:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            parts.append(f"{color}{level}{Colors.RESET}")
        else:
            parts.append(level)

        # Logger name (shortened)
        name = record.name
        if len(name) > 20:
            name = "..." + name[-17:]
        if self.color:
            parts.append(f"{Colors.CYAN}{name:>20}{Colors.RESET}")
        else:
            parts.append(f"{name:>20}")

        # Message
        parts.append(record.getMessage())

        result = " | ".join(parts)

        # Context
        if self.include_context:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in (
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "exc_info",
                    "exc_text",
                    "thread",
                    "threadName",
                    "message",
                    "taskName",
                ):
                    extra[key] = value

            if extra:
                context_str = " ".join(f"{k}={v}" for k, v in extra.items())
                if self.color:
                    result += f" {Colors.GRAY}[{context_str}]{Colors.RESET}"
                else:
                    result += f" [{context_str}]"

        # Exception
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        return result


class ClipVideoLogger(logging.Logger):
    """Custom logger with context support."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._context: dict[str, Any] = {}

    def with_context(self, **context: Any) -> "ClipVideoLogger":
        """Create a new logger with additional context.

        Args:
            **context: Context key-value pairs

        Returns:
            Logger with context bound
        """
        new_logger = ClipVideoLogger(self.name, self.level)
        new_logger.handlers = self.handlers
        new_logger._context = {**self._context, **context}
        return new_logger

    def _log(
        self,
        level: int,
        msg: object,
        args: tuple,
        exc_info: Any = None,
        extra: dict | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        """Override _log to include context."""
        if extra is None:
            extra = {}

        # Merge context
        merged_extra = {**self._context, **extra}

        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=merged_extra,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
        )


# Global configuration
_config: LogConfig = LogConfig()
_initialized: bool = False


def configure_logging(config: LogConfig | None = None) -> None:
    """Configure global logging settings.

    Args:
        config: Logging configuration
    """
    global _config, _initialized

    if config:
        _config = config

    # Set logging class
    logging.setLoggerClass(ClipVideoLogger)

    # Map our level to Python logging level
    level_map = {
        LogLevel.QUIET: logging.ERROR,
        LogLevel.NORMAL: logging.WARNING,
        LogLevel.VERBOSE: logging.INFO,
        LogLevel.DEBUG: logging.DEBUG,
    }
    log_level = level_map[_config.level]

    # Get root logger for clip_video
    root_logger = logging.getLogger("clip_video")
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_formatter = StructuredFormatter(
        json_format=_config.json_format,
        include_timestamp=_config.include_timestamp,
        include_context=_config.include_context,
        color=_config.color and sys.stderr.isatty(),
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if _config.log_file:
        _config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_config.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = StructuredFormatter(
            json_format=_config.json_format,
            include_timestamp=True,
            include_context=True,
            color=False,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> ClipVideoLogger:
    """Get a logger for the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    global _initialized

    if not _initialized:
        configure_logging()

    logger = logging.getLogger(name)
    if not isinstance(logger, ClipVideoLogger):
        # Wrap in our custom logger
        custom_logger = ClipVideoLogger(name)
        custom_logger.handlers = logger.handlers
        custom_logger.level = logger.level
        return custom_logger

    return logger  # type: ignore


def set_verbosity(level: LogLevel) -> None:
    """Set global verbosity level.

    Args:
        level: Verbosity level
    """
    global _config
    _config.level = level
    configure_logging(_config)


def enable_file_logging(log_file: Path) -> None:
    """Enable logging to a file.

    Args:
        log_file: Path to log file
    """
    global _config
    _config.log_file = log_file
    configure_logging(_config)


class LogContext:
    """Context manager for temporary logging context.

    Example:
        with LogContext(operation="transcribe", video="test.mp4"):
            # All logs in this block will include the context
            logger.info("Starting transcription")
    """

    def __init__(self, **context: Any):
        """Initialize with context.

        Args:
            **context: Context key-value pairs
        """
        self.context = context
        self._old_factory = None

    def __enter__(self) -> "LogContext":
        # Store the old factory
        self._old_factory = logging.getLogRecordFactory()

        # Create new factory that adds our context
        context = self.context

        def factory(
            name: str,
            level: int,
            fn: str,
            lno: int,
            msg: str,
            args: tuple,
            exc_info: Any,
            func: str | None = None,
            sinfo: str | None = None,
            **kwargs: Any,
        ) -> logging.LogRecord:
            record = self._old_factory(
                name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs
            )
            for key, value in context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        # Restore the old factory
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)


# Convenience functions for common operations
def log_operation_start(logger: logging.Logger, operation: str, **context: Any) -> None:
    """Log the start of an operation.

    Args:
        logger: Logger to use
        operation: Operation name
        **context: Additional context
    """
    logger.info(f"Starting: {operation}", extra=context)


def log_operation_complete(
    logger: logging.Logger,
    operation: str,
    duration: float | None = None,
    **context: Any,
) -> None:
    """Log the completion of an operation.

    Args:
        logger: Logger to use
        operation: Operation name
        duration: Optional duration in seconds
        **context: Additional context
    """
    if duration is not None:
        context["duration_seconds"] = round(duration, 2)
    logger.info(f"Completed: {operation}", extra=context)


def log_operation_failed(
    logger: logging.Logger,
    operation: str,
    error: Exception,
    **context: Any,
) -> None:
    """Log a failed operation.

    Args:
        logger: Logger to use
        operation: Operation name
        error: Error that occurred
        **context: Additional context
    """
    context["error_type"] = type(error).__name__
    context["error_message"] = str(error)
    logger.error(f"Failed: {operation}", extra=context)
