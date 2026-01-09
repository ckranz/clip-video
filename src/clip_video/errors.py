"""Error handling and retry logic for clip-video.

Provides comprehensive error handling with:
- Custom exception hierarchy
- Retry logic with exponential backoff
- API rate limit handling
- State file protection
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

from clip_video.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ErrorCategory(str, Enum):
    """Categories of errors for handling decisions."""

    TRANSIENT = "transient"  # Network, timeout - can retry
    RATE_LIMIT = "rate_limit"  # API rate limit - wait and retry
    VALIDATION = "validation"  # Bad input - don't retry
    CONFIGURATION = "configuration"  # Bad config - don't retry
    RESOURCE = "resource"  # Missing file/resource - don't retry
    EXTERNAL = "external"  # External service error - may retry
    INTERNAL = "internal"  # Bug in code - don't retry


class ClipVideoError(Exception):
    """Base exception for clip-video errors.

    Attributes:
        message: Human-readable error message
        category: Error category for handling
        context: Additional context information
        recoverable: Whether the error is recoverable
    """

    category: ErrorCategory = ErrorCategory.INTERNAL

    def __init__(
        self,
        message: str,
        context: dict | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (context: {self.context})"
        return self.message


class TransientError(ClipVideoError):
    """Transient error that can be retried.

    Examples: network timeouts, temporary service unavailability.
    """

    category = ErrorCategory.TRANSIENT

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, context, recoverable=True)


class RateLimitError(ClipVideoError):
    """API rate limit exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    category = ErrorCategory.RATE_LIMIT

    def __init__(
        self,
        message: str,
        retry_after: float = 60.0,
        context: dict | None = None,
    ):
        super().__init__(message, context, recoverable=True)
        self.retry_after = retry_after


class ValidationError(ClipVideoError):
    """Input validation error.

    Examples: invalid file format, missing required field.
    """

    category = ErrorCategory.VALIDATION

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, context, recoverable=False)


class ConfigurationError(ClipVideoError):
    """Configuration error.

    Examples: missing API key, invalid settings.
    """

    category = ErrorCategory.CONFIGURATION

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, context, recoverable=False)


class ResourceError(ClipVideoError):
    """Resource not found or unavailable.

    Examples: missing video file, missing model.
    """

    category = ErrorCategory.RESOURCE

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, context, recoverable=False)


class ExternalServiceError(ClipVideoError):
    """External service error.

    Examples: API errors, transcription service errors.
    """

    category = ErrorCategory.EXTERNAL

    def __init__(
        self,
        message: str,
        context: dict | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message, context, recoverable=recoverable)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_errors: Error types that should be retried
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple = (TransientError, RateLimitError, ExternalServiceError)


@dataclass
class RetryState:
    """State of a retry operation.

    Attributes:
        attempt: Current attempt number (1-indexed)
        total_attempts: Maximum attempts
        last_error: Last error encountered
        delays: List of delays between attempts
        started_at: When retry operation started
    """

    attempt: int = 0
    total_attempts: int = 3
    last_error: Exception | None = None
    delays: list[float] = field(default_factory=list)
    started_at: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    rate_limit_delay: float | None = None,
) -> float:
    """Calculate delay before next retry.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration
        rate_limit_delay: Optional delay from rate limit error

    Returns:
        Delay in seconds
    """
    # Use rate limit delay if provided and it's longer
    if rate_limit_delay is not None:
        base_delay = rate_limit_delay
    else:
        # Exponential backoff
        base_delay = config.initial_delay * (config.exponential_base ** (attempt - 1))

    # Cap at max delay
    delay = min(base_delay, config.max_delay)

    # Add jitter if configured
    if config.jitter:
        jitter_range = delay * 0.25  # 25% jitter
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(0.1, delay)  # Ensure positive

    return delay


def is_retryable(error: Exception, config: RetryConfig) -> bool:
    """Check if an error should be retried.

    Args:
        error: The error to check
        config: Retry configuration

    Returns:
        True if the error should be retried
    """
    # Check if it's one of our retryable error types
    if isinstance(error, config.retryable_errors):
        if hasattr(error, "recoverable"):
            return error.recoverable
        return True

    # Check for common transient error patterns
    error_str = str(error).lower()
    transient_patterns = [
        "timeout",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "service unavailable",
        "502",
        "503",
        "504",
    ]

    return any(pattern in error_str for pattern in transient_patterns)


def retry_with_backoff(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration

    Returns:
        Decorator function
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            state = RetryState(total_attempts=config.max_attempts)

            while state.attempt < config.max_attempts:
                state.attempt += 1

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    state.last_error = e

                    # Check if we should retry
                    if not is_retryable(e, config):
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {e}",
                            extra={"error_type": type(e).__name__},
                        )
                        raise

                    # Check if we have more attempts
                    if state.attempt >= config.max_attempts:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for {func.__name__}",
                            extra={
                                "error": str(e),
                                "attempts": state.attempt,
                            },
                        )
                        raise

                    # Calculate delay
                    rate_limit_delay = None
                    if isinstance(e, RateLimitError):
                        rate_limit_delay = e.retry_after

                    delay = calculate_delay(state.attempt, config, rate_limit_delay)
                    state.delays.append(delay)

                    logger.info(
                        f"Retrying {func.__name__} in {delay:.1f}s "
                        f"(attempt {state.attempt}/{config.max_attempts})",
                        extra={
                            "error": str(e),
                            "delay": delay,
                            "attempt": state.attempt,
                        },
                    )

                    time.sleep(delay)

            # This should not be reached, but just in case
            if state.last_error:
                raise state.last_error
            raise RuntimeError("Retry loop exited without result")

        return wrapper

    return decorator


class ErrorContext:
    """Context manager for error handling with automatic cleanup.

    Ensures state files are not corrupted by catching errors and
    providing consistent error handling.
    """

    def __init__(
        self,
        operation: str,
        rollback: Callable[[], None] | None = None,
        context: dict | None = None,
    ):
        """Initialize error context.

        Args:
            operation: Name of the operation being performed
            rollback: Optional rollback function to call on error
            context: Additional context to include in errors
        """
        self.operation = operation
        self.rollback = rollback
        self.context = context or {}
        self.error: Exception | None = None

    def __enter__(self) -> "ErrorContext":
        logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        if exc_val is not None:
            self.error = exc_val

            # Log the error
            logger.error(
                f"Error in {self.operation}: {exc_val}",
                extra={
                    "operation": self.operation,
                    "error_type": type(exc_val).__name__,
                    **self.context,
                },
            )

            # Execute rollback if provided
            if self.rollback:
                try:
                    logger.info(f"Rolling back {self.operation}")
                    self.rollback()
                except Exception as rollback_error:
                    logger.error(
                        f"Rollback failed for {self.operation}: {rollback_error}",
                    )

        else:
            logger.debug(f"Completed operation: {self.operation}")

        # Don't suppress the exception
        return False


def wrap_external_error(
    error: Exception,
    service: str,
    operation: str,
) -> ClipVideoError:
    """Wrap an external error in a ClipVideoError.

    Args:
        error: Original error
        service: Name of external service
        operation: Operation being performed

    Returns:
        Wrapped ClipVideoError
    """
    error_str = str(error).lower()

    # Check for rate limiting
    if "rate limit" in error_str or "429" in error_str:
        # Try to extract retry-after
        retry_after = 60.0  # Default
        if hasattr(error, "retry_after"):
            retry_after = float(error.retry_after)

        return RateLimitError(
            f"Rate limit exceeded for {service}: {operation}",
            retry_after=retry_after,
            context={"service": service, "operation": operation},
        )

    # Check for transient errors
    transient_patterns = ["timeout", "connection", "temporary", "unavailable"]
    if any(pattern in error_str for pattern in transient_patterns):
        return TransientError(
            f"Transient error from {service}: {operation} - {error}",
            context={"service": service, "operation": operation},
        )

    # Default to external service error
    return ExternalServiceError(
        f"Error from {service}: {operation} - {error}",
        context={"service": service, "operation": operation},
        recoverable=True,
    )


def format_error_for_display(error: Exception) -> str:
    """Format an error message for user display.

    Args:
        error: Error to format

    Returns:
        Human-readable error message
    """
    if isinstance(error, ClipVideoError):
        category = error.category.value
        base_message = error.message

        if error.context:
            context_str = ", ".join(f"{k}={v}" for k, v in error.context.items())
            return f"[{category}] {base_message} ({context_str})"

        return f"[{category}] {base_message}"

    return f"[error] {type(error).__name__}: {error}"
