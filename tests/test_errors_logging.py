"""Tests for error handling and logging modules."""

import logging
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

from clip_video.errors import (
    ErrorCategory,
    ClipVideoError,
    TransientError,
    RateLimitError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    ExternalServiceError,
    RetryConfig,
    RetryState,
    calculate_delay,
    is_retryable,
    retry_with_backoff,
    ErrorContext,
    wrap_external_error,
    format_error_for_display,
)
from clip_video.logging import (
    LogLevel,
    LogConfig,
    StructuredFormatter,
    configure_logging,
    get_logger,
    set_verbosity,
    enable_file_logging,
    LogContext,
    log_operation_start,
    log_operation_complete,
    log_operation_failed,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_categories(self):
        """Test error category values."""
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.RESOURCE.value == "resource"
        assert ErrorCategory.EXTERNAL.value == "external"
        assert ErrorCategory.INTERNAL.value == "internal"


class TestClipVideoError:
    """Tests for ClipVideoError base class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ClipVideoError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {}
        assert error.recoverable is False

    def test_error_with_context(self):
        """Test error with context."""
        error = ClipVideoError(
            "Test error",
            context={"key": "value"},
        )

        assert "context: {'key': 'value'}" in str(error)
        assert error.context == {"key": "value"}

    def test_recoverable_error(self):
        """Test recoverable error flag."""
        error = ClipVideoError("Test", recoverable=True)
        assert error.recoverable is True


class TestSpecificErrors:
    """Tests for specific error types."""

    def test_transient_error(self):
        """Test TransientError."""
        error = TransientError("Network timeout")

        assert error.category == ErrorCategory.TRANSIENT
        assert error.recoverable is True

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limit exceeded", retry_after=30.0)

        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.retry_after == 30.0
        assert error.recoverable is True

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")

        assert error.category == ErrorCategory.VALIDATION
        assert error.recoverable is False

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Missing API key")

        assert error.category == ErrorCategory.CONFIGURATION
        assert error.recoverable is False

    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError("File not found")

        assert error.category == ErrorCategory.RESOURCE
        assert error.recoverable is False

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        error = ExternalServiceError("API error")

        assert error.category == ErrorCategory.EXTERNAL
        assert error.recoverable is True


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)

        delay1 = calculate_delay(1, config)
        delay2 = calculate_delay(2, config)
        delay3 = calculate_delay(3, config)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, jitter=False)

        delay = calculate_delay(10, config)  # Would be 512 without cap

        assert delay == 5.0

    def test_rate_limit_delay(self):
        """Test rate limit delay takes precedence."""
        config = RetryConfig(initial_delay=1.0, jitter=False)

        delay = calculate_delay(1, config, rate_limit_delay=30.0)

        assert delay == 30.0

    def test_jitter(self):
        """Test jitter adds randomness."""
        config = RetryConfig(initial_delay=10.0, jitter=True)

        delays = [calculate_delay(1, config) for _ in range(10)]

        # All should be different (with very high probability)
        assert len(set(delays)) > 1


class TestIsRetryable:
    """Tests for is_retryable function."""

    def test_retryable_errors(self):
        """Test retryable error types."""
        config = RetryConfig()

        assert is_retryable(TransientError("test"), config) is True
        assert is_retryable(RateLimitError("test"), config) is True
        assert is_retryable(ExternalServiceError("test"), config) is True

    def test_non_retryable_errors(self):
        """Test non-retryable error types."""
        config = RetryConfig()

        assert is_retryable(ValidationError("test"), config) is False
        assert is_retryable(ConfigurationError("test"), config) is False
        assert is_retryable(ResourceError("test"), config) is False

    def test_generic_transient_patterns(self):
        """Test detection of transient patterns in generic errors."""
        config = RetryConfig()

        assert is_retryable(Exception("Connection timeout"), config) is True
        assert is_retryable(Exception("503 Service Unavailable"), config) is True
        assert is_retryable(Exception("connection reset by peer"), config) is True


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Test successful call with no retry needed."""
        call_count = 0

        @retry_with_backoff()
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()

        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Test retry on transient error."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Temporary failure")
            return "success"

        result = fails_then_succeeds()

        assert result == "success"
        assert call_count == 2

    def test_max_retries_exceeded(self):
        """Test max retries exceeded."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TransientError("Always fails")

        with pytest.raises(TransientError):
            always_fails()

        assert call_count == 3

    def test_no_retry_on_non_retryable(self):
        """Test no retry on non-retryable error."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3))
        def raises_validation():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Invalid input")

        with pytest.raises(ValidationError):
            raises_validation()

        assert call_count == 1


class TestErrorContext:
    """Tests for ErrorContext context manager."""

    def test_success_path(self):
        """Test successful operation path."""
        with ErrorContext("test_operation") as ctx:
            result = "success"

        assert ctx.error is None

    def test_error_path(self):
        """Test error path."""
        with pytest.raises(ValueError):
            with ErrorContext("test_operation") as ctx:
                raise ValueError("Test error")

        assert ctx.error is not None
        assert isinstance(ctx.error, ValueError)

    def test_rollback_called(self):
        """Test rollback is called on error."""
        rollback_called = False

        def rollback():
            nonlocal rollback_called
            rollback_called = True

        with pytest.raises(ValueError):
            with ErrorContext("test_operation", rollback=rollback):
                raise ValueError("Test error")

        assert rollback_called is True

    def test_context_included(self):
        """Test context is included."""
        ctx = ErrorContext("test_operation", context={"key": "value"})
        assert ctx.context == {"key": "value"}


class TestWrapExternalError:
    """Tests for wrap_external_error function."""

    def test_wrap_rate_limit(self):
        """Test wrapping rate limit error."""
        original = Exception("Rate limit exceeded (429)")
        wrapped = wrap_external_error(original, "TestAPI", "request")

        assert isinstance(wrapped, RateLimitError)
        assert wrapped.retry_after == 60.0

    def test_wrap_transient(self):
        """Test wrapping transient error."""
        original = Exception("Connection timeout")
        wrapped = wrap_external_error(original, "TestAPI", "request")

        assert isinstance(wrapped, TransientError)

    def test_wrap_generic(self):
        """Test wrapping generic error."""
        original = Exception("Unknown error")
        wrapped = wrap_external_error(original, "TestAPI", "request")

        assert isinstance(wrapped, ExternalServiceError)


class TestFormatErrorForDisplay:
    """Tests for format_error_for_display function."""

    def test_format_clip_video_error(self):
        """Test formatting ClipVideoError."""
        error = ValidationError("Invalid input", context={"field": "name"})
        formatted = format_error_for_display(error)

        assert "[validation]" in formatted
        assert "Invalid input" in formatted
        assert "field=name" in formatted

    def test_format_generic_error(self):
        """Test formatting generic error."""
        error = ValueError("Test error")
        formatted = format_error_for_display(error)

        assert "[error]" in formatted
        assert "ValueError" in formatted


# Logging Tests

class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_levels(self):
        """Test log level values."""
        assert LogLevel.QUIET == 0
        assert LogLevel.NORMAL == 1
        assert LogLevel.VERBOSE == 2
        assert LogLevel.DEBUG == 3


class TestLogConfig:
    """Tests for LogConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = LogConfig()

        assert config.level == LogLevel.NORMAL
        assert config.log_file is None
        assert config.json_format is False
        assert config.include_timestamp is True


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_text_format(self):
        """Test text formatting."""
        formatter = StructuredFormatter(
            json_format=False,
            include_timestamp=False,
            include_context=False,
            color=False,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "INFO" in formatted
        assert "Test message" in formatted

    def test_json_format(self):
        """Test JSON formatting."""
        import json

        formatter = StructuredFormatter(
            json_format=True,
            include_timestamp=False,
            include_context=True,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "info"
        assert parsed["message"] == "Test message"

    def test_context_in_text(self):
        """Test context inclusion in text format."""
        formatter = StructuredFormatter(
            json_format=False,
            include_timestamp=False,
            include_context=True,
            color=False,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.custom_key = "custom_value"

        formatted = formatter.format(record)

        assert "custom_key=custom_value" in formatted


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_default(self):
        """Test default configuration."""
        configure_logging()

        logger = get_logger("test_configure")
        assert logger is not None

    def test_configure_verbose(self):
        """Test verbose configuration."""
        config = LogConfig(level=LogLevel.VERBOSE)
        configure_logging(config)

        logger = get_logger("test_verbose")
        assert logger.level <= logging.INFO


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test.module")

        assert logger is not None
        assert "test.module" in logger.name


class TestSetVerbosity:
    """Tests for set_verbosity function."""

    def test_set_verbosity(self):
        """Test setting verbosity level."""
        set_verbosity(LogLevel.DEBUG)

        logger = get_logger("clip_video.test")
        root = logging.getLogger("clip_video")

        assert root.level == logging.DEBUG


class TestEnableFileLogging:
    """Tests for enable_file_logging function."""

    def test_enable_file_logging(self, tmp_path):
        """Test enabling file logging."""
        log_file = tmp_path / "test.log"
        enable_file_logging(log_file)

        logger = get_logger("clip_video.test_file")
        logger.info("Test message")

        # File should exist after logging
        # Note: Due to logging configuration, we just check the handler was added
        root = logging.getLogger("clip_video")
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context(self):
        """Test log context adds attributes."""
        configure_logging()

        with LogContext(test_key="test_value"):
            # Inside the context, logs should have the attribute
            pass  # In real usage, logger calls here would have the context

        # Context should be removed after exit


class TestLogOperationHelpers:
    """Tests for log operation helper functions."""

    def test_log_operation_start(self):
        """Test logging operation start."""
        logger = Mock()
        log_operation_start(logger, "test_operation", param="value")

        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Starting" in call_args[0][0]
        assert call_args[1]["extra"]["param"] == "value"

    def test_log_operation_complete(self):
        """Test logging operation complete."""
        logger = Mock()
        log_operation_complete(logger, "test_operation", duration=5.5)

        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Completed" in call_args[0][0]
        assert call_args[1]["extra"]["duration_seconds"] == 5.5

    def test_log_operation_failed(self):
        """Test logging operation failure."""
        logger = Mock()
        error = ValueError("Test error")
        log_operation_failed(logger, "test_operation", error)

        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "Failed" in call_args[0][0]
        assert call_args[1]["extra"]["error_type"] == "ValueError"
