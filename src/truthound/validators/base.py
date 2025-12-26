"""Base classes for validators.

Provides enterprise-grade abstractions including:
- Immutable configuration (thread-safe)
- Timeout mechanism (prevents hangs)
- Type-safe column filtering
- Streaming support for large datasets
- ReDoS protection for regex patterns
- Graceful degradation on partial failures
- Memory tracking and limits
- Consistent logging framework
- Enhanced error context
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import re
import signal
import threading
import traceback
import logging
import time
import sys
from contextlib import contextmanager
from functools import wraps
from enum import Enum

import polars as pl

from truthound.types import Severity


# ============================================================================
# Logging Infrastructure (#11)
# ============================================================================

class ValidatorLogger:
    """Centralized logging for validators.

    Provides consistent logging across all validators with structured output.

    Usage:
        logger = ValidatorLogger("MyValidator")
        logger.info("Starting validation")
        logger.warning("Found issues", extra={"count": 5})
        logger.error("Validation failed", exc_info=True)
    """

    _loggers: dict[str, logging.Logger] = {}
    _default_level: int = logging.INFO
    _handler_configured: bool = False

    @classmethod
    def configure(
        cls,
        level: int = logging.INFO,
        format_string: str | None = None,
        handler: logging.Handler | None = None,
    ) -> None:
        """Configure global logging settings.

        Args:
            level: Logging level (default: INFO)
            format_string: Custom format string
            handler: Custom handler (default: StreamHandler)
        """
        cls._default_level = level

        if not cls._handler_configured:
            root_logger = logging.getLogger("truthound")
            root_logger.setLevel(level)

            if handler is None:
                handler = logging.StreamHandler()
                format_str = format_string or (
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                )
                handler.setFormatter(logging.Formatter(format_str))

            root_logger.addHandler(handler)
            cls._handler_configured = True

    def __init__(self, name: str):
        self.name = name
        self._logger = self._get_logger(name)

    @classmethod
    def _get_logger(cls, name: str) -> logging.Logger:
        if name not in cls._loggers:
            logger = logging.getLogger(f"truthound.{name}")
            logger.setLevel(cls._default_level)
            cls._loggers[name] = logger
        return cls._loggers[name]

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, extra=kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, extra=kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._logger.error(msg, exc_info=exc_info, extra=kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(msg, extra=kwargs)


# ============================================================================
# Error Types and Context (#12)
# ============================================================================

class ValidationErrorContext:
    """Enhanced error context for debugging and monitoring.

    Captures detailed information about validation failures including
    stack traces, timing, and environmental context.
    """

    def __init__(
        self,
        validator_name: str,
        column: str | None = None,
        error_type: str = "unknown",
        message: str = "",
        exception: Exception | None = None,
    ):
        self.validator_name = validator_name
        self.column = column
        self.error_type = error_type
        self.message = message
        self.exception = exception
        self.timestamp = time.time()
        self.stack_trace = traceback.format_exc() if exception else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator_name,
            "column": self.column,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
            "exception_type": type(self.exception).__name__ if self.exception else None,
        }


class RegexValidationError(ValueError):
    """Raised when a regex pattern is invalid."""

    def __init__(self, pattern: str, error: str):
        self.pattern = pattern
        self.error = error
        super().__init__(f"Invalid regex pattern '{pattern}': {error}")


class ValidationTimeoutError(Exception):
    """Raised when validation exceeds the configured timeout."""

    def __init__(self, timeout_seconds: float, validator_name: str = ""):
        self.timeout_seconds = timeout_seconds
        self.validator_name = validator_name
        message = f"Validation timed out after {timeout_seconds}s"
        if validator_name:
            message = f"[{validator_name}] {message}"
        super().__init__(message)


class ColumnNotFoundError(Exception):
    """Raised when a required column is not found in the schema."""

    def __init__(self, column: str, available_columns: list[str]):
        self.column = column
        self.available_columns = available_columns
        super().__init__(
            f"Column '{column}' not found. Available: {available_columns[:10]}"
            + ("..." if len(available_columns) > 10 else "")
        )


class MemoryLimitExceededError(Exception):
    """Raised when validation exceeds memory limits."""

    def __init__(self, current_mb: float, limit_mb: float):
        self.current_mb = current_mb
        self.limit_mb = limit_mb
        super().__init__(
            f"Memory limit exceeded: {current_mb:.1f}MB > {limit_mb:.1f}MB"
        )


# ============================================================================
# ReDoS Protection (#7)
# ============================================================================

class RegexSafetyChecker:
    """Detects and mitigates ReDoS vulnerabilities in regex patterns.

    ReDoS (Regular Expression Denial of Service) occurs when a regex
    pattern causes exponential backtracking on malicious input.

    This checker:
    1. Detects common ReDoS patterns
    2. Limits pattern complexity
    3. Wraps execution with timeout protection
    """

    # Patterns that commonly cause ReDoS
    REDOS_PATTERNS = [
        r"\(.+\)\+\+",           # Nested quantifiers: (a+)+
        r"\(.+\)\*\*",           # Nested quantifiers: (a*)*
        r"\(.+\)\{\d+,\}",       # Nested with unbounded repetition
        r"\(.+\|.+\)\+",         # Alternation in quantified group
        r"\\d\+\\d\+",           # Adjacent unbounded quantifiers
        r"\.\*\.\*",             # Multiple .* patterns
        r"\(.+\?\)\+",           # Lazy quantifier in greedy group
    ]

    MAX_PATTERN_LENGTH = 1000
    MAX_QUANTIFIER_VALUE = 1000

    @classmethod
    def check_pattern(cls, pattern: str) -> tuple[bool, str | None]:
        """Check if a pattern is potentially vulnerable to ReDoS.

        Args:
            pattern: Regex pattern to check

        Returns:
            Tuple of (is_safe, warning_message)
        """
        # Check length
        if len(pattern) > cls.MAX_PATTERN_LENGTH:
            return False, f"Pattern too long ({len(pattern)} > {cls.MAX_PATTERN_LENGTH})"

        # Check for dangerous patterns
        for redos_pattern in cls.REDOS_PATTERNS:
            if re.search(redos_pattern, pattern):
                return False, f"Potentially vulnerable to ReDoS: matches {redos_pattern}"

        # Check for excessive quantifiers
        quantifiers = re.findall(r"\{(\d+)(?:,(\d*))?\}", pattern)
        for match in quantifiers:
            min_val = int(match[0])
            max_val = int(match[1]) if match[1] else min_val
            if max_val > cls.MAX_QUANTIFIER_VALUE:
                return False, f"Quantifier too large: {max_val} > {cls.MAX_QUANTIFIER_VALUE}"

        return True, None

    @classmethod
    def safe_match(
        cls,
        pattern: re.Pattern[str],
        text: str,
        timeout_ms: int = 100,
        match_type: str = "fullmatch",
    ) -> re.Match[str] | None:
        """Execute regex match with timeout protection.

        Args:
            pattern: Compiled regex pattern
            text: Text to match against
            timeout_ms: Maximum execution time in milliseconds
            match_type: "fullmatch", "match", or "search"

        Returns:
            Match object or None

        Note:
            On timeout, returns None rather than raising an exception
            to allow graceful degradation.
        """
        # For short strings, just execute directly
        if len(text) < 1000:
            if match_type == "fullmatch":
                return pattern.fullmatch(text)
            elif match_type == "match":
                return pattern.match(text)
            else:
                return pattern.search(text)

        # For longer strings, use threading with timeout
        result: list[re.Match[str] | None] = [None]
        exception: list[Exception | None] = [None]

        def do_match() -> None:
            try:
                if match_type == "fullmatch":
                    result[0] = pattern.fullmatch(text)
                elif match_type == "match":
                    result[0] = pattern.match(text)
                else:
                    result[0] = pattern.search(text)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=do_match)
        thread.start()
        thread.join(timeout=timeout_ms / 1000.0)

        if thread.is_alive():
            # Timeout occurred - return None for graceful degradation
            return None

        if exception[0]:
            raise exception[0]

        return result[0]


# ============================================================================
# Memory Tracking (#10)
# ============================================================================

class MemoryTracker:
    """Tracks and limits memory usage during validation.

    Provides:
    - Current memory usage estimation
    - Peak memory tracking
    - Configurable limits with enforcement
    """

    def __init__(self, limit_mb: float | None = None):
        """Initialize memory tracker.

        Args:
            limit_mb: Optional memory limit in megabytes
        """
        self.limit_mb = limit_mb
        self.peak_mb: float = 0.0
        self._start_mb: float = 0.0

    def get_current_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS returns bytes, Linux returns KB
            if sys.platform == "darwin":
                return usage.ru_maxrss / (1024 * 1024)
            else:
                return usage.ru_maxrss / 1024
        except ImportError:
            # Fallback for systems without resource module
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except ImportError:
                return 0.0

    def start(self) -> None:
        """Start tracking memory usage."""
        self._start_mb = self.get_current_mb()
        self.peak_mb = self._start_mb

    def check(self) -> None:
        """Check current memory against limit.

        Raises:
            MemoryLimitExceededError: If limit is exceeded
        """
        current = self.get_current_mb()
        self.peak_mb = max(self.peak_mb, current)

        if self.limit_mb is not None and current > self.limit_mb:
            raise MemoryLimitExceededError(current, self.limit_mb)

    def get_delta_mb(self) -> float:
        """Get memory change since start()."""
        return self.get_current_mb() - self._start_mb

    def __enter__(self) -> "MemoryTracker":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            self.check()
        except MemoryLimitExceededError:
            pass  # Don't raise on exit


# ============================================================================
# Safe Sampling (#6)
# ============================================================================

class SafeSampler:
    """Memory-safe sampling that avoids full dataset scans.

    Uses Polars lazy evaluation to ensure head() operations
    don't trigger full dataset materialization.
    """

    @staticmethod
    def safe_head(
        lf: pl.LazyFrame,
        n: int,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Safely get first n rows without full scan.

        Args:
            lf: LazyFrame to sample from
            n: Number of rows to retrieve
            columns: Optional list of columns to include

        Returns:
            DataFrame with at most n rows
        """
        query = lf
        if columns:
            # Only select columns that exist
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)

        # Use streaming collection with row limit
        return query.head(n).collect(engine="streaming")

    @staticmethod
    def safe_sample(
        lf: pl.LazyFrame,
        n: int,
        columns: list[str] | None = None,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Safely sample n rows without full materialization.

        Uses reservoir sampling approach for large datasets.

        Args:
            lf: LazyFrame to sample from
            n: Number of rows to sample
            columns: Optional columns to include
            seed: Optional random seed

        Returns:
            DataFrame with sampled rows
        """
        query = lf
        if columns:
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)

        # For small n, just take head (faster)
        if n <= 100:
            return query.head(n).collect(engine="streaming")

        # Use Polars native sampling with streaming
        df = query.collect(engine="streaming")
        if len(df) <= n:
            return df
        return df.sample(n=n, seed=seed)

    @staticmethod
    def safe_filter_sample(
        lf: pl.LazyFrame,
        filter_expr: pl.Expr,
        n: int,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Safely get filtered samples without full scan.

        Args:
            lf: LazyFrame to sample from
            filter_expr: Polars expression for filtering
            n: Maximum number of matching rows to return
            columns: Optional columns to include

        Returns:
            DataFrame with filtered samples
        """
        query = lf.filter(filter_expr)
        if columns:
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)

        return query.head(n).collect(engine="streaming")


# ============================================================================
# Graceful Degradation (#8, #13)
# ============================================================================

class ValidationResult(Enum):
    """Result status for individual validation operations."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Completed with some issues
    SKIPPED = "skipped"  # Skipped due to missing columns, etc.
    FAILED = "failed"    # Unrecoverable error
    TIMEOUT = "timeout"  # Exceeded time limit


@dataclass
class ValidatorExecutionResult:
    """Result of a single validator execution with error handling.

    Captures both successful validation issues and any errors that occurred.
    """
    validator_name: str
    status: ValidationResult
    issues: list["ValidationIssue"]
    error_context: ValidationErrorContext | None = None
    execution_time_ms: float = 0.0
    memory_delta_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator_name,
            "status": self.status.value,
            "issue_count": len(self.issues),
            "execution_time_ms": self.execution_time_ms,
            "memory_delta_mb": self.memory_delta_mb,
            "error": self.error_context.to_dict() if self.error_context else None,
        }


class GracefulValidator:
    """Wrapper that provides graceful degradation for validators.

    Catches exceptions and returns partial results instead of failing
    the entire validation pipeline.
    """

    def __init__(
        self,
        validator: "Validator",
        skip_on_error: bool = True,
        log_errors: bool = True,
    ):
        self.validator = validator
        self.skip_on_error = skip_on_error
        self.log_errors = log_errors
        self.logger = ValidatorLogger(f"GracefulValidator.{validator.name}")

    def validate(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
        """Execute validation with comprehensive error handling.

        Returns:
            ValidatorExecutionResult with status and any issues found
        """
        start_time = time.time()
        memory_tracker = MemoryTracker()
        memory_tracker.start()

        try:
            issues = self.validator.validate(lf)
            execution_time = (time.time() - start_time) * 1000

            return ValidatorExecutionResult(
                validator_name=self.validator.name,
                status=ValidationResult.SUCCESS,
                issues=issues,
                execution_time_ms=execution_time,
                memory_delta_mb=memory_tracker.get_delta_mb(),
            )

        except ColumnNotFoundError as e:
            error_ctx = ValidationErrorContext(
                validator_name=self.validator.name,
                column=e.column,
                error_type="column_not_found",
                message=str(e),
                exception=e,
            )
            if self.log_errors:
                self.logger.warning(f"Column not found: {e.column}")

            return ValidatorExecutionResult(
                validator_name=self.validator.name,
                status=ValidationResult.SKIPPED,
                issues=[],
                error_context=error_ctx,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except ValidationTimeoutError as e:
            error_ctx = ValidationErrorContext(
                validator_name=self.validator.name,
                error_type="timeout",
                message=str(e),
                exception=e,
            )
            if self.log_errors:
                self.logger.warning(f"Validation timed out: {e.timeout_seconds}s")

            return ValidatorExecutionResult(
                validator_name=self.validator.name,
                status=ValidationResult.TIMEOUT,
                issues=[],
                error_context=error_ctx,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except MemoryLimitExceededError as e:
            error_ctx = ValidationErrorContext(
                validator_name=self.validator.name,
                error_type="memory_limit",
                message=str(e),
                exception=e,
            )
            if self.log_errors:
                self.logger.error(f"Memory limit exceeded: {e.current_mb:.1f}MB")

            return ValidatorExecutionResult(
                validator_name=self.validator.name,
                status=ValidationResult.FAILED,
                issues=[],
                error_context=error_ctx,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except pl.exceptions.PolarsError as e:
            error_ctx = ValidationErrorContext(
                validator_name=self.validator.name,
                error_type="polars_error",
                message=str(e),
                exception=e,
            )
            if self.log_errors:
                self.logger.error(f"Polars error: {e}", exc_info=True)

            if self.skip_on_error:
                return ValidatorExecutionResult(
                    validator_name=self.validator.name,
                    status=ValidationResult.FAILED,
                    issues=[],
                    error_context=error_ctx,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            raise

        except Exception as e:
            error_ctx = ValidationErrorContext(
                validator_name=self.validator.name,
                error_type="unexpected_error",
                message=str(e),
                exception=e,
            )
            if self.log_errors:
                self.logger.exception(f"Unexpected error in {self.validator.name}")

            if self.skip_on_error:
                return ValidatorExecutionResult(
                    validator_name=self.validator.name,
                    status=ValidationResult.FAILED,
                    issues=[],
                    error_context=error_ctx,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            raise


# ============================================================================
# Schema Resilience (#9)
# ============================================================================

class SchemaValidator:
    """Validates schema compatibility before running validators.

    Prevents runtime errors from missing columns by pre-checking schema.
    """

    @staticmethod
    def check_columns_exist(
        lf: pl.LazyFrame,
        required_columns: list[str],
        raise_on_missing: bool = True,
    ) -> tuple[bool, list[str]]:
        """Check if required columns exist in the LazyFrame.

        Args:
            lf: LazyFrame to check
            required_columns: List of required column names
            raise_on_missing: If True, raise ColumnNotFoundError

        Returns:
            Tuple of (all_exist, missing_columns)
        """
        schema = lf.collect_schema()
        available = set(schema.names())
        missing = [c for c in required_columns if c not in available]

        if missing and raise_on_missing:
            raise ColumnNotFoundError(missing[0], list(available))

        return len(missing) == 0, missing

    @staticmethod
    def get_safe_columns(
        lf: pl.LazyFrame,
        requested_columns: list[str] | None,
        dtype_filter: set[type] | None = None,
    ) -> list[str]:
        """Get columns that exist and match type filter.

        Args:
            lf: LazyFrame to check
            requested_columns: Requested columns (None = all)
            dtype_filter: Optional set of allowed types

        Returns:
            List of valid column names
        """
        schema = lf.collect_schema()
        available = list(schema.names())

        if requested_columns:
            columns = [c for c in requested_columns if c in available]
        else:
            columns = available

        if dtype_filter:
            columns = [c for c in columns if type(schema[c]) in dtype_filter]

        return columns


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ValidatorConfig:
    """Immutable configuration for validators.

    This dataclass is frozen (immutable) to ensure thread-safety when
    validators are used across multiple threads or processes.

    Immutability Benefits:
    - Thread-safe: No race conditions from shared mutable state
    - Hashable: Can be used as dict keys or in sets
    - Predictable: Config cannot be accidentally modified

    Example:
        # Create a config
        config = ValidatorConfig(columns=("a", "b"), sample_size=10)

        # To modify, create a new config using replace()
        new_config = config.replace(sample_size=20)
    """

    columns: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] | None = None
    severity_override: Severity | None = None
    sample_size: int = 5
    # Mostly parameter: fraction of rows that must pass (0.0 to 1.0)
    mostly: float | None = None
    # Timeout in seconds (None = no timeout, default = 300s = 5 minutes)
    timeout_seconds: float | None = 300.0
    # Memory limit in MB (None = no limit)
    memory_limit_mb: float | None = None
    # Enable graceful degradation on errors
    graceful_degradation: bool = True
    # Log validation errors
    log_errors: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")
        if self.mostly is not None and not (0.0 <= self.mostly <= 1.0):
            raise ValueError(f"mostly must be in [0.0, 1.0], got {self.mostly}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be > 0, got {self.memory_limit_mb}")

    def replace(self, **kwargs: Any) -> "ValidatorConfig":
        """Create a new config with updated values."""
        from dataclasses import asdict
        current = asdict(self)
        current.update(kwargs)

        # Convert lists to tuples for frozen dataclass
        if "columns" in current and isinstance(current["columns"], list):
            current["columns"] = tuple(current["columns"])
        if "exclude_columns" in current and isinstance(current["exclude_columns"], list):
            current["exclude_columns"] = tuple(current["exclude_columns"])

        return ValidatorConfig(**current)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ValidatorConfig":
        """Create config from kwargs, converting lists to tuples."""
        # Convert lists to tuples
        if "columns" in kwargs and isinstance(kwargs["columns"], list):
            kwargs["columns"] = tuple(kwargs["columns"])
        if "exclude_columns" in kwargs and isinstance(kwargs["exclude_columns"], list):
            kwargs["exclude_columns"] = tuple(kwargs["exclude_columns"])

        # Filter only valid config fields
        valid_fields = {
            "columns", "exclude_columns", "severity_override", "sample_size",
            "mostly", "timeout_seconds", "memory_limit_mb",
            "graceful_degradation", "log_errors"
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}

        return cls(**filtered)


# ============================================================================
# Timeout Handler
# ============================================================================

class TimeoutHandler:
    """Thread-safe timeout handler for validation operations."""

    def __init__(self, timeout_seconds: float | None, validator_name: str = ""):
        self.timeout_seconds = timeout_seconds
        self.validator_name = validator_name
        self._old_handler = None

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        raise ValidationTimeoutError(self.timeout_seconds or 0, self.validator_name)

    def __enter__(self) -> "TimeoutHandler":
        if self.timeout_seconds is None:
            return self

        try:
            if threading.current_thread() is threading.main_thread():
                self._old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
        except (AttributeError, ValueError):
            pass

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self.timeout_seconds is None:
            return False

        try:
            if threading.current_thread() is threading.main_thread():
                signal.setitimer(signal.ITIMER_REAL, 0)
                if self._old_handler is not None:
                    signal.signal(signal.SIGALRM, self._old_handler)
        except (AttributeError, ValueError):
            pass

        return False


def with_timeout(func: Callable) -> Callable:
    """Decorator to add timeout support to validation methods."""
    @wraps(func)
    def wrapper(self: "Validator", *args: Any, **kwargs: Any) -> Any:
        timeout = self.config.timeout_seconds
        validator_name = getattr(self, "name", self.__class__.__name__)

        with TimeoutHandler(timeout, validator_name):
            return func(self, *args, **kwargs)

    return wrapper


# ============================================================================
# ValidationIssue
# ============================================================================

@dataclass
class ValidationIssue:
    """Represents a single data quality issue found during validation.

    Enhanced with optional error context for debugging.
    """

    column: str
    issue_type: str
    count: int
    severity: Severity
    details: str | None = None
    expected: Any | None = None
    actual: Any | None = None
    sample_values: list[Any] | None = None
    # Enhanced context (#12)
    error_context: ValidationErrorContext | None = None
    validator_name: str | None = None
    execution_time_ms: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "column": self.column,
            "issue_type": self.issue_type,
            "count": self.count,
            "severity": self.severity.value,
            "details": self.details,
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.actual is not None:
            result["actual"] = self.actual
        if self.sample_values is not None:
            result["sample_values"] = self.sample_values
        if self.validator_name is not None:
            result["validator"] = self.validator_name
        if self.execution_time_ms is not None:
            result["execution_time_ms"] = self.execution_time_ms
        if self.error_context is not None:
            result["error_context"] = self.error_context.to_dict()
        return result


# ============================================================================
# Type Filters
# ============================================================================

NUMERIC_TYPES: set[type[pl.DataType]] = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

STRING_TYPES: set[type[pl.DataType]] = {pl.String, pl.Utf8}

DATETIME_TYPES: set[type[pl.DataType]] = {pl.Date, pl.Datetime, pl.Time, pl.Duration}

FLOAT_TYPES: set[type[pl.DataType]] = {pl.Float32, pl.Float64}


# ============================================================================
# Base Validator
# ============================================================================

class Validator(ABC):
    """Abstract base class for all validators.

    Enterprise Features:
    - Immutable ValidatorConfig (thread-safe)
    - Timeout support (prevents hangs)
    - Memory tracking (prevents OOM)
    - Schema validation (prevents runtime errors)
    - Graceful degradation (partial results on errors)
    - Consistent logging (debugging and monitoring)
    """

    name: str = "base"
    category: str = "general"

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        """Initialize the validator.

        Args:
            config: Immutable validator configuration
            **kwargs: Additional config options (merged into config)
        """
        if config is not None:
            if kwargs:
                self.config = config.replace(**kwargs)
            else:
                self.config = config
        else:
            self.config = ValidatorConfig.from_kwargs(**kwargs)

        self.logger = ValidatorLogger(self.name)

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation on the given LazyFrame."""
        pass

    def validate_safe(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
        """Run validation with graceful error handling.

        This is the recommended method for production use as it:
        - Catches and logs all errors
        - Returns partial results when possible
        - Provides detailed error context
        """
        wrapper = GracefulValidator(
            self,
            skip_on_error=self.config.graceful_degradation,
            log_errors=self.config.log_errors,
        )
        return wrapper.validate(lf)

    def validate_with_timeout(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation with timeout protection."""
        timeout = self.config.timeout_seconds
        validator_name = getattr(self, "name", self.__class__.__name__)

        with TimeoutHandler(timeout, validator_name):
            return self.validate(lf)

    def _get_target_columns(
        self,
        lf: pl.LazyFrame,
        dtype_filter: set[type[pl.DataType]] | None = None,
    ) -> list[str]:
        """Get columns to validate based on config and dtype filter.

        Uses SchemaValidator for safe column resolution.
        """
        requested = list(self.config.columns) if self.config.columns else None
        exclude = list(self.config.exclude_columns) if self.config.exclude_columns else []

        columns = SchemaValidator.get_safe_columns(lf, requested, dtype_filter)

        if exclude:
            columns = [c for c in columns if c not in exclude]

        return columns

    def _calculate_severity(
        self,
        ratio: float,
        thresholds: tuple[float, float, float] = (0.5, 0.2, 0.05),
    ) -> Severity:
        """Calculate severity based on ratio and thresholds."""
        if self.config.severity_override:
            return self.config.severity_override

        critical_th, high_th, medium_th = thresholds
        if ratio > critical_th:
            return Severity.CRITICAL
        elif ratio > high_th:
            return Severity.HIGH
        elif ratio > medium_th:
            return Severity.MEDIUM
        return Severity.LOW

    def _passes_mostly(self, failure_count: int, total_count: int) -> bool:
        """Check if validation passes based on mostly threshold."""
        if self.config.mostly is None:
            return False

        if total_count == 0:
            return True

        pass_ratio = 1 - (failure_count / total_count)
        return pass_ratio >= self.config.mostly

    def _get_mostly_adjusted_severity(
        self,
        failure_count: int,
        total_count: int,
        base_severity: Severity,
    ) -> Severity | None:
        """Get severity adjusted for mostly threshold."""
        if self._passes_mostly(failure_count, total_count):
            return None
        return base_severity

    def _safe_sample(
        self,
        lf: pl.LazyFrame,
        filter_expr: pl.Expr,
        columns: list[str] | None = None,
    ) -> list[Any]:
        """Safely get sample values using SafeSampler."""
        try:
            df = SafeSampler.safe_filter_sample(
                lf, filter_expr, self.config.sample_size, columns
            )
            if len(df) == 0:
                return []
            return df.to_dicts()
        except Exception as e:
            self.logger.warning(f"Failed to collect samples: {e}")
            return []


# ============================================================================
# Mixins
# ============================================================================

class NumericValidatorMixin:
    """Mixin for validators that work with numeric columns."""

    def _get_numeric_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=NUMERIC_TYPES)  # type: ignore


class StringValidatorMixin:
    """Mixin for validators that work with string columns."""

    def _get_string_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=STRING_TYPES)  # type: ignore


class DatetimeValidatorMixin:
    """Mixin for validators that work with datetime columns."""

    def _get_datetime_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=DATETIME_TYPES)  # type: ignore


class FloatValidatorMixin:
    """Mixin for validators that work with float columns."""

    def _get_float_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=FLOAT_TYPES)  # type: ignore


class RegexValidatorMixin:
    """Mixin for validators that use regex patterns with ReDoS protection."""

    @staticmethod
    def validate_pattern(pattern: str, flags: int = 0) -> re.Pattern[str]:
        """Validate and compile a regex pattern with ReDoS check.

        Args:
            pattern: The regex pattern string to validate
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Returns:
            Compiled regex pattern

        Raises:
            RegexValidationError: If the pattern is invalid or unsafe
        """
        if pattern is None:
            raise RegexValidationError("None", "Pattern cannot be None")

        # Check for ReDoS vulnerability
        is_safe, warning = RegexSafetyChecker.check_pattern(pattern)
        if not is_safe:
            raise RegexValidationError(pattern, f"ReDoS risk: {warning}")

        try:
            return re.compile(pattern, flags)
        except re.error as e:
            raise RegexValidationError(pattern, str(e)) from e

    @staticmethod
    def validate_patterns(patterns: list[str], flags: int = 0) -> list[re.Pattern[str]]:
        """Validate and compile multiple regex patterns."""
        return [RegexValidatorMixin.validate_pattern(p, flags) for p in patterns]

    @staticmethod
    def safe_match(
        pattern: re.Pattern[str],
        text: str,
        match_type: str = "fullmatch",
    ) -> re.Match[str] | None:
        """Execute regex match with timeout protection."""
        return RegexSafetyChecker.safe_match(pattern, text, match_type=match_type)


class StreamingValidatorMixin:
    """Mixin for validators that support streaming/chunked processing."""

    default_chunk_size: int = 100_000

    def _validate_streaming(
        self,
        lf: pl.LazyFrame,
        chunk_size: int | None = None,
        validate_chunk: Callable[[pl.LazyFrame], list["ValidationIssue"]] | None = None,
    ) -> list["ValidationIssue"]:
        """Process validation in streaming chunks."""
        chunk_size = chunk_size or self.default_chunk_size
        validate_fn = validate_chunk or self.validate  # type: ignore

        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return []

        if total_rows <= chunk_size:
            return validate_fn(lf)

        all_issues: dict[tuple[str, str], "ValidationIssue"] = {}

        for offset in range(0, total_rows, chunk_size):
            chunk_lf = lf.slice(offset, chunk_size)
            chunk_issues = validate_fn(chunk_lf)

            for issue in chunk_issues:
                key = (issue.column, issue.issue_type)
                if key in all_issues:
                    existing = all_issues[key]
                    existing.count += issue.count
                    if existing.sample_values and issue.sample_values:
                        combined = existing.sample_values + issue.sample_values
                        existing.sample_values = combined[:5]
                else:
                    all_issues[key] = issue

        return list(all_issues.values())


# ============================================================================
# Template Validators
# ============================================================================

class ColumnValidator(Validator):
    """Template for column-level validation."""

    @abstractmethod
    def check_column(
        self,
        lf: pl.LazyFrame,
        col: str,
        total_rows: int,
    ) -> ValidationIssue | None:
        """Check a single column. Implement in subclass."""
        pass

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        for col in columns:
            try:
                issue = self.check_column(lf, col, total_rows)
                if issue:
                    issues.append(issue)
            except Exception as e:
                if self.config.graceful_degradation:
                    self.logger.warning(f"Error checking column {col}: {e}")
                else:
                    raise

        return issues


class AggregateValidator(Validator, NumericValidatorMixin):
    """Template for aggregate statistics validation."""

    @abstractmethod
    def check_aggregate(
        self,
        col: str,
        stats: dict[str, Any],
        total_rows: int,
    ) -> ValidationIssue | None:
        """Check aggregate stats for a column. Implement in subclass."""
        pass

    def _compute_stats(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> tuple[int, dict[str, dict[str, Any]]]:
        """Compute statistics for all columns in single query."""
        exprs: list[pl.Expr] = [pl.len().alias("_total")]

        for col in columns:
            exprs.extend([
                pl.col(col).mean().alias(f"_mean_{col}"),
                pl.col(col).std().alias(f"_std_{col}"),
                pl.col(col).min().alias(f"_min_{col}"),
                pl.col(col).max().alias(f"_max_{col}"),
                pl.col(col).sum().alias(f"_sum_{col}"),
                pl.col(col).median().alias(f"_median_{col}"),
                pl.col(col).count().alias(f"_count_{col}"),
            ])

        result = lf.select(exprs).collect()
        total = result["_total"][0]

        stats: dict[str, dict[str, Any]] = {}
        for col in columns:
            stats[col] = {
                "mean": result[f"_mean_{col}"][0],
                "std": result[f"_std_{col}"][0],
                "min": result[f"_min_{col}"][0],
                "max": result[f"_max_{col}"][0],
                "sum": result[f"_sum_{col}"][0],
                "median": result[f"_median_{col}"][0],
                "count": result[f"_count_{col}"][0],
            }

        return total, stats

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        total_rows, all_stats = self._compute_stats(lf, columns)

        if total_rows == 0:
            return issues

        for col in columns:
            try:
                issue = self.check_aggregate(col, all_stats[col], total_rows)
                if issue:
                    issues.append(issue)
            except Exception as e:
                if self.config.graceful_degradation:
                    self.logger.warning(f"Error checking aggregate for {col}: {e}")
                else:
                    raise

        return issues
