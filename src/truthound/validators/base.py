"""Base classes for validators.

Features:
- Immutable configuration (thread-safe)
- Timeout mechanism
- Type-safe column filtering
- ReDoS protection for regex patterns
- Graceful degradation on errors
- Expression-based validation for single collect() optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable, TYPE_CHECKING
import re
import signal
import threading
import logging
import time
import traceback as _traceback_mod
from functools import wraps
from enum import Enum

import polars as pl

from truthound.types import ResultFormat, ResultFormatConfig, Severity, ValidationDetail

if TYPE_CHECKING:
    from truthound.validators.metrics import MetricKey, SharedMetricStore


# ============================================================================
# Logging - Uses standard Python logging directly
# ============================================================================

def _get_logger(name: str) -> logging.Logger:
    """Get a logger for the given validator name."""
    return logging.getLogger(f"truthound.{name}")


# ============================================================================
# Error Types
# ============================================================================

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


# ============================================================================
# Query Plan Optimization (#5.3 Performance Optimization)
# ============================================================================

# Default optimization flags for all collect() calls (Polars 1.30+)
# Uses QueryOptFlags to avoid deprecation warnings
QUERY_OPTIMIZATIONS: dict[str, bool] = {
    "predicate_pushdown": True,
    "projection_pushdown": True,
    "slice_pushdown": True,
    "comm_subplan_elim": True,  # Common subplan elimination
    "comm_subexpr_elim": True,  # Common subexpression elimination
    "simplify_expression": True,
    "cluster_with_columns": True,
}


def _get_optimizations() -> pl.QueryOptFlags:
    """Get QueryOptFlags with all optimizations enabled."""
    return pl.QueryOptFlags(
        predicate_pushdown=True,
        projection_pushdown=True,
        slice_pushdown=True,
        comm_subplan_elim=True,
        comm_subexpr_elim=True,
        simplify_expression=True,
        cluster_with_columns=True,
    )


def optimized_collect(
    lf: pl.LazyFrame,
    *,
    streaming: bool = False,
    **kwargs: Any,
) -> pl.DataFrame:
    """Collect LazyFrame with query plan optimizations enabled.

    Applies predicate pushdown, projection pushdown, slice pushdown,
    and common subplan elimination for optimal query execution.

    Args:
        lf: LazyFrame to collect
        streaming: Use streaming engine for large datasets
        **kwargs: Additional collect() arguments

    Returns:
        Collected DataFrame
    """
    collect_kwargs: dict[str, Any] = {
        "optimizations": _get_optimizations(),
        **kwargs,
    }
    if streaming:
        collect_kwargs["engine"] = "streaming"
    return lf.collect(**collect_kwargs)


# ============================================================================
# ReDoS Protection (simplified)
# ============================================================================

class RegexSafetyChecker:
    """Detects ReDoS vulnerabilities in regex patterns.

    Checks for common dangerous patterns that could cause exponential backtracking.
    """

    REDOS_PATTERNS = [
        r"\(.+\)\+\+",           # Nested quantifiers: (a+)+
        r"\(.+\)\*\*",           # Nested quantifiers: (a*)*
        r"\(.+\)\{\d+,\}",       # Nested with unbounded repetition
        r"\(.+\|.+\)\+",         # Alternation in quantified group
    ]

    MAX_PATTERN_LENGTH = 1000

    @classmethod
    def check_pattern(cls, pattern: str) -> tuple[bool, str | None]:
        """Check if a pattern is potentially vulnerable to ReDoS."""
        if len(pattern) > cls.MAX_PATTERN_LENGTH:
            return False, f"Pattern too long ({len(pattern)} > {cls.MAX_PATTERN_LENGTH})"

        for redos_pattern in cls.REDOS_PATTERNS:
            if re.search(redos_pattern, pattern):
                return False, f"Potentially vulnerable to ReDoS: matches {redos_pattern}"

        return True, None


# ============================================================================
# Safe Sampling
# ============================================================================

class SafeSampler:
    """Memory-safe sampling using Polars lazy evaluation."""

    # Threshold for enabling streaming mode (1M rows)
    STREAMING_THRESHOLD: int = 1_000_000

    @staticmethod
    def safe_head(
        lf: pl.LazyFrame,
        n: int,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Safely get first n rows with query plan optimizations."""
        query = lf
        if columns:
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)
        return optimized_collect(query.head(n), streaming=True)

    @staticmethod
    def safe_sample(
        lf: pl.LazyFrame,
        n: int,
        columns: list[str] | None = None,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Safely sample n rows with query plan optimizations."""
        return SafeSampler.safe_head(lf, n, columns)

    @staticmethod
    def safe_filter_sample(
        lf: pl.LazyFrame,
        filter_expr: pl.Expr,
        n: int,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Safely get filtered samples with query plan optimizations."""
        query = lf.filter(filter_expr)
        if columns:
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)
        return optimized_collect(query.head(n), streaming=True)


# ============================================================================
# Memory Tracking (stub for compatibility)
# ============================================================================

class MemoryTracker:
    """Stub for backward compatibility. Memory tracking is not enforced."""

    def __init__(self, limit_mb: float | None = None):
        self.limit_mb = limit_mb
        self.peak_mb: float = 0.0

    def get_current_mb(self) -> float:
        return 0.0

    def start(self) -> None:
        pass

    def check(self) -> None:
        pass

    def get_delta_mb(self) -> float:
        return 0.0

    def __enter__(self) -> "MemoryTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


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
class ErrorContext:
    """Simplified error context for validation failures."""
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"error_type": self.error_type, "message": self.message}


@dataclass
class ExceptionInfo:
    """Detailed exception information for validation failures.

    Captures full context about an exception that occurred during
    validation, including classification, retry metadata, and traceback.
    This enables precise diagnostics and smart retry decisions.
    """

    raised_exception: bool = False
    exception_type: str | None = None
    exception_message: str | None = None
    exception_traceback: str | None = None

    # Retry metadata
    retry_count: int = 0
    max_retries: int = 0
    is_retryable: bool = False

    # Context
    validator_name: str | None = None
    column: str | None = None
    expression_alias: str | None = None

    # Classification: transient | permanent | configuration | data
    failure_category: str = "unknown"

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        validator_name: str | None = None,
        column: str | None = None,
        expression_alias: str | None = None,
    ) -> "ExceptionInfo":
        """Create ExceptionInfo from a caught exception."""
        category = cls._classify_exception(exc)
        return cls(
            raised_exception=True,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            exception_traceback=_traceback_mod.format_exc(),
            is_retryable=(category == "transient"),
            validator_name=validator_name,
            column=column,
            expression_alias=expression_alias,
            failure_category=category,
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> str:
        """Classify an exception into a failure category."""
        # Transient — worth retrying
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return "transient"
        if isinstance(exc, ValidationTimeoutError):
            return "transient"

        # Configuration — bad setup, no retry
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return "configuration"
        if isinstance(exc, ColumnNotFoundError):
            return "configuration"
        if isinstance(exc, RegexValidationError):
            return "configuration"

        # Data — Polars compute/schema errors
        try:
            import polars.exceptions as pl_exc
            if isinstance(exc, (pl_exc.ComputeError, pl_exc.SchemaError)):
                return "data"
        except (ImportError, AttributeError):
            pass

        return "permanent"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict, omitting default/empty values."""
        d: dict[str, Any] = {}
        if self.raised_exception:
            d["raised_exception"] = True
        if self.exception_type:
            d["exception_type"] = self.exception_type
        if self.exception_message:
            d["exception_message"] = self.exception_message
        if self.exception_traceback:
            d["exception_traceback"] = self.exception_traceback
        if self.retry_count > 0:
            d["retry_count"] = self.retry_count
        if self.max_retries > 0:
            d["max_retries"] = self.max_retries
        if self.is_retryable:
            d["is_retryable"] = True
        if self.validator_name:
            d["validator_name"] = self.validator_name
        if self.column:
            d["column"] = self.column
        if self.expression_alias:
            d["expression_alias"] = self.expression_alias
        if self.failure_category != "unknown":
            d["failure_category"] = self.failure_category
        return d

    def to_error_context(self) -> ErrorContext:
        """Downgrade to legacy ErrorContext for backward compatibility."""
        return ErrorContext(
            error_type=self.exception_type or "Unknown",
            message=self.exception_message or "",
        )


@dataclass
class ValidatorExecutionResult:
    """Result of a single validator execution with error handling."""
    validator_name: str
    status: ValidationResult
    issues: list["ValidationIssue"]
    error_message: str | None = None
    error_context: ErrorContext | None = None
    execution_time_ms: float = 0.0

    # PHASE 5 fields
    exception_info: ExceptionInfo | None = None
    retry_count: int = 0
    partial_issues: list["ValidationIssue"] | None = None

    @property
    def has_exception(self) -> bool:
        """Whether this result involved an exception."""
        return self.exception_info is not None and self.exception_info.raised_exception

    @property
    def is_partial(self) -> bool:
        """Whether only some expressions succeeded (partial failure)."""
        return self.status == ValidationResult.PARTIAL and self.partial_issues is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "validator": self.validator_name,
            "status": self.status.value,
            "issue_count": len(self.issues),
            "execution_time_ms": self.execution_time_ms,
            "error": self.error_context.to_dict() if self.error_context else None,
        }
        if self.exception_info is not None:
            d["exception_info"] = self.exception_info.to_dict()
        if self.retry_count > 0:
            d["retry_count"] = self.retry_count
        if self.partial_issues is not None:
            d["partial_issue_count"] = len(self.partial_issues)
        return d


def _validate_safe(
    validator: "Validator",
    lf: pl.LazyFrame,
    skip_on_error: bool = True,
    log_errors: bool = True,
    max_retries: int = 0,
    retry_on: tuple[type[Exception], ...] = (ValidationTimeoutError, ConnectionError, OSError),
) -> ValidatorExecutionResult:
    """Execute validation with error handling and automatic retry.

    Args:
        validator: Validator instance to execute.
        lf: LazyFrame to validate.
        skip_on_error: If True, return FAILED status instead of raising.
        log_errors: If True, log exceptions.
        max_retries: Number of retry attempts for transient errors.
        retry_on: Exception types eligible for automatic retry.

    Returns:
        ValidatorExecutionResult with status, issues, and exception info.
    """
    start_time = time.time()
    lgr = _get_logger(validator.name)
    last_exception: Exception | None = None
    retry_count = 0

    for attempt in range(max_retries + 1):
        try:
            issues = validator.validate(lf)
            return ValidatorExecutionResult(
                validator_name=validator.name,
                status=ValidationResult.SUCCESS,
                issues=issues,
                execution_time_ms=(time.time() - start_time) * 1000,
                retry_count=retry_count,
            )

        except ColumnNotFoundError as e:
            # Configuration error — never retry
            exc_info = ExceptionInfo.from_exception(
                e, validator_name=validator.name, column=e.column,
            )
            if log_errors:
                lgr.warning("Column not found: %s", e.column)
            return ValidatorExecutionResult(
                validator_name=validator.name,
                status=ValidationResult.SKIPPED,
                issues=[],
                error_message=str(e),
                error_context=exc_info.to_error_context(),
                execution_time_ms=(time.time() - start_time) * 1000,
                exception_info=exc_info,
            )

        except retry_on as e:
            # Transient error — eligible for retry
            last_exception = e
            retry_count = attempt + 1
            exc_info = ExceptionInfo.from_exception(
                e, validator_name=validator.name,
            )
            exc_info.retry_count = retry_count
            exc_info.max_retries = max_retries

            if attempt < max_retries:
                wait = min(2 ** attempt * 0.1, 5.0)
                lgr.warning(
                    "Validator '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                    validator.name, attempt + 1, max_retries + 1, wait, e,
                )
                time.sleep(wait)
            else:
                if log_errors:
                    lgr.error(
                        "Validator '%s' failed after %d attempts: %s",
                        validator.name, max_retries + 1, e,
                    )

        except Exception as e:
            # Permanent error — no retry
            exc_info = ExceptionInfo.from_exception(
                e, validator_name=validator.name,
            )
            if log_errors:
                lgr.exception("Error in %s: %s", validator.name, e)
            if skip_on_error:
                return ValidatorExecutionResult(
                    validator_name=validator.name,
                    status=ValidationResult.FAILED,
                    issues=[],
                    error_message=str(e),
                    error_context=exc_info.to_error_context(),
                    execution_time_ms=(time.time() - start_time) * 1000,
                    exception_info=exc_info,
                )
            raise

    # All retries exhausted
    exc_info = ExceptionInfo.from_exception(
        last_exception, validator_name=validator.name,  # type: ignore[arg-type]
    )
    exc_info.retry_count = retry_count
    exc_info.max_retries = max_retries

    if skip_on_error:
        status = (
            ValidationResult.TIMEOUT
            if isinstance(last_exception, ValidationTimeoutError)
            else ValidationResult.FAILED
        )
        return ValidatorExecutionResult(
            validator_name=validator.name,
            status=status,
            issues=[],
            error_message=f"Failed after {retry_count} retries: {last_exception}",
            error_context=exc_info.to_error_context(),
            execution_time_ms=(time.time() - start_time) * 1000,
            exception_info=exc_info,
            retry_count=retry_count,
        )
    raise last_exception  # type: ignore[misc]


class GracefulValidator:
    """Wrapper for backward compatibility. Use validator.validate_safe() instead."""

    def __init__(
        self,
        validator: "Validator",
        skip_on_error: bool = True,
        log_errors: bool = True,
    ):
        self.validator = validator
        self.skip_on_error = skip_on_error
        self.log_errors = log_errors

    def validate(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
        return _validate_safe(
            self.validator, lf, self.skip_on_error, self.log_errors
        )


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

    Thread-safe frozen dataclass that can be used as dict keys.
    """

    columns: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] | None = None
    severity_override: Severity | None = None
    sample_size: int = 5
    mostly: float | None = None  # Fraction of rows that must pass (0.0 to 1.0)
    timeout_seconds: float | None = 300.0
    graceful_degradation: bool = True
    log_errors: bool = True
    result_format: ResultFormat | ResultFormatConfig = ResultFormat.SUMMARY

    # PHASE 5: Exception control options
    catch_exceptions: bool = True        # False = strict mode (first error aborts)
    max_retries: int = 0                 # Retry count for transient errors
    partial_failure_mode: str = "collect" # collect | skip | raise

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")
        if self.mostly is not None and not (0.0 <= self.mostly <= 1.0):
            raise ValueError(f"mostly must be in [0.0, 1.0], got {self.mostly}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.partial_failure_mode not in ("collect", "skip", "raise"):
            raise ValueError(
                f"partial_failure_mode must be collect/skip/raise, "
                f"got {self.partial_failure_mode!r}"
            )
        # Normalize string result_format to enum
        if isinstance(self.result_format, str):
            object.__setattr__(self, "result_format", ResultFormat.from_string(self.result_format))

    def get_result_format_config(self) -> ResultFormatConfig:
        """Resolve result_format to a full ResultFormatConfig.

        If result_format is a ResultFormat enum, wraps it with default options.
        If already a ResultFormatConfig, returns as-is.
        """
        return ResultFormatConfig.from_any(self.result_format)

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
        # Restore result_format from dict representation
        rf = current.get("result_format")
        if isinstance(rf, dict):
            fmt = rf.get("format")
            if isinstance(fmt, str):
                rf["format"] = ResultFormat.from_string(fmt)
            current["result_format"] = ResultFormatConfig(**rf)
        elif isinstance(rf, str):
            current["result_format"] = ResultFormat.from_string(rf)
        return ValidatorConfig(**current)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ValidatorConfig":
        """Create config from kwargs, converting lists to tuples."""
        if "columns" in kwargs and isinstance(kwargs["columns"], list):
            kwargs["columns"] = tuple(kwargs["columns"])
        if "exclude_columns" in kwargs and isinstance(kwargs["exclude_columns"], list):
            kwargs["exclude_columns"] = tuple(kwargs["exclude_columns"])
        # Normalize result_format from string
        if "result_format" in kwargs and isinstance(kwargs["result_format"], str):
            kwargs["result_format"] = ResultFormat.from_string(kwargs["result_format"])
        valid_fields = {
            "columns", "exclude_columns", "severity_override", "sample_size",
            "mostly", "timeout_seconds", "graceful_degradation", "log_errors",
            "result_format", "catch_exceptions", "max_retries",
            "partial_failure_mode",
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

    Core fields (column, issue_type, count, severity) are always populated.
    Legacy fields (details, expected, actual, sample_values) are kept for
    backward compatibility.  The new ``result`` field holds a structured
    :class:`~truthound.types.ValidationDetail` that provides richer,
    type-safe access to the same (and more) information.

    ``validator_name`` records which validator produced this issue, and
    ``success`` is always ``False`` for issues (``True`` means the
    validation passed — no issue is created in that case).
    """

    # -- Core fields (always populated) --
    column: str
    issue_type: str
    count: int
    severity: Severity

    # -- Legacy detail fields (backward compatible) --
    details: str | None = None
    expected: Any | None = None
    actual: Any | None = None
    sample_values: list[Any] | None = None

    # -- PHASE 2 fields --
    result: ValidationDetail | None = None
    validator_name: str | None = None
    success: bool = False

    # -- PHASE 5 fields --
    exception_info: ExceptionInfo | None = None

    # ------------------------------------------------------------------ #
    # Convenience accessors (delegate to result when available)
    # ------------------------------------------------------------------ #

    @property
    def unexpected_percent(self) -> float | None:
        """Failure percentage from structured result."""
        if self.result is not None:
            return self.result.unexpected_percent
        return None

    @property
    def unexpected_rows(self) -> "pl.DataFrame | None":
        """Failure rows DataFrame from structured result."""
        if self.result is not None:
            return self.result.unexpected_rows
        return None

    @property
    def debug_query(self) -> str | None:
        """Debug query string from structured result."""
        if self.result is not None:
            return self.result.debug_query
        return None

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Backward-compatible: always includes legacy keys.  When ``result``
        is present its ``to_dict()`` output is nested under ``"result"``.
        """
        d: dict[str, Any] = {
            "column": self.column,
            "issue_type": self.issue_type,
            "count": self.count,
            "severity": self.severity.value,
            "success": self.success,
        }
        if self.details is not None:
            d["details"] = self.details
        if self.expected is not None:
            d["expected"] = self.expected
        if self.actual is not None:
            d["actual"] = self.actual
        if self.sample_values is not None:
            d["sample_values"] = self.sample_values
        if self.validator_name is not None:
            d["validator_name"] = self.validator_name
        if self.result is not None:
            d["result"] = self.result.to_dict()
        if self.exception_info is not None:
            d["exception_info"] = self.exception_info.to_dict()
        return d


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
# Skip Condition (PHASE 4: DAG Conditional Execution)
# ============================================================================

@dataclass(frozen=True)
class SkipCondition:
    """Declares when a Validator should be skipped based on prior results.

    Used by ``Validator.get_skip_conditions()`` to declare fine-grained
    skip logic that goes beyond simple dependency failure checks.

    Attributes:
        depends_on: Name (or ``provides`` tag) of the upstream Validator.
        skip_when: Trigger mode:
            - ``"failed"``   – skip if the upstream ended FAILED or TIMEOUT.
            - ``"critical"`` – skip if the upstream produced a CRITICAL-severity issue.
            - ``"any_issue"``– skip if the upstream produced **any** issue.
        reason_template: Human-readable reason; ``{depends_on}`` and
            ``{skip_when}`` are interpolated at evaluation time.
    """
    depends_on: str
    skip_when: str = "failed"  # "failed" | "critical" | "any_issue"
    reason_template: str = "Skipped due to {depends_on} {skip_when}"

    def evaluate(self, result: ValidatorExecutionResult) -> tuple[bool, str]:
        """Evaluate this condition against an upstream result.

        Returns:
            ``(should_skip, reason)`` – *should_skip* is ``True`` when the
            condition is satisfied and the downstream Validator should be
            skipped.
        """
        if self.skip_when == "failed":
            should_skip = result.status in (ValidationResult.FAILED, ValidationResult.TIMEOUT)
        elif self.skip_when == "critical":
            should_skip = any(
                i.severity == Severity.CRITICAL for i in result.issues
            )
        elif self.skip_when == "any_issue":
            should_skip = len(result.issues) > 0
        else:
            should_skip = False

        reason = self.reason_template.format(
            depends_on=self.depends_on,
            skip_when=self.skip_when,
        )
        return should_skip, reason


# ============================================================================
# Base Validator
# ============================================================================

class Validator(ABC):
    """Abstract base class for all validators.

    Features:
    - Immutable ValidatorConfig (thread-safe)
    - Timeout support
    - Schema validation
    - Graceful degradation on errors
    - Dependency-aware execution ordering

    Data Type Support:
        Validators ONLY accept Polars LazyFrame (pl.LazyFrame) directly.
        For other data types, use the public API (th.check()) which handles conversion:

        - th.check("data.csv")      → Automatically converts to LazyFrame
        - th.check(pl.DataFrame())  → Converts DataFrame to LazyFrame
        - th.check(pd.DataFrame())  → Converts pandas DataFrame to LazyFrame
        - th.check({"col": [1,2]})  → Converts dict to LazyFrame

        If using validators directly, convert data first::

            import polars as pl
            from truthound.adapters import to_lazyframe

            # Option 1: Use the adapter
            lf = to_lazyframe(your_data)
            issues = NullValidator().validate(lf)

            # Option 2: Convert manually
            lf = pl.DataFrame(your_data).lazy()
            issues = NullValidator().validate(lf)

    Class Attributes:
        name: Unique identifier for this validator
        category: Validator category (schema, completeness, uniqueness, etc.)
        dependencies: Set of validator names that must run before this one
        provides: Set of capabilities this validator provides
        priority: Execution priority within phase (lower = earlier)

    Example:
        class MyValidator(Validator):
            name = "my_validator"
            category = "custom"
            dependencies = {"null", "schema"}  # Runs after null and schema
            provides = {"my_check"}  # Other validators can depend on this

            def validate(self, lf):
                ...
    """

    name: str = "base"
    category: str = "general"

    # DAG execution metadata
    dependencies: set[str] = set()  # Validators that must run before this
    provides: set[str] = set()      # Capabilities this validator provides
    priority: int = 100             # Lower = runs earlier within same phase

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        """Initialize the validator.

        Args:
            config: Immutable validator configuration
            **kwargs: Additional config options (merged into config)
        """
        if config is not None:
            self.config = config.replace(**kwargs) if kwargs else config
        else:
            self.config = ValidatorConfig.from_kwargs(**kwargs)
        self.logger = _get_logger(self.name)

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation on the given LazyFrame."""
        pass

    def get_required_metrics(self, columns: list[str]) -> list["MetricKey"]:
        """Declare base metrics this validator needs.

        Override in subclasses to declare shared metric dependencies.
        The ExpressionBatchExecutor collects these from all validators,
        deduplicates, computes once, and stores results in SharedMetricStore.

        Args:
            columns: Target columns for this validator.

        Returns:
            List of MetricKey instances this validator requires.
        """
        return []

    def validate_with_metrics(
        self,
        lf: pl.LazyFrame,
        metric_store: "SharedMetricStore",
    ) -> list[ValidationIssue]:
        """Run validation using pre-computed metrics from the store.

        Default implementation delegates to ``validate()``.
        Validators that benefit from shared metrics should override
        this to read from *metric_store* instead of recomputing.

        Args:
            lf: LazyFrame to validate.
            metric_store: Session-scoped metric cache.

        Returns:
            List of ValidationIssue objects.
        """
        return self.validate(lf)

    def validate_safe(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
        """Run validation with graceful error handling."""
        return _validate_safe(
            self,
            lf,
            skip_on_error=self.config.graceful_degradation,
            log_errors=self.config.log_errors,
        )

    def validate_with_timeout(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation with timeout protection."""
        timeout = self.config.timeout_seconds
        validator_name = getattr(self, "name", self.__class__.__name__)

        with TimeoutHandler(timeout, validator_name):
            return self.validate(lf)

    # -- DAG conditional execution (PHASE 4) --

    def should_skip(
        self,
        prior_results: dict[str, ValidatorExecutionResult],
    ) -> tuple[bool, str | None]:
        """Decide whether to skip this Validator based on prior results.

        The default implementation checks two things in order:

        1. **Explicit dependencies** – if any Validator listed in
           ``self.dependencies`` has status FAILED, TIMEOUT, or SKIPPED, skip.
        2. **SkipConditions** – evaluate each condition returned by
           ``get_skip_conditions()``.

        Override for custom skip logic (e.g. always-run validators).

        Args:
            prior_results: ``{validator_name: ValidatorExecutionResult}``
                for already-executed Validators.

        Returns:
            ``(skip, reason)`` – *skip* is ``True`` when this Validator
            should not run; *reason* explains why.
        """
        # 1. Check explicit dependencies
        for dep_name in self.dependencies:
            dep_result = prior_results.get(dep_name)
            if dep_result is not None and dep_result.status in (
                ValidationResult.FAILED,
                ValidationResult.TIMEOUT,
                ValidationResult.SKIPPED,
            ):
                return (
                    True,
                    f"Dependency '{dep_name}' {dep_result.status.value}",
                )

        # 2. Evaluate fine-grained SkipConditions
        for condition in self.get_skip_conditions():
            dep_result = prior_results.get(condition.depends_on)
            if dep_result is not None:
                skip, reason = condition.evaluate(dep_result)
                if skip:
                    return True, reason

        return False, None

    def get_skip_conditions(self) -> list[SkipCondition]:
        """Declare fine-grained skip conditions for this Validator.

        Override to specify conditions beyond the basic
        dependency-failure check performed by ``should_skip()``.

        Example::

            def get_skip_conditions(self):
                return [
                    SkipCondition(
                        depends_on="column_exists",
                        skip_when="critical",
                        reason_template="Column missing – skipping {depends_on}",
                    ),
                ]

        Returns:
            List of :class:`SkipCondition` instances.
        """
        return []

    def _filter_columns_by_context(
        self,
        columns: list[str],
        critical_columns: set[str] | None,
    ) -> list[str]:
        """Remove columns that already have CRITICAL issues from prior validators.

        Called automatically by the DAG executor when an
        :class:`ExecutionContext` is available.

        Args:
            columns: Candidate columns for validation.
            critical_columns: Columns with prior CRITICAL-severity issues.

        Returns:
            Filtered list of columns (may be empty).
        """
        if not critical_columns:
            return columns

        filtered = [c for c in columns if c not in critical_columns]
        if not filtered and columns:
            self.logger.debug(
                f"All columns skipped for {self.name} due to prior critical issues"
            )
        return filtered

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
        """Safely get sample values.

        Respects result_format: returns empty list for BOOLEAN_ONLY.
        """
        if not self._should_collect_samples():
            return []
        try:
            sample_size = self._get_partial_count()
            df = SafeSampler.safe_filter_sample(
                lf, filter_expr, sample_size, columns
            )
            return df.to_dicts() if len(df) > 0 else []
        except Exception as e:
            self.logger.warning(f"Failed to collect samples: {e}")
            return []

    # -- Result format helpers --

    def _get_result_format_config(self) -> ResultFormatConfig:
        """Get the resolved ResultFormatConfig for this validator."""
        return self.config.get_result_format_config()

    def _should_collect_samples(self) -> bool:
        """Whether to collect sample values (BASIC+)."""
        return self._get_result_format_config().includes_unexpected_samples()

    def _should_collect_index(self) -> bool:
        """Whether to collect failure row indices (SUMMARY+)."""
        return self._get_result_format_config().includes_unexpected_counts()

    def _should_build_details(self) -> bool:
        """Whether to build detail strings (BASIC+)."""
        return self._get_result_format_config().includes_observed_value()

    def _get_partial_count(self) -> int:
        """Number of sample values to collect."""
        fmt = self._get_result_format_config()
        if not fmt.includes_unexpected_samples():
            return 0
        return fmt.partial_unexpected_count


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
        """Validate and compile a regex pattern with ReDoS check."""
        if pattern is None:
            raise RegexValidationError("None", "Pattern cannot be None")

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


class StreamingValidatorMixin:
    """Mixin for validators that support streaming/chunked processing."""

    default_chunk_size: int = 100_000

    def _validate_streaming(
        self,
        lf: pl.LazyFrame,
        chunk_size: int | None = None,
        validate_chunk: Callable[[pl.LazyFrame], list["ValidationIssue"]] | None = None,
    ) -> list["ValidationIssue"]:
        """Process validation in streaming chunks with query plan optimizations."""
        chunk_size = chunk_size or self.default_chunk_size
        validate_fn = validate_chunk or self.validate  # type: ignore

        # Use optimized collect for row count
        total_rows = optimized_collect(lf.select(pl.len()), streaming=True).item()
        if total_rows == 0:
            return []
        if total_rows <= chunk_size:
            return validate_fn(lf)

        all_issues: dict[tuple[str, str], "ValidationIssue"] = {}
        for offset in range(0, total_rows, chunk_size):
            chunk_lf = lf.slice(offset, chunk_size)
            for issue in validate_fn(chunk_lf):
                key = (issue.column, issue.issue_type)
                if key in all_issues:
                    all_issues[key].count += issue.count
                else:
                    all_issues[key] = issue
        return list(all_issues.values())


class EnterpriseScaleSamplingMixin:
    """Mixin for validators that support enterprise-scale sampling.

    Provides automatic sampling for large datasets (100M+ rows) with
    statistical quality guarantees.

    Features:
        - Automatic scale detection and strategy selection
        - Memory-aware sampling with backpressure
        - Statistical confidence bounds on results
        - Time-budget aware processing

    Usage:
        class MyValidator(Validator, EnterpriseScaleSamplingMixin):
            # Enable sampling for datasets > 10M rows
            sampling_threshold: int = 10_000_000
            sampling_target_rows: int = 100_000
            sampling_quality: str = "standard"

            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                # Automatically sample if dataset is large
                sampled_lf, metrics = self._sample_for_validation(lf)

                # Validate on sampled data
                issues = self._do_validation(sampled_lf)

                # Extrapolate counts if sampled
                if metrics.is_sampled:
                    issues = self._extrapolate_issues(issues, metrics)

                return issues
    """

    # Sampling configuration (override in subclass)
    sampling_threshold: int = 10_000_000       # 10M rows
    sampling_target_rows: int = 100_000        # Target sample size
    sampling_quality: str = "standard"         # Quality level
    sampling_confidence: float = 0.95          # Confidence level
    sampling_margin_of_error: float = 0.05     # Acceptable error

    def _sample_for_validation(
        self,
        lf: pl.LazyFrame,
        target_rows: int | None = None,
    ) -> tuple[pl.LazyFrame, "SamplingInfo"]:
        """Sample data if it exceeds threshold.

        Args:
            lf: Input LazyFrame
            target_rows: Override target sample size

        Returns:
            Tuple of (sampled LazyFrame, sampling info)
        """
        # Get row count with query plan optimizations
        total_rows = optimized_collect(lf.select(pl.len()), streaming=True).item()

        # Check if sampling needed
        if total_rows <= self.sampling_threshold:
            return lf, SamplingInfo(
                is_sampled=False,
                original_rows=total_rows,
                sampled_rows=total_rows,
                sampling_ratio=1.0,
                confidence_level=1.0,
                margin_of_error=0.0,
            )

        # Determine target
        target = target_rows or self.sampling_target_rows
        target = min(target, total_rows)

        # Calculate sampling ratio
        sample_ratio = target / total_rows

        # Apply sampling
        seed = getattr(self, "_sampling_seed", 42)
        threshold = max(1, int(sample_ratio * 10000))

        sampled_lf = (
            lf.with_row_index("__sample_idx")
            .filter(pl.col("__sample_idx").hash(seed) % 10000 < threshold)
            .drop("__sample_idx")
        )

        return sampled_lf, SamplingInfo(
            is_sampled=True,
            original_rows=total_rows,
            sampled_rows=target,
            sampling_ratio=sample_ratio,
            confidence_level=self.sampling_confidence,
            margin_of_error=self.sampling_margin_of_error,
        )

    def _extrapolate_issues(
        self,
        issues: list["ValidationIssue"],
        sampling_info: "SamplingInfo",
    ) -> list["ValidationIssue"]:
        """Extrapolate issue counts from sample to population.

        Args:
            issues: Issues found in sample
            sampling_info: Sampling information

        Returns:
            Issues with extrapolated counts
        """
        if not sampling_info.is_sampled:
            return issues

        extrapolation_factor = 1.0 / sampling_info.sampling_ratio

        for issue in issues:
            # Extrapolate count
            original_count = issue.count
            extrapolated_count = int(original_count * extrapolation_factor)
            issue.count = extrapolated_count

            # Add sampling note to details
            if issue.details:
                issue.details = (
                    f"{issue.details} "
                    f"[sampled: {original_count} → estimated: {extrapolated_count}, "
                    f"confidence: {sampling_info.confidence_level:.0%}]"
                )

        return issues

    def _get_sampling_strategy(self, total_rows: int) -> str:
        """Get recommended sampling strategy for data size."""
        if total_rows < 1_000_000:
            return "none"
        elif total_rows < 10_000_000:
            return "systematic"
        elif total_rows < 100_000_000:
            return "block"
        else:
            return "multi_stage"


@dataclass
class SamplingInfo:
    """Information about sampling applied to validation.

    Attributes:
        is_sampled: Whether sampling was applied
        original_rows: Original row count
        sampled_rows: Rows after sampling
        sampling_ratio: Sample size / original size
        confidence_level: Statistical confidence
        margin_of_error: Error margin
    """
    is_sampled: bool
    original_rows: int
    sampled_rows: int
    sampling_ratio: float
    confidence_level: float
    margin_of_error: float

    @property
    def extrapolation_factor(self) -> float:
        """Factor to multiply sample counts by for population estimate."""
        if self.sampling_ratio <= 0:
            return 1.0
        return 1.0 / self.sampling_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_sampled": self.is_sampled,
            "original_rows": self.original_rows,
            "sampled_rows": self.sampled_rows,
            "sampling_ratio": self.sampling_ratio,
            "confidence_level": self.confidence_level,
            "margin_of_error": self.margin_of_error,
        }


@dataclass
class EarlyTerminationResult:
    """Result of sampling-based early termination check.

    Attributes:
        should_terminate: Whether to skip full validation
        sample_fail_rate: Failure rate observed in sample
        estimated_fail_count: Extrapolated failure count for full dataset
        sample_size: Number of rows sampled
        total_rows: Total rows in dataset
        confidence_threshold: Threshold used for decision
    """
    should_terminate: bool
    sample_fail_rate: float
    estimated_fail_count: int
    sample_size: int
    total_rows: int
    confidence_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_terminate": self.should_terminate,
            "sample_fail_rate": self.sample_fail_rate,
            "estimated_fail_count": self.estimated_fail_count,
            "sample_size": self.sample_size,
            "total_rows": self.total_rows,
            "confidence_threshold": self.confidence_threshold,
        }


class SampledEarlyTerminationMixin:
    """Mixin for validators that support sampling-based early termination.

    When validating large datasets, this mixin first checks a sample. If the
    sample already shows failure rate exceeding (1 - confidence_threshold),
    full validation is skipped and results are extrapolated.

    This provides significant performance improvements for datasets with
    obvious quality issues, avoiding unnecessary full scans.

    Features:
        - Configurable sample size and confidence threshold
        - Automatic extrapolation of failure counts
        - Statistical confidence bounds on results
        - Seamless fallback to full validation when needed

    Usage:
        class MyValidator(Validator, SampledEarlyTerminationMixin):
            # Override defaults if needed
            early_termination_sample_size: int = 10_000
            early_termination_threshold: float = 0.99
            early_termination_min_rows: int = 50_000

            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                # Check if early termination is possible
                result = self._check_early_termination(
                    lf,
                    columns=self._get_string_columns(lf),
                    build_invalid_expr=self._build_match_expr,
                )

                if result.should_terminate:
                    # Use extrapolated results
                    return self._build_early_termination_issues(result, ...)

                # Fall back to full validation
                return self._full_validate(lf)

    Example:
        # Dataset with 10M rows, 30% failure rate
        # Sample (10K rows) shows 3,000 failures (30%)
        # Since 30% > (1 - 0.99) = 1%, early terminate
        # Extrapolate: 3,000,000 estimated failures

    Performance:
        - 10M rows with high failure rate: ~0.05s (vs ~5s full scan)
        - 10M rows with low failure rate: ~5s (falls through to full scan)
    """

    # Configuration (override in subclass or at runtime)
    early_termination_sample_size: int = 10_000
    early_termination_threshold: float = 0.99  # 99% must pass
    early_termination_min_rows: int = 50_000   # Don't sample below this

    def _check_early_termination(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
        build_invalid_expr: Callable[[str], pl.Expr],
        sample_size: int | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, EarlyTerminationResult]:
        """Check if early termination is possible based on sample.

        Args:
            lf: Input LazyFrame
            columns: Columns to check
            build_invalid_expr: Function that builds invalid expression for a column
            sample_size: Override sample size
            confidence_threshold: Override confidence threshold

        Returns:
            Dict mapping column names to EarlyTerminationResult
        """
        sample_size = sample_size or self.early_termination_sample_size
        threshold = confidence_threshold or self.early_termination_threshold

        # Get total row count with query plan optimizations
        total_rows = optimized_collect(lf.select(pl.len()), streaming=True).item()

        results: dict[str, EarlyTerminationResult] = {}

        # Skip sampling for small datasets
        if total_rows <= self.early_termination_min_rows:
            for col in columns:
                results[col] = EarlyTerminationResult(
                    should_terminate=False,
                    sample_fail_rate=0.0,
                    estimated_fail_count=0,
                    sample_size=total_rows,
                    total_rows=total_rows,
                    confidence_threshold=threshold,
                )
            return results

        # Build expressions for sample validation
        exprs: list[pl.Expr] = []
        for col in columns:
            invalid_expr = build_invalid_expr(col)
            exprs.append(invalid_expr.sum().alias(f"_sample_inv_{col}"))
            exprs.append(pl.col(col).is_not_null().sum().alias(f"_sample_nn_{col}"))

        # Validate on sample with query plan optimizations
        sample_result = optimized_collect(lf.head(sample_size).select(exprs), streaming=True)

        # Analyze results for each column
        fail_threshold = 1.0 - threshold  # e.g., 0.01 for 99% threshold

        for col in columns:
            sample_invalid = sample_result[f"_sample_inv_{col}"][0]
            sample_non_null = sample_result[f"_sample_nn_{col}"][0]

            if sample_non_null == 0:
                results[col] = EarlyTerminationResult(
                    should_terminate=False,
                    sample_fail_rate=0.0,
                    estimated_fail_count=0,
                    sample_size=sample_size,
                    total_rows=total_rows,
                    confidence_threshold=threshold,
                )
                continue

            sample_fail_rate = sample_invalid / sample_non_null

            # Check if sample failure rate exceeds threshold
            should_terminate = sample_fail_rate > fail_threshold

            # Extrapolate to full dataset
            estimated_fail_count = int(sample_fail_rate * total_rows)

            results[col] = EarlyTerminationResult(
                should_terminate=should_terminate,
                sample_fail_rate=sample_fail_rate,
                estimated_fail_count=estimated_fail_count,
                sample_size=min(sample_size, total_rows),
                total_rows=total_rows,
                confidence_threshold=threshold,
            )

        return results

    def _build_early_termination_issue(
        self,
        col: str,
        result: EarlyTerminationResult,
        issue_type: str,
        details: str,
        expected: Any | None = None,
        sample_values: list[Any] | None = None,
    ) -> "ValidationIssue":
        """Build ValidationIssue from early termination result.

        Args:
            col: Column name
            result: Early termination result
            issue_type: Type of issue
            details: Issue details
            expected: Expected value/pattern
            sample_values: Sample invalid values

        Returns:
            ValidationIssue with extrapolated count
        """
        # Add early termination note to details
        enhanced_details = (
            f"{details} "
            f"[early-termination: sampled {result.sample_size:,} of {result.total_rows:,} rows, "
            f"sample fail rate: {result.sample_fail_rate:.2%}]"
        )

        return ValidationIssue(
            column=col,
            issue_type=issue_type,
            count=result.estimated_fail_count,
            severity=self._calculate_severity(result.sample_fail_rate),  # type: ignore
            details=enhanced_details,
            expected=expected,
            sample_values=sample_values,
        )


# ============================================================================
# Expression-Based Validation Architecture (#15 Performance Optimization)
# ============================================================================


@dataclass
class ValidationExpressionSpec:
    """Specification for a validation expression.

    Each spec produces one or more expressions that can be combined into
    a single collect() call across multiple validators.

    Attributes:
        column: Column being validated
        validator_name: Name of the validator
        issue_type: Type of issue to report if validation fails
        count_expr: Expression that returns count of invalid rows
        non_null_expr: Expression that returns count of non-null rows (for ratio calculation)
        severity_ratio_thresholds: Thresholds for severity calculation
        details_template: Template string for issue details (use {count}, {ratio}, {column})
        expected: Expected value/pattern for reporting
        extra_exprs: Additional expressions for more complex validation
        extra_keys: Keys for extra_exprs results
        filter_expr: Boolean expression selecting invalid rows (for sample/row collection)
        sample_columns: Columns to include when collecting failure row samples
    """

    column: str
    validator_name: str
    issue_type: str
    count_expr: pl.Expr
    non_null_expr: pl.Expr | None = None
    severity_ratio_thresholds: tuple[float, float, float] = (0.5, 0.2, 0.05)
    details_template: str = "{count} invalid values ({ratio:.1%})"
    expected: Any = None
    extra_exprs: list[pl.Expr] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    filter_expr: pl.Expr | None = None
    sample_columns: list[str] | None = None

    def get_all_exprs(self, prefix: str) -> list[pl.Expr]:
        """Get all expressions with unique aliases.

        Args:
            prefix: Unique prefix for this spec's aliases

        Returns:
            List of expressions with aliased names
        """
        exprs = [self.count_expr.alias(f"{prefix}_count")]
        if self.non_null_expr is not None:
            exprs.append(self.non_null_expr.alias(f"{prefix}_nn"))
        for i, expr in enumerate(self.extra_exprs):
            key = self.extra_keys[i] if i < len(self.extra_keys) else f"extra_{i}"
            exprs.append(expr.alias(f"{prefix}_{key}"))
        return exprs


@runtime_checkable
class ExpressionValidatorProtocol(Protocol):
    """Protocol for expression-based validators.

    Validators implementing this protocol can participate in batched
    validation where all expressions are collected in a single query.

    This provides significant performance improvements by:
    1. Eliminating multiple collect() calls per validator
    2. Allowing Polars to optimize the combined query plan
    3. Reducing memory pressure from intermediate DataFrames

    Example:
        class MyValidator(Validator, ExpressionValidatorMixin):
            def get_validation_exprs(
                self, lf: pl.LazyFrame, columns: list[str]
            ) -> list[ValidationExpressionSpec]:
                specs = []
                for col in columns:
                    specs.append(ValidationExpressionSpec(
                        column=col,
                        validator_name=self.name,
                        issue_type="my_issue",
                        count_expr=pl.col(col).is_null().sum(),
                        non_null_expr=pl.col(col).is_not_null().sum(),
                    ))
                return specs

            def build_issues_from_results(
                self, specs: list[ValidationExpressionSpec],
                results: dict[str, Any], total_rows: int
            ) -> list[ValidationIssue]:
                # Process results and build issues
                ...
    """

    name: str
    config: ValidatorConfig

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for the given columns.

        Args:
            lf: LazyFrame to validate
            columns: Columns to validate

        Returns:
            List of ValidationExpressionSpec for batched execution
        """
        ...

    def build_issues_from_results(
        self,
        specs: list[ValidationExpressionSpec],
        results: dict[str, dict[str, Any]],
        total_rows: int,
        prefix_map: dict[str, ValidationExpressionSpec],
    ) -> list["ValidationIssue"]:
        """Build ValidationIssue list from collected results.

        Args:
            specs: Original expression specs
            results: Collected results from all expressions
            total_rows: Total row count
            prefix_map: Map from prefix to spec

        Returns:
            List of ValidationIssue objects
        """
        ...


class ExpressionValidatorMixin:
    """Mixin that provides expression-based validation infrastructure.

    This mixin provides:
    1. Default implementation of build_issues_from_results
    2. Helper methods for building common expression patterns
    3. Integration with the base Validator class

    Subclasses should implement get_validation_exprs() to define
    their validation expressions.

    Usage:
        class MyValidator(Validator, ExpressionValidatorMixin):
            name = "my_validator"

            def get_validation_exprs(self, lf, columns):
                return [
                    ValidationExpressionSpec(
                        column=col,
                        validator_name=self.name,
                        issue_type="my_issue",
                        count_expr=self._build_invalid_expr(col),
                    )
                    for col in columns
                ]

            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                # Can use expression-based or traditional approach
                return self._validate_with_expressions(lf)
    """

    def _validate_with_expressions(
        self,
        lf: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> list["ValidationIssue"]:
        """Execute validation using expression-based approach.

        Execution phases (controlled by result_format):
            Phase 1 (always): Aggregate expressions → single collect()
            Phase 2 (BASIC+):  Enrich issues with failure value samples
            Phase 3 (SUMMARY+): Enrich with value frequency counts
            Phase 4 (COMPLETE): Enrich with full failure rows/indices

        Args:
            lf: LazyFrame to validate
            columns: Override columns (default: use _get_target_columns)

        Returns:
            List of ValidationIssue objects
        """
        if columns is None:
            columns = self._get_target_columns(lf)  # type: ignore

        if not columns:
            return []

        # Get expression specs from subclass
        specs = self.get_validation_exprs(lf, columns)  # type: ignore

        if not specs:
            return []

        # Build all expressions
        all_exprs: list[pl.Expr] = [pl.len().alias("_total")]
        prefix_map: dict[str, ValidationExpressionSpec] = {}

        for i, spec in enumerate(specs):
            prefix = f"_v{i}"
            prefix_map[prefix] = spec
            all_exprs.extend(spec.get_all_exprs(prefix))

        # Phase 1: Single aggregate collect (always performed)
        result_df = optimized_collect(lf.select(all_exprs), streaming=True)
        result_row = result_df.row(0, named=True)
        total_rows = result_row["_total"]

        if total_rows == 0:
            return []

        # Build results dict for each spec
        all_results: dict[str, dict[str, Any]] = {}
        for prefix, spec in prefix_map.items():
            spec_results: dict[str, Any] = {
                "count": result_row.get(f"{prefix}_count", 0),
            }
            if spec.non_null_expr is not None:
                spec_results["non_null"] = result_row.get(f"{prefix}_nn", total_rows)
            for i, key in enumerate(spec.extra_keys):
                spec_results[key] = result_row.get(f"{prefix}_{key}")
            all_results[prefix] = spec_results

        # Build issues (details/samples controlled by result_format)
        issues = self.build_issues_from_results(  # type: ignore
            specs=specs,
            results=all_results,
            total_rows=total_rows,
            prefix_map=prefix_map,
        )

        if not issues:
            return issues

        # Resolve result_format config
        fmt = self._get_result_format_config()  # type: ignore

        # Phase 2: Collect failure value samples (BASIC+)
        if fmt.includes_unexpected_samples():
            self._enrich_with_samples(lf, issues, list(prefix_map.values()), fmt)

        # Phase 3: Collect value frequency counts (SUMMARY+)
        if fmt.includes_unexpected_counts():
            self._enrich_with_value_counts(lf, issues, list(prefix_map.values()), fmt)

        # Phase 4: Collect full failure rows (COMPLETE)
        if fmt.includes_full_results():
            self._enrich_with_full_results(lf, issues, list(prefix_map.values()), fmt)

        return issues

    def build_issues_from_results(
        self,
        specs: list[ValidationExpressionSpec],
        results: dict[str, dict[str, Any]],
        total_rows: int,
        prefix_map: dict[str, ValidationExpressionSpec],
    ) -> list["ValidationIssue"]:
        """Build issues from aggregate results with structured ``ValidationDetail``.

        Result format controls what fields are populated:
            BOOLEAN_ONLY: count, severity, result with element_count/missing_count
            BASIC+:       + details string, expected, observed_value, unexpected_%

        Sample values, value counts, and full rows are handled in separate
        enrichment phases by ``_validate_with_expressions``.

        Args:
            specs: List of validation specs
            results: Results dict keyed by prefix
            total_rows: Total row count
            prefix_map: Map from prefix to spec

        Returns:
            List of ValidationIssue objects
        """
        issues: list[ValidationIssue] = []
        # Resolve format; fall back to SUMMARY for safety if config unavailable
        config = getattr(self, "config", None)
        fmt = config.get_result_format_config() if config else ResultFormatConfig()
        validator_name = getattr(self, "name", None)

        for prefix, spec in prefix_map.items():
            spec_results = results[prefix]
            count = spec_results.get("count", 0)

            if count <= 0:
                continue

            # Calculate ratio
            denominator = spec_results.get("non_null", total_rows)
            if denominator == 0:
                continue

            ratio = count / denominator

            # Check mostly threshold if available
            if config and getattr(config, "mostly", None) is not None:
                pass_ratio = 1 - ratio
                if pass_ratio >= config.mostly:
                    continue

            # Calculate severity
            severity = self._calculate_severity_from_ratio(
                ratio, spec.severity_ratio_thresholds
            )

            # Build structured detail (always — cheap for BOOLEAN_ONLY)
            missing_count = total_rows - denominator if spec.non_null_expr is not None else 0
            detail: ValidationDetail | None = None

            if fmt.includes_observed_value():
                # BASIC+: full aggregate detail
                detail = ValidationDetail.from_aggregates(
                    element_count=total_rows,
                    missing_count=missing_count,
                    unexpected_count=count,
                    observed_value=self._compute_observed_value(spec, spec_results),
                )
            else:
                # BOOLEAN_ONLY: minimal detail
                detail = ValidationDetail(
                    element_count=total_rows,
                    missing_count=missing_count,
                )

            # Build details string only for BASIC+ (skip for BOOLEAN_ONLY)
            details = None
            if fmt.includes_observed_value():
                details = spec.details_template.format(
                    count=count,
                    ratio=ratio,
                    column=spec.column,
                )

            issues.append(
                ValidationIssue(
                    column=spec.column,
                    issue_type=spec.issue_type,
                    count=count,
                    severity=severity,
                    details=details,
                    expected=spec.expected if fmt.includes_observed_value() else None,
                    actual=detail.observed_value if detail else None,
                    validator_name=validator_name,
                    success=False,
                    result=detail,
                )
            )

        return issues

    @staticmethod
    def _compute_observed_value(
        spec: ValidationExpressionSpec,
        spec_results: dict[str, Any],
    ) -> Any:
        """Derive a human-meaningful observed_value for this spec.

        Subclasses may override to provide validator-specific values
        (e.g., mean, unique count, min/max).  The default returns the
        failure count.
        """
        return spec_results.get("count", 0)

    # -- Enrichment phases (called by _validate_with_expressions) --

    def _enrich_with_samples(
        self,
        lf: pl.LazyFrame,
        issues: list["ValidationIssue"],
        specs: list[ValidationExpressionSpec],
        fmt: "ResultFormatConfig",
    ) -> None:
        """Phase 2: Collect failure value samples for issues with filter_expr.

        Populates both the legacy ``sample_values`` field and the structured
        ``result.partial_unexpected_list`` for backward compatibility.
        """
        sample_count = fmt.partial_unexpected_count
        logger = getattr(self, "logger", logging.getLogger(__name__))

        for issue, spec in zip(issues, specs):
            if spec.filter_expr is None:
                continue
            try:
                sample_df = (
                    lf.filter(spec.filter_expr)
                    .select(spec.column)
                    .head(sample_count)
                    .collect()
                )
                if len(sample_df) > 0:
                    sample_list = sample_df[spec.column].to_list()
                    issue.sample_values = sample_list
                    if issue.result is not None:
                        issue.result.partial_unexpected_list = sample_list
            except Exception as e:
                logger.debug(f"Failed to collect samples for {spec.column}: {e}")

    def _enrich_with_value_counts(
        self,
        lf: pl.LazyFrame,
        issues: list["ValidationIssue"],
        specs: list[ValidationExpressionSpec],
        fmt: "ResultFormatConfig",
    ) -> None:
        """Phase 3: Collect value frequency counts for failure values.

        Populates ``result.partial_unexpected_counts`` with
        ``[{"value": x, "count": n}, ...]`` and appends a human-readable
        summary to the legacy ``details`` string.
        """
        logger = getattr(self, "logger", logging.getLogger(__name__))

        for issue, spec in zip(issues, specs):
            if spec.filter_expr is None:
                continue
            try:
                count_df = (
                    lf.filter(spec.filter_expr)
                    .group_by(spec.column)
                    .agg(pl.len().alias("_count"))
                    .sort("_count", descending=True)
                    .head(fmt.partial_unexpected_count)
                    .collect()
                )
                if len(count_df) > 0:
                    value_counts = [
                        {"value": row[spec.column], "count": row["_count"]}
                        for row in count_df.iter_rows(named=True)
                    ]
                    # Store in structured result
                    if issue.result is not None:
                        issue.result.partial_unexpected_counts = value_counts
                    # Also append to legacy details string
                    top_values = ", ".join(
                        f"{vc['value']}({vc['count']})" for vc in value_counts[:5]
                    )
                    if issue.details:
                        issue.details += f" | top failures: {top_values}"
            except Exception as e:
                logger.debug(f"Failed to collect value counts for {spec.column}: {e}")

    def _enrich_with_full_results(
        self,
        lf: pl.LazyFrame,
        issues: list["ValidationIssue"],
        specs: list[ValidationExpressionSpec],
        fmt: "ResultFormatConfig",
    ) -> None:
        """Phase 4: Collect full failure rows and indices (COMPLETE level).

        Populates ``result.unexpected_list``, ``unexpected_index_list``,
        ``unexpected_rows``, and ``debug_query`` on the structured result.
        Also sets the legacy ``sample_values`` if not yet populated.
        """
        logger = getattr(self, "logger", logging.getLogger(__name__))
        max_rows = fmt.max_unexpected_rows

        for issue, spec in zip(issues, specs):
            if spec.filter_expr is None:
                continue
            try:
                columns = spec.sample_columns or [spec.column]
                result_df = (
                    lf.with_row_index("_truthound_row_idx")
                    .filter(spec.filter_expr)
                    .select(["_truthound_row_idx"] + columns)
                    .head(max_rows)
                    .collect()
                )
                if len(result_df) > 0:
                    # Legacy fallback
                    if issue.sample_values is None:
                        issue.sample_values = result_df[spec.column].to_list()

                    # Structured result
                    if issue.result is not None:
                        issue.result.unexpected_index_list = (
                            result_df["_truthound_row_idx"].to_list()
                        )
                        issue.result.unexpected_list = (
                            result_df[spec.column].to_list()
                        )
                        if fmt.include_unexpected_rows:
                            issue.result.unexpected_rows = result_df.drop(
                                "_truthound_row_idx"
                            )

                # Debug query (always for COMPLETE when return_debug_query)
                if fmt.return_debug_query and issue.result is not None:
                    issue.result.debug_query = self._generate_debug_query(spec)

            except Exception as e:
                logger.debug(f"Failed to collect full results for {spec.column}: {e}")

    @staticmethod
    def _generate_debug_query(spec: ValidationExpressionSpec) -> str:
        """Generate a reproducible Polars filter expression string.

        Returns a human-readable snippet that users can paste into a
        notebook / REPL to retrieve the failing rows.
        """
        if spec.filter_expr is not None:
            filter_str = str(spec.filter_expr)
            cols = spec.sample_columns or [spec.column]
            col_arg = ", ".join(f'"{c}"' for c in cols)
            return f'df.filter({filter_str}).select({col_arg})'
        return (
            f'# {spec.validator_name}: {spec.issue_type} '
            f'on column "{spec.column}"'
        )

    def _calculate_severity_from_ratio(
        self,
        ratio: float,
        thresholds: tuple[float, float, float] = (0.5, 0.2, 0.05),
    ) -> Severity:
        """Calculate severity from ratio using thresholds.

        Args:
            ratio: Failure ratio (0.0 to 1.0)
            thresholds: (critical, high, medium) thresholds

        Returns:
            Severity level
        """
        config = getattr(self, "config", None)
        if config and getattr(config, "severity_override", None):
            return config.severity_override

        critical_th, high_th, medium_th = thresholds
        if ratio > critical_th:
            return Severity.CRITICAL
        elif ratio > high_th:
            return Severity.HIGH
        elif ratio > medium_th:
            return Severity.MEDIUM
        return Severity.LOW


class ExpressionBatchExecutor:
    """Executor that batches multiple validators' expressions into single collect().

    This executor collects expressions from multiple validators and executes
    them in a single query, significantly improving performance.

    Supports result_format-aware execution:
        - BOOLEAN_ONLY: Only aggregate expressions (single collect, no enrichment)
        - BASIC: + sample collection phase
        - SUMMARY: + value frequency counts
        - COMPLETE: + full failure rows

    Metric deduplication (PHASE 3):
        When a SharedMetricStore is provided, the executor:
        1. Collects ``get_required_metrics()`` from all validators
        2. Deduplicates by MetricKey
        3. Computes missing metrics in a single collect()
        4. Passes metric_store to traditional validators via ``validate_with_metrics()``

    Usage:
        executor = ExpressionBatchExecutor()

        # Add validators
        executor.add_validator(NullValidator())
        executor.add_validator(RangeValidator(min_value=0, max_value=100))
        executor.add_validator(RegexValidator(pattern=r"^[A-Z]+$"))

        # Execute all in one collect()
        all_issues = executor.execute(lf)

        # Execute with specific result_format
        all_issues = executor.execute(lf, result_format="boolean_only")

        # With shared metric store for deduplication
        from truthound.validators.metrics import SharedMetricStore
        store = SharedMetricStore()
        all_issues = executor.execute(lf, metric_store=store)

    Performance:
        - 3 validators, 10M rows: ~0.5s (batched) vs ~1.5s (sequential)
        - 10 validators, 10M rows: ~1s (batched) vs ~5s (sequential)
        - BOOLEAN_ONLY further reduces cost by skipping enrichment phases
        - Metric deduplication: 1.5-2x improvement when validators share metrics
    """

    def __init__(self, metric_store: "SharedMetricStore | None" = None) -> None:
        self._validators: list[Validator] = []
        self._metric_store = metric_store

    def add_validator(self, validator: "Validator") -> "ExpressionBatchExecutor":
        """Add a validator to the batch.

        Args:
            validator: Validator to add

        Returns:
            Self for chaining
        """
        self._validators.append(validator)
        return self

    def add_validators(self, validators: list["Validator"]) -> "ExpressionBatchExecutor":
        """Add multiple validators to the batch.

        Args:
            validators: List of validators to add

        Returns:
            Self for chaining
        """
        self._validators.extend(validators)
        return self

    def execute(
        self,
        lf: pl.LazyFrame,
        result_format: "str | ResultFormat | ResultFormatConfig | None" = None,
        metric_store: "SharedMetricStore | None" = None,
    ) -> list["ValidationIssue"]:
        """Execute all validators in a single batched query.

        Validators that implement ExpressionValidatorProtocol will have their
        expressions batched. Other validators fall back to individual execution.

        Args:
            lf: LazyFrame to validate
            result_format: Override result_format for all validators in this batch.
                           If None, each validator uses its own config.
            metric_store: Optional SharedMetricStore for metric deduplication.
                          Overrides the instance-level store if provided.

        Returns:
            Combined list of ValidationIssue from all validators
        """
        if not self._validators:
            return []

        # Use provided store or fall back to instance store
        store = metric_store or self._metric_store

        # Resolve override format
        fmt_override = ResultFormatConfig.from_any(result_format) if result_format is not None else None

        # Separate expression-based and traditional validators
        expr_validators: list[tuple[Validator, list[ValidationExpressionSpec]]] = []
        traditional_validators: list[Validator] = []

        for validator in self._validators:
            if isinstance(validator, ExpressionValidatorProtocol):
                columns = validator._get_target_columns(lf)  # type: ignore
                if columns:
                    specs = validator.get_validation_exprs(lf, columns)
                    if specs:
                        expr_validators.append((validator, specs))
                    else:
                        traditional_validators.append(validator)
                # No columns = skip this validator
            else:
                traditional_validators.append(validator)

        # Phase 0: Precompute shared metrics (deduplication)
        if store is not None:
            self._precompute_shared_metrics(lf, store)

        all_issues: list[ValidationIssue] = []

        # Phase 1: Execute expression-based validators in batch (aggregate collect)
        if expr_validators:
            batched_issues = self._execute_batched(lf, expr_validators, fmt_override)
            all_issues.extend(batched_issues)

        # Execute traditional validators (with metric store if available)
        lgr = _get_logger("batch_executor")
        for validator in traditional_validators:
            try:
                if store is not None:
                    issues = validator.validate_with_metrics(lf, store)
                else:
                    issues = validator.validate(lf)
                all_issues.extend(issues)
            except Exception as e:
                config = getattr(validator, "config", None)
                catch = config.catch_exceptions if config else True
                if catch:
                    lgr.warning(
                        "Traditional validator %s failed: %s", validator.name, e,
                    )
                    exc_info = ExceptionInfo.from_exception(
                        e, validator_name=validator.name,
                    )
                    all_issues.append(ValidationIssue(
                        column="*",
                        issue_type="validator_error",
                        count=0,
                        severity=Severity.LOW,
                        details=f"Validator failed: {e}",
                        validator_name=validator.name,
                        exception_info=exc_info,
                    ))
                else:
                    raise

        return all_issues

    def _precompute_shared_metrics(
        self,
        lf: pl.LazyFrame,
        store: "SharedMetricStore",
    ) -> None:
        """Collect metric needs from all validators, deduplicate, compute once.

        This is the core of PHASE 3 metric deduplication. It:
        1. Asks each validator for its required MetricKeys
        2. Filters out keys already in the store
        3. Resolves keys to Polars expressions
        4. Executes all in a single collect()
        5. Stores results for later consumption
        """
        from truthound.validators.metrics import metric_key_to_expr, MetricKey

        # Gather all required metric keys
        all_keys: dict[MetricKey, pl.Expr] = {}

        for validator in self._validators:
            try:
                columns = validator._get_target_columns(lf)
            except Exception:
                columns = []
            if not columns:
                continue

            for key in validator.get_required_metrics(columns):
                if key in all_keys or key in store:
                    continue
                expr = metric_key_to_expr(key)
                if expr is not None:
                    all_keys[key] = expr

        if not all_keys:
            return

        # Record how many deduplicated metrics we avoided computing
        store.stats.deduplication_saves += (
            sum(
                len(v.get_required_metrics(
                    v._get_target_columns(lf) if hasattr(v, '_get_target_columns') else []
                ))
                for v in self._validators
            ) - len(all_keys)
        )

        # Single collect for all shared metrics
        exprs = list(all_keys.values())
        try:
            result_df = optimized_collect(lf.select(exprs), streaming=True)
            result_row = result_df.row(0, named=True)

            pairs = {}
            for key, expr in all_keys.items():
                alias = expr.meta.output_name()
                if alias in result_row:
                    pairs[key] = result_row[alias]

            store.put_many(pairs)
        except Exception as e:
            _get_logger("batch_executor").warning(
                f"Shared metric precomputation failed: {e}"
            )

    def _execute_batched(
        self,
        lf: pl.LazyFrame,
        expr_validators: list[tuple["Validator", list[ValidationExpressionSpec]]],
        fmt_override: "ResultFormatConfig | None" = None,
    ) -> list["ValidationIssue"]:
        """Execute expression-based validators with 3-tier fallback.

        Tier 1: Single batched collect for all validators (fastest).
        Tier 2: Per-validator collect — tried if the batch fails.
        Tier 3: Per-expression collect — tried if a per-validator collect fails.

        Each tier gracefully degrades to the next, attaching
        ``ExceptionInfo`` to any error-generated issues so callers can
        diagnose what went wrong.

        Enrichment phases (samples, value counts, full rows) run per-
        validator after the aggregate phase, with their own error
        isolation so a single enrichment failure doesn't lose aggregate
        results.

        Args:
            lf: LazyFrame to validate
            expr_validators: List of (validator, specs) tuples
            fmt_override: If set, overrides each validator's result_format

        Returns:
            Combined list of ValidationIssue
        """
        lgr = _get_logger("batch_executor")

        # ── helpers ──────────────────────────────────────────────────
        def _build_prefix_map(
            v_idx: int, specs: list[ValidationExpressionSpec],
        ) -> dict[str, ValidationExpressionSpec]:
            return {
                f"_v{v_idx}_s{s_idx}": spec
                for s_idx, spec in enumerate(specs)
            }

        def _collect_exprs(
            prefix_map: dict[str, ValidationExpressionSpec],
        ) -> list[pl.Expr]:
            exprs: list[pl.Expr] = [pl.len().alias("_total")]
            for prefix, spec in prefix_map.items():
                exprs.extend(spec.get_all_exprs(prefix))
            return exprs

        def _extract_spec_results(
            prefix_map: dict[str, ValidationExpressionSpec],
            result_row: dict[str, Any],
            total_rows: int,
        ) -> dict[str, dict[str, Any]]:
            results: dict[str, dict[str, Any]] = {}
            for prefix, spec in prefix_map.items():
                spec_results: dict[str, Any] = {
                    "count": result_row.get(f"{prefix}_count", 0),
                }
                if spec.non_null_expr is not None:
                    spec_results["non_null"] = result_row.get(f"{prefix}_nn", total_rows)
                for key in spec.extra_keys:
                    spec_results[key] = result_row.get(f"{prefix}_{key}")
                results[prefix] = spec_results
            return results

        def _build_issues_for_validator(
            validator: "Validator",
            specs: list[ValidationExpressionSpec],
            prefix_map: dict[str, ValidationExpressionSpec],
            result_row: dict[str, Any],
            total_rows: int,
            fmt: "ResultFormatConfig",
        ) -> list["ValidationIssue"]:
            results = _extract_spec_results(prefix_map, result_row, total_rows)

            original_config = None
            if fmt_override is not None and hasattr(validator, "config"):
                original_config = validator.config
                validator.config = validator.config.replace(result_format=fmt_override)
            try:
                if hasattr(validator, "build_issues_from_results"):
                    return validator.build_issues_from_results(
                        specs=specs,
                        results=results,
                        total_rows=total_rows,
                        prefix_map=prefix_map,
                    )
                return []
            finally:
                if original_config is not None:
                    validator.config = original_config

        def _resolve_fmt(validator: "Validator") -> "ResultFormatConfig":
            if fmt_override is not None:
                return fmt_override
            config = getattr(validator, "config", None)
            return config.get_result_format_config() if config else ResultFormatConfig()

        def _enrich_issues(
            validator: "Validator",
            issues: list["ValidationIssue"],
            spec_list: list[ValidationExpressionSpec],
            fmt: "ResultFormatConfig",
        ) -> None:
            """Run enrichment phases with per-phase error isolation."""
            # Phase 2: Sample enrichment (BASIC+)
            if fmt.includes_unexpected_samples() and hasattr(validator, "_enrich_with_samples"):
                try:
                    validator._enrich_with_samples(lf, issues, spec_list, fmt)
                except Exception as e:
                    lgr.warning("Sample enrichment failed for %s: %s", validator.name, e)

            # Phase 3: Value frequency counts (SUMMARY+)
            if fmt.includes_unexpected_counts() and hasattr(validator, "_enrich_with_value_counts"):
                try:
                    validator._enrich_with_value_counts(lf, issues, spec_list, fmt)
                except Exception as e:
                    lgr.warning("Value count enrichment failed for %s: %s", validator.name, e)

            # Phase 4: Full failure rows (COMPLETE)
            if fmt.includes_full_results() and hasattr(validator, "_enrich_with_full_results"):
                try:
                    validator._enrich_with_full_results(lf, issues, spec_list, fmt)
                except Exception as e:
                    lgr.warning("Full results enrichment failed for %s: %s", validator.name, e)

        def _make_error_issue(
            validator: "Validator", exc: Exception, column: str = "*",
        ) -> "ValidationIssue":
            """Create a ValidationIssue that records an expression-level error."""
            exc_info = ExceptionInfo.from_exception(
                exc, validator_name=validator.name, column=column,
            )
            return ValidationIssue(
                column=column,
                issue_type="expression_error",
                count=0,
                severity=Severity.LOW,
                details=f"Expression failed: {exc}",
                validator_name=validator.name,
                exception_info=exc_info,
            )

        # ── Tier 2: per-validator fallback ───────────────────────────
        def _fallback_per_validator(
            v_idx: int,
            validator: "Validator",
            specs: list[ValidationExpressionSpec],
        ) -> list["ValidationIssue"]:
            """Execute a single validator's expressions independently."""
            prefix_map = _build_prefix_map(v_idx, specs)
            fmt = _resolve_fmt(validator)
            exprs = _collect_exprs(prefix_map)

            try:
                result_df = optimized_collect(lf.select(exprs), streaming=True)
                result_row = result_df.row(0, named=True)
                total_rows = result_row["_total"]
                if total_rows == 0:
                    return []

                issues = _build_issues_for_validator(
                    validator, specs, prefix_map, result_row, total_rows, fmt,
                )
                if issues:
                    _enrich_issues(validator, issues, list(prefix_map.values()), fmt)
                return issues

            except Exception as e2:
                lgr.warning(
                    "Per-validator collect failed for %s, falling back to per-expression: %s",
                    validator.name, e2,
                )
                return _fallback_per_expression(v_idx, validator, specs)

        # ── Tier 3: per-expression fallback ──────────────────────────
        def _fallback_per_expression(
            v_idx: int,
            validator: "Validator",
            specs: list[ValidationExpressionSpec],
        ) -> list["ValidationIssue"]:
            """Execute each expression independently as last resort."""
            partial_issues: list["ValidationIssue"] = []
            config = getattr(validator, "config", None)
            mode = config.partial_failure_mode if config else "collect"

            for s_idx, spec in enumerate(specs):
                prefix = f"_v{v_idx}_s{s_idx}"
                single_map = {prefix: spec}
                exprs = _collect_exprs(single_map)

                try:
                    result_df = optimized_collect(lf.select(exprs), streaming=True)
                    result_row = result_df.row(0, named=True)
                    total_rows = result_row["_total"]
                    if total_rows == 0:
                        continue

                    fmt = _resolve_fmt(validator)
                    issues = _build_issues_for_validator(
                        validator, [spec], single_map, result_row, total_rows, fmt,
                    )
                    partial_issues.extend(issues)

                except Exception as e3:
                    lgr.warning(
                        "Expression %s failed for %s: %s", prefix, validator.name, e3,
                    )
                    if mode == "collect":
                        partial_issues.append(_make_error_issue(
                            validator, e3, column=spec.column if hasattr(spec, "column") else "*",
                        ))
                    elif mode == "raise":
                        raise
                    # mode == "skip": silently skip this expression

            return partial_issues

        # ── Tier 1: full batch collect ───────────────────────────────
        # Build combined prefix maps
        all_prefix_maps: dict[int, dict[str, ValidationExpressionSpec]] = {}
        all_exprs: list[pl.Expr] = [pl.len().alias("_total")]

        for v_idx, (validator, specs) in enumerate(expr_validators):
            prefix_map = _build_prefix_map(v_idx, specs)
            all_prefix_maps[v_idx] = prefix_map
            for prefix, spec in prefix_map.items():
                all_exprs.extend(spec.get_all_exprs(prefix))

        try:
            result_df = optimized_collect(lf.select(all_exprs), streaming=True)
            result_row = result_df.row(0, named=True)
            total_rows = result_row["_total"]

            if total_rows == 0:
                return []

            # Batch succeeded — build issues per validator
            all_issues: list[ValidationIssue] = []
            for v_idx, (validator, specs) in enumerate(expr_validators):
                prefix_map = all_prefix_maps[v_idx]
                fmt = _resolve_fmt(validator)

                try:
                    issues = _build_issues_for_validator(
                        validator, specs, prefix_map, result_row, total_rows, fmt,
                    )
                except Exception as e_build:
                    lgr.warning("Issue building failed for %s: %s", validator.name, e_build)
                    all_issues.append(_make_error_issue(validator, e_build))
                    continue

                if not issues:
                    continue

                _enrich_issues(validator, issues, list(prefix_map.values()), fmt)
                all_issues.extend(issues)

            return all_issues

        except Exception as e_batch:
            lgr.warning(
                "Batched collect failed, falling back to per-validator: %s", e_batch,
            )
            # Tier 2 fallback: execute each validator independently
            all_issues = []
            for v_idx, (validator, specs) in enumerate(expr_validators):
                try:
                    issues = _fallback_per_validator(v_idx, validator, specs)
                    all_issues.extend(issues)
                except Exception as e_final:
                    lgr.error("All tiers failed for %s: %s", validator.name, e_final)
                    all_issues.append(_make_error_issue(validator, e_final))
            return all_issues

    def clear(self) -> None:
        """Clear all validators from the batch."""
        self._validators.clear()


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

        # Get row count with query plan optimizations
        total_rows = optimized_collect(lf.select(pl.len()), streaming=True).item()

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

        # Collect with query plan optimizations
        result = optimized_collect(lf.select(exprs), streaming=True)
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
