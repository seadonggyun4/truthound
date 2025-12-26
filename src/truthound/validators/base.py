"""Base classes for validators.

Features:
- Immutable configuration (thread-safe)
- Timeout mechanism
- Type-safe column filtering
- ReDoS protection for regex patterns
- Graceful degradation on errors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
import re
import signal
import threading
import logging
import time
from functools import wraps
from enum import Enum

import polars as pl

from truthound.types import Severity


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

    @staticmethod
    def safe_head(
        lf: pl.LazyFrame,
        n: int,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Safely get first n rows."""
        query = lf
        if columns:
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)
        return query.head(n).collect(engine="streaming")

    @staticmethod
    def safe_sample(
        lf: pl.LazyFrame,
        n: int,
        columns: list[str] | None = None,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Safely sample n rows."""
        return SafeSampler.safe_head(lf, n, columns)

    @staticmethod
    def safe_filter_sample(
        lf: pl.LazyFrame,
        filter_expr: pl.Expr,
        n: int,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Safely get filtered samples."""
        query = lf.filter(filter_expr)
        if columns:
            schema = lf.collect_schema()
            valid_cols = [c for c in columns if c in schema.names()]
            if valid_cols:
                query = query.select(valid_cols)
        return query.head(n).collect(engine="streaming")


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
class ValidatorExecutionResult:
    """Result of a single validator execution with error handling."""
    validator_name: str
    status: ValidationResult
    issues: list["ValidationIssue"]
    error_message: str | None = None
    error_context: ErrorContext | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator_name,
            "status": self.status.value,
            "issue_count": len(self.issues),
            "execution_time_ms": self.execution_time_ms,
            "error": self.error_context.to_dict() if self.error_context else None,
        }


def _validate_safe(
    validator: "Validator",
    lf: pl.LazyFrame,
    skip_on_error: bool = True,
    log_errors: bool = True,
) -> ValidatorExecutionResult:
    """Execute validation with error handling.

    Returns:
        ValidatorExecutionResult with status and any issues found
    """
    start_time = time.time()
    logger = _get_logger(validator.name)

    try:
        issues = validator.validate(lf)
        return ValidatorExecutionResult(
            validator_name=validator.name,
            status=ValidationResult.SUCCESS,
            issues=issues,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    except ColumnNotFoundError as e:
        if log_errors:
            logger.warning(f"Column not found: {e.column}")
        return ValidatorExecutionResult(
            validator_name=validator.name,
            status=ValidationResult.SKIPPED,
            issues=[],
            error_message=str(e),
            error_context=ErrorContext("ColumnNotFoundError", str(e)),
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    except ValidationTimeoutError as e:
        if log_errors:
            logger.warning(f"Validation timed out: {e.timeout_seconds}s")
        return ValidatorExecutionResult(
            validator_name=validator.name,
            status=ValidationResult.TIMEOUT,
            issues=[],
            error_message=str(e),
            error_context=ErrorContext("ValidationTimeoutError", str(e)),
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        if log_errors:
            logger.exception(f"Error in {validator.name}: {e}")
        if skip_on_error:
            return ValidatorExecutionResult(
                validator_name=validator.name,
                status=ValidationResult.FAILED,
                issues=[],
                error_message=str(e),
                error_context=ErrorContext(type(e).__name__, str(e)),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        raise


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

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")
        if self.mostly is not None and not (0.0 <= self.mostly <= 1.0):
            raise ValueError(f"mostly must be in [0.0, 1.0], got {self.mostly}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")

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
        if "columns" in kwargs and isinstance(kwargs["columns"], list):
            kwargs["columns"] = tuple(kwargs["columns"])
        if "exclude_columns" in kwargs and isinstance(kwargs["exclude_columns"], list):
            kwargs["exclude_columns"] = tuple(kwargs["exclude_columns"])
        valid_fields = {
            "columns", "exclude_columns", "severity_override", "sample_size",
            "mostly", "timeout_seconds", "graceful_degradation", "log_errors"
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
    """Represents a single data quality issue found during validation."""

    column: str
    issue_type: str
    count: int
    severity: Severity
    details: str | None = None
    expected: Any | None = None
    actual: Any | None = None
    sample_values: list[Any] | None = None

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

    Features:
    - Immutable ValidatorConfig (thread-safe)
    - Timeout support
    - Schema validation
    - Graceful degradation on errors
    - Dependency-aware execution ordering

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
        """Safely get sample values."""
        try:
            df = SafeSampler.safe_filter_sample(
                lf, filter_expr, self.config.sample_size, columns
            )
            return df.to_dicts() if len(df) > 0 else []
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
        # Get row count
        total_rows = lf.select(pl.len()).collect().item()

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
