"""Structured error handling for the profiler module.

This module provides a comprehensive error handling system with:
- Typed exception hierarchy
- Error context preservation
- Structured error collection
- Logging integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar, Generic

logger = logging.getLogger("truthound.profiler")


# =============================================================================
# Error Severity and Categories
# =============================================================================


class ErrorSeverity(str, Enum):
    """Severity levels for profiling errors."""

    WARNING = "warning"   # Non-fatal, profiling continues
    ERROR = "error"       # Component failed, partial results
    CRITICAL = "critical" # Profiling cannot continue


class ErrorCategory(str, Enum):
    """Categories of profiling errors."""

    ANALYSIS = "analysis"       # Error during column/table analysis
    PATTERN = "pattern"         # Error during pattern detection
    TYPE_INFERENCE = "type_inference"  # Error during type inference
    IO = "io"                   # File/data loading errors
    MEMORY = "memory"           # Memory-related errors
    TIMEOUT = "timeout"         # Operation timeout
    VALIDATION = "validation"   # Invalid input/configuration
    INTERNAL = "internal"       # Internal/unexpected errors


# =============================================================================
# Exception Hierarchy
# =============================================================================


class ProfilerError(Exception):
    """Base exception for all profiler errors.

    This provides structured error information that can be
    collected, logged, and reported consistently.
    """

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        column: str | None = None,
        analyzer: str | None = None,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.column = column
        self.analyzer = analyzer
        self.cause = cause
        self.context = context or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "column": self.column,
            "analyzer": self.analyzer,
            "cause": str(self.cause) if self.cause else None,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.message}"]
        if self.column:
            parts.append(f"column={self.column}")
        if self.analyzer:
            parts.append(f"analyzer={self.analyzer}")
        if self.cause:
            parts.append(f"cause={self.cause}")
        return " | ".join(parts)


class AnalysisError(ProfilerError):
    """Error during column or table analysis."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.ANALYSIS, **kwargs)


class PatternError(ProfilerError):
    """Error during pattern detection."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.PATTERN, **kwargs)


class TypeInferenceError(ProfilerError):
    """Error during type inference."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.TYPE_INFERENCE, **kwargs)


class IOError(ProfilerError):
    """Error during file/data operations."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.IO, **kwargs)


class MemoryError(ProfilerError):
    """Memory-related error (e.g., OOM during profiling)."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


class TimeoutError(ProfilerError):
    """Operation timeout error."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)


class ValidationError(ProfilerError):
    """Invalid input or configuration error."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


# =============================================================================
# Error Collector
# =============================================================================


@dataclass
class ErrorRecord:
    """Record of an error that occurred during profiling."""

    error: ProfilerError
    recovered: bool = False  # Whether profiling continued after this error

    def to_dict(self) -> dict[str, Any]:
        d = self.error.to_dict()
        d["recovered"] = self.recovered
        return d


class ErrorCollector:
    """Collects and manages errors during profiling.

    This allows profiling to continue even when individual
    components fail, while still tracking all errors.

    Example:
        collector = ErrorCollector()

        with collector.catch(column="email", analyzer="pattern"):
            # This will be caught and recorded
            risky_operation()

        # Check results
        if collector.has_errors:
            for error in collector.errors:
                print(error)
    """

    def __init__(
        self,
        *,
        fail_fast: bool = False,
        log_errors: bool = True,
        max_errors: int = 1000,
    ):
        """Initialize error collector.

        Args:
            fail_fast: If True, raise on first error instead of collecting
            log_errors: If True, log errors as they occur
            max_errors: Maximum errors to collect before raising
        """
        self._errors: list[ErrorRecord] = []
        self.fail_fast = fail_fast
        self.log_errors = log_errors
        self.max_errors = max_errors

    @property
    def errors(self) -> list[ErrorRecord]:
        """Get all collected errors."""
        return list(self._errors)

    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self._errors) > 0

    @property
    def has_critical(self) -> bool:
        """Check if any critical errors occurred."""
        return any(
            e.error.severity == ErrorSeverity.CRITICAL
            for e in self._errors
        )

    @property
    def error_count(self) -> int:
        """Get total error count."""
        return len(self._errors)

    def add(
        self,
        error: ProfilerError | Exception,
        *,
        recovered: bool = True,
        column: str | None = None,
        analyzer: str | None = None,
    ) -> None:
        """Add an error to the collection.

        Args:
            error: The error to add
            recovered: Whether profiling continued after this error
            column: Column being processed when error occurred
            analyzer: Analyzer that was running when error occurred
        """
        # Wrap non-ProfilerError exceptions
        if not isinstance(error, ProfilerError):
            error = ProfilerError(
                str(error),
                column=column,
                analyzer=analyzer,
                cause=error,
            )
        else:
            # Update context if provided
            if column and not error.column:
                error.column = column
            if analyzer and not error.analyzer:
                error.analyzer = analyzer

        record = ErrorRecord(error=error, recovered=recovered)
        self._errors.append(record)

        # Logging
        if self.log_errors:
            log_method = {
                ErrorSeverity.WARNING: logger.warning,
                ErrorSeverity.ERROR: logger.error,
                ErrorSeverity.CRITICAL: logger.critical,
            }.get(error.severity, logger.error)
            log_method(str(error))

        # Fail fast check
        if self.fail_fast:
            raise error

        # Max errors check
        if len(self._errors) >= self.max_errors:
            raise ProfilerError(
                f"Maximum error count ({self.max_errors}) exceeded",
                severity=ErrorSeverity.CRITICAL,
            )

    def catch(
        self,
        *,
        column: str | None = None,
        analyzer: str | None = None,
        default: Any = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> "ErrorCatcher":
        """Context manager for catching and recording errors.

        Args:
            column: Column being processed
            analyzer: Analyzer running
            default: Default value to return on error
            severity: Severity level for caught errors

        Returns:
            Context manager that catches exceptions

        Example:
            with collector.catch(column="email", default=None) as catcher:
                result = analyze_column()
            # If error occurred, result will be None
            result = catcher.result or default
        """
        return ErrorCatcher(
            self,
            column=column,
            analyzer=analyzer,
            default=default,
            severity=severity,
        )

    def get_by_column(self, column: str) -> list[ErrorRecord]:
        """Get errors for a specific column."""
        return [e for e in self._errors if e.error.column == column]

    def get_by_category(self, category: ErrorCategory) -> list[ErrorRecord]:
        """Get errors of a specific category."""
        return [e for e in self._errors if e.error.category == category]

    def get_by_severity(self, severity: ErrorSeverity) -> list[ErrorRecord]:
        """Get errors of a specific severity."""
        return [e for e in self._errors if e.error.severity == severity]

    def clear(self) -> None:
        """Clear all collected errors."""
        self._errors.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_errors": len(self._errors),
            "has_critical": self.has_critical,
            "by_severity": {
                sev.value: len(self.get_by_severity(sev))
                for sev in ErrorSeverity
            },
            "by_category": {
                cat.value: len(self.get_by_category(cat))
                for cat in ErrorCategory
            },
            "errors": [e.to_dict() for e in self._errors],
        }

    def summary(self) -> str:
        """Get a summary of collected errors."""
        if not self._errors:
            return "No errors"

        lines = [f"Collected {len(self._errors)} error(s):"]
        for sev in ErrorSeverity:
            count = len(self.get_by_severity(sev))
            if count > 0:
                lines.append(f"  {sev.value}: {count}")
        return "\n".join(lines)


class ErrorCatcher:
    """Context manager for catching errors within ErrorCollector."""

    def __init__(
        self,
        collector: ErrorCollector,
        *,
        column: str | None = None,
        analyzer: str | None = None,
        default: Any = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ):
        self.collector = collector
        self.column = column
        self.analyzer = analyzer
        self.default = default
        self.severity = severity
        self.result: Any = None
        self.error: Exception | None = None

    def __enter__(self) -> "ErrorCatcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None:
            self.error = exc_val

            # Create or update error
            if isinstance(exc_val, ProfilerError):
                exc_val.severity = self.severity
            else:
                exc_val = ProfilerError(
                    str(exc_val),
                    column=self.column,
                    analyzer=self.analyzer,
                    severity=self.severity,
                    cause=exc_val,
                )

            self.collector.add(
                exc_val,
                recovered=True,
                column=self.column,
                analyzer=self.analyzer,
            )
            self.result = self.default
            return True  # Suppress exception

        return False


# =============================================================================
# Decorator for Error Handling
# =============================================================================

T = TypeVar("T")


def with_error_handling(
    *,
    column_param: str | None = None,
    analyzer_name: str | None = None,
    default: Any = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Decorator for methods that should use error collection.

    The decorated method's class must have an `error_collector` attribute.

    Args:
        column_param: Name of parameter containing column name
        analyzer_name: Name of the analyzer
        default: Default value on error
        severity: Severity level for errors

    Example:
        class MyAnalyzer:
            error_collector: ErrorCollector

            @with_error_handling(column_param="column", analyzer_name="my_analyzer")
            def analyze(self, column: str, data):
                # Errors here will be caught and collected
                ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        def wrapper(self, *args, **kwargs) -> T | None:
            collector = getattr(self, "error_collector", None)
            if collector is None:
                # No collector, just run normally
                return func(self, *args, **kwargs)

            # Extract column name if specified
            column = None
            if column_param:
                # Try kwargs first, then positional args
                if column_param in kwargs:
                    column = kwargs[column_param]
                else:
                    # Get from function signature
                    import inspect
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if column_param in params:
                        idx = params.index(column_param) - 1  # -1 for self
                        if idx < len(args):
                            column = args[idx]

            with collector.catch(
                column=column,
                analyzer=analyzer_name or func.__name__,
                default=default,
                severity=severity,
            ) as catcher:
                catcher.result = func(self, *args, **kwargs)
                return catcher.result

            return catcher.result

        return wrapper
    return decorator
