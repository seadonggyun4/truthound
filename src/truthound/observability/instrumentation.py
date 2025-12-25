"""Instrumentation utilities for automatic observability.

This module provides decorators and utilities for automatic instrumentation
of code with logging, metrics, and tracing.

Features:
    - Function decorators for automatic metrics/tracing
    - Checkpoint instrumentation for validation metrics
    - Automatic error tracking
    - Performance timing
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, ParamSpec, TypeVar

from truthound.observability.logging import get_logger, StructuredLogger, log_context
from truthound.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    get_metrics,
)
from truthound.observability.context import create_span, SpanStatus

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# Function Decorators
# =============================================================================


def traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add tracing to a function.

    Creates a span for each function call, capturing timing and errors.

    Args:
        name: Span name (defaults to function name).
        attributes: Static attributes to add to span.

    Example:
        >>> @traced()
        ... def process_data(data):
        ...     return validate(data)
        >>>
        >>> @traced("custom.span.name", attributes={"component": "validator"})
        ... def validate(data):
        ...     pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with create_span(span_name, attributes=attributes) as span:
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("function.name", func.__qualname__)
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timed(
    metric_name: str | None = None,
    *,
    labels: dict[str, str] | None = None,
    buckets: list[float] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to measure function execution time.

    Records execution time to a histogram metric.

    Args:
        metric_name: Histogram name (defaults to function_duration_seconds).
        labels: Static labels to add.
        buckets: Custom histogram buckets.

    Example:
        >>> @timed("validation_duration_seconds")
        ... def validate(data):
        ...     return check_data(data)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        histogram: Histogram | None = None

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal histogram

            if histogram is None:
                collector = get_metrics()
                name = metric_name or f"{func.__module__}_{func.__qualname__}_duration_seconds"
                name = name.replace(".", "_").lower()
                histogram = collector.histogram(
                    name,
                    f"Duration of {func.__qualname__}",
                    buckets=buckets,
                    labels=list(labels.keys()) if labels else [],
                )

            with histogram.time(**(labels or {})):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def counted(
    metric_name: str | None = None,
    *,
    labels: dict[str, str] | None = None,
    count_exceptions: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to count function calls.

    Records call count and optionally exception count.

    Args:
        metric_name: Counter name.
        labels: Static labels.
        count_exceptions: Also count exceptions.

    Example:
        >>> @counted("validations_total", labels={"type": "null_check"})
        ... def check_nulls(data):
        ...     return data.isnull().sum()
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        counter: Counter | None = None
        exception_counter: Counter | None = None

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal counter, exception_counter

            if counter is None:
                collector = get_metrics()
                name = metric_name or f"{func.__module__}_{func.__qualname__}_total"
                name = name.replace(".", "_").lower()
                counter = collector.counter(
                    name,
                    f"Total calls to {func.__qualname__}",
                    labels=list(labels.keys()) if labels else [],
                )

                if count_exceptions:
                    exception_counter = collector.counter(
                        f"{name}_exceptions",
                        f"Exceptions in {func.__qualname__}",
                        labels=list(labels.keys()) if labels else [],
                    )

            counter.inc(**(labels or {}))

            try:
                return func(*args, **kwargs)
            except Exception:
                if exception_counter:
                    exception_counter.inc(**(labels or {}))
                raise

        return wrapper

    return decorator


# =============================================================================
# Checkpoint Instrumentation
# =============================================================================


@dataclass
class CheckpointMetrics:
    """Metrics for checkpoint monitoring.

    Provides pre-defined metrics for validation checkpoints.
    """

    # Counters
    validations_total: Counter
    validations_failed: Counter
    issues_total: Counter

    # Gauges
    active_validations: Gauge
    last_validation_timestamp: Gauge

    # Histograms
    validation_duration: Histogram
    issues_per_validation: Histogram


class CheckpointInstrumentation:
    """Automatic instrumentation for checkpoints.

    Provides metrics and logging for validation checkpoints.

    Example:
        >>> instrumentation = CheckpointInstrumentation()
        >>>
        >>> # Use context manager for validation
        >>> with instrumentation.validation("daily_check", "users.csv"):
        ...     result = run_validation()
        ...     instrumentation.record_result(result)
    """

    def __init__(
        self,
        *,
        collector: MetricsCollector | None = None,
        logger: StructuredLogger | None = None,
        prefix: str = "truthound",
    ) -> None:
        """Initialize checkpoint instrumentation.

        Args:
            collector: MetricsCollector (uses global if None).
            logger: Logger (uses global if None).
            prefix: Metric name prefix.
        """
        self._collector = collector or get_metrics()
        self._logger = logger or get_logger("truthound.checkpoint")
        self._prefix = prefix
        self._metrics = self._create_metrics()

    def _create_metrics(self) -> CheckpointMetrics:
        """Create checkpoint metrics."""
        p = self._prefix

        return CheckpointMetrics(
            validations_total=self._collector.counter(
                f"{p}_validations_total",
                "Total number of validations run",
                labels=["checkpoint", "status"],
            ),
            validations_failed=self._collector.counter(
                f"{p}_validations_failed_total",
                "Number of failed validations",
                labels=["checkpoint", "reason"],
            ),
            issues_total=self._collector.counter(
                f"{p}_issues_total",
                "Total number of issues found",
                labels=["checkpoint", "severity"],
            ),
            active_validations=self._collector.gauge(
                f"{p}_active_validations",
                "Number of validations currently running",
                labels=["checkpoint"],
            ),
            last_validation_timestamp=self._collector.gauge(
                f"{p}_last_validation_timestamp",
                "Timestamp of last validation",
                labels=["checkpoint"],
            ),
            validation_duration=self._collector.histogram(
                f"{p}_validation_duration_seconds",
                "Validation execution time",
                buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
                labels=["checkpoint"],
            ),
            issues_per_validation=self._collector.histogram(
                f"{p}_issues_per_validation",
                "Number of issues found per validation",
                buckets=[0, 1, 5, 10, 50, 100, 500, 1000],
                labels=["checkpoint"],
            ),
        )

    @property
    def metrics(self) -> CheckpointMetrics:
        """Get checkpoint metrics."""
        return self._metrics

    @contextmanager
    def validation(
        self,
        checkpoint_name: str,
        data_asset: str,
    ) -> Iterator["ValidationContext"]:
        """Context manager for validation tracking.

        Automatically tracks timing, counts, and logging.

        Args:
            checkpoint_name: Name of the checkpoint.
            data_asset: Data asset being validated.

        Yields:
            ValidationContext for recording results.
        """
        ctx = ValidationContext(
            checkpoint_name=checkpoint_name,
            data_asset=data_asset,
            instrumentation=self,
        )

        # Track active validations
        self._metrics.active_validations.inc(checkpoint=checkpoint_name)

        # Log start
        self._logger.info(
            "Validation started",
            checkpoint=checkpoint_name,
            data_asset=data_asset,
        )

        start_time = time.perf_counter()

        try:
            with create_span(
                f"validation.{checkpoint_name}",
                attributes={
                    "checkpoint.name": checkpoint_name,
                    "checkpoint.data_asset": data_asset,
                },
            ) as span:
                ctx._span = span
                yield ctx

                # Record success
                if not ctx._recorded:
                    ctx.record_success()

        except Exception as e:
            ctx.record_failure(str(e))
            raise

        finally:
            # Record duration
            duration = time.perf_counter() - start_time
            self._metrics.validation_duration.observe(
                duration, checkpoint=checkpoint_name
            )

            # Update timestamp
            self._metrics.last_validation_timestamp.set(
                time.time(), checkpoint=checkpoint_name
            )

            # Decrement active
            self._metrics.active_validations.dec(checkpoint=checkpoint_name)

            # Log completion
            self._logger.info(
                "Validation completed",
                checkpoint=checkpoint_name,
                data_asset=data_asset,
                duration_seconds=round(duration, 3),
                status=ctx._status,
            )

    def record_issues(
        self,
        checkpoint_name: str,
        issues: dict[str, int],
    ) -> None:
        """Record issue counts by severity.

        Args:
            checkpoint_name: Checkpoint name.
            issues: Dict mapping severity to count.
        """
        total = 0
        for severity, count in issues.items():
            self._metrics.issues_total.add(
                count, checkpoint=checkpoint_name, severity=severity
            )
            total += count

        self._metrics.issues_per_validation.observe(
            total, checkpoint=checkpoint_name
        )


@dataclass
class ValidationContext:
    """Context for tracking a single validation run."""

    checkpoint_name: str
    data_asset: str
    instrumentation: CheckpointInstrumentation
    _span: Any = None
    _recorded: bool = False
    _status: str = "unknown"

    def record_success(
        self,
        total_issues: int = 0,
        issues_by_severity: dict[str, int] | None = None,
    ) -> None:
        """Record successful validation.

        Args:
            total_issues: Total number of issues found.
            issues_by_severity: Issues by severity level.
        """
        self._recorded = True
        self._status = "success"

        metrics = self.instrumentation._metrics
        metrics.validations_total.inc(
            checkpoint=self.checkpoint_name, status="success"
        )

        if issues_by_severity:
            self.instrumentation.record_issues(
                self.checkpoint_name, issues_by_severity
            )
        elif total_issues > 0:
            metrics.issues_per_validation.observe(
                total_issues, checkpoint=self.checkpoint_name
            )

        if self._span:
            self._span.set_status(SpanStatus.OK)
            self._span.set_attribute("validation.total_issues", total_issues)

    def record_failure(
        self,
        reason: str,
        exception: Exception | None = None,
    ) -> None:
        """Record failed validation.

        Args:
            reason: Failure reason.
            exception: Optional exception.
        """
        self._recorded = True
        self._status = "failure"

        metrics = self.instrumentation._metrics
        metrics.validations_total.inc(
            checkpoint=self.checkpoint_name, status="failure"
        )
        metrics.validations_failed.inc(
            checkpoint=self.checkpoint_name, reason=reason[:50]
        )

        if self._span:
            self._span.set_status(SpanStatus.ERROR, reason)
            if exception:
                self._span.set_attribute("exception.type", type(exception).__name__)
                self._span.set_attribute("exception.message", str(exception))

        self.instrumentation._logger.error(
            "Validation failed",
            checkpoint=self.checkpoint_name,
            reason=reason,
            exception=str(exception) if exception else None,
        )

    def record_warning(
        self,
        total_issues: int,
        issues_by_severity: dict[str, int] | None = None,
    ) -> None:
        """Record validation with warnings.

        Args:
            total_issues: Total issues found.
            issues_by_severity: Issues by severity.
        """
        self._recorded = True
        self._status = "warning"

        metrics = self.instrumentation._metrics
        metrics.validations_total.inc(
            checkpoint=self.checkpoint_name, status="warning"
        )

        if issues_by_severity:
            self.instrumentation.record_issues(
                self.checkpoint_name, issues_by_severity
            )

        if self._span:
            self._span.set_attribute("validation.total_issues", total_issues)


# =============================================================================
# Convenience Functions
# =============================================================================

_checkpoint_instrumentation: CheckpointInstrumentation | None = None


def get_checkpoint_instrumentation() -> CheckpointInstrumentation:
    """Get the global checkpoint instrumentation.

    Returns:
        Global CheckpointInstrumentation instance.
    """
    global _checkpoint_instrumentation
    if _checkpoint_instrumentation is None:
        _checkpoint_instrumentation = CheckpointInstrumentation()
    return _checkpoint_instrumentation


def instrument_checkpoint(
    checkpoint_name: str,
    data_asset: str,
) -> "contextmanager[ValidationContext]":
    """Convenience function to instrument a checkpoint.

    Args:
        checkpoint_name: Checkpoint name.
        data_asset: Data asset being validated.

    Returns:
        Context manager yielding ValidationContext.
    """
    return get_checkpoint_instrumentation().validation(
        checkpoint_name, data_asset
    )


# =============================================================================
# Async Instrumentation
# =============================================================================


def async_traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Async version of traced decorator.

    Args:
        name: Span name.
        attributes: Span attributes.

    Example:
        >>> @async_traced()
        ... async def process_async(data):
        ...     return await validate_async(data)
    """
    import asyncio

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with create_span(span_name, attributes=attributes) as span:
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("function.name", func.__qualname__)
                span.set_attribute("function.async", True)
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def async_timed(
    metric_name: str | None = None,
    *,
    labels: dict[str, str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Async version of timed decorator."""
    import asyncio

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        histogram: Histogram | None = None

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal histogram

            if histogram is None:
                collector = get_metrics()
                name = metric_name or f"{func.__module__}_{func.__qualname__}_duration_seconds"
                name = name.replace(".", "_").lower()
                histogram = collector.histogram(
                    name,
                    f"Duration of {func.__qualname__}",
                    labels=list(labels.keys()) if labels else [],
                )

            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                histogram.observe(duration, **(labels or {}))

        return wrapper

    return decorator
