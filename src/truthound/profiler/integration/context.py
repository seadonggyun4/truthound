"""Execution context builders and utilities.

This module provides fluent builders for creating execution contexts
with various configuration options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from truthound.profiler.integration.protocols import (
    ExecutionContext,
    ExecutionListener,
    ProgressReporter,
)

if TYPE_CHECKING:
    from truthound.profiler.generators.suite_generator import ValidationSuite
    from truthound.profiler.generators.base import GeneratedRule


@dataclass
class ExecutionConfig:
    """Extended execution configuration.

    Attributes:
        parallel: Whether to run validators in parallel.
        fail_fast: Whether to stop on first failure.
        max_workers: Maximum parallel workers.
        timeout_seconds: Per-validator timeout.
        dry_run: Simulate without execution.
        collect_metrics: Track detailed metrics.
        skip_disabled: Skip disabled rules.
        filter_categories: Only run rules in these categories.
        filter_columns: Only run rules for these columns.
        min_confidence: Minimum confidence level.
        extra: Additional context data.
    """

    parallel: bool = False
    fail_fast: bool = False
    max_workers: int | None = None
    timeout_seconds: float | None = None
    dry_run: bool = False
    collect_metrics: bool = True
    skip_disabled: bool = True
    filter_categories: list[str] | None = None
    filter_columns: list[str] | None = None
    min_confidence: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> ExecutionContext:
        """Convert to ExecutionContext."""
        return ExecutionContext(
            parallel=self.parallel,
            fail_fast=self.fail_fast,
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
            dry_run=self.dry_run,
            collect_metrics=self.collect_metrics,
            skip_disabled=self.skip_disabled,
            extra={
                **self.extra,
                "filter_categories": self.filter_categories,
                "filter_columns": self.filter_columns,
                "min_confidence": self.min_confidence,
            },
        )


class ExecutionContextBuilder:
    """Fluent builder for execution contexts.

    Example:
        context = (
            ExecutionContextBuilder()
            .parallel(max_workers=4)
            .with_timeout(30.0)
            .fail_fast()
            .filter_by_category("schema", "completeness")
            .build()
        )

        result = executor.execute(suite, data, context)
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._config = ExecutionConfig()
        self._listeners: list[ExecutionListener] = []
        self._progress_reporter: ProgressReporter | None = None
        self._rule_filter: Callable[["GeneratedRule"], bool] | None = None

    def parallel(self, max_workers: int | None = None) -> "ExecutionContextBuilder":
        """Enable parallel execution.

        Args:
            max_workers: Maximum number of workers.
        """
        self._config.parallel = True
        self._config.max_workers = max_workers
        return self

    def sequential(self) -> "ExecutionContextBuilder":
        """Use sequential execution."""
        self._config.parallel = False
        return self

    def fail_fast(self, enabled: bool = True) -> "ExecutionContextBuilder":
        """Enable or disable fail-fast mode.

        Args:
            enabled: Whether to enable fail-fast.
        """
        self._config.fail_fast = enabled
        return self

    def with_timeout(self, seconds: float) -> "ExecutionContextBuilder":
        """Set per-validator timeout.

        Args:
            seconds: Timeout in seconds.
        """
        self._config.timeout_seconds = seconds
        return self

    def dry_run(self, enabled: bool = True) -> "ExecutionContextBuilder":
        """Enable or disable dry run mode.

        Args:
            enabled: Whether to enable dry run.
        """
        self._config.dry_run = enabled
        return self

    def collect_metrics(self, enabled: bool = True) -> "ExecutionContextBuilder":
        """Enable or disable metrics collection.

        Args:
            enabled: Whether to collect metrics.
        """
        self._config.collect_metrics = enabled
        return self

    def filter_by_category(self, *categories: str) -> "ExecutionContextBuilder":
        """Only run rules in specified categories.

        Args:
            categories: Category names to include.
        """
        self._config.filter_categories = list(categories)
        return self

    def filter_by_columns(self, *columns: str) -> "ExecutionContextBuilder":
        """Only run rules for specified columns.

        Args:
            columns: Column names to include.
        """
        self._config.filter_columns = list(columns)
        return self

    def with_min_confidence(self, level: str) -> "ExecutionContextBuilder":
        """Set minimum confidence level.

        Args:
            level: Minimum confidence ('low', 'medium', 'high').
        """
        self._config.min_confidence = level
        return self

    def with_listener(self, listener: ExecutionListener) -> "ExecutionContextBuilder":
        """Add an execution listener.

        Args:
            listener: Listener to add.
        """
        self._listeners.append(listener)
        return self

    def with_progress(self, reporter: ProgressReporter) -> "ExecutionContextBuilder":
        """Set progress reporter.

        Args:
            reporter: Progress reporter.
        """
        self._progress_reporter = reporter
        return self

    def with_rule_filter(
        self,
        filter_fn: Callable[["GeneratedRule"], bool],
    ) -> "ExecutionContextBuilder":
        """Set custom rule filter.

        Args:
            filter_fn: Filter function that returns True for rules to include.
        """
        self._rule_filter = filter_fn
        return self

    def with_extra(self, key: str, value: Any) -> "ExecutionContextBuilder":
        """Add extra context data.

        Args:
            key: Data key.
            value: Data value.
        """
        self._config.extra[key] = value
        return self

    def build(self) -> ExecutionContext:
        """Build the execution context."""
        ctx = self._config.to_context()

        if self._listeners:
            ctx.extra["listeners"] = self._listeners
        if self._progress_reporter:
            ctx.extra["progress_reporter"] = self._progress_reporter
        if self._rule_filter:
            ctx.extra["rule_filter"] = self._rule_filter

        return ctx

    @classmethod
    def for_ci(cls) -> "ExecutionContextBuilder":
        """Create builder preset for CI environments.

        - Parallel execution
        - No fail-fast (run all tests)
        - Metrics collection enabled
        """
        return cls().parallel().fail_fast(False).collect_metrics(True)

    @classmethod
    def for_development(cls) -> "ExecutionContextBuilder":
        """Create builder preset for development.

        - Sequential execution
        - Fail-fast enabled
        - Metrics collection enabled
        """
        return cls().sequential().fail_fast(True).collect_metrics(True)

    @classmethod
    def for_production(cls) -> "ExecutionContextBuilder":
        """Create builder preset for production.

        - Parallel execution
        - Timeout protection
        - No fail-fast
        """
        return cls().parallel(max_workers=4).with_timeout(30.0).fail_fast(False)


def create_context(
    parallel: bool = False,
    fail_fast: bool = False,
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
    **extra: Any,
) -> ExecutionContext:
    """Create an execution context with common options.

    This is a convenience function for simple context creation.

    Args:
        parallel: Enable parallel execution.
        fail_fast: Stop on first failure.
        max_workers: Maximum parallel workers.
        timeout_seconds: Per-validator timeout.
        dry_run: Simulate without execution.
        **extra: Additional context data.

    Returns:
        Configured execution context.

    Example:
        context = create_context(parallel=True, fail_fast=True)
        result = executor.execute(suite, data, context)
    """
    return ExecutionContext(
        parallel=parallel,
        fail_fast=fail_fast,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
        extra=extra,
    )


class ConsoleProgressReporter(ProgressReporter):
    """Simple console-based progress reporter."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize reporter.

        Args:
            verbose: Whether to print detailed output.
        """
        self._verbose = verbose

    def report_progress(
        self,
        current: int,
        total: int,
        current_rule: str | None = None,
    ) -> None:
        """Report current progress."""
        if self._verbose:
            rule_info = f" - {current_rule}" if current_rule else ""
            print(f"[{current}/{total}]{rule_info}")

    def report_result(
        self,
        rule_name: str,
        success: bool,
        message: str | None = None,
    ) -> None:
        """Report result of a single rule."""
        status = "PASS" if success else "FAIL"
        msg_info = f": {message}" if message else ""
        print(f"  {status}: {rule_name}{msg_info}")


class SimpleExecutionListener(ExecutionListener):
    """Simple execution listener that logs events."""

    def __init__(self, logger_name: str = "truthound.execution") -> None:
        """Initialize listener.

        Args:
            logger_name: Name of the logger to use.
        """
        import logging
        self._logger = logging.getLogger(logger_name)

    def on_suite_start(
        self,
        suite: "ValidationSuite",
        context: ExecutionContext,
    ) -> None:
        """Called when suite execution starts."""
        self._logger.info(f"Starting suite: {suite.name} ({len(suite)} rules)")

    def on_rule_start(self, rule: "GeneratedRule") -> None:
        """Called when a rule execution starts."""
        self._logger.debug(f"Starting rule: {rule.name}")

    def on_rule_complete(
        self,
        rule: "GeneratedRule",
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called when a rule execution completes."""
        status = "passed" if success else "failed"
        self._logger.debug(f"Rule {rule.name} {status} in {duration_ms:.2f}ms")

    def on_suite_complete(self, result: "ExecutionResult") -> None:
        """Called when suite execution completes."""
        self._logger.info(
            f"Suite complete: {result.passed_rules}/{result.executed_rules} passed "
            f"({result.pass_rate:.1f}%) in {result.execution_time_ms:.2f}ms"
        )

    def on_error(self, rule: "GeneratedRule", error: Exception) -> None:
        """Called when an error occurs."""
        self._logger.error(f"Error in rule {rule.name}: {error}")
