"""Protocol definitions for profiler-validator integration.

This module defines the interfaces for executing validation suites
and converting generated rules to validators.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeVar, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import polars as pl
    from truthound.profiler.generators.suite_generator import ValidationSuite
    from truthound.profiler.generators.base import GeneratedRule
    from truthound.validators.base import Validator
    from truthound.report import Report

T = TypeVar("T")


@dataclass
class ExecutionResult:
    """Result of suite execution.

    Attributes:
        success: Whether all validations passed.
        report: Full validation report.
        executed_rules: Number of rules executed.
        passed_rules: Number of rules that passed.
        failed_rules: Number of rules that failed.
        skipped_rules: Number of rules that were skipped.
        execution_time_ms: Total execution time in milliseconds.
        errors: Any errors encountered during execution.
        metadata: Additional execution metadata.
    """

    success: bool
    report: Any  # Report type
    executed_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    skipped_rules: int = 0
    execution_time_ms: float = 0.0
    errors: list[Exception] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        if self.executed_rules == 0:
            return 100.0
        return (self.passed_rules / self.executed_rules) * 100

    def complete(self) -> None:
        """Mark execution as complete."""
        self.completed_at = datetime.now()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.execution_time_ms = delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "executed_rules": self.executed_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "skipped_rules": self.skipped_rules,
            "pass_rate": self.pass_rate,
            "execution_time_ms": self.execution_time_ms,
            "error_count": len(self.errors),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionContext:
    """Context for suite execution.

    Attributes:
        parallel: Whether to run validators in parallel.
        fail_fast: Whether to stop on first failure.
        max_workers: Maximum number of parallel workers.
        timeout_seconds: Maximum execution time per validator.
        dry_run: Whether to simulate execution without running validators.
        collect_metrics: Whether to collect detailed metrics.
        skip_disabled: Whether to skip disabled rules.
        extra: Additional context data.
    """

    parallel: bool = False
    fail_fast: bool = False
    max_workers: int | None = None
    timeout_seconds: float | None = None
    dry_run: bool = False
    collect_metrics: bool = True
    skip_disabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ValidationExecutor(Protocol):
    """Protocol for suite execution implementations."""

    @abstractmethod
    def execute(
        self,
        suite: "ValidationSuite",
        data: Any,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute a validation suite against data.

        Args:
            suite: The validation suite to execute.
            data: Data to validate (LazyFrame, DataFrame, file path, etc.).
            context: Execution context options.

        Returns:
            Execution result with report and metrics.
        """
        ...

    @abstractmethod
    async def execute_async(
        self,
        suite: "ValidationSuite",
        data: Any,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute a validation suite asynchronously.

        Args:
            suite: The validation suite to execute.
            data: Data to validate.
            context: Execution context options.

        Returns:
            Execution result with report and metrics.
        """
        ...


@runtime_checkable
class ValidatorFactory(Protocol):
    """Protocol for creating validators from generated rules."""

    @abstractmethod
    def create_validator(self, rule: "GeneratedRule") -> "Validator":
        """Create a validator from a generated rule.

        Args:
            rule: The generated rule to convert.

        Returns:
            A Validator instance.

        Raises:
            ValueError: If the rule cannot be converted.
        """
        ...

    @abstractmethod
    def supports_rule(self, rule: "GeneratedRule") -> bool:
        """Check if this factory supports the given rule.

        Args:
            rule: The rule to check.

        Returns:
            True if the factory can create a validator for this rule.
        """
        ...


class ExecutionListener(Protocol):
    """Protocol for execution event listeners."""

    def on_suite_start(self, suite: "ValidationSuite", context: ExecutionContext) -> None:
        """Called when suite execution starts."""
        ...

    def on_rule_start(self, rule: "GeneratedRule") -> None:
        """Called when a rule execution starts."""
        ...

    def on_rule_complete(
        self,
        rule: "GeneratedRule",
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called when a rule execution completes."""
        ...

    def on_suite_complete(self, result: ExecutionResult) -> None:
        """Called when suite execution completes."""
        ...

    def on_error(self, rule: "GeneratedRule", error: Exception) -> None:
        """Called when an error occurs during rule execution."""
        ...


class ProgressReporter(Protocol):
    """Protocol for reporting execution progress."""

    def report_progress(
        self,
        current: int,
        total: int,
        current_rule: str | None = None,
    ) -> None:
        """Report current progress.

        Args:
            current: Current rule number.
            total: Total number of rules.
            current_rule: Name of the current rule.
        """
        ...

    def report_result(
        self,
        rule_name: str,
        success: bool,
        message: str | None = None,
    ) -> None:
        """Report result of a single rule execution.

        Args:
            rule_name: Name of the rule.
            success: Whether the rule passed.
            message: Optional result message.
        """
        ...
