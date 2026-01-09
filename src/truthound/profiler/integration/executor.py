"""Suite execution implementations.

This module provides various execution strategies for running
validation suites against data.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import polars as pl

from truthound.profiler.integration.protocols import (
    ExecutionContext,
    ExecutionResult,
    ValidationExecutor,
    ExecutionListener,
    ProgressReporter,
)
from truthound.profiler.integration.adapters import (
    ValidatorRegistry,
    create_validator_from_rule,
)

if TYPE_CHECKING:
    from truthound.profiler.generators.suite_generator import ValidationSuite
    from truthound.profiler.generators.base import GeneratedRule
    from truthound.validators.base import Validator
    from truthound.report import Report

logger = logging.getLogger(__name__)


class SuiteExecutor(ValidationExecutor):
    """Standard suite executor.

    Executes validation suites by converting rules to validators
    and running them against the provided data.

    Example:
        executor = SuiteExecutor(parallel=True, fail_fast=False)
        result = executor.execute(suite, data)

        if result.success:
            print(f"All {result.executed_rules} rules passed!")
        else:
            print(f"Failed: {result.failed_rules} rules")
            print(result.report)
    """

    def __init__(
        self,
        parallel: bool = False,
        fail_fast: bool = False,
        max_workers: int | None = None,
        timeout_seconds: float | None = None,
        registry: ValidatorRegistry | None = None,
        listeners: list[ExecutionListener] | None = None,
        progress_reporter: ProgressReporter | None = None,
    ):
        """Initialize the executor.

        Args:
            parallel: Whether to run validators in parallel.
            fail_fast: Whether to stop on first failure.
            max_workers: Maximum number of parallel workers.
            timeout_seconds: Maximum execution time per validator.
            registry: Custom validator registry.
            listeners: Event listeners.
            progress_reporter: Progress reporter.
        """
        self._parallel = parallel
        self._fail_fast = fail_fast
        self._max_workers = max_workers
        self._timeout_seconds = timeout_seconds
        self._registry = registry or ValidatorRegistry.get_instance()
        self._listeners = listeners or []
        self._progress_reporter = progress_reporter

    def execute(
        self,
        suite: "ValidationSuite",
        data: Any,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute a validation suite.

        Args:
            suite: The validation suite to execute.
            data: Data to validate.
            context: Execution context options.

        Returns:
            Execution result with report and metrics.
        """
        ctx = context or ExecutionContext(
            parallel=self._parallel,
            fail_fast=self._fail_fast,
            max_workers=self._max_workers,
            timeout_seconds=self._timeout_seconds,
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener.on_suite_start(suite, ctx)
            except Exception as e:
                logger.warning(f"Listener error on suite start: {e}")

        # Handle strictness as either enum or string
        strictness_value = (
            suite.strictness.value
            if hasattr(suite.strictness, 'value')
            else suite.strictness
        )
        result = ExecutionResult(
            success=True,
            report=None,
            started_at=datetime.now(),
            metadata={"suite_name": suite.name, "strictness": strictness_value},
        )

        # Convert data to LazyFrame
        lf = self._to_lazyframe(data)

        # Get rules to execute
        rules = list(suite.rules)
        total_rules = len(rules)
        result.metadata["total_rules"] = total_rules

        if ctx.dry_run:
            return self._execute_dry_run(suite, rules, result)

        # Convert rules to validators
        validators: list[tuple["GeneratedRule", "Validator"]] = []
        for rule in rules:
            try:
                validator = create_validator_from_rule(rule)
                validators.append((rule, validator))
            except Exception as e:
                logger.warning(f"Failed to create validator for rule {rule.name}: {e}")
                result.skipped_rules += 1
                result.errors.append(e)

        # Execute validators
        if ctx.parallel and len(validators) > 1:
            self._execute_parallel(validators, lf, ctx, result)
        else:
            self._execute_sequential(validators, lf, ctx, result)

        # Calculate final result
        result.success = result.failed_rules == 0
        result.complete()

        # Create combined report
        result.report = self._create_report(result)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener.on_suite_complete(result)
            except Exception as e:
                logger.warning(f"Listener error on suite complete: {e}")

        return result

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
        # Use thread pool for async execution
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute(suite, data, context),
        )

    def _to_lazyframe(self, data: Any) -> pl.LazyFrame:
        """Convert data to LazyFrame."""
        if isinstance(data, pl.LazyFrame):
            return data
        if isinstance(data, pl.DataFrame):
            return data.lazy()
        if isinstance(data, str):
            # Assume file path
            if data.endswith(".parquet"):
                return pl.scan_parquet(data)
            elif data.endswith(".csv"):
                return pl.scan_csv(data)
            else:
                return pl.read_csv(data).lazy()

        # Try using adapters
        try:
            from truthound.adapters import to_lazyframe
            return to_lazyframe(data)
        except Exception as e:
            raise ValueError(f"Cannot convert data to LazyFrame: {e}") from e

    def _execute_sequential(
        self,
        validators: list[tuple["GeneratedRule", "Validator"]],
        lf: pl.LazyFrame,
        ctx: ExecutionContext,
        result: ExecutionResult,
    ) -> None:
        """Execute validators sequentially."""
        total = len(validators)

        for idx, (rule, validator) in enumerate(validators):
            # Report progress
            if self._progress_reporter:
                self._progress_reporter.report_progress(idx + 1, total, rule.name)

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener.on_rule_start(rule)
                except Exception as e:
                    logger.debug(f"Listener error: {e}")

            start_time = time.perf_counter()
            success = False
            error: Exception | None = None

            try:
                # Execute validator
                validation_result = validator.validate(lf)
                # validate() returns list[ValidationIssue] - empty means valid
                if hasattr(validation_result, 'is_valid'):
                    # If it has is_valid attribute (e.g., ValidatorExecutionResult)
                    success = validation_result.is_valid
                elif isinstance(validation_result, list):
                    # Standard return: empty list = valid, non-empty = issues found
                    success = len(validation_result) == 0
                else:
                    # Default: trust the result
                    success = bool(validation_result)
                result.executed_rules += 1

                if success:
                    result.passed_rules += 1
                else:
                    result.failed_rules += 1

            except Exception as e:
                error = e
                result.failed_rules += 1
                result.errors.append(e)
                logger.error(f"Validator {rule.name} failed with error: {e}")

                for listener in self._listeners:
                    try:
                        listener.on_error(rule, e)
                    except Exception:
                        pass

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Report result
            if self._progress_reporter:
                msg = str(error) if error else None
                self._progress_reporter.report_result(rule.name, success, msg)

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener.on_rule_complete(rule, success, duration_ms)
                except Exception:
                    pass

            # Check fail fast
            if ctx.fail_fast and not success:
                result.skipped_rules += len(validators) - idx - 1
                break

    def _execute_parallel(
        self,
        validators: list[tuple["GeneratedRule", "Validator"]],
        lf: pl.LazyFrame,
        ctx: ExecutionContext,
        result: ExecutionResult,
    ) -> None:
        """Execute validators in parallel."""
        max_workers = ctx.max_workers or min(32, len(validators))

        def run_validator(
            rule_validator: tuple["GeneratedRule", "Validator"],
        ) -> tuple["GeneratedRule", bool, float, Exception | None]:
            rule, validator = rule_validator
            start = time.perf_counter()
            error = None
            success = False

            try:
                validation_result = validator.validate(lf)
                # validate() returns list[ValidationIssue] - empty means valid
                if hasattr(validation_result, 'is_valid'):
                    success = validation_result.is_valid
                elif isinstance(validation_result, list):
                    success = len(validation_result) == 0
                else:
                    success = bool(validation_result)
            except Exception as e:
                error = e

            duration = (time.perf_counter() - start) * 1000
            return (rule, success, duration, error)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_validator, v): v for v in validators
            }

            completed = 0
            total = len(validators)

            for future in concurrent.futures.as_completed(futures):
                rule, success, duration, error = future.result()
                completed += 1
                result.executed_rules += 1

                if error:
                    result.failed_rules += 1
                    result.errors.append(error)
                elif success:
                    result.passed_rules += 1
                else:
                    result.failed_rules += 1

                if self._progress_reporter:
                    self._progress_reporter.report_progress(completed, total, rule.name)
                    self._progress_reporter.report_result(
                        rule.name,
                        success and error is None,
                        str(error) if error else None,
                    )

    def _execute_dry_run(
        self,
        suite: "ValidationSuite",
        rules: list["GeneratedRule"],
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Execute a dry run without actually running validators."""
        for rule in rules:
            try:
                # Just try to create the validator
                create_validator_from_rule(rule)
                result.executed_rules += 1
                result.passed_rules += 1
            except Exception as e:
                result.executed_rules += 1
                result.failed_rules += 1
                result.errors.append(e)

        result.success = result.failed_rules == 0
        result.metadata["dry_run"] = True
        result.complete()
        return result

    def _create_report(self, result: ExecutionResult) -> dict[str, Any]:
        """Create a report from the execution result."""
        return {
            "summary": {
                "success": result.success,
                "executed": result.executed_rules,
                "passed": result.passed_rules,
                "failed": result.failed_rules,
                "skipped": result.skipped_rules,
                "pass_rate": result.pass_rate,
                "execution_time_ms": result.execution_time_ms,
            },
            "errors": [str(e) for e in result.errors],
            "metadata": result.metadata,
        }


class AsyncSuiteExecutor(SuiteExecutor):
    """Async-first suite executor.

    Uses asyncio for execution, suitable for I/O-bound validations.
    """

    async def execute_async(
        self,
        suite: "ValidationSuite",
        data: Any,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute a validation suite asynchronously."""
        ctx = context or ExecutionContext(
            parallel=self._parallel,
            fail_fast=self._fail_fast,
        )

        result = ExecutionResult(
            success=True,
            report=None,
            started_at=datetime.now(),
            metadata={"suite_name": suite.name},
        )

        lf = self._to_lazyframe(data)
        rules = list(suite.rules)

        # Convert rules to validators
        validators: list[tuple["GeneratedRule", "Validator"]] = []
        for rule in rules:
            try:
                validator = create_validator_from_rule(rule)
                validators.append((rule, validator))
            except Exception as e:
                result.skipped_rules += 1
                result.errors.append(e)

        # Execute with asyncio
        if ctx.parallel:
            tasks = [
                self._run_validator_async(rule, validator, lf)
                for rule, validator in validators
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (rule, _), r in zip(validators, results):
                result.executed_rules += 1
                if isinstance(r, Exception):
                    result.failed_rules += 1
                    result.errors.append(r)
                elif r:
                    result.passed_rules += 1
                else:
                    result.failed_rules += 1
        else:
            for rule, validator in validators:
                try:
                    success = await self._run_validator_async(rule, validator, lf)
                    result.executed_rules += 1
                    if success:
                        result.passed_rules += 1
                    else:
                        result.failed_rules += 1
                except Exception as e:
                    result.executed_rules += 1
                    result.failed_rules += 1
                    result.errors.append(e)

                if ctx.fail_fast and result.failed_rules > 0:
                    break

        result.success = result.failed_rules == 0
        result.complete()
        result.report = self._create_report(result)
        return result

    async def _run_validator_async(
        self,
        rule: "GeneratedRule",
        validator: "Validator",
        lf: pl.LazyFrame,
    ) -> bool:
        """Run a single validator asynchronously."""
        def run_validate() -> bool:
            result = validator.validate(lf)
            if hasattr(result, 'is_valid'):
                return result.is_valid
            elif isinstance(result, list):
                return len(result) == 0
            return bool(result)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_validate)


class DryRunExecutor(SuiteExecutor):
    """Executor that only validates rule conversion without execution."""

    def execute(
        self,
        suite: "ValidationSuite",
        data: Any,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute dry run."""
        ctx = context or ExecutionContext(dry_run=True)
        ctx = ExecutionContext(
            parallel=ctx.parallel,
            fail_fast=ctx.fail_fast,
            max_workers=ctx.max_workers,
            timeout_seconds=ctx.timeout_seconds,
            dry_run=True,
            collect_metrics=ctx.collect_metrics,
        )
        return super().execute(suite, data, ctx)


class ParallelExecutor(SuiteExecutor):
    """Executor optimized for parallel execution."""

    def __init__(
        self,
        max_workers: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(parallel=True, max_workers=max_workers, **kwargs)


def create_executor(
    strategy: str = "standard",
    **kwargs: Any,
) -> SuiteExecutor:
    """Factory function for creating executors.

    Args:
        strategy: Execution strategy ('standard', 'parallel', 'async', 'dry_run').
        **kwargs: Additional executor options.

    Returns:
        Configured executor instance.

    Example:
        executor = create_executor("parallel", max_workers=4)
        result = executor.execute(suite, data)
    """
    strategies = {
        "standard": SuiteExecutor,
        "parallel": ParallelExecutor,
        "async": AsyncSuiteExecutor,
        "dry_run": DryRunExecutor,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")

    return strategies[strategy](**kwargs)
