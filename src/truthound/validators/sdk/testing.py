"""Testing utilities for custom validators.

This module provides a comprehensive testing framework for validators:
- ValidatorTestCase: Base class for validator unit tests
- Test data generators and fixtures
- Assertion helpers for validation results
- Performance benchmarking utilities

Example:
    class TestMyValidator(ValidatorTestCase):
        validator_class = MyValidator

        def test_detects_violations(self):
            df = self.create_df({"col1": [1, -1, 2, -2]})
            result = self.validate(df)
            self.assert_has_issue("col1", "negative_value", 2)

        def test_no_issues_for_valid_data(self):
            df = self.create_df({"col1": [1, 2, 3]})
            result = self.validate(df)
            self.assert_no_issues()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar
from unittest import TestCase

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorConfig,
    ValidatorExecutionResult,
)
from truthound.types import Severity


T = TypeVar("T", bound=Validator)


@dataclass
class ValidatorTestResult:
    """Result of a validator test run.

    Attributes:
        issues: List of validation issues found
        execution_time_ms: Time taken in milliseconds
        error: Any error that occurred
        passed: Whether the test passed (based on expectations)
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: Exception | None = None
    passed: bool = True

    @property
    def issue_count(self) -> int:
        """Total number of issues found."""
        return len(self.issues)

    @property
    def total_violations(self) -> int:
        """Sum of all violation counts."""
        return sum(issue.count for issue in self.issues)

    def get_issues_for_column(self, column: str) -> list[ValidationIssue]:
        """Get issues for a specific column."""
        return [i for i in self.issues if i.column == column]

    def get_issues_by_type(self, issue_type: str) -> list[ValidationIssue]:
        """Get issues of a specific type."""
        return [i for i in self.issues if i.issue_type == issue_type]

    def has_issue(
        self,
        column: str | None = None,
        issue_type: str | None = None,
        min_count: int | None = None,
    ) -> bool:
        """Check if a matching issue exists."""
        for issue in self.issues:
            if column and issue.column != column:
                continue
            if issue_type and issue.issue_type != issue_type:
                continue
            if min_count and issue.count < min_count:
                continue
            return True
        return False


class ValidatorTestCase(TestCase):
    """Base class for validator unit tests.

    Provides convenient methods for testing validators:
    - create_df: Create test DataFrames
    - validate: Run validation and capture results
    - assert_*: Various assertion helpers

    Example:
        class TestNullValidator(ValidatorTestCase):
            validator_class = NullValidator

            def test_finds_nulls(self):
                df = self.create_df({
                    "name": ["Alice", None, "Bob", None],
                    "age": [25, 30, None, 35],
                })
                self.validate(df)
                self.assert_has_issue("name", "null_value", min_count=2)
                self.assert_has_issue("age", "null_value", min_count=1)

            def test_no_issues_when_no_nulls(self):
                df = self.create_df({
                    "name": ["Alice", "Bob"],
                    "age": [25, 30],
                })
                self.validate(df)
                self.assert_no_issues()
    """

    validator_class: type[Validator] | None = None
    default_config: ValidatorConfig | None = None

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self._last_result: ValidatorTestResult | None = None
        self._validator: Validator | None = None

    def create_validator(
        self,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ) -> Validator:
        """Create a validator instance.

        Args:
            config: Validator configuration
            **kwargs: Additional config options

        Returns:
            Validator instance

        Raises:
            ValueError: If validator_class is not set
        """
        if self.validator_class is None:
            raise ValueError(
                "validator_class must be set on the test class, "
                "or override create_validator()"
            )

        effective_config = config or self.default_config
        self._validator = self.validator_class(effective_config, **kwargs)
        return self._validator

    def create_df(self, data: dict[str, list[Any]]) -> pl.LazyFrame:
        """Create a test LazyFrame from a dictionary.

        Args:
            data: Dictionary mapping column names to values

        Returns:
            LazyFrame for testing
        """
        return pl.LazyFrame(data)

    def create_large_df(
        self,
        rows: int = 1_000_000,
        schema: dict[str, type] | None = None,
        seed: int = 42,
    ) -> pl.LazyFrame:
        """Create a large test DataFrame for performance testing.

        Args:
            rows: Number of rows
            schema: Column name to type mapping
            seed: Random seed for reproducibility

        Returns:
            Large LazyFrame for testing
        """
        import random

        random.seed(seed)

        if schema is None:
            schema = {
                "id": int,
                "value": float,
                "name": str,
            }

        data: dict[str, list[Any]] = {}

        for col, dtype in schema.items():
            if dtype == int:
                data[col] = list(range(rows))
            elif dtype == float:
                data[col] = [random.random() * 100 for _ in range(rows)]
            elif dtype == str:
                data[col] = [f"value_{i}" for i in range(rows)]
            else:
                data[col] = [None] * rows

        return pl.LazyFrame(data)

    def validate(
        self,
        lf: pl.LazyFrame,
        validator: Validator | None = None,
        **kwargs: Any,
    ) -> ValidatorTestResult:
        """Run validation and store the result.

        Args:
            lf: LazyFrame to validate
            validator: Validator to use (creates default if None)
            **kwargs: Additional config options for new validator

        Returns:
            ValidatorTestResult with issues and timing
        """
        if validator is None:
            if self._validator is None:
                self.create_validator(**kwargs)
            validator = self._validator

        start_time = time.time()
        error: Exception | None = None
        issues: list[ValidationIssue] = []

        try:
            issues = validator.validate(lf)  # type: ignore
        except Exception as e:
            error = e

        execution_time = (time.time() - start_time) * 1000

        self._last_result = ValidatorTestResult(
            issues=issues,
            execution_time_ms=execution_time,
            error=error,
        )

        return self._last_result

    def validate_safe(
        self,
        lf: pl.LazyFrame,
        validator: Validator | None = None,
        **kwargs: Any,
    ) -> ValidatorExecutionResult:
        """Run validation with error handling.

        Args:
            lf: LazyFrame to validate
            validator: Validator to use
            **kwargs: Additional config options

        Returns:
            ValidatorExecutionResult with full status
        """
        if validator is None:
            if self._validator is None:
                self.create_validator(**kwargs)
            validator = self._validator

        return validator.validate_safe(lf)  # type: ignore

    @property
    def last_result(self) -> ValidatorTestResult:
        """Get the last validation result.

        Raises:
            AssertionError: If no validation has been run
        """
        if self._last_result is None:
            raise AssertionError("No validation has been run. Call validate() first.")
        return self._last_result

    def assert_no_issues(self) -> None:
        """Assert that no issues were found."""
        result = self.last_result
        if result.issues:
            issue_summary = "\n".join(
                f"  - {i.column}: {i.issue_type} ({i.count})" for i in result.issues
            )
            self.fail(
                f"Expected no issues, but found {len(result.issues)}:\n{issue_summary}"
            )

    def assert_has_issue(
        self,
        column: str | None = None,
        issue_type: str | None = None,
        min_count: int | None = None,
        exact_count: int | None = None,
        severity: Severity | None = None,
    ) -> None:
        """Assert that a matching issue exists.

        Args:
            column: Expected column (None = any)
            issue_type: Expected issue type (None = any)
            min_count: Minimum violation count
            exact_count: Exact violation count
            severity: Expected severity level
        """
        result = self.last_result

        for issue in result.issues:
            if column and issue.column != column:
                continue
            if issue_type and issue.issue_type != issue_type:
                continue
            if severity and issue.severity != severity:
                continue
            if min_count and issue.count < min_count:
                continue
            if exact_count and issue.count != exact_count:
                continue
            return  # Found matching issue

        # Build failure message
        criteria = []
        if column:
            criteria.append(f"column='{column}'")
        if issue_type:
            criteria.append(f"issue_type='{issue_type}'")
        if min_count:
            criteria.append(f"min_count={min_count}")
        if exact_count:
            criteria.append(f"exact_count={exact_count}")
        if severity:
            criteria.append(f"severity={severity.value}")

        found_summary = (
            "\n".join(
                f"  - {i.column}: {i.issue_type} ({i.count}, {i.severity.value})"
                for i in result.issues
            )
            if result.issues
            else "  (none)"
        )

        self.fail(
            f"Expected issue matching {', '.join(criteria)}, "
            f"but found:\n{found_summary}"
        )

    def assert_issue_count(self, expected: int) -> None:
        """Assert the total number of distinct issues.

        Args:
            expected: Expected number of issues
        """
        result = self.last_result
        self.assertEqual(
            len(result.issues),
            expected,
            f"Expected {expected} issues, found {len(result.issues)}",
        )

    def assert_total_violations(self, expected: int) -> None:
        """Assert the sum of all violation counts.

        Args:
            expected: Expected total violations
        """
        result = self.last_result
        total = sum(i.count for i in result.issues)
        self.assertEqual(
            total,
            expected,
            f"Expected {expected} total violations, found {total}",
        )

    def assert_no_error(self) -> None:
        """Assert that no error occurred during validation."""
        result = self.last_result
        if result.error:
            self.fail(f"Expected no error, but got: {result.error}")

    def assert_error(self, error_type: type[Exception] | None = None) -> None:
        """Assert that an error occurred.

        Args:
            error_type: Expected error type (None = any error)
        """
        result = self.last_result
        if not result.error:
            self.fail("Expected an error, but validation succeeded")
        if error_type and not isinstance(result.error, error_type):
            self.fail(
                f"Expected {error_type.__name__}, "
                f"but got {type(result.error).__name__}: {result.error}"
            )

    def assert_performance(
        self,
        max_ms: float,
        rows: int | None = None,
    ) -> None:
        """Assert that validation completed within time limit.

        Args:
            max_ms: Maximum allowed milliseconds
            rows: Row count for per-row calculation (optional)
        """
        result = self.last_result
        if result.execution_time_ms > max_ms:
            per_row = ""
            if rows:
                per_row = f" ({result.execution_time_ms / rows:.4f} ms/row)"
            self.fail(
                f"Validation took {result.execution_time_ms:.2f}ms, "
                f"expected <= {max_ms}ms{per_row}"
            )


# ============================================================================
# Test Data Generators
# ============================================================================


def create_test_dataframe(
    data: dict[str, list[Any]] | None = None,
    rows: int = 100,
    columns: list[str] | None = None,
    include_nulls: bool = False,
    null_probability: float = 0.1,
    seed: int = 42,
) -> pl.LazyFrame:
    """Create a test DataFrame with configurable properties.

    Args:
        data: Explicit data (if provided, other args ignored)
        rows: Number of rows to generate
        columns: Column names to generate
        include_nulls: Whether to include null values
        null_probability: Probability of null values
        seed: Random seed

    Returns:
        Test LazyFrame
    """
    if data is not None:
        return pl.LazyFrame(data)

    import random

    random.seed(seed)

    if columns is None:
        columns = ["id", "name", "value", "date"]

    generated_data: dict[str, list[Any]] = {}

    for col in columns:
        if col == "id":
            values: list[Any] = list(range(rows))
        elif col == "name":
            values = [f"item_{i}" for i in range(rows)]
        elif col == "value":
            values = [random.random() * 100 for _ in range(rows)]
        elif col == "date":
            from datetime import date, timedelta

            base = date(2024, 1, 1)
            values = [base + timedelta(days=i) for i in range(rows)]
        else:
            values = [f"{col}_{i}" for i in range(rows)]

        if include_nulls:
            values = [
                None if random.random() < null_probability else v for v in values
            ]

        generated_data[col] = values

    return pl.LazyFrame(generated_data)


def create_edge_case_data() -> dict[str, pl.LazyFrame]:
    """Create a collection of edge case test DataFrames.

    Returns:
        Dictionary mapping case names to LazyFrames
    """
    cases: dict[str, pl.LazyFrame] = {}

    # Empty DataFrame
    cases["empty"] = pl.LazyFrame({"col": []})

    # Single row
    cases["single_row"] = pl.LazyFrame({"col": [1]})

    # All nulls
    cases["all_nulls"] = pl.LazyFrame({"col": [None, None, None]})

    # Mixed types (will be Object)
    cases["uniform_values"] = pl.LazyFrame({"col": [1, 1, 1, 1, 1]})

    # Large values
    cases["large_values"] = pl.LazyFrame(
        {"col": [10**15, 10**16, 10**17]}
    )

    # Small values
    cases["small_values"] = pl.LazyFrame(
        {"col": [1e-15, 1e-16, 1e-17]}
    )

    # Unicode strings
    cases["unicode"] = pl.LazyFrame(
        {"col": ["한글", "日本語", "中文", "العربية", "🎉"]}
    )

    # Empty strings
    cases["empty_strings"] = pl.LazyFrame(
        {"col": ["", "", "value", ""]}
    )

    # Whitespace strings
    cases["whitespace"] = pl.LazyFrame(
        {"col": ["  ", "\t", "\n", "value"]}
    )

    # Special floats
    cases["special_floats"] = pl.LazyFrame(
        {"col": [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]}
    )

    return cases


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_no_issues(issues: list[ValidationIssue]) -> None:
    """Assert that the issue list is empty.

    Args:
        issues: List of validation issues

    Raises:
        AssertionError: If issues are present
    """
    if issues:
        summary = "\n".join(
            f"  - {i.column}: {i.issue_type} ({i.count})" for i in issues
        )
        raise AssertionError(f"Expected no issues, found {len(issues)}:\n{summary}")


def assert_has_issue(
    issues: list[ValidationIssue],
    column: str | None = None,
    issue_type: str | None = None,
    min_count: int = 1,
) -> ValidationIssue:
    """Assert that a matching issue exists and return it.

    Args:
        issues: List of validation issues
        column: Expected column (None = any)
        issue_type: Expected issue type (None = any)
        min_count: Minimum violation count

    Returns:
        The matching issue

    Raises:
        AssertionError: If no matching issue found
    """
    for issue in issues:
        if column and issue.column != column:
            continue
        if issue_type and issue.issue_type != issue_type:
            continue
        if issue.count < min_count:
            continue
        return issue

    criteria = []
    if column:
        criteria.append(f"column='{column}'")
    if issue_type:
        criteria.append(f"issue_type='{issue_type}'")
    criteria.append(f"min_count={min_count}")

    found = (
        "\n".join(f"  - {i.column}: {i.issue_type} ({i.count})" for i in issues)
        if issues
        else "  (none)"
    )

    raise AssertionError(
        f"Expected issue matching {', '.join(criteria)}, found:\n{found}"
    )


def assert_issue_count(issues: list[ValidationIssue], expected: int) -> None:
    """Assert the total number of issues.

    Args:
        issues: List of validation issues
        expected: Expected count

    Raises:
        AssertionError: If count doesn't match
    """
    if len(issues) != expected:
        raise AssertionError(
            f"Expected {expected} issues, found {len(issues)}"
        )


# ============================================================================
# Performance Testing
# ============================================================================


@dataclass
class BenchmarkResult:
    """Result of a validator benchmark.

    Attributes:
        validator_name: Name of the validator
        row_count: Number of rows tested
        iterations: Number of iterations run
        mean_ms: Mean execution time in ms
        min_ms: Minimum execution time
        max_ms: Maximum execution time
        std_ms: Standard deviation
        throughput_rows_per_sec: Rows processed per second
    """

    validator_name: str
    row_count: int
    iterations: int
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    throughput_rows_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator_name,
            "rows": self.row_count,
            "iterations": self.iterations,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "throughput": round(self.throughput_rows_per_sec, 0),
        }


def benchmark_validator(
    validator: Validator,
    lf: pl.LazyFrame,
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Benchmark a validator's performance.

    Args:
        validator: Validator to benchmark
        lf: LazyFrame to validate
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)

    Returns:
        BenchmarkResult with timing statistics
    """
    import statistics

    # Get row count
    row_count = lf.select(pl.len()).collect().item()

    # Warmup
    for _ in range(warmup):
        validator.validate(lf)

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        start = time.time()
        validator.validate(lf)
        times.append((time.time() - start) * 1000)

    mean_ms = statistics.mean(times)
    throughput = (row_count / mean_ms) * 1000 if mean_ms > 0 else 0

    return BenchmarkResult(
        validator_name=validator.name,
        row_count=row_count,
        iterations=iterations,
        mean_ms=mean_ms,
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        throughput_rows_per_sec=throughput,
    )


class ValidatorBenchmark:
    """Utility class for benchmarking validators.

    Example:
        benchmark = ValidatorBenchmark()
        benchmark.add_validator(NullValidator())
        benchmark.add_validator(UniqueValidator())

        results = benchmark.run(row_counts=[1000, 10000, 100000])
        benchmark.print_report()
    """

    def __init__(self) -> None:
        self._validators: list[Validator] = []
        self._results: list[BenchmarkResult] = []

    def add_validator(self, validator: Validator) -> "ValidatorBenchmark":
        """Add a validator to benchmark.

        Args:
            validator: Validator instance

        Returns:
            Self for chaining
        """
        self._validators.append(validator)
        return self

    def run(
        self,
        row_counts: list[int] | None = None,
        iterations: int = 10,
        data_generator: Callable[[int], pl.LazyFrame] | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmarks for all validators.

        Args:
            row_counts: List of row counts to test
            iterations: Iterations per benchmark
            data_generator: Function to generate test data

        Returns:
            List of benchmark results
        """
        if row_counts is None:
            row_counts = [1000, 10000, 100000]

        if data_generator is None:
            data_generator = lambda n: create_test_dataframe(rows=n)

        self._results = []

        for rows in row_counts:
            lf = data_generator(rows)
            for validator in self._validators:
                result = benchmark_validator(validator, lf, iterations)
                self._results.append(result)

        return self._results

    def print_report(self) -> None:
        """Print a formatted benchmark report."""
        if not self._results:
            print("No benchmark results. Run benchmarks first.")
            return

        # Group by validator
        by_validator: dict[str, list[BenchmarkResult]] = {}
        for r in self._results:
            if r.validator_name not in by_validator:
                by_validator[r.validator_name] = []
            by_validator[r.validator_name].append(r)

        print("\n" + "=" * 70)
        print("VALIDATOR BENCHMARK REPORT")
        print("=" * 70)

        for name, results in by_validator.items():
            print(f"\n{name}:")
            print("-" * 50)
            print(f"{'Rows':>12} {'Mean (ms)':>12} {'Throughput':>15}")
            print("-" * 50)
            for r in sorted(results, key=lambda x: x.row_count):
                print(
                    f"{r.row_count:>12,} {r.mean_ms:>12.2f} "
                    f"{r.throughput_rows_per_sec:>12,.0f}/s"
                )

        print("\n" + "=" * 70)
