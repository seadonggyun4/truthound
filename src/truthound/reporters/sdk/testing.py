"""Testing utilities for reporter development.

This module provides tools for testing custom reporters including
mock data generators, assertion helpers, and test case base classes.

Example:
    >>> from truthound.reporters.sdk.testing import (
    ...     ReporterTestCase,
    ...     create_mock_result,
    ...     assert_valid_output,
    ... )
    >>>
    >>> class TestMyReporter(ReporterTestCase):
    ...     reporter_class = MyCustomReporter
    ...
    ...     def test_render_output(self):
    ...         result = create_mock_result(passed=5, failed=3)
    ...         output = self.reporter.render(result)
    ...         assert_valid_output(output, format="json")
"""

from __future__ import annotations

import json
import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from xml.etree import ElementTree

__all__ = [
    # Test case base
    "ReporterTestCase",
    # Mock data generators
    "create_mock_result",
    "create_mock_results",
    "create_mock_validator_result",
    "MockResultBuilder",
    # Assertions
    "assert_valid_output",
    "assert_json_valid",
    "assert_xml_valid",
    "assert_csv_valid",
    "assert_contains_patterns",
    # Fixtures
    "create_sample_data",
    "create_edge_case_data",
    "create_stress_test_data",
    # Utilities
    "capture_output",
    "benchmark_reporter",
]


class Severity(Enum):
    """Severity levels for mock validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MockValidatorResult:
    """Mock validator result for testing."""

    validator_name: str
    column: Optional[str] = None
    passed: bool = True
    message: str = ""
    severity: Severity = Severity.ERROR
    details: Dict[str, Any] = field(default_factory=dict)
    row_count: int = 100
    failed_count: int = 0
    execution_time_ms: float = 10.0
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.message:
            if self.passed:
                self.message = f"Validation passed for {self.column or 'data'}"
            else:
                self.message = f"Validation failed for {self.column or 'data'}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validator_name": self.validator_name,
            "column": self.column,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "row_count": self.row_count,
            "failed_count": self.failed_count,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class MockValidationResult:
    """Mock validation result for testing reporters."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_asset: str = "test_data.csv"
    validator_results: List[MockValidatorResult] = field(default_factory=list)
    success: bool = True
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    row_count: int = 1000
    columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now() - timedelta(seconds=5)
        if self.completed_at is None:
            self.completed_at = datetime.now()
        if not self.columns:
            self.columns = ["id", "name", "email", "age", "status"]
        if not self.validator_results:
            # Generate some default results
            self._generate_default_results()

    def _generate_default_results(self):
        """Generate default validator results."""
        self.validator_results = [
            MockValidatorResult(
                validator_name="not_null",
                column="id",
                passed=True,
                row_count=self.row_count,
            ),
            MockValidatorResult(
                validator_name="unique",
                column="id",
                passed=True,
                row_count=self.row_count,
            ),
            MockValidatorResult(
                validator_name="email_format",
                column="email",
                passed=self.success,
                failed_count=0 if self.success else 5,
                row_count=self.row_count,
            ),
        ]

    @property
    def passed_count(self) -> int:
        """Number of passed validations."""
        return sum(1 for r in self.validator_results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of failed validations."""
        return sum(1 for r in self.validator_results if not r.passed)

    @property
    def total_count(self) -> int:
        """Total number of validations."""
        return len(self.validator_results)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_count == 0:
            return 100.0
        return (self.passed_count / self.total_count) * 100

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "run_id": self.run_id,
            "data_asset": self.data_asset,
            "success": self.success,
            "validator_results": [r.to_dict() for r in self.validator_results],
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "row_count": self.row_count,
            "columns": self.columns,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class MockResultBuilder:
    """Fluent builder for creating mock validation results.

    Example:
        >>> result = (
        ...     MockResultBuilder()
        ...     .with_data_asset("users.csv")
        ...     .with_row_count(10000)
        ...     .add_passed("not_null", column="id")
        ...     .add_passed("unique", column="id")
        ...     .add_failed("email_format", column="email", failed_count=50)
        ...     .build()
        ... )
    """

    def __init__(self):
        self._run_id: str = str(uuid.uuid4())
        self._data_asset: str = "test_data.csv"
        self._validator_results: List[MockValidatorResult] = []
        self._metadata: Dict[str, Any] = {}
        self._row_count: int = 1000
        self._columns: List[str] = []
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None

    def with_run_id(self, run_id: str) -> "MockResultBuilder":
        """Set the run ID."""
        self._run_id = run_id
        return self

    def with_data_asset(self, data_asset: str) -> "MockResultBuilder":
        """Set the data asset name."""
        self._data_asset = data_asset
        return self

    def with_row_count(self, count: int) -> "MockResultBuilder":
        """Set the row count."""
        self._row_count = count
        return self

    def with_columns(self, columns: List[str]) -> "MockResultBuilder":
        """Set the column names."""
        self._columns = columns
        return self

    def with_metadata(self, metadata: Dict[str, Any]) -> "MockResultBuilder":
        """Set metadata."""
        self._metadata = metadata
        return self

    def with_timestamps(
        self,
        started_at: datetime,
        completed_at: Optional[datetime] = None,
    ) -> "MockResultBuilder":
        """Set timestamps."""
        self._started_at = started_at
        self._completed_at = completed_at or datetime.now()
        return self

    def add_result(self, result: MockValidatorResult) -> "MockResultBuilder":
        """Add a validator result."""
        self._validator_results.append(result)
        return self

    def add_passed(
        self,
        validator_name: str,
        column: Optional[str] = None,
        **kwargs,
    ) -> "MockResultBuilder":
        """Add a passed validation result."""
        result = MockValidatorResult(
            validator_name=validator_name,
            column=column,
            passed=True,
            row_count=self._row_count,
            **kwargs,
        )
        self._validator_results.append(result)
        return self

    def add_failed(
        self,
        validator_name: str,
        column: Optional[str] = None,
        failed_count: int = 1,
        severity: Severity = Severity.ERROR,
        **kwargs,
    ) -> "MockResultBuilder":
        """Add a failed validation result."""
        result = MockValidatorResult(
            validator_name=validator_name,
            column=column,
            passed=False,
            failed_count=failed_count,
            severity=severity,
            row_count=self._row_count,
            **kwargs,
        )
        self._validator_results.append(result)
        return self

    def add_warning(
        self,
        validator_name: str,
        column: Optional[str] = None,
        message: str = "",
        **kwargs,
    ) -> "MockResultBuilder":
        """Add a warning validation result."""
        result = MockValidatorResult(
            validator_name=validator_name,
            column=column,
            passed=True,
            severity=Severity.WARNING,
            message=message,
            row_count=self._row_count,
            **kwargs,
        )
        self._validator_results.append(result)
        return self

    def build(self) -> MockValidationResult:
        """Build the mock validation result."""
        success = all(r.passed for r in self._validator_results)

        return MockValidationResult(
            run_id=self._run_id,
            data_asset=self._data_asset,
            validator_results=self._validator_results,
            success=success,
            started_at=self._started_at,
            completed_at=self._completed_at,
            metadata=self._metadata,
            row_count=self._row_count,
            columns=self._columns or ["id", "name", "email", "age", "status"],
        )


def create_mock_result(
    passed: int = 5,
    failed: int = 0,
    warnings: int = 0,
    row_count: int = 1000,
    data_asset: str = "test_data.csv",
) -> MockValidationResult:
    """Create a mock validation result with specified counts.

    Args:
        passed: Number of passed validations.
        failed: Number of failed validations.
        warnings: Number of warning validations.
        row_count: Number of rows in the mock data.
        data_asset: Name of the data asset.

    Returns:
        MockValidationResult with the specified configuration.

    Example:
        >>> result = create_mock_result(passed=10, failed=2)
        >>> print(result.success_rate)  # 83.33%
    """
    builder = MockResultBuilder().with_data_asset(data_asset).with_row_count(row_count)

    validators = ["not_null", "unique", "range", "format", "regex", "length"]
    columns = ["id", "name", "email", "age", "status", "created_at"]

    for i in range(passed):
        builder.add_passed(
            validators[i % len(validators)],
            column=columns[i % len(columns)],
        )

    for i in range(failed):
        builder.add_failed(
            validators[i % len(validators)],
            column=columns[i % len(columns)],
            failed_count=max(1, row_count // 100),
        )

    for i in range(warnings):
        builder.add_warning(
            validators[i % len(validators)],
            column=columns[i % len(columns)],
            message=f"Warning: potential issue in {columns[i % len(columns)]}",
        )

    return builder.build()


def create_mock_results(
    count: int = 5,
    success_rate: float = 0.8,
    row_count: int = 1000,
) -> List[MockValidationResult]:
    """Create multiple mock validation results.

    Args:
        count: Number of results to create.
        success_rate: Ratio of successful results (0.0 to 1.0).
        row_count: Number of rows per result.

    Returns:
        List of MockValidationResult objects.

    Example:
        >>> results = create_mock_results(count=10, success_rate=0.7)
        >>> successful = [r for r in results if r.success]
    """
    import random

    results = []
    for i in range(count):
        is_success = random.random() < success_rate
        passed = random.randint(3, 10)
        failed = 0 if is_success else random.randint(1, 3)

        result = create_mock_result(
            passed=passed,
            failed=failed,
            row_count=row_count,
            data_asset=f"data_{i:03d}.csv",
        )
        results.append(result)

    return results


def create_mock_validator_result(
    validator_name: str = "not_null",
    column: Optional[str] = "id",
    passed: bool = True,
    **kwargs,
) -> MockValidatorResult:
    """Create a single mock validator result.

    Args:
        validator_name: Name of the validator.
        column: Column being validated.
        passed: Whether validation passed.
        **kwargs: Additional arguments for MockValidatorResult.

    Returns:
        MockValidatorResult instance.
    """
    return MockValidatorResult(
        validator_name=validator_name,
        column=column,
        passed=passed,
        **kwargs,
    )


class ReporterTestCase(ABC):
    """Base class for reporter test cases.

    Subclass this to create test suites for your custom reporters.
    Provides common setup, teardown, and utility methods.

    Example:
        >>> class TestMyReporter(ReporterTestCase):
        ...     reporter_class = MyCustomReporter
        ...     reporter_kwargs = {"indent": 2}
        ...
        ...     def test_basic_render(self):
        ...         result = self.create_result(passed=5, failed=1)
        ...         output = self.reporter.render(result)
        ...         self.assert_valid_output(output)
        ...
        ...     def test_empty_result(self):
        ...         result = self.create_result(passed=0, failed=0)
        ...         output = self.reporter.render(result)
        ...         self.assertIsNotNone(output)
    """

    reporter_class: Optional[Type] = None
    reporter_kwargs: Dict[str, Any] = {}

    def setUp(self):
        """Set up test fixtures."""
        if self.reporter_class is not None:
            self.reporter = self.reporter_class(**self.reporter_kwargs)
        else:
            self.reporter = None

    def tearDown(self):
        """Clean up after tests."""
        self.reporter = None

    def create_result(self, **kwargs) -> MockValidationResult:
        """Create a mock validation result with the given parameters."""
        return create_mock_result(**kwargs)

    def create_results(self, count: int = 5, **kwargs) -> List[MockValidationResult]:
        """Create multiple mock validation results."""
        return create_mock_results(count=count, **kwargs)

    def assert_valid_output(
        self,
        output: Any,
        format: Optional[str] = None,
    ) -> None:
        """Assert that output is valid for the expected format."""
        assert_valid_output(output, format=format)

    def assert_json_structure(
        self,
        output: Any,
        required_keys: Optional[List[str]] = None,
    ) -> None:
        """Assert JSON output has required structure."""
        if isinstance(output, str):
            data = json.loads(output)
        else:
            data = output

        if required_keys:
            for key in required_keys:
                assert key in data, f"Missing required key: {key}"

    def assert_contains_all(self, output: str, substrings: List[str]) -> None:
        """Assert output contains all specified substrings."""
        for substring in substrings:
            assert substring in output, f"Missing expected content: {substring}"


def assert_valid_output(
    output: Any,
    format: Optional[str] = None,
) -> None:
    """Assert that output is valid.

    Args:
        output: The output to validate.
        format: Expected format (json, xml, csv, text).

    Raises:
        AssertionError: If output is invalid.

    Example:
        >>> output = '{"success": true}'
        >>> assert_valid_output(output, format="json")  # OK
    """
    assert output is not None, "Output is None"

    if format is None:
        return

    format = format.lower()

    if format == "json":
        assert_json_valid(output)
    elif format == "xml":
        assert_xml_valid(output)
    elif format == "csv":
        assert_csv_valid(output)
    elif format == "text":
        assert isinstance(output, str), f"Expected string, got {type(output)}"


def assert_json_valid(output: Any) -> Dict[str, Any]:
    """Assert that output is valid JSON.

    Args:
        output: JSON string or dict to validate.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        AssertionError: If output is not valid JSON.
    """
    if isinstance(output, dict):
        return output

    assert isinstance(output, str), f"Expected string or dict, got {type(output)}"

    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON: {e}")


def assert_xml_valid(output: Any) -> ElementTree.Element:
    """Assert that output is valid XML.

    Args:
        output: XML string or Element to validate.

    Returns:
        Parsed XML as Element.

    Raises:
        AssertionError: If output is not valid XML.
    """
    if isinstance(output, ElementTree.Element):
        return output

    assert isinstance(
        output, (str, bytes)
    ), f"Expected string, bytes, or Element, got {type(output)}"

    try:
        return ElementTree.fromstring(output)
    except ElementTree.ParseError as e:
        raise AssertionError(f"Invalid XML: {e}")


def assert_csv_valid(output: Any, has_header: bool = True) -> List[List[str]]:
    """Assert that output is valid CSV.

    Args:
        output: CSV string to validate.
        has_header: Whether CSV has a header row.

    Returns:
        Parsed CSV as list of rows.

    Raises:
        AssertionError: If output is not valid CSV.
    """
    import csv
    from io import StringIO

    assert isinstance(output, str), f"Expected string, got {type(output)}"

    try:
        reader = csv.reader(StringIO(output))
        rows = list(reader)
        assert len(rows) > 0, "CSV is empty"
        return rows
    except Exception as e:
        raise AssertionError(f"Invalid CSV: {e}")


def assert_contains_patterns(
    output: str,
    patterns: List[str],
    regex: bool = False,
) -> None:
    """Assert that output contains all specified patterns.

    Args:
        output: Output string to check.
        patterns: List of patterns to find.
        regex: If True, treat patterns as regular expressions.

    Raises:
        AssertionError: If any pattern is not found.

    Example:
        >>> output = "Total: 100 rows, 5 errors"
        >>> assert_contains_patterns(output, ["Total:", "errors"])
    """
    import re

    for pattern in patterns:
        if regex:
            match = re.search(pattern, output)
            assert match is not None, f"Pattern not found: {pattern}"
        else:
            assert pattern in output, f"Pattern not found: {pattern}"


def create_sample_data() -> Dict[str, List[MockValidationResult]]:
    """Create sample data for testing various scenarios.

    Returns:
        Dictionary mapping scenario names to mock results.

    Example:
        >>> samples = create_sample_data()
        >>> for name, results in samples.items():
        ...     print(f"{name}: {len(results)} results")
    """
    return {
        "all_passed": [create_mock_result(passed=10, failed=0)],
        "all_failed": [create_mock_result(passed=0, failed=10)],
        "mixed": [create_mock_result(passed=7, failed=3)],
        "empty": [create_mock_result(passed=0, failed=0)],
        "single_passed": [create_mock_result(passed=1, failed=0)],
        "single_failed": [create_mock_result(passed=0, failed=1)],
        "large_dataset": [create_mock_result(passed=50, failed=10, row_count=1000000)],
        "multiple_runs": create_mock_results(count=5, success_rate=0.8),
    }


def create_edge_case_data() -> Dict[str, MockValidationResult]:
    """Create edge case data for testing boundary conditions.

    Returns:
        Dictionary mapping edge case names to mock results.

    Example:
        >>> edge_cases = create_edge_case_data()
        >>> for name, result in edge_cases.items():
        ...     output = reporter.render(result)
        ...     assert_valid_output(output, format="json")
    """
    return {
        "empty_validators": MockResultBuilder().build(),
        "zero_rows": MockResultBuilder()
        .with_row_count(0)
        .add_passed("not_null", column="id")
        .build(),
        "unicode_data_asset": MockResultBuilder()
        .with_data_asset("데이터_파일_🔥.csv")
        .add_passed("not_null", column="이름")
        .build(),
        "special_chars": MockResultBuilder()
        .with_data_asset('file"with<special>chars&.csv')
        .add_failed(
            "format",
            column="col<with>chars",
            message='Error: "value" is <invalid>',
        )
        .build(),
        "long_column_names": MockResultBuilder()
        .add_passed(
            "not_null",
            column="this_is_a_very_long_column_name_that_might_cause_formatting_issues_in_some_reporters",
        )
        .build(),
        "many_validators": MockResultBuilder()
        .with_row_count(1000)
        ._build_many_validators(),
    }


def _build_many_validators(builder: MockResultBuilder) -> MockValidationResult:
    """Helper to build result with many validators."""
    for i in range(100):
        if i % 5 == 0:
            builder.add_failed(f"validator_{i:03d}", column=f"col_{i % 10}")
        else:
            builder.add_passed(f"validator_{i:03d}", column=f"col_{i % 10}")
    return builder.build()


# Monkey-patch the method
MockResultBuilder._build_many_validators = lambda self: _build_many_validators(self)


def create_stress_test_data(
    num_validators: int = 1000,
    num_rows: int = 10000000,
) -> MockValidationResult:
    """Create stress test data for performance testing.

    Args:
        num_validators: Number of validator results to generate.
        num_rows: Number of rows in mock dataset.

    Returns:
        MockValidationResult for stress testing.

    Example:
        >>> import time
        >>> data = create_stress_test_data(num_validators=10000)
        >>> start = time.time()
        >>> output = reporter.render(data)
        >>> elapsed = time.time() - start
        >>> print(f"Render time: {elapsed:.2f}s")
    """
    import random

    builder = MockResultBuilder().with_row_count(num_rows).with_data_asset("stress_test.csv")

    validators = ["not_null", "unique", "range", "format", "regex", "length", "custom"]
    columns = [f"col_{i}" for i in range(50)]

    for i in range(num_validators):
        if random.random() < 0.9:  # 90% pass rate
            builder.add_passed(
                random.choice(validators),
                column=random.choice(columns),
            )
        else:
            builder.add_failed(
                random.choice(validators),
                column=random.choice(columns),
                failed_count=random.randint(1, num_rows // 1000),
            )

    return builder.build()


@dataclass
class CapturedOutput:
    """Container for captured reporter output."""

    content: Any
    duration_ms: float
    memory_bytes: int
    exception: Optional[Exception] = None


def capture_output(
    reporter: Any,
    result: MockValidationResult,
    method: str = "render",
) -> CapturedOutput:
    """Capture reporter output with timing and memory information.

    Args:
        reporter: The reporter instance.
        result: Mock result to render.
        method: Method name to call on reporter.

    Returns:
        CapturedOutput with content, timing, and memory info.

    Example:
        >>> captured = capture_output(my_reporter, result)
        >>> print(f"Output length: {len(captured.content)}")
        >>> print(f"Time: {captured.duration_ms:.2f}ms")
    """
    import time
    import tracemalloc

    tracemalloc.start()
    start_time = time.perf_counter()
    exception = None
    content = None

    try:
        render_func = getattr(reporter, method)
        content = render_func(result)
    except Exception as e:
        exception = e

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CapturedOutput(
        content=content,
        duration_ms=(end_time - start_time) * 1000,
        memory_bytes=peak,
        exception=exception,
    )


@dataclass
class BenchmarkResult:
    """Result of a reporter benchmark."""

    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    peak_memory_bytes: int
    output_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "p50_time_ms": self.p50_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "output_size_bytes": self.output_size_bytes,
        }

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(\n"
            f"  iterations={self.iterations},\n"
            f"  avg_time={self.avg_time_ms:.2f}ms,\n"
            f"  p95_time={self.p95_time_ms:.2f}ms,\n"
            f"  peak_memory={self.peak_memory_bytes / 1024:.2f}KB,\n"
            f"  output_size={self.output_size_bytes / 1024:.2f}KB\n"
            f")"
        )


def benchmark_reporter(
    reporter: Any,
    result: Optional[MockValidationResult] = None,
    iterations: int = 100,
    warmup: int = 5,
    method: str = "render",
) -> BenchmarkResult:
    """Benchmark reporter performance.

    Args:
        reporter: The reporter instance to benchmark.
        result: Mock result to render. If None, creates a default one.
        iterations: Number of iterations to run.
        warmup: Number of warmup iterations.
        method: Method name to call on reporter.

    Returns:
        BenchmarkResult with performance statistics.

    Example:
        >>> result = create_mock_result(passed=100, failed=10)
        >>> benchmark = benchmark_reporter(my_reporter, result, iterations=100)
        >>> print(f"Average: {benchmark.avg_time_ms:.2f}ms")
        >>> print(f"P95: {benchmark.p95_time_ms:.2f}ms")
    """
    import statistics
    import time
    import tracemalloc

    if result is None:
        result = create_mock_result(passed=10, failed=2)

    render_func = getattr(reporter, method)

    # Warmup
    for _ in range(warmup):
        render_func(result)

    # Benchmark
    times_ms = []
    peak_memory = 0
    output_size = 0

    tracemalloc.start()

    for _ in range(iterations):
        start = time.perf_counter()
        output = render_func(result)
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

        current, peak = tracemalloc.get_traced_memory()
        peak_memory = max(peak_memory, peak)

        if output_size == 0:
            if isinstance(output, str):
                output_size = len(output.encode("utf-8"))
            elif isinstance(output, bytes):
                output_size = len(output)
            elif isinstance(output, dict):
                output_size = len(json.dumps(output).encode("utf-8"))

    tracemalloc.stop()

    times_ms.sort()

    return BenchmarkResult(
        iterations=iterations,
        total_time_ms=sum(times_ms),
        avg_time_ms=statistics.mean(times_ms),
        min_time_ms=min(times_ms),
        max_time_ms=max(times_ms),
        p50_time_ms=times_ms[len(times_ms) // 2],
        p95_time_ms=times_ms[int(len(times_ms) * 0.95)],
        p99_time_ms=times_ms[int(len(times_ms) * 0.99)],
        peak_memory_bytes=peak_memory,
        output_size_bytes=output_size,
    )
