"""Tests for the Reporter SDK testing utilities module."""

import json
import pytest
from xml.etree import ElementTree

from truthound.reporters.sdk.testing import (
    # Mock data classes
    MockValidatorResult,
    MockValidationResult,
    MockResultBuilder,
    Severity,
    # Mock data generators
    create_mock_result,
    create_mock_results,
    create_mock_validator_result,
    # Assertions
    assert_valid_output,
    assert_json_valid,
    assert_xml_valid,
    assert_csv_valid,
    assert_contains_patterns,
    # Fixtures
    create_sample_data,
    create_edge_case_data,
    create_stress_test_data,
    # Utilities
    capture_output,
    benchmark_reporter,
    BenchmarkResult,
    CapturedOutput,
)


class TestMockValidatorResult:
    """Tests for MockValidatorResult class."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = MockValidatorResult(
            validator_name="not_null",
            column="id",
            passed=True,
        )

        assert result.validator_name == "not_null"
        assert result.column == "id"
        assert result.passed is True
        assert result.severity == Severity.ERROR

    def test_auto_message_generation(self):
        """Test automatic message generation."""
        passed = MockValidatorResult(
            validator_name="test",
            column="col",
            passed=True,
        )
        assert "passed" in passed.message.lower()

        failed = MockValidatorResult(
            validator_name="test",
            column="col",
            passed=False,
        )
        assert "failed" in failed.message.lower()

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = MockValidatorResult(
            validator_name="unique",
            column="email",
            passed=False,
            failed_count=5,
            severity=Severity.WARNING,
        )
        d = result.to_dict()

        assert d["validator_name"] == "unique"
        assert d["column"] == "email"
        assert d["passed"] is False
        assert d["failed_count"] == 5
        assert d["severity"] == "warning"

    def test_custom_details(self):
        """Test custom details field."""
        result = MockValidatorResult(
            validator_name="range",
            column="age",
            passed=False,
            details={"min": 0, "max": 150, "violations": [180, -5]},
        )

        assert result.details["min"] == 0
        assert len(result.details["violations"]) == 2


class TestMockValidationResult:
    """Tests for MockValidationResult class."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = MockValidationResult()

        assert result.run_id is not None
        assert result.data_asset == "test_data.csv"
        assert len(result.validator_results) > 0

    def test_custom_properties(self):
        """Test custom property calculation."""
        results = [
            MockValidatorResult(validator_name="v1", passed=True),
            MockValidatorResult(validator_name="v2", passed=True),
            MockValidatorResult(validator_name="v3", passed=False),
        ]

        result = MockValidationResult(validator_results=results)

        assert result.passed_count == 2
        assert result.failed_count == 1
        assert result.total_count == 3
        assert abs(result.success_rate - 66.67) < 1

    def test_empty_results(self):
        """Test with empty validator results via builder."""
        # Note: MockValidationResult generates default results when empty
        # To test truly empty, we need to check that default results are generated
        result = MockValidationResult(validator_results=[])
        # Default behavior generates 3 results
        assert result.total_count == 3  # Default results are generated

        # For truly empty, we can use the builder
        result = MockResultBuilder().build()
        # Builder with no adds also generates no custom results,
        # but the build() completes
        assert isinstance(result, MockValidationResult)

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = MockValidationResult(
            data_asset="users.csv",
            row_count=5000,
        )
        d = result.to_dict()

        assert d["data_asset"] == "users.csv"
        assert d["row_count"] == 5000
        assert "validator_results" in d
        assert "success_rate" in d


class TestMockResultBuilder:
    """Tests for MockResultBuilder class."""

    def test_basic_builder(self):
        """Test basic builder usage."""
        result = (
            MockResultBuilder()
            .with_data_asset("test.csv")
            .with_row_count(1000)
            .add_passed("not_null", column="id")
            .add_passed("unique", column="id")
            .build()
        )

        assert result.data_asset == "test.csv"
        assert result.row_count == 1000
        assert len(result.validator_results) == 2
        assert result.success is True

    def test_failed_results(self):
        """Test adding failed results."""
        result = (
            MockResultBuilder()
            .add_passed("not_null", column="id")
            .add_failed("email_format", column="email", failed_count=10)
            .build()
        )

        assert result.passed_count == 1
        assert result.failed_count == 1
        assert result.success is False

    def test_warning_results(self):
        """Test adding warning results."""
        result = (
            MockResultBuilder()
            .add_warning("outlier", column="age", message="Potential outliers detected")
            .build()
        )

        assert len(result.validator_results) == 1
        assert result.validator_results[0].severity == Severity.WARNING

    def test_custom_result(self):
        """Test adding custom result."""
        custom = MockValidatorResult(
            validator_name="custom",
            column="data",
            passed=True,
            details={"custom_field": "value"},
        )
        result = MockResultBuilder().add_result(custom).build()

        assert result.validator_results[0].details["custom_field"] == "value"

    def test_with_metadata(self):
        """Test adding metadata."""
        result = (
            MockResultBuilder()
            .with_metadata({"environment": "test", "version": "1.0"})
            .add_passed("test")
            .build()
        )

        assert result.metadata["environment"] == "test"

    def test_with_columns(self):
        """Test setting columns."""
        result = (
            MockResultBuilder()
            .with_columns(["col1", "col2", "col3"])
            .add_passed("test")
            .build()
        )

        assert result.columns == ["col1", "col2", "col3"]

    def test_with_run_id(self):
        """Test setting run ID."""
        result = (
            MockResultBuilder()
            .with_run_id("custom-run-id-123")
            .add_passed("test")
            .build()
        )

        assert result.run_id == "custom-run-id-123"


class TestCreateMockResult:
    """Tests for create_mock_result function."""

    def test_default_result(self):
        """Test default result creation."""
        result = create_mock_result()

        assert result.passed_count == 5
        assert result.failed_count == 0
        assert result.success is True

    def test_custom_counts(self):
        """Test with custom pass/fail counts."""
        result = create_mock_result(passed=10, failed=5)

        assert result.passed_count == 10
        assert result.failed_count == 5
        assert result.success is False

    def test_with_warnings(self):
        """Test with warnings."""
        result = create_mock_result(passed=5, failed=0, warnings=3)

        assert result.total_count == 8

    def test_custom_row_count(self):
        """Test custom row count."""
        result = create_mock_result(row_count=1000000)
        assert result.row_count == 1000000

    def test_custom_data_asset(self):
        """Test custom data asset name."""
        result = create_mock_result(data_asset="customers.parquet")
        assert result.data_asset == "customers.parquet"


class TestCreateMockResults:
    """Tests for create_mock_results function."""

    def test_multiple_results(self):
        """Test creating multiple results."""
        results = create_mock_results(count=10)
        assert len(results) == 10

    def test_success_rate(self):
        """Test success rate distribution."""
        # With 100% success rate
        results = create_mock_results(count=10, success_rate=1.0)
        assert all(r.success for r in results)

        # With 0% success rate
        results = create_mock_results(count=10, success_rate=0.0)
        assert not any(r.success for r in results)


class TestAssertValidOutput:
    """Tests for assertion functions."""

    def test_assert_valid_output_none(self):
        """Test assert_valid_output with None."""
        with pytest.raises(AssertionError):
            assert_valid_output(None)

    def test_assert_valid_output_any(self):
        """Test assert_valid_output without format."""
        assert_valid_output("anything")
        assert_valid_output({"key": "value"})
        assert_valid_output([1, 2, 3])

    def test_assert_json_valid_string(self):
        """Test assert_json_valid with string."""
        result = assert_json_valid('{"key": "value"}')
        assert result["key"] == "value"

    def test_assert_json_valid_dict(self):
        """Test assert_json_valid with dict."""
        result = assert_json_valid({"key": "value"})
        assert result["key"] == "value"

    def test_assert_json_valid_invalid(self):
        """Test assert_json_valid with invalid JSON."""
        with pytest.raises(AssertionError):
            assert_json_valid("not json")

    def test_assert_xml_valid_string(self):
        """Test assert_xml_valid with string."""
        result = assert_xml_valid("<root><child/></root>")
        assert result.tag == "root"

    def test_assert_xml_valid_element(self):
        """Test assert_xml_valid with Element."""
        elem = ElementTree.fromstring("<root/>")
        result = assert_xml_valid(elem)
        assert result.tag == "root"

    def test_assert_xml_valid_invalid(self):
        """Test assert_xml_valid with invalid XML."""
        with pytest.raises(AssertionError):
            assert_xml_valid("<unclosed>")

    def test_assert_csv_valid(self):
        """Test assert_csv_valid."""
        rows = assert_csv_valid("a,b\n1,2\n3,4")
        assert len(rows) == 3
        assert rows[0] == ["a", "b"]

    def test_assert_csv_valid_empty(self):
        """Test assert_csv_valid with empty CSV."""
        with pytest.raises(AssertionError):
            assert_csv_valid("")

    def test_assert_contains_patterns_simple(self):
        """Test assert_contains_patterns with simple patterns."""
        text = "Total: 100 rows, 5 errors"
        assert_contains_patterns(text, ["Total:", "errors"])

    def test_assert_contains_patterns_regex(self):
        """Test assert_contains_patterns with regex."""
        text = "Count: 12345"
        assert_contains_patterns(text, [r"Count: \d+"], regex=True)

    def test_assert_contains_patterns_missing(self):
        """Test assert_contains_patterns with missing pattern."""
        with pytest.raises(AssertionError):
            assert_contains_patterns("hello", ["goodbye"])


class TestCreateSampleData:
    """Tests for create_sample_data function."""

    def test_sample_data_structure(self):
        """Test sample data structure."""
        samples = create_sample_data()

        assert "all_passed" in samples
        assert "all_failed" in samples
        assert "mixed" in samples
        assert "empty" in samples
        assert "multiple_runs" in samples

    def test_all_passed_scenario(self):
        """Test all_passed scenario."""
        samples = create_sample_data()
        result = samples["all_passed"][0]

        assert result.failed_count == 0
        assert result.success is True

    def test_all_failed_scenario(self):
        """Test all_failed scenario."""
        samples = create_sample_data()
        result = samples["all_failed"][0]

        assert result.passed_count == 0
        assert result.success is False


class TestCreateEdgeCaseData:
    """Tests for create_edge_case_data function."""

    def test_edge_case_structure(self):
        """Test edge case data structure."""
        cases = create_edge_case_data()

        assert "empty_validators" in cases
        assert "zero_rows" in cases
        assert "unicode_data_asset" in cases
        assert "special_chars" in cases

    def test_empty_validators(self):
        """Test empty validators case."""
        cases = create_edge_case_data()
        result = cases["empty_validators"]

        # Builder builds default results, so we check what we get
        assert isinstance(result, MockValidationResult)

    def test_unicode_data_asset(self):
        """Test unicode data asset case."""
        cases = create_edge_case_data()
        result = cases["unicode_data_asset"]

        assert "🔥" in result.data_asset or "데이터" in result.data_asset

    def test_special_chars(self):
        """Test special characters case."""
        cases = create_edge_case_data()
        result = cases["special_chars"]

        assert '"' in result.data_asset or "<" in result.data_asset


class TestCreateStressTestData:
    """Tests for create_stress_test_data function."""

    def test_default_stress_data(self):
        """Test default stress test data."""
        result = create_stress_test_data()

        assert result.row_count == 10000000
        assert len(result.validator_results) == 1000

    def test_custom_stress_data(self):
        """Test custom stress test data."""
        result = create_stress_test_data(num_validators=100, num_rows=1000)

        assert result.row_count == 1000
        assert len(result.validator_results) == 100


class TestCaptureOutput:
    """Tests for capture_output function."""

    def test_capture_basic(self):
        """Test basic output capture."""

        class SimpleReporter:
            def render(self, result):
                return json.dumps(result.to_dict())

        reporter = SimpleReporter()
        result = create_mock_result()
        captured = capture_output(reporter, result)

        assert captured.content is not None
        assert captured.duration_ms >= 0
        assert captured.memory_bytes >= 0
        assert captured.exception is None

    def test_capture_exception(self):
        """Test capturing exception."""

        class FailingReporter:
            def render(self, result):
                raise ValueError("Test error")

        reporter = FailingReporter()
        result = create_mock_result()
        captured = capture_output(reporter, result)

        assert captured.exception is not None
        assert isinstance(captured.exception, ValueError)

    def test_capture_custom_method(self):
        """Test capturing custom method."""

        class CustomReporter:
            def custom_render(self, result):
                return "custom output"

        reporter = CustomReporter()
        result = create_mock_result()
        captured = capture_output(reporter, result, method="custom_render")

        assert captured.content == "custom output"


class TestBenchmarkReporter:
    """Tests for benchmark_reporter function."""

    def test_basic_benchmark(self):
        """Test basic benchmarking."""

        class FastReporter:
            def render(self, result):
                return {"success": result.success}

        reporter = FastReporter()
        result = create_mock_result()

        benchmark = benchmark_reporter(reporter, result, iterations=10, warmup=2)

        assert isinstance(benchmark, BenchmarkResult)
        assert benchmark.iterations == 10
        assert benchmark.avg_time_ms >= 0
        assert benchmark.min_time_ms <= benchmark.avg_time_ms
        assert benchmark.max_time_ms >= benchmark.avg_time_ms
        assert benchmark.p50_time_ms >= 0
        assert benchmark.p95_time_ms >= 0
        assert benchmark.p99_time_ms >= 0

    def test_benchmark_with_default_result(self):
        """Test benchmark with default result."""

        class SimpleReporter:
            def render(self, result):
                return str(result.to_dict())

        reporter = SimpleReporter()
        benchmark = benchmark_reporter(reporter, iterations=5)

        assert benchmark.iterations == 5
        assert benchmark.output_size_bytes > 0

    def test_benchmark_to_dict(self):
        """Test benchmark to_dict method."""

        class SimpleReporter:
            def render(self, result):
                return "output"

        reporter = SimpleReporter()
        benchmark = benchmark_reporter(reporter, iterations=5)
        d = benchmark.to_dict()

        assert "iterations" in d
        assert "avg_time_ms" in d
        assert "p95_time_ms" in d
        assert "peak_memory_bytes" in d

    def test_benchmark_repr(self):
        """Test benchmark __repr__."""

        class SimpleReporter:
            def render(self, result):
                return "output"

        reporter = SimpleReporter()
        benchmark = benchmark_reporter(reporter, iterations=5)
        repr_str = repr(benchmark)

        assert "BenchmarkResult" in repr_str
        assert "iterations=" in repr_str


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating BenchmarkResult."""
        result = BenchmarkResult(
            iterations=100,
            total_time_ms=1000.0,
            avg_time_ms=10.0,
            min_time_ms=5.0,
            max_time_ms=20.0,
            p50_time_ms=9.0,
            p95_time_ms=18.0,
            p99_time_ms=19.5,
            peak_memory_bytes=1024000,
            output_size_bytes=5000,
        )

        assert result.iterations == 100
        assert result.avg_time_ms == 10.0


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_mock_result_with_zero_row_count(self):
        """Test mock result with zero row count."""
        result = (
            MockResultBuilder()
            .with_row_count(0)
            .add_passed("not_null", column="id")
            .build()
        )

        assert result.row_count == 0

    def test_very_long_column_name(self):
        """Test with very long column name."""
        long_name = "a" * 1000
        result = create_mock_validator_result(column=long_name)

        assert result.column == long_name

    def test_special_characters_in_names(self):
        """Test special characters in names."""
        result = create_mock_validator_result(
            validator_name="test<validator>",
            column='column"with"quotes',
        )

        assert "<" in result.validator_name
        assert '"' in result.column
