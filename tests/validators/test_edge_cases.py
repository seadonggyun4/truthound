"""Edge case tests for validators (#22).

This module provides comprehensive edge case testing for:
- Empty DataFrames
- Single row/column DataFrames
- All-null columns
- Mixed types
- Unicode and special characters
- Extreme values
- Boundary conditions
- Concurrent access
- Memory limits
- Error recovery
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    ValidatorConfig,
    ValidatorExecutionResult,
    ValidationResult,
    GracefulValidator,
    Validator,
    SafeSampler,
    MemoryTracker,
    SchemaValidator,
    TimeoutHandler,
    ValidationTimeoutError,
    ColumnNotFoundError,
)
from truthound.validators.utils import (
    SeverityCalculator,
    calculate_severity,
    calculate_severity_from_counts,
    check_mostly_threshold,
    MostlyResult,
    format_issue_count,
    format_range,
    safe_divide,
    clamp,
    get_columns_by_dtype,
    filter_existing_columns,
    validate_columns_exist,
    NUMERIC_DTYPES,
    STRING_DTYPES,
    DEFAULT_SEVERITY_THRESHOLDS,
    STRICT_SEVERITY_THRESHOLDS,
)


# =============================================================================
# Empty DataFrame Edge Cases
# =============================================================================


class TestEmptyDataFrame:
    """Test validators with empty DataFrames."""

    def test_empty_dataframe_no_columns(self):
        """Empty DataFrame with no columns should not crash."""
        df = pl.DataFrame()
        lf = df.lazy()

        # Schema should be empty
        schema = lf.collect_schema()
        assert len(schema.names()) == 0

    def test_empty_dataframe_with_columns(self):
        """Empty DataFrame with columns should work correctly."""
        df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.String})
        lf = df.lazy()

        # Should have schema but no rows
        schema = lf.collect_schema()
        assert len(schema.names()) == 2
        assert lf.collect().height == 0

    def test_safe_head_on_empty(self):
        """SafeSampler.safe_head should handle empty DataFrame."""
        df = pl.DataFrame({"a": []}).cast({"a": pl.Int64})
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 100)
        assert len(result) == 0

    def test_safe_filter_sample_on_empty(self):
        """SafeSampler.safe_filter_sample should handle empty result."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        lf = df.lazy()

        # Filter that matches nothing
        result = SafeSampler.safe_filter_sample(
            lf, pl.col("a") > 100, 10, ["a"]
        )
        assert len(result) == 0


# =============================================================================
# Single Row/Column Edge Cases
# =============================================================================


class TestSingleRowColumn:
    """Test with minimal data sizes."""

    def test_single_row(self):
        """Single row DataFrame should work."""
        df = pl.DataFrame({"a": [42], "b": ["test"]})
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 10)
        assert len(result) == 1

    def test_single_column(self):
        """Single column DataFrame should work."""
        df = pl.DataFrame({"only_col": [1, 2, 3]})
        lf = df.lazy()

        cols = get_columns_by_dtype(lf, NUMERIC_DTYPES)
        assert cols == ["only_col"]

    def test_single_value(self):
        """DataFrame with single value should work."""
        df = pl.DataFrame({"x": [None]})
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 1)
        assert len(result) == 1
        assert result["x"][0] is None


# =============================================================================
# Null Value Edge Cases
# =============================================================================


class TestNullValues:
    """Test handling of null values."""

    def test_all_null_column(self):
        """Column with all nulls should be handled."""
        df = pl.DataFrame({"a": [None, None, None]})
        lf = df.lazy()

        # Filter for non-null should return empty
        result = SafeSampler.safe_filter_sample(
            lf, pl.col("a").is_not_null(), 10, ["a"]
        )
        assert len(result) == 0

    def test_mixed_nulls(self):
        """Mixed null and non-null values should work."""
        df = pl.DataFrame({"a": [1, None, 3, None, 5]})
        lf = df.lazy()

        result = SafeSampler.safe_filter_sample(
            lf, pl.col("a").is_not_null(), 10, ["a"]
        )
        assert len(result) == 3

    def test_null_in_string_column(self):
        """Null in string column should be handled."""
        df = pl.DataFrame({"s": ["a", None, "b", None]})
        lf = df.lazy()

        cols = get_columns_by_dtype(lf, STRING_DTYPES)
        assert "s" in cols


# =============================================================================
# Unicode and Special Characters
# =============================================================================


class TestUnicodeSpecialChars:
    """Test handling of unicode and special characters."""

    def test_unicode_column_names(self):
        """Unicode column names should work."""
        df = pl.DataFrame({
            "이름": ["홍길동", "김철수"],
            "年齢": [25, 30],
            "नाम": ["अमित", "राज"],
        })
        lf = df.lazy()

        schema = lf.collect_schema()
        assert "이름" in schema.names()
        assert "年齢" in schema.names()

    def test_unicode_values(self):
        """Unicode values should be handled correctly."""
        df = pl.DataFrame({
            "text": [
                "Hello, 世界! 🌍",
                "Привет мир",
                "مرحبا بالعالم",
                "שלום עולם",
            ]
        })
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 10)
        assert len(result) == 4

    def test_special_characters_in_values(self):
        """Special characters in values should work."""
        df = pl.DataFrame({
            "data": [
                "line1\nline2",
                "tab\there",
                "quote\"here",
                "backslash\\here",
                "\x00null\x00byte",
            ]
        })
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 10)
        assert len(result) == 5

    def test_empty_strings(self):
        """Empty strings should be distinct from nulls."""
        df = pl.DataFrame({
            "s": ["", None, " ", "  "],
        })
        lf = df.lazy()

        # Empty string is not null
        result = lf.filter(pl.col("s").is_not_null()).collect()
        assert len(result) == 3


# =============================================================================
# Extreme Values
# =============================================================================


class TestExtremeValues:
    """Test handling of extreme numeric values."""

    def test_very_large_integers(self):
        """Very large integers should be handled."""
        df = pl.DataFrame({
            "big": [2**62, 2**63 - 1, -(2**62)],
        })
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 10)
        assert len(result) == 3

    def test_very_small_floats(self):
        """Very small floats should be handled."""
        df = pl.DataFrame({
            "small": [1e-300, 1e-308, float("inf"), float("-inf")],
        })
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 10)
        assert len(result) == 4

    def test_nan_values(self):
        """NaN values should be handled."""
        df = pl.DataFrame({
            "f": [1.0, float("nan"), 3.0, float("nan")],
        })
        lf = df.lazy()

        # NaN is not null in Polars
        result = lf.filter(pl.col("f").is_not_null()).collect()
        assert len(result) == 4  # NaN is not null

    def test_infinity_values(self):
        """Infinity values should be handled."""
        df = pl.DataFrame({
            "f": [float("inf"), float("-inf"), 0.0],
        })
        lf = df.lazy()

        result = SafeSampler.safe_head(lf, 10)
        assert len(result) == 3


# =============================================================================
# Severity Calculation Edge Cases
# =============================================================================


class TestSeverityCalculation:
    """Test severity calculation edge cases."""

    def test_zero_ratio(self):
        """Zero ratio should return LOW."""
        assert calculate_severity(0.0) == Severity.LOW

    def test_one_ratio(self):
        """100% ratio should return CRITICAL."""
        assert calculate_severity(1.0) == Severity.CRITICAL

    def test_boundary_values(self):
        """Test exact boundary values."""
        # Default thresholds: (0.5, 0.2, 0.05)
        assert calculate_severity(0.05) == Severity.LOW  # Exactly at boundary
        assert calculate_severity(0.051) == Severity.MEDIUM
        assert calculate_severity(0.2) == Severity.MEDIUM
        assert calculate_severity(0.201) == Severity.HIGH
        assert calculate_severity(0.5) == Severity.HIGH
        assert calculate_severity(0.501) == Severity.CRITICAL

    def test_override_always_wins(self):
        """Override should always return specified severity."""
        assert calculate_severity(0.0, override=Severity.CRITICAL) == Severity.CRITICAL
        assert calculate_severity(1.0, override=Severity.LOW) == Severity.LOW

    def test_zero_total_count(self):
        """Zero total count should return LOW."""
        assert calculate_severity_from_counts(0, 0) == Severity.LOW

    def test_calculator_validation(self):
        """Invalid thresholds should raise error."""
        with pytest.raises(ValueError, match="must be in"):
            SeverityCalculator(thresholds=(1.5, 0.5, 0.1))

        with pytest.raises(ValueError, match="descending order"):
            SeverityCalculator(thresholds=(0.1, 0.5, 0.3))

    def test_negative_counts(self):
        """Negative counts should raise error."""
        calc = SeverityCalculator()
        with pytest.raises(ValueError, match="cannot be negative"):
            calc.from_counts(-1, 100)

        with pytest.raises(ValueError, match="cannot be negative"):
            calc.from_counts(0, -1)

    def test_failure_exceeds_total(self):
        """Failure count exceeding total should raise error."""
        calc = SeverityCalculator()
        with pytest.raises(ValueError, match="cannot exceed"):
            calc.from_counts(101, 100)


# =============================================================================
# Mostly Threshold Edge Cases
# =============================================================================


class TestMostlyThreshold:
    """Test mostly threshold edge cases."""

    def test_none_mostly(self):
        """None mostly should never pass."""
        result = check_mostly_threshold(10, 100, None)
        assert not result.passes

    def test_zero_total(self):
        """Zero total should always pass."""
        result = check_mostly_threshold(0, 0, 0.95)
        assert result.passes
        assert result.pass_ratio == 1.0

    def test_exact_threshold(self):
        """Exact threshold value should pass."""
        # 5 failures out of 100 = 95% pass rate
        result = check_mostly_threshold(5, 100, 0.95)
        assert result.passes
        assert result.pass_ratio == 0.95

    def test_just_below_threshold(self):
        """Just below threshold should fail."""
        # 6 failures out of 100 = 94% pass rate
        result = check_mostly_threshold(6, 100, 0.95)
        assert not result.passes
        assert result.pass_ratio == 0.94

    def test_all_pass(self):
        """Zero failures should pass any threshold."""
        result = check_mostly_threshold(0, 100, 0.99)
        assert result.passes
        assert result.pass_ratio == 1.0

    def test_all_fail(self):
        """All failures should fail any threshold."""
        result = check_mostly_threshold(100, 100, 0.01)
        assert not result.passes
        assert result.pass_ratio == 0.0


# =============================================================================
# Schema Validation Edge Cases
# =============================================================================


class TestSchemaValidation:
    """Test schema validation edge cases."""

    def test_empty_column_list(self):
        """Empty column list should succeed."""
        df = pl.DataFrame({"a": [1], "b": [2]})
        lf = df.lazy()

        all_exist, missing = validate_columns_exist(lf, [])
        assert all_exist
        assert missing == []

    def test_all_columns_missing(self):
        """All columns missing should report all."""
        df = pl.DataFrame({"a": [1]})
        lf = df.lazy()

        all_exist, missing = validate_columns_exist(lf, ["x", "y", "z"])
        assert not all_exist
        assert missing == ["x", "y", "z"]

    def test_duplicate_column_requests(self):
        """Duplicate column requests should be handled."""
        df = pl.DataFrame({"a": [1], "b": [2]})
        lf = df.lazy()

        result = filter_existing_columns(lf, ["a", "a", "a"])
        assert result == ["a", "a", "a"]  # Preserves duplicates


# =============================================================================
# Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Test thread safety."""

    def test_concurrent_severity_calculation(self):
        """Severity calculation should be thread-safe."""
        results = []
        calc = SeverityCalculator()

        def calculate_many():
            for i in range(100):
                ratio = i / 100.0
                sev = calc.from_ratio(ratio)
                results.append((ratio, sev))

        threads = [threading.Thread(target=calculate_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1000

    def test_concurrent_memory_tracker(self):
        """MemoryTracker should be thread-safe."""
        tracker = MemoryTracker(limit_mb=1000)

        def track():
            for _ in range(100):
                tracker.get_current_mb()
                tracker.get_delta_mb()

        threads = [threading.Thread(target=track) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# =============================================================================
# Formatting Edge Cases
# =============================================================================


class TestFormatting:
    """Test formatting utilities."""

    def test_format_issue_count_singular(self):
        """Singular count should use singular noun."""
        assert format_issue_count(1) == "1 value"
        assert format_issue_count(1, noun="row") == "1 row"

    def test_format_issue_count_plural(self):
        """Plural count should use plural noun."""
        assert format_issue_count(5) == "5 values"
        assert format_issue_count(0) == "0 values"

    def test_format_issue_count_with_percentage(self):
        """Percentage should be calculated correctly."""
        result = format_issue_count(25, 100)
        assert "25.0%" in result

    def test_format_issue_count_large_numbers(self):
        """Large numbers should be formatted with commas."""
        result = format_issue_count(1_000_000)
        assert "1,000,000" in result

    def test_format_range_both_bounds(self):
        """Both bounds should show bracket notation."""
        assert format_range(0, 100) == "[0, 100]"
        assert format_range(0, 100, inclusive=False) == "(0, 100)"

    def test_format_range_one_bound(self):
        """Single bound should show comparison."""
        assert format_range(0, None) == ">= 0"
        assert format_range(None, 100) == "<= 100"

    def test_format_range_no_bounds(self):
        """No bounds should show any value."""
        assert format_range(None, None) == "any value"


# =============================================================================
# Helper Function Edge Cases
# =============================================================================


class TestHelperFunctions:
    """Test helper function edge cases."""

    def test_safe_divide_zero_denominator(self):
        """Division by zero should return default."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1

    def test_safe_divide_normal(self):
        """Normal division should work."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(1, 3) == pytest.approx(0.333, rel=1e-2)

    def test_clamp_within_range(self):
        """Value within range should be unchanged."""
        assert clamp(5, 0, 10) == 5

    def test_clamp_below_min(self):
        """Value below min should be clamped."""
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above_max(self):
        """Value above max should be clamped."""
        assert clamp(15, 0, 10) == 10


# =============================================================================
# GracefulValidator Edge Cases
# =============================================================================


class TestGracefulValidator:
    """Test GracefulValidator edge cases."""

    def test_validator_returns_empty_list(self):
        """Validator returning empty list should succeed."""
        class EmptyValidator(Validator):
            name = "empty"
            category = "test"

            def validate(self, lf):
                return []

        v = EmptyValidator()
        graceful = GracefulValidator(v)
        result = graceful.validate(pl.DataFrame({"a": [1]}).lazy())

        assert result.status == ValidationResult.SUCCESS
        assert len(result.issues) == 0

    def test_validator_raises_column_not_found(self):
        """ColumnNotFoundError should result in SKIPPED status."""
        class MissingColumnValidator(Validator):
            name = "missing"
            category = "test"

            def validate(self, lf):
                raise ColumnNotFoundError("missing_col", ["a", "b"])

        v = MissingColumnValidator()
        graceful = GracefulValidator(v)
        result = graceful.validate(pl.DataFrame({"a": [1]}).lazy())

        assert result.status == ValidationResult.SKIPPED
        assert result.error_context is not None

    def test_validator_raises_generic_error(self):
        """Generic error should result in FAILED status."""
        class FailingValidator(Validator):
            name = "failing"
            category = "test"

            def validate(self, lf):
                raise RuntimeError("Something went wrong")

        v = FailingValidator()
        graceful = GracefulValidator(v, skip_on_error=True)
        result = graceful.validate(pl.DataFrame({"a": [1]}).lazy())

        assert result.status == ValidationResult.FAILED
        assert result.error_context is not None
        assert "RuntimeError" in str(result.error_context.to_dict())


# =============================================================================
# ValidatorConfig Edge Cases
# =============================================================================


class TestValidatorConfig:
    """Test ValidatorConfig edge cases."""

    def test_frozen_config(self):
        """Config should be immutable."""
        config = ValidatorConfig(sample_size=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            config.sample_size = 20

    def test_config_replace(self):
        """Replace should create new config."""
        config = ValidatorConfig(sample_size=10)
        new_config = config.replace(sample_size=20)

        assert config.sample_size == 10
        assert new_config.sample_size == 20

    def test_config_validation_mostly_range(self):
        """Mostly must be in [0, 1]."""
        with pytest.raises(ValueError, match="mostly"):
            ValidatorConfig(mostly=1.5)

        with pytest.raises(ValueError, match="mostly"):
            ValidatorConfig(mostly=-0.1)

    def test_config_validation_sample_size(self):
        """Sample size must be non-negative."""
        with pytest.raises(ValueError, match="sample_size"):
            ValidatorConfig(sample_size=-1)


# =============================================================================
# Column Type Detection Edge Cases
# =============================================================================


class TestColumnTypeDetection:
    """Test column type detection edge cases."""

    def test_no_matching_types(self):
        """No columns matching type should return empty list."""
        df = pl.DataFrame({"s": ["a", "b"]})
        lf = df.lazy()

        result = get_columns_by_dtype(lf, NUMERIC_DTYPES)
        assert result == []

    def test_all_matching_types(self):
        """All columns matching should return all."""
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        lf = df.lazy()

        result = get_columns_by_dtype(lf, NUMERIC_DTYPES)
        assert set(result) == {"a", "b", "c"}

    def test_mixed_types(self):
        """Mixed types should filter correctly."""
        df = pl.DataFrame({
            "int_col": [1, 2],
            "str_col": ["a", "b"],
            "float_col": [1.0, 2.0],
        })
        lf = df.lazy()

        numeric = get_columns_by_dtype(lf, NUMERIC_DTYPES)
        assert set(numeric) == {"int_col", "float_col"}

        strings = get_columns_by_dtype(lf, STRING_DTYPES)
        assert strings == ["str_col"]
