"""Tests for NaN validators."""

import pytest
import polars as pl
import math

from truthound.validators.completeness.nan import (
    NaNValidator,
    NotNaNValidator,
    NaNRatioValidator,
    InfinityValidator,
    FiniteValidator,
)
from truthound.types import Severity


class TestNaNValidator:
    """Tests for NaNValidator."""

    def test_detects_nan_values(self):
        """Should detect NaN values in float columns."""
        lf = pl.LazyFrame({
            "values": [1.0, float("nan"), 3.0, float("nan"), 5.0],
        })

        validator = NaNValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "values"
        assert issues[0].issue_type == "nan"
        assert issues[0].count == 2

    def test_ignores_null_values(self):
        """NaN validator should not count NULL values."""
        lf = pl.LazyFrame({
            "values": [1.0, None, 3.0, None, 5.0],
        })

        validator = NaNValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_only_checks_float_columns(self):
        """Should only check float columns, not int or string."""
        lf = pl.LazyFrame({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.0, float("nan"), 3.0],
        })

        validator = NaNValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "float_col"

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        lf = pl.LazyFrame({"values": pl.Series([], dtype=pl.Float64)})

        validator = NaNValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_column_filter(self):
        """Should respect column filter."""
        lf = pl.LazyFrame({
            "col1": [float("nan"), 2.0],
            "col2": [float("nan"), 2.0],
        })

        validator = NaNValidator(columns=["col1"])
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "col1"


class TestNotNaNValidator:
    """Tests for NotNaNValidator."""

    def test_reports_high_severity(self):
        """Should report HIGH severity by default."""
        lf = pl.LazyFrame({
            "values": [1.0, float("nan"), 3.0],
        })

        validator = NotNaNValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH

    def test_no_nan_passes(self):
        """Should pass when no NaN values."""
        lf = pl.LazyFrame({
            "values": [1.0, 2.0, 3.0],
        })

        validator = NotNaNValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestNaNRatioValidator:
    """Tests for NaNRatioValidator."""

    def test_ratio_within_threshold(self):
        """Should pass when NaN ratio is within threshold."""
        lf = pl.LazyFrame({
            "values": [float("nan")] + [1.0] * 99,  # 1% NaN
        })

        validator = NaNRatioValidator(max_ratio=0.05)  # 5% allowed
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_ratio_exceeds_threshold(self):
        """Should fail when NaN ratio exceeds threshold."""
        lf = pl.LazyFrame({
            "values": [float("nan")] * 10 + [1.0] * 90,  # 10% NaN
        })

        validator = NaNRatioValidator(max_ratio=0.05)  # 5% allowed
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert "10.0%" in issues[0].details

    def test_invalid_max_ratio(self):
        """Should raise error for invalid max_ratio."""
        with pytest.raises(ValueError):
            NaNRatioValidator(max_ratio=1.5)

        with pytest.raises(ValueError):
            NaNRatioValidator(max_ratio=-0.1)


class TestInfinityValidator:
    """Tests for InfinityValidator."""

    def test_detects_positive_infinity(self):
        """Should detect positive infinity."""
        lf = pl.LazyFrame({
            "values": [1.0, float("inf"), 3.0],
        })

        validator = InfinityValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_detects_negative_infinity(self):
        """Should detect negative infinity."""
        lf = pl.LazyFrame({
            "values": [1.0, float("-inf"), 3.0],
        })

        validator = InfinityValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1

    def test_detects_both_infinities(self):
        """Should detect both positive and negative infinity."""
        lf = pl.LazyFrame({
            "values": [float("inf"), float("-inf"), 3.0],
        })

        validator = InfinityValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2


class TestFiniteValidator:
    """Tests for FiniteValidator."""

    def test_detects_nan_and_infinity(self):
        """Should detect both NaN and infinity."""
        lf = pl.LazyFrame({
            "values": [float("nan"), float("inf"), float("-inf"), 1.0],
        })

        validator = FiniteValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 3
        assert issues[0].issue_type == "not_finite"

    def test_passes_all_finite(self):
        """Should pass when all values are finite."""
        lf = pl.LazyFrame({
            "values": [1.0, 2.0, 3.0, -100.0, 0.0],
        })

        validator = FiniteValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_ignores_null(self):
        """Should not count NULL as non-finite."""
        lf = pl.LazyFrame({
            "values": [1.0, None, 3.0],
        })

        validator = FiniteValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0
