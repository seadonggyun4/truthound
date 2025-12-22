"""Tests for validators."""

import polars as pl
import pytest

from truthound.validators import (
    DuplicateValidator,
    FormatValidator,
    NullValidator,
    OutlierValidator,
    RangeValidator,
    TypeValidator,
    UniqueValidator,
)


class TestNullValidator:
    """Tests for NullValidator."""

    def test_detects_nulls(self):
        """Test detection of null values."""
        df = pl.DataFrame({"col": [1, None, 3, None, 5]})
        validator = NullValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "col"
        assert issues[0].issue_type == "null"
        assert issues[0].count == 2

    def test_no_nulls(self):
        """Test with no null values."""
        df = pl.DataFrame({"col": [1, 2, 3, 4, 5]})
        validator = NullValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0

    def test_severity_based_on_percentage(self):
        """Test severity assignment based on null percentage."""
        # Low null rate
        df_low = pl.DataFrame({"col": [1, 2, 3, 4, None]})  # 20%
        issues_low = NullValidator().validate(df_low.lazy())
        assert issues_low[0].severity.value == "medium"

        # High null rate
        df_high = pl.DataFrame({"col": [None, None, None, 4, 5]})  # 60%
        issues_high = NullValidator().validate(df_high.lazy())
        assert issues_high[0].severity.value == "critical"


class TestDuplicateValidator:
    """Tests for DuplicateValidator."""

    def test_detects_duplicates(self):
        """Test detection of duplicate rows."""
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        validator = DuplicateValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "duplicate_row"
        # is_duplicated() marks ALL rows that have duplicates (both copies)
        assert issues[0].count == 2

    def test_no_duplicates(self):
        """Test with no duplicate rows."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        validator = DuplicateValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0


class TestRangeValidator:
    """Tests for RangeValidator."""

    def test_detects_negative_age(self):
        """Test detection of negative age values."""
        df = pl.DataFrame({"age": [25, -5, 30, 150, 45]})
        validator = RangeValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "age"
        assert issues[0].issue_type == "out_of_range"

    def test_detects_out_of_range_percentage(self):
        """Test detection of percentage values out of 0-100 range."""
        df = pl.DataFrame({"score": [50, 75, 110, -10, 100]})
        validator = RangeValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # 110 and -10


class TestOutlierValidator:
    """Tests for OutlierValidator."""

    def test_detects_outliers(self):
        """Test detection of statistical outliers."""
        # Values: 10, 11, 12, 13, 14, 15, 100 (outlier)
        df = pl.DataFrame({"value": [10, 11, 12, 13, 14, 15, 100]})
        validator = OutlierValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "outlier"

    def test_no_outliers(self):
        """Test with normally distributed data."""
        df = pl.DataFrame({"value": [10, 11, 12, 13, 14, 15, 16]})
        validator = OutlierValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0


class TestFormatValidator:
    """Tests for FormatValidator."""

    def test_detects_invalid_email(self):
        """Test detection of invalid email format."""
        df = pl.DataFrame(
            {"email": ["alice@example.com", "invalid-email", "bob@test.org", "not-an-email"]}
        )
        validator = FormatValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "email"
        assert issues[0].issue_type == "invalid_format"
        assert issues[0].count == 2

    def test_valid_emails(self):
        """Test with all valid emails."""
        df = pl.DataFrame({"email": ["alice@example.com", "bob@test.org", "charlie@demo.net"]})
        validator = FormatValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0


class TestUniqueValidator:
    """Tests for UniqueValidator."""

    def test_detects_duplicate_ids(self):
        """Test detection of duplicate values in ID column."""
        df = pl.DataFrame({"user_id": [1, 2, 3, 1, 4]})
        validator = UniqueValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "user_id"
        assert issues[0].issue_type == "unique_violation"

    def test_unique_ids(self):
        """Test with all unique IDs."""
        df = pl.DataFrame({"user_id": [1, 2, 3, 4, 5]})
        validator = UniqueValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0


class TestTypeValidator:
    """Tests for TypeValidator."""

    def test_detects_mixed_types(self):
        """Test detection of mixed types in string column."""
        df = pl.DataFrame({"data": ["apple", "123", "banana", "456", "cherry"]})
        validator = TypeValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "mixed_type"

    def test_consistent_types(self):
        """Test with consistent string values."""
        df = pl.DataFrame({"data": ["apple", "banana", "cherry"]})
        validator = TypeValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0
