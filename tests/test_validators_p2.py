"""Tests for P2 validators: LIKE pattern, dateutil parsing, within-record uniqueness, column pair sets."""

import polars as pl
import pytest

from truthound.types import Severity


# =============================================================================
# LikePatternValidator Tests
# =============================================================================


class TestLikePatternValidator:
    """Tests for SQL LIKE pattern validators."""

    def test_like_pattern_percent_wildcard(self):
        """Test LIKE pattern with % wildcard."""
        from truthound.validators import LikePatternValidator

        df = pl.DataFrame({
            "filename": ["report_2024.pdf", "summary_2024.pdf", "image.jpg", "data.csv"],
        })

        validator = LikePatternValidator(
            column="filename",
            pattern="%.pdf",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # image.jpg, data.csv don't match

    def test_like_pattern_underscore_wildcard(self):
        """Test LIKE pattern with _ wildcard."""
        from truthound.validators import LikePatternValidator

        df = pl.DataFrame({
            "code": ["A1", "B2", "C3", "AB1", "A12"],
        })

        validator = LikePatternValidator(
            column="code",
            pattern="_1",  # Single char followed by "1"
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 4  # B2, C3, AB1, A12 don't match

    def test_like_pattern_mixed_wildcards(self):
        """Test LIKE pattern with both wildcards."""
        from truthound.validators import LikePatternValidator

        df = pl.DataFrame({
            "email": [
                "user@example.com",
                "admin@test.org",
                "support@company.net",
                "invalid",
            ],
        })

        validator = LikePatternValidator(
            column="email",
            pattern="%@%.%",  # Contains @ and . (basic email pattern)
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # "invalid" doesn't match

    def test_like_pattern_starts_with(self):
        """Test LIKE pattern for starts with."""
        from truthound.validators import LikePatternValidator

        df = pl.DataFrame({
            "product_code": ["PRD-001", "PRD-002", "SKU-001", "ITM-003"],
        })

        validator = LikePatternValidator(
            column="product_code",
            pattern="PRD-%",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # SKU-001, ITM-003 don't match

    def test_like_pattern_case_insensitive(self):
        """Test LIKE pattern with case insensitivity."""
        from truthound.validators import LikePatternValidator

        df = pl.DataFrame({
            "status": ["ACTIVE", "Active", "active", "inactive"],
        })

        validator = LikePatternValidator(
            column="status",
            pattern="active%",
            case_sensitive=False,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # "inactive" doesn't match "active%"

    def test_like_pattern_all_match(self):
        """Test LIKE pattern when all values match."""
        from truthound.validators import LikePatternValidator

        df = pl.DataFrame({
            "prefix": ["ABC-1", "ABC-2", "ABC-3"],
        })

        validator = LikePatternValidator(
            column="prefix",
            pattern="ABC-%",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_not_like_pattern(self):
        """Test NOT LIKE pattern validator."""
        from truthound.validators import NotLikePatternValidator

        df = pl.DataFrame({
            "filename": ["data.txt", "report.pdf", "secret.bak", "backup.bak"],
        })

        validator = NotLikePatternValidator(
            column="filename",
            pattern="%.bak",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # secret.bak, backup.bak match the forbidden pattern


# =============================================================================
# DateutilParseableValidator Tests
# =============================================================================


class TestDateutilParseableValidator:
    """Tests for flexible date parsing validator."""

    def test_parseable_various_formats(self):
        """Test parsing various date formats."""
        from truthound.validators import DateutilParseableValidator

        df = pl.DataFrame({
            "date_str": [
                "2024-01-15",
                "01/15/2024",
                "January 15, 2024",
                "15-Jan-2024",
            ],
        })

        validator = DateutilParseableValidator(column="date_str")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_parseable_with_time(self):
        """Test parsing dates with time components."""
        from truthound.validators import DateutilParseableValidator

        df = pl.DataFrame({
            "datetime_str": [
                "2024-01-15 10:30:00",
                "2024-01-15T10:30:00Z",
                "Jan 15 2024 10:30 AM",
            ],
        })

        validator = DateutilParseableValidator(column="datetime_str")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_parseable_invalid_dates(self):
        """Test detecting invalid date strings."""
        from truthound.validators import DateutilParseableValidator

        df = pl.DataFrame({
            "date_str": [
                "2024-01-15",
                "not a date",
                "12345",
                "hello world",
            ],
        })

        validator = DateutilParseableValidator(column="date_str")
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 3  # 3 invalid date strings

    def test_parseable_with_fuzzy(self):
        """Test fuzzy parsing for embedded dates."""
        from truthound.validators import DateutilParseableValidator

        df = pl.DataFrame({
            "text": [
                "Meeting on 2024-01-15",
                "Due date: January 20, 2024",
                "no date here",
            ],
        })

        validator = DateutilParseableValidator(column="text", fuzzy=True)
        issues = validator.validate(df.lazy())

        # With fuzzy, first two should parse, third should fail
        assert len(issues) == 1
        assert issues[0].count == 1

    def test_parseable_null_handling(self):
        """Test null value handling."""
        from truthound.validators import DateutilParseableValidator

        df = pl.DataFrame({
            "date_str": ["2024-01-15", None, "2024-02-20"],
        })

        validator = DateutilParseableValidator(column="date_str")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0  # Nulls are ignored by default


# =============================================================================
# UniqueWithinRecordValidator Tests
# =============================================================================


class TestUniqueWithinRecordValidator:
    """Tests for within-record uniqueness validators."""

    def test_unique_within_record_valid(self):
        """Test when all rows have unique values."""
        from truthound.validators import UniqueWithinRecordValidator

        df = pl.DataFrame({
            "choice_1": ["A", "B", "C"],
            "choice_2": ["B", "C", "A"],
            "choice_3": ["C", "A", "B"],
        })

        validator = UniqueWithinRecordValidator(
            columns=["choice_1", "choice_2", "choice_3"],
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_unique_within_record_duplicates(self):
        """Test when rows have duplicate values."""
        from truthound.validators import UniqueWithinRecordValidator

        df = pl.DataFrame({
            "primary_contact": ["Alice", "Bob", "Charlie"],
            "secondary_contact": ["Bob", "Bob", "David"],  # Row 2: Bob appears twice
        })

        validator = UniqueWithinRecordValidator(
            columns=["primary_contact", "secondary_contact"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # Only row 2 has duplicate

    def test_unique_within_record_ignore_nulls(self):
        """Test that nulls are ignored by default."""
        from truthound.validators import UniqueWithinRecordValidator

        df = pl.DataFrame({
            "col1": ["A", None, "C"],
            "col2": ["A", None, "D"],  # Row 1: both A, Row 2: both null
        })

        validator = UniqueWithinRecordValidator(
            columns=["col1", "col2"],
            ignore_nulls=True,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # Only row 1 has duplicate (A, A)

    def test_unique_within_record_include_nulls(self):
        """Test with null checking enabled."""
        from truthound.validators import UniqueWithinRecordValidator

        df = pl.DataFrame({
            "col1": [None, "B"],
            "col2": [None, "C"],
        })

        validator = UniqueWithinRecordValidator(
            columns=["col1", "col2"],
            ignore_nulls=False,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # Row 1: (None, None) counts as duplicate

    def test_unique_within_record_multiple_columns(self):
        """Test with many columns."""
        from truthound.validators import UniqueWithinRecordValidator

        df = pl.DataFrame({
            "q1": ["A", "B"],
            "q2": ["B", "B"],  # Row 2 has duplicate B
            "q3": ["C", "C"],  # Row 2 has duplicate C (but already flagged)
            "q4": ["D", "D"],
        })

        validator = UniqueWithinRecordValidator(
            columns=["q1", "q2", "q3", "q4"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # Row 2 has duplicates

    def test_all_columns_unique_within_record(self):
        """Test AllColumnsUniqueWithinRecordValidator."""
        from truthound.validators import AllColumnsUniqueWithinRecordValidator

        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": [2, 2, 4],  # Row 2: value 2 appears twice
            "c": [3, 4, 5],
        })

        validator = AllColumnsUniqueWithinRecordValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1


# =============================================================================
# ColumnPairInSetValidator Tests
# =============================================================================


class TestColumnPairInSetValidator:
    """Tests for column pair set validators."""

    def test_column_pair_valid_combinations(self):
        """Test with all valid combinations."""
        from truthound.validators import ColumnPairInSetValidator

        df = pl.DataFrame({
            "country": ["US", "UK", "JP"],
            "currency": ["USD", "GBP", "JPY"],
        })

        validator = ColumnPairInSetValidator(
            column_a="country",
            column_b="currency",
            valid_pairs=[
                ("US", "USD"),
                ("UK", "GBP"),
                ("JP", "JPY"),
                ("KR", "KRW"),
            ],
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_pair_invalid_combinations(self):
        """Test with invalid combinations."""
        from truthound.validators import ColumnPairInSetValidator

        df = pl.DataFrame({
            "country": ["US", "UK", "JP", "US"],
            "currency": ["USD", "EUR", "JPY", "GBP"],  # UK-EUR and US-GBP are invalid
        })

        validator = ColumnPairInSetValidator(
            column_a="country",
            column_b="currency",
            valid_pairs=[
                ("US", "USD"),
                ("UK", "GBP"),
                ("JP", "JPY"),
            ],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # UK-EUR and US-GBP are invalid

    def test_column_pair_with_nulls(self):
        """Test null handling in column pairs."""
        from truthound.validators import ColumnPairInSetValidator

        df = pl.DataFrame({
            "dept": ["eng", None, "sales"],
            "role": ["dev", "manager", None],
        })

        validator = ColumnPairInSetValidator(
            column_a="dept",
            column_b="role",
            valid_pairs=[("eng", "dev"), ("sales", "rep")],
            ignore_nulls=True,
        )
        issues = validator.validate(df.lazy())

        # Row 1 is valid, Rows 2-3 have nulls (ignored)
        assert len(issues) == 0

    def test_column_pair_not_ignore_nulls(self):
        """Test with null checking enabled."""
        from truthound.validators import ColumnPairInSetValidator

        df = pl.DataFrame({
            "dept": ["eng", None],
            "role": ["dev", "manager"],
        })

        validator = ColumnPairInSetValidator(
            column_a="dept",
            column_b="role",
            valid_pairs=[("eng", "dev")],
            ignore_nulls=False,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # (None, "manager") is invalid

    def test_column_pair_not_in_set(self):
        """Test forbidden pairs validator."""
        from truthound.validators import ColumnPairNotInSetValidator

        df = pl.DataFrame({
            "status": ["open", "closed", "resolved", "closed"],
            "priority": ["high", "urgent", "low", "low"],
        })

        validator = ColumnPairNotInSetValidator(
            column_a="status",
            column_b="priority",
            forbidden_pairs=[
                ("closed", "urgent"),
                ("resolved", "urgent"),
            ],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # (closed, urgent) is forbidden

    def test_column_pair_empty_set(self):
        """Test with empty valid pairs set."""
        from truthound.validators import ColumnPairInSetValidator

        df = pl.DataFrame({
            "a": ["x"],
            "b": ["y"],
        })

        validator = ColumnPairInSetValidator(
            column_a="a",
            column_b="b",
            valid_pairs=[],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # All pairs invalid when set is empty


# =============================================================================
# Integration Tests
# =============================================================================


class TestP2Integration:
    """Integration tests for P2 validators."""

    def test_like_pattern_with_mostly(self):
        """Test LIKE pattern with mostly parameter."""
        from truthound.validators import LikePatternValidator
        from truthound.validators.base import ValidatorConfig

        df = pl.DataFrame({
            "code": ["PRD-001", "PRD-002", "SKU-001"] * 100,  # 300 rows
        })

        validator = LikePatternValidator(
            column="code",
            pattern="PRD-%",
            config=ValidatorConfig(mostly=0.6),  # Allow 40% failures
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0  # 66% pass > 60% threshold

    def test_validator_registry_p2(self):
        """Test that P2 validators are registered."""
        from truthound.validators import registry

        # Check LIKE pattern validators
        assert registry.get("like_pattern") is not None
        assert registry.get("not_like_pattern") is not None

        # Check dateutil validator
        assert registry.get("dateutil_parseable") is not None

        # Check within-record validators
        assert registry.get("unique_within_record") is not None
        assert registry.get("all_columns_unique_within_record") is not None

        # Check column pair validators
        assert registry.get("column_pair_in_set") is not None
        assert registry.get("column_pair_not_in_set") is not None

    def test_combined_p2_validators(self):
        """Test multiple P2 validators on same dataset."""
        from truthound.validators import (
            LikePatternValidator,
            UniqueWithinRecordValidator,
            ColumnPairInSetValidator,
        )

        df = pl.DataFrame({
            "id": ["PRD-001", "PRD-002", "SKU-003"],
            "choice_1": ["A", "B", "C"],
            "choice_2": ["B", "B", "D"],  # Row 2 has duplicate B
            "country": ["US", "UK", "JP"],
            "currency": ["USD", "EUR", "JPY"],  # UK-EUR is invalid
        })

        # Validator 1: Check ID pattern
        v1 = LikePatternValidator(column="id", pattern="PRD-%")
        issues1 = v1.validate(df.lazy())
        assert len(issues1) == 1  # SKU-003 doesn't match

        # Validator 2: Check choice uniqueness
        v2 = UniqueWithinRecordValidator(columns=["choice_1", "choice_2"])
        issues2 = v2.validate(df.lazy())
        assert len(issues2) == 1  # Row 2 has duplicate

        # Validator 3: Check country-currency pairs
        v3 = ColumnPairInSetValidator(
            column_a="country",
            column_b="currency",
            valid_pairs=[("US", "USD"), ("UK", "GBP"), ("JP", "JPY")],
        )
        issues3 = v3.validate(df.lazy())
        assert len(issues3) == 1  # UK-EUR is invalid
