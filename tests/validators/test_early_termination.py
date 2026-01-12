"""Tests for sampling-based early termination in validators."""

import pytest
import polars as pl

from truthound.validators.base import (
    SampledEarlyTerminationMixin,
    EarlyTerminationResult,
    Validator,
    ValidationIssue,
    StringValidatorMixin,
)
from truthound.validators.string.regex import RegexValidator
from truthound.validators.string.charset import AlphanumericValidator
from truthound.validators.string.like_pattern import LikePatternValidator, NotLikePatternValidator
from truthound.validators.string.format import EmailValidator, PhoneValidator, UuidValidator
from truthound.validators.string.json import JsonParseableValidator


class TestEarlyTerminationResult:
    """Tests for EarlyTerminationResult dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        result = EarlyTerminationResult(
            should_terminate=True,
            sample_fail_rate=0.3,
            estimated_fail_count=300_000,
            sample_size=10_000,
            total_rows=1_000_000,
            confidence_threshold=0.99,
        )

        d = result.to_dict()
        assert d["should_terminate"] is True
        assert d["sample_fail_rate"] == 0.3
        assert d["estimated_fail_count"] == 300_000
        assert d["sample_size"] == 10_000
        assert d["total_rows"] == 1_000_000
        assert d["confidence_threshold"] == 0.99


class TestSampledEarlyTerminationMixin:
    """Tests for the SampledEarlyTerminationMixin."""

    def test_small_dataset_no_sampling(self):
        """Should not sample for datasets below threshold."""
        # Create validator with mixin
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
        )
        # Override min_rows for testing
        validator.early_termination_min_rows = 1000

        # Small dataset (below threshold)
        lf = pl.LazyFrame({
            "col": ["ABC", "invalid", "DEF"] * 100  # 300 rows
        })

        results = validator._check_early_termination(
            lf,
            columns=["col"],
            build_invalid_expr=validator._build_match_expr,
        )

        assert results["col"].should_terminate is False
        assert results["col"].total_rows == 300

    def test_high_failure_rate_triggers_early_termination(self):
        """Should trigger early termination when sample has high failure rate."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
        )
        # Lower thresholds for testing
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50
        validator.early_termination_threshold = 0.99

        # Dataset with 50% failure rate (way above 1% threshold)
        lf = pl.LazyFrame({
            "col": ["ABC", "invalid"] * 500  # 1000 rows, 50% invalid
        })

        results = validator._check_early_termination(
            lf,
            columns=["col"],
            build_invalid_expr=validator._build_match_expr,
        )

        assert results["col"].should_terminate is True
        assert results["col"].sample_fail_rate > 0.01  # Above threshold

    def test_low_failure_rate_no_early_termination(self):
        """Should not trigger early termination when sample has low failure rate."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50
        validator.early_termination_threshold = 0.99

        # Dataset with very low failure rate (0.1%)
        valid_values = ["ABC"] * 999
        invalid_values = ["invalid"]
        lf = pl.LazyFrame({
            "col": valid_values + invalid_values
        })

        results = validator._check_early_termination(
            lf,
            columns=["col"],
            build_invalid_expr=validator._build_match_expr,
        )

        # With only 0.1% failure rate, shouldn't trigger early termination
        assert results["col"].sample_fail_rate < 0.05

    def test_extrapolation_calculation(self):
        """Should correctly extrapolate failure count."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 100
        validator.early_termination_threshold = 0.99

        # Dataset with 30% failure rate
        lf = pl.LazyFrame({
            "col": ["ABC", "ABC", "invalid"] * 500  # 1500 rows, 33% invalid
        })

        results = validator._check_early_termination(
            lf,
            columns=["col"],
            build_invalid_expr=validator._build_match_expr,
        )

        # Should extrapolate based on sample rate
        total_rows = results["col"].total_rows
        estimated = results["col"].estimated_fail_count
        sample_rate = results["col"].sample_fail_rate

        # Estimated should be approximately sample_rate * total_rows
        expected_estimate = int(sample_rate * total_rows)
        assert estimated == expected_estimate


class TestRegexValidatorEarlyTermination:
    """Tests for early termination in RegexValidator."""

    def test_early_termination_produces_valid_issues(self):
        """Early termination should produce valid ValidationIssue objects."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High failure rate dataset
        lf = pl.LazyFrame({
            "col": ["ABC", "invalid", "bad", "wrong"] * 250  # 1000 rows, 75% invalid
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "col"
        assert issues[0].issue_type == "regex_mismatch"
        assert "early-termination" in issues[0].details
        assert issues[0].count > 0

    def test_full_validation_fallback(self):
        """Should fall back to full validation for low failure rates."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # Low failure rate dataset
        lf = pl.LazyFrame({
            "col": ["ABC"] * 990 + ["invalid"] * 10  # 1% failure rate
        })

        issues = validator.validate(lf)

        # Should still detect the issue through full validation
        assert len(issues) == 1
        # Full validation doesn't have "early-termination" in details
        assert "early-termination" not in issues[0].details


class TestAlphanumericValidatorEarlyTermination:
    """Tests for early termination in AlphanumericValidator."""

    def test_early_termination_with_special_chars(self):
        """Should detect non-alphanumeric chars with early termination."""
        validator = AlphanumericValidator(columns=["col"])
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High failure rate - special characters
        lf = pl.LazyFrame({
            "col": ["abc123", "invalid@#$", "bad!chars"] * 400
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "non_alphanumeric"


class TestLikePatternValidatorEarlyTermination:
    """Tests for early termination in LikePatternValidator."""

    def test_early_termination_like_pattern(self):
        """Should use early termination for LIKE pattern validation."""
        validator = LikePatternValidator(
            pattern="PRD-%",
            column="code",
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High failure rate
        lf = pl.LazyFrame({
            "code": ["PRD-001", "INVALID-001", "WRONG-002"] * 400
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "like_pattern_mismatch"


class TestNotLikePatternValidatorEarlyTermination:
    """Tests for early termination in NotLikePatternValidator."""

    def test_early_termination_not_like_pattern(self):
        """Should use early termination for NOT LIKE pattern validation."""
        validator = NotLikePatternValidator(
            pattern="%test%",
            column="name",
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High "failure" rate (values matching the forbidden pattern)
        lf = pl.LazyFrame({
            "name": ["test_value", "some_test", "testing123", "valid"] * 300
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "like_pattern_unexpected_match"


class TestFormatValidatorsEarlyTermination:
    """Tests for early termination in format validators."""

    def test_email_validator_early_termination(self):
        """EmailValidator should support early termination."""
        validator = EmailValidator(columns=["email"])
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High failure rate - invalid emails
        lf = pl.LazyFrame({
            "email": ["user@example.com", "invalid", "not-an-email"] * 400
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_email"

    def test_uuid_validator_early_termination(self):
        """UuidValidator should support early termination."""
        validator = UuidValidator(columns=["id"])
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High failure rate - invalid UUIDs
        lf = pl.LazyFrame({
            "id": [
                "550e8400-e29b-41d4-a716-446655440000",
                "invalid-uuid",
                "not-a-uuid",
            ] * 400
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_uuid"


class TestJsonParseableValidatorEarlyTermination:
    """Tests for early termination in JsonParseableValidator."""

    def test_json_validator_early_termination(self):
        """JsonParseableValidator should support early termination."""
        validator = JsonParseableValidator(columns=["data"])
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # High failure rate - invalid JSON
        lf = pl.LazyFrame({
            "data": ['{"valid": true}', "not json", "{invalid}"] * 400
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_json"


class TestEarlyTerminationPerformance:
    """Tests for performance characteristics of early termination."""

    def test_early_termination_samples_only_head(self):
        """Early termination should only process sample_size rows initially."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 100

        # Large dataset where early termination should kick in
        # First 100 rows have 50% failure rate
        data = (["ABC", "invalid"] * 50) + (["VALID"] * 900)  # 1000 rows total

        lf = pl.LazyFrame({"col": data})

        results = validator._check_early_termination(
            lf,
            columns=["col"],
            build_invalid_expr=validator._build_match_expr,
        )

        # Sample should see ~50% failure rate (from first 100 rows)
        # This should trigger early termination
        assert results["col"].sample_size == 100
        assert results["col"].sample_fail_rate > 0.3  # High enough to trigger


class TestEarlyTerminationEdgeCases:
    """Tests for edge cases in early termination."""

    def test_all_null_column(self):
        """Should handle all-null columns gracefully."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        lf = pl.LazyFrame({
            "col": [None] * 200
        })

        issues = validator.validate(lf)
        assert len(issues) == 0

    def test_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
        )
        validator.early_termination_min_rows = 100

        lf = pl.LazyFrame({"col": []})

        issues = validator.validate(lf)
        assert len(issues) == 0

    def test_multiple_columns_mixed_termination(self):
        """Should handle multiple columns with different termination decisions."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # col1: high failure rate (should early terminate)
        # col2: low failure rate (should do full validation)
        col1_data = ["ABC", "invalid"] * 500  # 50% invalid
        col2_data = ["ABC"] * 990 + ["invalid"] * 10  # 1% invalid

        lf = pl.LazyFrame({
            "col1": col1_data,
            "col2": col2_data,
        })

        issues = validator.validate(lf)

        # Both columns should have issues
        col_names = [issue.column for issue in issues]
        assert "col1" in col_names
        assert "col2" in col_names

    def test_mostly_threshold_with_early_termination(self):
        """Early termination should respect mostly threshold."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
            mostly=0.5,  # Allow up to 50% failures
        )
        validator.early_termination_min_rows = 100
        validator.early_termination_sample_size = 50

        # 40% failure rate (below mostly threshold)
        lf = pl.LazyFrame({
            "col": ["ABC", "ABC", "ABC", "invalid", "bad"] * 200
        })

        issues = validator.validate(lf)

        # Should not report issues because failure rate < (1 - mostly)
        # However, early termination triggers, so it checks mostly against extrapolated count
        # 40% > 1% threshold, so early termination happens
        # But mostly=0.5 means we accept up to 50% failures
        # So the issue should be suppressed
        assert len(issues) == 0

    def test_custom_sample_size_and_threshold(self):
        """Should support custom sample size and threshold."""
        validator = RegexValidator(
            pattern=r"^[A-Z]+$",
            columns=["col"],
        )
        # Custom configuration
        validator.early_termination_sample_size = 200
        validator.early_termination_threshold = 0.95  # 5% failure threshold
        validator.early_termination_min_rows = 500

        # Distribute failures evenly so sample sees them
        # 6% failure rate distributed throughout the dataset
        data = []
        for i in range(1000):
            if i % 16 == 0:  # ~6.25% failure rate
                data.append("invalid")
            else:
                data.append("ABC")

        lf = pl.LazyFrame({"col": data})

        results = validator._check_early_termination(
            lf,
            columns=["col"],
            build_invalid_expr=validator._build_match_expr,
        )

        # 6% > 5% threshold, so should trigger early termination
        assert results["col"].should_terminate is True
        assert results["col"].confidence_threshold == 0.95
