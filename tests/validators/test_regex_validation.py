"""Tests for regex pattern validation at construction time."""

import pytest
import polars as pl

from truthound.validators.base import RegexValidationError, RegexValidatorMixin
from truthound.validators.string.regex import RegexValidator
from truthound.validators.string.regex_extended import (
    RegexListValidator,
    NotMatchRegexValidator,
    NotMatchRegexListValidator,
)


class TestRegexValidatorMixin:
    """Tests for RegexValidatorMixin."""

    def test_validate_valid_pattern(self):
        """Should compile valid patterns."""
        pattern = RegexValidatorMixin.validate_pattern(r"^[A-Z]{3}-\d{4}$")
        assert pattern is not None
        assert pattern.fullmatch("ABC-1234")

    def test_validate_invalid_pattern(self):
        """Should raise RegexValidationError for invalid patterns."""
        with pytest.raises(RegexValidationError) as exc_info:
            RegexValidatorMixin.validate_pattern(r"[invalid(")

        assert "Invalid regex pattern" in str(exc_info.value)
        assert "[invalid(" in str(exc_info.value)

    def test_validate_none_pattern(self):
        """Should raise RegexValidationError for None pattern."""
        with pytest.raises(RegexValidationError) as exc_info:
            RegexValidatorMixin.validate_pattern(None)

        assert "Pattern cannot be None" in str(exc_info.value)

    def test_validate_patterns_list(self):
        """Should validate all patterns in a list."""
        patterns = [r"^\d+$", r"^[a-z]+$", r"^[A-Z]+$"]
        compiled = RegexValidatorMixin.validate_patterns(patterns)

        assert len(compiled) == 3
        assert compiled[0].fullmatch("123")
        assert compiled[1].fullmatch("abc")
        assert compiled[2].fullmatch("ABC")

    def test_validate_patterns_list_with_invalid(self):
        """Should raise on first invalid pattern in list."""
        patterns = [r"^\d+$", r"[invalid(", r"^[a-z]+$"]

        with pytest.raises(RegexValidationError) as exc_info:
            RegexValidatorMixin.validate_patterns(patterns)

        assert "[invalid(" in str(exc_info.value)


class TestRegexValidatorConstruction:
    """Tests for RegexValidator construction-time validation."""

    def test_valid_pattern_construction(self):
        """Should construct successfully with valid pattern."""
        validator = RegexValidator(pattern=r"^[A-Z]{3}-\d{4}$")
        assert validator.pattern == r"^[A-Z]{3}-\d{4}$"

    def test_missing_pattern_raises_clear_error(self):
        """Should raise RegexValidationError with clear message when pattern is missing.

        This is important for YAML configuration where users might specify:
            validators:
              - name: regex   # Missing pattern!
        """
        with pytest.raises(RegexValidationError) as exc_info:
            RegexValidator()  # No pattern argument

        error_msg = str(exc_info.value)
        # Error should explain what's wrong and how to fix it
        assert "pattern" in error_msg.lower()
        assert "required" in error_msg.lower()
        # Should include usage examples
        assert "RegexValidator" in error_msg or "YAML" in error_msg

    def test_none_pattern_raises_clear_error(self):
        """Should raise RegexValidationError when pattern is explicitly None."""
        with pytest.raises(RegexValidationError) as exc_info:
            RegexValidator(pattern=None)

        error_msg = str(exc_info.value)
        assert "pattern" in error_msg.lower()
        assert "required" in error_msg.lower()

    def test_invalid_pattern_construction(self):
        """Should fail at construction with invalid pattern."""
        with pytest.raises(RegexValidationError):
            RegexValidator(pattern=r"[invalid(")

    def test_empty_pattern_construction(self):
        """Should allow empty pattern (matches empty string)."""
        validator = RegexValidator(pattern=r"")
        assert validator.pattern == ""

    def test_case_insensitive_flag(self):
        """Should support case insensitive matching."""
        validator = RegexValidator(
            pattern=r"^[a-z]+$",
            case_insensitive=True,
        )

        lf = pl.LazyFrame({"values": ["abc", "ABC", "AbC"]})
        issues = validator.validate(lf)

        # All should match with case insensitive
        assert len(issues) == 0


class TestRegexListValidatorConstruction:
    """Tests for RegexListValidator construction-time validation."""

    def test_valid_patterns_construction(self):
        """Should construct with valid patterns."""
        validator = RegexListValidator(
            patterns=[r"^\d{4}-\d{2}-\d{2}$", r"^\d{2}/\d{2}/\d{4}$"]
        )
        assert len(validator._compiled) == 2

    def test_invalid_pattern_in_list(self):
        """Should fail if any pattern is invalid."""
        with pytest.raises(RegexValidationError):
            RegexListValidator(
                patterns=[r"^\d+$", r"[invalid(", r"^[a-z]+$"]
            )


class TestNotMatchRegexValidatorConstruction:
    """Tests for NotMatchRegexValidator construction-time validation."""

    def test_valid_pattern_construction(self):
        """Should construct with valid pattern."""
        validator = NotMatchRegexValidator(pattern=r"\d{3}-\d{2}-\d{4}")
        assert validator.pattern == r"\d{3}-\d{2}-\d{4}"

    def test_invalid_pattern_construction(self):
        """Should fail with invalid pattern."""
        with pytest.raises(RegexValidationError):
            NotMatchRegexValidator(pattern=r"[invalid(")


class TestNotMatchRegexListValidatorConstruction:
    """Tests for NotMatchRegexListValidator construction-time validation."""

    def test_valid_patterns_construction(self):
        """Should construct with valid patterns."""
        validator = NotMatchRegexListValidator(
            patterns=[r"\d{3}-\d{2}-\d{4}", r"\d{16}"]
        )
        assert len(validator._compiled) == 2

    def test_invalid_pattern_in_list(self):
        """Should fail if any pattern is invalid."""
        with pytest.raises(RegexValidationError):
            NotMatchRegexListValidator(
                patterns=[r"\d+", r"[invalid("]
            )


class TestRegexValidationErrorMessage:
    """Tests for RegexValidationError message formatting."""

    def test_error_message_includes_pattern(self):
        """Error message should include the invalid pattern."""
        try:
            RegexValidatorMixin.validate_pattern(r"[invalid(")
        except RegexValidationError as e:
            assert "[invalid(" in str(e)
            assert e.pattern == "[invalid("

    def test_error_message_includes_reason(self):
        """Error message should include the reason."""
        try:
            RegexValidatorMixin.validate_pattern(r"[invalid(")
        except RegexValidationError as e:
            # re.error message should be included
            assert len(e.error) > 0


class TestRegexValidatorFunctionality:
    """Tests for RegexValidator functionality after construction."""

    def test_matches_correctly(self):
        """Should correctly match patterns."""
        validator = RegexValidator(
            pattern=r"^[A-Z]{2}\d{3}$",
            columns=["code"],
        )

        lf = pl.LazyFrame({
            "code": ["AB123", "CD456", "invalid", "EF789"],
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1
        assert "invalid" in issues[0].sample_values

    def test_complex_pattern(self):
        """Should handle complex patterns."""
        # Email-like pattern
        validator = RegexValidator(
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )

        lf = pl.LazyFrame({
            "email": ["user@example.com", "invalid", "test@test.co.uk"],
        })

        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1
