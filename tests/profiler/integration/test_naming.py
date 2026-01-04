"""Tests for validator name resolution."""

from __future__ import annotations

import pytest

from truthound.profiler.integration.naming import resolve_validator_name


class TestResolveValidatorName:
    """Test resolve_validator_name function."""

    def test_pascal_case_with_validator_suffix(self):
        """Test PascalCase names with Validator suffix."""
        assert resolve_validator_name("ColumnTypeValidator") == "column_type"
        assert resolve_validator_name("NotNullValidator") == "not_null"
        assert resolve_validator_name("UniqueValidator") == "unique"
        assert resolve_validator_name("NullValidator") == "null"

    def test_pascal_case_without_suffix(self):
        """Test PascalCase names without Validator suffix."""
        assert resolve_validator_name("ColumnType") == "column_type"
        assert resolve_validator_name("NotNull") == "not_null"
        assert resolve_validator_name("Null") == "null"

    def test_snake_case(self):
        """Test snake_case names."""
        assert resolve_validator_name("column_type") == "column_type"
        assert resolve_validator_name("not_null") == "not_null"
        assert resolve_validator_name("null") == "null"

    def test_snake_case_with_suffix(self):
        """Test snake_case with _validator suffix."""
        assert resolve_validator_name("column_type_validator") == "column_type"
        assert resolve_validator_name("null_validator") == "null"

    def test_kebab_case(self):
        """Test kebab-case names."""
        assert resolve_validator_name("column-type") == "column_type"
        assert resolve_validator_name("not-null") == "not_null"

    def test_kebab_case_with_suffix(self):
        """Test kebab-case with -validator suffix."""
        assert resolve_validator_name("column-type-validator") == "column_type"

    def test_mixed_case(self):
        """Test various mixed case inputs."""
        assert resolve_validator_name("COLUMN_TYPE") == "column_type"
        assert resolve_validator_name("Column.Type") == "column_type"

    def test_single_word(self):
        """Test single word names."""
        assert resolve_validator_name("Null") == "null"
        assert resolve_validator_name("null") == "null"
        assert resolve_validator_name("Type") == "type"

    def test_empty_raises(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_validator_name("")

    def test_caching(self):
        """Test that results are cached (same result for same input)."""
        result1 = resolve_validator_name("ColumnTypeValidator")
        result2 = resolve_validator_name("ColumnTypeValidator")
        assert result1 == result2 == "column_type"


class TestCommonValidatorNames:
    """Test common validator name conversions."""

    @pytest.mark.parametrize("input_name,expected", [
        # Schema validators
        ("ColumnExistsValidator", "column_exists"),
        ("ColumnTypeValidator", "column_type"),
        ("RowCountValidator", "row_count"),
        ("TableSchemaValidator", "table_schema"),
        # Completeness validators
        ("NullValidator", "null"),
        ("NotNullValidator", "not_null"),
        ("CompletenessRatioValidator", "completeness_ratio"),
        # Uniqueness validators
        ("UniqueValidator", "unique"),
        ("DuplicateValidator", "duplicate"),
        ("PrimaryKeyValidator", "primary_key"),
        # Distribution validators
        ("BetweenValidator", "between"),
        ("RangeValidator", "range"),
        ("OutlierValidator", "outlier"),
        # String validators
        ("RegexValidator", "regex"),
        ("EmailValidator", "email"),
        ("UrlValidator", "url"),
    ])
    def test_common_validators(self, input_name: str, expected: str):
        """Test common validator name conversions."""
        assert resolve_validator_name(input_name) == expected
