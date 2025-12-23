"""Tests for table metadata validators."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from truthound.validators.table import (
    TableValidator,
    TableRowCountRangeValidator,
    TableRowCountExactValidator,
    TableRowCountCompareValidator,
    TableNotEmptyValidator,
    TableColumnCountValidator,
    TableRequiredColumnsValidator,
    TableForbiddenColumnsValidator,
    TableFreshnessValidator,
    TableDataRecencyValidator,
    TableUpdateFrequencyValidator,
    TableSchemaMatchValidator,
    TableSchemaCompareValidator,
    TableColumnTypesValidator,
    TableMemorySizeValidator,
    TableRowToColumnRatioValidator,
    TableDimensionsValidator,
)


# =============================================================================
# Row Count Validators
# =============================================================================


class TestTableRowCountRangeValidator:
    """Tests for TableRowCountRangeValidator."""

    def test_row_count_in_range(self):
        """Test that row count in range passes."""
        df = pl.DataFrame({"a": range(100)})
        validator = TableRowCountRangeValidator(min_rows=50, max_rows=200)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_row_count_below_minimum(self):
        """Test that row count below minimum fails."""
        df = pl.DataFrame({"a": range(10)})
        validator = TableRowCountRangeValidator(min_rows=50)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "row_count_below_minimum"
        assert issues[0].count == 10

    def test_row_count_above_maximum(self):
        """Test that row count above maximum fails."""
        df = pl.DataFrame({"a": range(200)})
        validator = TableRowCountRangeValidator(max_rows=100)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "row_count_above_maximum"

    def test_missing_bounds_raises(self):
        """Test that missing both bounds raises error."""
        with pytest.raises(ValueError, match="At least one"):
            TableRowCountRangeValidator()

    def test_invalid_bounds_raises(self):
        """Test that min > max raises error."""
        with pytest.raises(ValueError, match="cannot be greater"):
            TableRowCountRangeValidator(min_rows=100, max_rows=50)


class TestTableRowCountExactValidator:
    """Tests for TableRowCountExactValidator."""

    def test_exact_count_match(self):
        """Test that exact count match passes."""
        df = pl.DataFrame({"a": range(100)})
        validator = TableRowCountExactValidator(expected_rows=100)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_exact_count_mismatch(self):
        """Test that count mismatch fails."""
        df = pl.DataFrame({"a": range(90)})
        validator = TableRowCountExactValidator(expected_rows=100)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "row_count_mismatch"

    def test_with_tolerance(self):
        """Test that tolerance allows deviation."""
        df = pl.DataFrame({"a": range(95)})
        validator = TableRowCountExactValidator(expected_rows=100, tolerance=10)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


class TestTableRowCountCompareValidator:
    """Tests for TableRowCountCompareValidator."""

    def test_equal_row_counts(self):
        """Test equal row counts pass."""
        main_df = pl.DataFrame({"a": range(100)})
        ref_df = pl.DataFrame({"b": range(100)})
        validator = TableRowCountCompareValidator(
            reference_table=ref_df.lazy(), comparison="equal"
        )
        issues = validator.validate(main_df.lazy())
        assert len(issues) == 0

    def test_unequal_row_counts(self):
        """Test unequal row counts fail."""
        main_df = pl.DataFrame({"a": range(100)})
        ref_df = pl.DataFrame({"b": range(50)})
        validator = TableRowCountCompareValidator(
            reference_table=ref_df.lazy(), comparison="equal"
        )
        issues = validator.validate(main_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "row_count_comparison_failed"

    def test_greater_comparison(self):
        """Test greater comparison."""
        main_df = pl.DataFrame({"a": range(100)})
        ref_df = pl.DataFrame({"b": range(50)})
        validator = TableRowCountCompareValidator(
            reference_table=ref_df.lazy(), comparison="greater"
        )
        issues = validator.validate(main_df.lazy())
        assert len(issues) == 0

    def test_with_tolerance_ratio(self):
        """Test equal comparison with tolerance ratio."""
        main_df = pl.DataFrame({"a": range(105)})
        ref_df = pl.DataFrame({"b": range(100)})
        validator = TableRowCountCompareValidator(
            reference_table=ref_df.lazy(), comparison="equal", tolerance_ratio=0.1
        )
        issues = validator.validate(main_df.lazy())
        assert len(issues) == 0


class TestTableNotEmptyValidator:
    """Tests for TableNotEmptyValidator."""

    def test_non_empty_table(self):
        """Test non-empty table passes."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        validator = TableNotEmptyValidator()
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_empty_table(self):
        """Test empty table fails."""
        df = pl.DataFrame({"a": []})
        validator = TableNotEmptyValidator()
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "table_is_empty"


# =============================================================================
# Column Count Validators
# =============================================================================


class TestTableColumnCountValidator:
    """Tests for TableColumnCountValidator."""

    def test_exact_column_count(self):
        """Test exact column count passes."""
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        validator = TableColumnCountValidator(expected_count=3)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_count_mismatch(self):
        """Test column count mismatch fails."""
        df = pl.DataFrame({"a": [1], "b": [2]})
        validator = TableColumnCountValidator(expected_count=3)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "column_count_mismatch"

    def test_column_count_range(self):
        """Test column count within range passes."""
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        validator = TableColumnCountValidator(min_count=2, max_count=5)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


class TestTableRequiredColumnsValidator:
    """Tests for TableRequiredColumnsValidator."""

    def test_all_required_present(self):
        """Test all required columns present passes."""
        df = pl.DataFrame({"id": [1], "name": ["a"], "email": ["a@b.c"]})
        validator = TableRequiredColumnsValidator(required_columns=["id", "name"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_missing_required_columns(self):
        """Test missing required columns fails."""
        df = pl.DataFrame({"id": [1]})
        validator = TableRequiredColumnsValidator(required_columns=["id", "name", "email"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "missing_required_columns"
        assert issues[0].count == 2  # name and email missing


class TestTableForbiddenColumnsValidator:
    """Tests for TableForbiddenColumnsValidator."""

    def test_no_forbidden_present(self):
        """Test no forbidden columns passes."""
        df = pl.DataFrame({"id": [1], "name": ["a"]})
        validator = TableForbiddenColumnsValidator(forbidden_columns=["ssn", "password"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_forbidden_columns_found(self):
        """Test forbidden columns found fails."""
        df = pl.DataFrame({"id": [1], "ssn": ["123-45-6789"]})
        validator = TableForbiddenColumnsValidator(forbidden_columns=["ssn", "password"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "forbidden_columns_found"


# =============================================================================
# Freshness Validators
# =============================================================================


class TestTableFreshnessValidator:
    """Tests for TableFreshnessValidator."""

    def test_fresh_data(self):
        """Test fresh data passes."""
        now = datetime.now()
        df = pl.DataFrame({"updated_at": [now, now - timedelta(hours=1)]})
        validator = TableFreshnessValidator(
            timestamp_column="updated_at",
            max_age_hours=24,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_stale_data(self):
        """Test stale data fails."""
        now = datetime.now()
        old_date = now - timedelta(days=10)
        df = pl.DataFrame({"updated_at": [old_date]})
        validator = TableFreshnessValidator(
            timestamp_column="updated_at",
            max_age_days=7,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "data_not_fresh"


class TestTableDataRecencyValidator:
    """Tests for TableDataRecencyValidator."""

    def test_sufficient_recent_data(self):
        """Test sufficient recent data passes."""
        now = datetime.now()
        recent = [now - timedelta(days=i) for i in range(10)]
        df = pl.DataFrame({"created_at": recent})
        validator = TableDataRecencyValidator(
            timestamp_column="created_at",
            max_age_days=30,
            min_recent_ratio=0.5,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_insufficient_recent_data(self):
        """Test insufficient recent data fails."""
        now = datetime.now()
        old = [now - timedelta(days=60 + i) for i in range(10)]
        df = pl.DataFrame({"created_at": old})
        validator = TableDataRecencyValidator(
            timestamp_column="created_at",
            max_age_days=30,
            min_recent_ratio=0.5,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "insufficient_recent_data"


# =============================================================================
# Schema Validators
# =============================================================================


class TestTableSchemaMatchValidator:
    """Tests for TableSchemaMatchValidator."""

    def test_schema_matches(self):
        """Test matching schema passes."""
        df = pl.DataFrame({"id": [1], "name": ["a"]})
        validator = TableSchemaMatchValidator(
            expected_schema={"id": pl.Int64, "name": pl.Utf8}
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_missing_columns(self):
        """Test missing columns fails."""
        df = pl.DataFrame({"id": [1]})
        validator = TableSchemaMatchValidator(
            expected_schema={"id": pl.Int64, "name": pl.Utf8}
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "schema_missing_columns"

    def test_type_mismatch(self):
        """Test type mismatch fails."""
        df = pl.DataFrame({"id": ["a"]})  # String instead of Int64
        validator = TableSchemaMatchValidator(expected_schema={"id": pl.Int64})
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "schema_type_mismatch"

    def test_extra_columns_strict(self):
        """Test extra columns in strict mode fails."""
        df = pl.DataFrame({"id": [1], "extra": [2]})
        validator = TableSchemaMatchValidator(
            expected_schema={"id": pl.Int64}, strict=True
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "schema_extra_columns"


class TestTableSchemaCompareValidator:
    """Tests for TableSchemaCompareValidator."""

    def test_same_schema(self):
        """Test same schema passes."""
        main_df = pl.DataFrame({"id": [1], "name": ["a"]})
        ref_df = pl.DataFrame({"id": [2], "name": ["b"]})
        validator = TableSchemaCompareValidator(reference_table=ref_df.lazy())
        issues = validator.validate(main_df.lazy())
        assert len(issues) == 0

    def test_different_schema(self):
        """Test different schema fails."""
        main_df = pl.DataFrame({"id": [1]})
        ref_df = pl.DataFrame({"id": [2], "name": ["b"]})
        validator = TableSchemaCompareValidator(reference_table=ref_df.lazy())
        issues = validator.validate(main_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "schema_missing_columns"


class TestTableColumnTypesValidator:
    """Tests for TableColumnTypesValidator."""

    def test_valid_types(self):
        """Test valid column types pass."""
        df = pl.DataFrame({"id": [1], "name": ["a"]})
        validator = TableColumnTypesValidator(
            column_types={"id": [pl.Int64], "name": [pl.Utf8]}
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_type(self):
        """Test invalid column type fails."""
        df = pl.DataFrame({"id": ["not_int"]})
        validator = TableColumnTypesValidator(column_types={"id": [pl.Int64, pl.Int32]})
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "column_type_invalid"


# =============================================================================
# Size Validators
# =============================================================================


class TestTableMemorySizeValidator:
    """Tests for TableMemorySizeValidator."""

    def test_size_within_bounds(self):
        """Test size within bounds passes."""
        df = pl.DataFrame({"a": range(100)})
        validator = TableMemorySizeValidator(max_size_mb=100)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_size_too_small(self):
        """Test size too small fails."""
        df = pl.DataFrame({"a": [1]})
        validator = TableMemorySizeValidator(min_size_bytes=10000)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "table_size_below_minimum"


class TestTableRowToColumnRatioValidator:
    """Tests for TableRowToColumnRatioValidator."""

    def test_ratio_in_range(self):
        """Test ratio in range passes."""
        df = pl.DataFrame({"a": range(100), "b": range(100)})  # 100 rows, 2 cols = 50 ratio
        validator = TableRowToColumnRatioValidator(min_ratio=10, max_ratio=100)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_ratio_too_low(self):
        """Test ratio too low (wide table) fails."""
        df = pl.DataFrame({f"col_{i}": [1] for i in range(10)})  # 1 row, 10 cols = 0.1 ratio
        validator = TableRowToColumnRatioValidator(min_ratio=5)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "row_column_ratio_too_low"


class TestTableDimensionsValidator:
    """Tests for TableDimensionsValidator."""

    def test_dimensions_valid(self):
        """Test valid dimensions pass."""
        df = pl.DataFrame({"a": range(100), "b": range(100), "c": range(100)})
        validator = TableDimensionsValidator(min_rows=50, max_rows=200, min_cols=2, max_cols=10)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_rows_below_minimum(self):
        """Test rows below minimum fails."""
        df = pl.DataFrame({"a": range(10)})
        validator = TableDimensionsValidator(min_rows=50)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "row_count_below_minimum"

    def test_cols_above_maximum(self):
        """Test columns above maximum fails."""
        df = pl.DataFrame({f"col_{i}": [1] for i in range(20)})
        validator = TableDimensionsValidator(max_cols=10)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "column_count_above_maximum"
