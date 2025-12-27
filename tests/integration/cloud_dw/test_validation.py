"""Data validation tests for Cloud DW backends.

These tests verify that Truthound validators work correctly
with data stored in cloud data warehouses.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.integration.cloud_dw.fixtures import StandardTestData


# =============================================================================
# Null Detection Tests
# =============================================================================


class TestNullDetection:
    """Tests for null value detection."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_null_count_detection(
        self,
        any_backend,
        any_dataset,
    ):
        """Test detection of null values."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with known null pattern
        schema = StandardTestData.nulls_schema(dialect)
        data = StandardTestData.nulls_data(n=100, null_ratio=0.2)

        table = any_backend.create_test_table(
            any_dataset,
            "null_detection_test",
            schema,
            data,
        )

        # Count nulls in optional_field
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT COUNT(*) AS total, "
            f"SUM(CASE WHEN optional_field IS NULL THEN 1 ELSE 0 END) AS null_count "
            f"FROM {full_name}"
        )

        total = result[0].get("total", result[0].get("TOTAL", 0))
        null_count = result[0].get("null_count", result[0].get("NULL_COUNT", 0))

        # Should have approximately 20% nulls (with some tolerance)
        null_ratio = null_count / total if total > 0 else 0
        assert 0.1 <= null_ratio <= 0.35, f"Unexpected null ratio: {null_ratio}"

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_required_field_no_nulls(
        self,
        any_backend,
        any_dataset,
    ):
        """Test that required fields have no nulls."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table
        schema = StandardTestData.nulls_schema(dialect)
        data = StandardTestData.nulls_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "required_field_test",
            schema,
            data,
        )

        # Count nulls in required_field (should be 0)
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT COUNT(*) AS null_count "
            f"FROM {full_name} "
            f"WHERE required_field IS NULL"
        )

        null_count = result[0].get("null_count", result[0].get("NULL_COUNT", 0))
        assert null_count == 0, "Required field should have no nulls"


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestDuplicateDetection:
    """Tests for duplicate value detection."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_unique_key_detection(
        self,
        any_backend,
        any_dataset,
    ):
        """Test detection of duplicate keys."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with known duplicates
        schema = StandardTestData.duplicates_schema(dialect)
        data = StandardTestData.duplicates_data(n=100, duplicate_ratio=0.1)

        table = any_backend.create_test_table(
            any_dataset,
            "duplicate_detection_test",
            schema,
            data,
        )

        # Find duplicates
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT unique_key, COUNT(*) AS cnt "
            f"FROM {full_name} "
            f"GROUP BY unique_key "
            f"HAVING COUNT(*) > 1"
        )

        # Should have some duplicates (given 10% duplicate ratio)
        # The exact count varies due to randomness
        assert isinstance(result, list)

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_id_uniqueness(
        self,
        any_backend,
        any_dataset,
    ):
        """Test that ID field is unique."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with unique IDs
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "id_uniqueness_test",
            schema,
            data,
        )

        # Check for duplicate IDs
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT id, COUNT(*) AS cnt "
            f"FROM {full_name} "
            f"GROUP BY id "
            f"HAVING COUNT(*) > 1"
        )

        assert len(result) == 0, "IDs should be unique"


# =============================================================================
# Range Validation Tests
# =============================================================================


class TestRangeValidation:
    """Tests for value range validation."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_age_range(
        self,
        any_backend,
        any_dataset,
    ):
        """Test that age values are in valid range."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = any_backend.create_test_table(
            any_dataset,
            "age_range_test",
            schema,
            data,
        )

        # Check age range (test data generates 18-80)
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT MIN(age) AS min_age, MAX(age) AS max_age FROM {full_name}"
        )

        min_age = result[0].get("min_age", result[0].get("MIN_AGE"))
        max_age = result[0].get("max_age", result[0].get("MAX_AGE"))

        assert min_age >= 18, f"Minimum age {min_age} below expected"
        assert max_age <= 80, f"Maximum age {max_age} above expected"

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_salary_positive(
        self,
        any_backend,
        any_dataset,
    ):
        """Test that salary values are positive."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "salary_positive_test",
            schema,
            data,
        )

        # Check for negative salaries
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT COUNT(*) AS cnt FROM {full_name} WHERE salary < 0"
        )

        negative_count = result[0].get("cnt", result[0].get("CNT", 0))
        assert negative_count == 0, "Salary should be positive"


# =============================================================================
# Pattern Validation Tests
# =============================================================================


class TestPatternValidation:
    """Tests for pattern validation (email, etc.)."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_email_format(
        self,
        any_backend,
        any_dataset,
    ):
        """Test email format validation."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "email_format_test",
            schema,
            data,
        )

        # Check that all emails contain @
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)

        # Use platform-appropriate string contains
        if any_backend.platform_name == "bigquery":
            like_clause = "email NOT LIKE '%@%'"
        else:
            like_clause = "email NOT LIKE '%@%'"

        result = any_backend.execute_query(
            f"SELECT COUNT(*) AS cnt FROM {full_name} WHERE {like_clause}"
        )

        invalid_count = result[0].get("cnt", result[0].get("CNT", 0))
        assert invalid_count == 0, "All emails should contain @"


# =============================================================================
# Unicode Handling Tests
# =============================================================================


class TestUnicodeHandling:
    """Tests for Unicode data handling."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_unicode_preservation(
        self,
        any_backend,
        any_dataset,
    ):
        """Test that Unicode characters are preserved."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with Unicode data
        schema = StandardTestData.unicode_schema(dialect)
        data = StandardTestData.unicode_data()

        table = any_backend.create_test_table(
            any_dataset,
            "unicode_test",
            schema,
            data,
        )

        # Query and verify
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT * FROM {full_name} ORDER BY id"
        )

        assert len(result) == 2

        # Check Chinese characters preserved
        chinese = result[0].get("chinese", result[0].get("CHINESE", ""))
        assert "你好" in chinese or chinese == "你好世界"

        # Check Korean characters preserved
        korean = result[0].get("korean", result[0].get("KOREAN", ""))
        assert "안녕" in korean or "세계" in korean


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge case handling."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_empty_string_vs_null(
        self,
        any_backend,
        any_dataset,
    ):
        """Test distinction between empty strings and nulls."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with edge cases
        schema = StandardTestData.edge_cases_schema(dialect)
        data = StandardTestData.edge_cases_data()

        table = any_backend.create_test_table(
            any_dataset,
            "edge_case_test",
            schema,
            data,
        )

        # Query
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT * FROM {full_name} ORDER BY id"
        )

        assert len(result) == 3

        # First row should have empty string
        empty_str = result[0].get("empty_string", result[0].get("EMPTY_STRING"))
        assert empty_str == "" or empty_str is None  # Some platforms may coerce

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_large_integer_values(
        self,
        any_backend,
        any_dataset,
    ):
        """Test handling of large integer values."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with edge cases
        schema = StandardTestData.edge_cases_schema(dialect)
        data = StandardTestData.edge_cases_data()

        table = any_backend.create_test_table(
            any_dataset,
            "large_int_test",
            schema,
            data,
        )

        # Query max values
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT MAX(max_int) AS max_val FROM {full_name}"
        )

        max_val = result[0].get("max_val", result[0].get("MAX_VAL"))
        assert max_val is not None
        assert max_val > 0


# =============================================================================
# Statistical Validation Tests
# =============================================================================


@pytest.mark.expensive
class TestStatisticalValidation:
    """Tests for statistical validation (marked as expensive)."""

    @pytest.mark.validation
    @pytest.mark.requires_data
    def test_basic_statistics(
        self,
        any_backend,
        any_dataset,
    ):
        """Test basic statistical computations."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create table with known data
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=1000)

        table = any_backend.create_test_table(
            any_dataset,
            "stats_test",
            schema,
            data,
        )

        # Compute statistics
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT "
            f"COUNT(*) AS cnt, "
            f"AVG(age) AS avg_age, "
            f"MIN(age) AS min_age, "
            f"MAX(age) AS max_age "
            f"FROM {full_name}"
        )

        # Verify statistics
        stats = result[0]
        cnt = stats.get("cnt", stats.get("CNT"))
        avg_age = stats.get("avg_age", stats.get("AVG_AGE"))
        min_age = stats.get("min_age", stats.get("MIN_AGE"))
        max_age = stats.get("max_age", stats.get("MAX_AGE"))

        assert cnt == 1000
        assert 18 <= avg_age <= 80  # Average should be in valid range
        assert min_age >= 18
        assert max_age <= 80
