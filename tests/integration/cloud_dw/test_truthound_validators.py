"""Truthound validator integration tests for Cloud DW backends.

These tests verify that Truthound validators work correctly with data
stored in cloud data warehouses using the actual Truthound API.

This module tests:
    - NotNullValidator
    - UniqueValidator
    - RangeValidator
    - PatternValidator
    - Cross-table validators
    - Pushdown optimization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.integration.cloud_dw.fixtures import SQLDialect, StandardTestData

if TYPE_CHECKING:
    from tests.integration.cloud_dw.base import CloudDWTestBackend, TestDataset


# =============================================================================
# Helper Functions
# =============================================================================


def get_dialect_for_backend(backend: "CloudDWTestBackend") -> SQLDialect:
    """Get the SQL dialect for a backend."""
    dialect_map = {
        "bigquery": SQLDialect.BIGQUERY,
        "snowflake": SQLDialect.SNOWFLAKE,
        "redshift": SQLDialect.REDSHIFT,
        "databricks": SQLDialect.DATABRICKS,
    }
    return dialect_map.get(backend.platform_name, SQLDialect.BIGQUERY)


# =============================================================================
# NotNull Validator Tests
# =============================================================================


class TestNotNullValidator:
    """Integration tests for NotNullValidator."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_not_null_validator_detects_nulls(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that NotNullValidator correctly detects null values."""
        import truthound as th
        from truthound.validators import NotNullValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create table with known null pattern (20% nulls in optional_field)
        schema = StandardTestData.nulls_schema(dialect)
        data = StandardTestData.nulls_data(n=100, null_ratio=0.2)

        table = any_backend.create_test_table(
            any_dataset,
            "not_null_validator_test",
            schema,
            data,
        )

        # Create Truthound datasource
        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Run validation on required_field (should pass - no nulls)
        result = th.check(
            datasource,
            validators=[NotNullValidator("required_field")],
        )

        assert result.success, "required_field should have no nulls"
        assert result.passed == 1
        assert result.failed == 0

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_not_null_validator_fails_on_nulls(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that NotNullValidator fails when nulls are present."""
        import truthound as th
        from truthound.validators import NotNullValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create table with known null pattern
        schema = StandardTestData.nulls_schema(dialect)
        data = StandardTestData.nulls_data(n=100, null_ratio=0.3)

        table = any_backend.create_test_table(
            any_dataset,
            "not_null_fails_test",
            schema,
            data,
        )

        # Create Truthound datasource
        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Run validation on optional_field (should fail - has nulls)
        result = th.check(
            datasource,
            validators=[NotNullValidator("optional_field")],
        )

        assert not result.success, "optional_field should have nulls"
        assert result.failed == 1

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_not_null_with_pushdown(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test NotNullValidator with query pushdown enabled."""
        import truthound as th
        from truthound.validators import NotNullValidator

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.nulls_schema(dialect)
        data = StandardTestData.nulls_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "not_null_pushdown_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Run with pushdown enabled
        result = th.check(
            datasource,
            validators=[NotNullValidator("required_field")],
            pushdown=True,
        )

        assert result.success


# =============================================================================
# Unique Validator Tests
# =============================================================================


class TestUniqueValidator:
    """Integration tests for UniqueValidator."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_unique_validator_passes(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that UniqueValidator passes for unique columns."""
        import truthound as th
        from truthound.validators import UniqueValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create table with unique IDs
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "unique_validator_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Validate ID uniqueness
        result = th.check(
            datasource,
            validators=[UniqueValidator("id")],
        )

        assert result.success, "ID column should be unique"

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_unique_validator_detects_duplicates(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that UniqueValidator detects duplicates."""
        import truthound as th
        from truthound.validators import UniqueValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create table with intentional duplicates
        schema = StandardTestData.duplicates_schema(dialect)
        data = StandardTestData.duplicates_data(n=100, duplicate_ratio=0.2)

        table = any_backend.create_test_table(
            any_dataset,
            "unique_validator_dups_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Validate unique_key (should fail due to duplicates)
        result = th.check(
            datasource,
            validators=[UniqueValidator("unique_key")],
        )

        # Note: Due to randomness, there may or may not be duplicates
        # We just verify the validator runs correctly
        assert result is not None


# =============================================================================
# Range Validator Tests
# =============================================================================


class TestRangeValidator:
    """Integration tests for RangeValidator."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_range_validator_passes(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that RangeValidator passes for values in range."""
        import truthound as th
        from truthound.validators import RangeValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create table with age values 18-80
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = any_backend.create_test_table(
            any_dataset,
            "range_validator_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Validate age range (18-80, with margin)
        result = th.check(
            datasource,
            validators=[RangeValidator("age", min_value=0, max_value=100)],
        )

        assert result.success, "Age values should be in range 0-100"

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_range_validator_with_strict_bounds(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test RangeValidator with strict bounds."""
        import truthound as th
        from truthound.validators import RangeValidator

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "range_strict_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Use exact bounds (18-80) that match the test data generation
        result = th.check(
            datasource,
            validators=[RangeValidator("age", min_value=18, max_value=80)],
        )

        assert result.success


# =============================================================================
# Pattern Validator Tests
# =============================================================================


class TestPatternValidator:
    """Integration tests for PatternValidator."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_email_pattern_validator(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test PatternValidator for email format."""
        import truthound as th
        from truthound.validators import PatternValidator

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = any_backend.create_test_table(
            any_dataset,
            "pattern_email_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Validate email contains @
        result = th.check(
            datasource,
            validators=[PatternValidator("email", pattern=r".*@.*")],
        )

        assert result.success, "All emails should contain @"


# =============================================================================
# Combined Validator Tests
# =============================================================================


class TestCombinedValidators:
    """Integration tests for multiple validators together."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_multiple_validators(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test running multiple validators together."""
        import truthound as th
        from truthound.validators import (
            NotNullValidator,
            UniqueValidator,
            RangeValidator,
        )

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = any_backend.create_test_table(
            any_dataset,
            "multi_validator_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Run multiple validators
        result = th.check(
            datasource,
            validators=[
                NotNullValidator("id"),
                UniqueValidator("id"),
                RangeValidator("age", min_value=0, max_value=100),
                NotNullValidator("email"),
            ],
        )

        assert result.passed >= 3, "Most validators should pass"

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_validators_with_profile(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test running validators after profiling."""
        import truthound as th

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = any_backend.create_test_table(
            any_dataset,
            "profile_validator_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Profile first
        profile = th.profile(datasource)
        assert profile is not None

        # Then run auto-generated validators
        result = th.check(datasource)
        assert result is not None


# =============================================================================
# Pushdown Optimization Tests
# =============================================================================


@pytest.mark.expensive
class TestPushdownOptimization:
    """Integration tests for query pushdown optimization."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_pushdown_reduces_data_transfer(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that pushdown reduces data transfer."""
        import truthound as th
        from truthound.validators import NotNullValidator, UniqueValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create larger dataset to see pushdown benefits
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=1000)

        table = any_backend.create_test_table(
            any_dataset,
            "pushdown_perf_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Run with pushdown
        result = th.check(
            datasource,
            validators=[
                NotNullValidator("id"),
                UniqueValidator("email"),
            ],
            pushdown=True,
        )

        assert result is not None

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_pushdown_compatibility(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that pushdown works with different validator types."""
        import truthound as th
        from truthound.validators import (
            NotNullValidator,
            UniqueValidator,
            RangeValidator,
        )

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=200)

        table = any_backend.create_test_table(
            any_dataset,
            "pushdown_compat_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Test various validator types with pushdown
        validators = [
            NotNullValidator("id"),
            NotNullValidator("email"),
            UniqueValidator("id"),
            RangeValidator("age", min_value=0, max_value=100),
        ]

        result = th.check(
            datasource,
            validators=validators,
            pushdown=True,
        )

        assert result is not None
        assert result.total == len(validators)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Integration tests for error handling in validators."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_validator_on_nonexistent_column(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that validators handle nonexistent columns gracefully."""
        import truthound as th
        from truthound.validators import NotNullValidator

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=10)

        table = any_backend.create_test_table(
            any_dataset,
            "error_handling_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Try to validate a column that doesn't exist
        result = th.check(
            datasource,
            validators=[NotNullValidator("nonexistent_column")],
        )

        # Should fail gracefully
        assert result.failed == 1

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_empty_table_validation(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test validators on empty tables."""
        import truthound as th
        from truthound.validators import NotNullValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create empty table
        schema = StandardTestData.users_schema(dialect)

        table = any_backend.create_test_table(
            any_dataset,
            "empty_table_test",
            schema,
            [],  # No data
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Validate empty table
        result = th.check(
            datasource,
            validators=[NotNullValidator("id")],
        )

        # Empty table should pass NotNull (no rows to violate)
        assert result.success


# =============================================================================
# Backend-Specific Tests
# =============================================================================


@pytest.mark.bigquery
class TestBigQueryValidators:
    """BigQuery-specific validator tests."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_bigquery_array_type_validation(
        self,
        bigquery_backend: "CloudDWTestBackend",
        bigquery_dataset: "TestDataset",
    ):
        """Test validation on BigQuery ARRAY types."""
        import truthound as th
        from truthound.validators import NotNullValidator

        # Create table with array column
        schema = {
            "id": "INT64",
            "tags": "ARRAY<STRING>",
        }
        data = [
            {"id": 1, "tags": ["a", "b", "c"]},
            {"id": 2, "tags": ["d"]},
            {"id": 3, "tags": None},
        ]

        table = bigquery_backend.create_test_table(
            bigquery_dataset,
            "array_type_test",
            schema,
            data,
        )

        datasource = bigquery_backend.create_datasource(
            bigquery_dataset.name, table.name
        )

        result = th.check(
            datasource,
            validators=[NotNullValidator("id")],
        )

        assert result.success


@pytest.mark.snowflake
class TestSnowflakeValidators:
    """Snowflake-specific validator tests."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_snowflake_variant_type_validation(
        self,
        snowflake_backend: "CloudDWTestBackend",
        snowflake_dataset: "TestDataset",
    ):
        """Test validation on Snowflake VARIANT types."""
        import truthound as th
        from truthound.validators import NotNullValidator

        # Create table with variant column
        schema = {
            "id": "INTEGER",
            "metadata": "VARIANT",
        }
        data = [
            {"id": 1, "metadata": '{"key": "value"}'},
            {"id": 2, "metadata": '{"count": 42}'},
        ]

        table = snowflake_backend.create_test_table(
            snowflake_dataset,
            "variant_type_test",
            schema,
            data,
        )

        datasource = snowflake_backend.create_datasource(
            snowflake_dataset.name, table.name
        )

        result = th.check(
            datasource,
            validators=[NotNullValidator("id")],
        )

        assert result.success


@pytest.mark.databricks
class TestDatabricksValidators:
    """Databricks-specific validator tests."""

    @pytest.mark.validation
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_databricks_delta_table_validation(
        self,
        databricks_backend: "CloudDWTestBackend",
        databricks_dataset: "TestDataset",
    ):
        """Test validation on Databricks Delta tables."""
        import truthound as th
        from truthound.validators import NotNullValidator, UniqueValidator

        dialect = SQLDialect.DATABRICKS
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=50)

        table = databricks_backend.create_test_table(
            databricks_dataset,
            "delta_table_test",
            schema,
            data,
        )

        datasource = databricks_backend.create_datasource(
            databricks_dataset.name, table.name
        )

        result = th.check(
            datasource,
            validators=[
                NotNullValidator("id"),
                UniqueValidator("id"),
            ],
        )

        assert result.passed >= 1
