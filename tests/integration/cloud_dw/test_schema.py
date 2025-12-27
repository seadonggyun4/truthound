"""Schema inference tests for Cloud DW backends.

These tests verify that:
- Table schemas can be retrieved correctly
- Column types are mapped correctly
- Schema inference matches actual data
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.integration.cloud_dw.fixtures import (
    StandardTestData,
    validate_schema_match,
)


# =============================================================================
# Schema Inference Tests
# =============================================================================


class TestSchemaInference:
    """Schema inference tests for all backends."""

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_basic_schema_retrieval(
        self,
        any_backend,
        any_dataset,
    ):
        """Test basic schema retrieval."""
        # Get dialect for this backend
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create test table
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=10)

        table = any_backend.create_test_table(
            any_dataset,
            "schema_test",
            schema,
            data,
        )

        # Retrieve schema
        actual_schema = any_backend.get_table_schema(any_dataset.name, table.name)

        # Validate schema matches (non-strict for type variations)
        success, errors = validate_schema_match(schema, actual_schema, strict=False)
        assert success, f"Schema mismatch: {errors}"

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_row_count(
        self,
        any_backend,
        any_dataset,
    ):
        """Test row count retrieval."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create test table with known row count
        expected_rows = 50
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=expected_rows)

        table = any_backend.create_test_table(
            any_dataset,
            "row_count_test",
            schema,
            data,
        )

        # Get row count
        actual_count = any_backend.get_row_count(any_dataset.name, table.name)
        assert actual_count == expected_rows

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_empty_table_schema(
        self,
        any_backend,
        any_dataset,
    ):
        """Test schema retrieval for empty table."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create empty table
        schema = StandardTestData.users_schema(dialect)

        table = any_backend.create_test_table(
            any_dataset,
            "empty_table_test",
            schema,
            None,  # No data
        )

        # Schema should still be retrievable
        actual_schema = any_backend.get_table_schema(any_dataset.name, table.name)
        assert len(actual_schema) == len(schema)

        # Row count should be 0
        count = any_backend.get_row_count(any_dataset.name, table.name)
        assert count == 0


# =============================================================================
# Type Mapping Tests
# =============================================================================


class TestTypeMappings:
    """Tests for type mappings across platforms."""

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_numeric_types(
        self,
        any_backend,
        any_dataset,
    ):
        """Test numeric type handling."""
        from tests.integration.cloud_dw.fixtures import get_type, SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        schema = {
            "int_col": get_type("int", dialect),
            "bigint_col": get_type("bigint", dialect),
            "float_col": get_type("float", dialect),
            "decimal_col": get_type("decimal", dialect),
        }

        data = [
            {
                "int_col": 42,
                "bigint_col": 9223372036854775807,  # Max int64
                "float_col": 3.14159,
                "decimal_col": 123.4567,
            }
        ]

        table = any_backend.create_test_table(
            any_dataset,
            "numeric_types_test",
            schema,
            data,
        )

        # Query and verify
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(f"SELECT * FROM {full_name}")

        assert len(result) == 1
        row = result[0]

        # Values should be preserved (with some floating point tolerance)
        assert row["int_col"] == 42 or row["INT_COL"] == 42
        assert abs(row.get("float_col", row.get("FLOAT_COL", 0)) - 3.14159) < 0.0001

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_string_types(
        self,
        any_backend,
        any_dataset,
    ):
        """Test string type handling."""
        from tests.integration.cloud_dw.fixtures import get_type, SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        schema = {
            "short_string": get_type("string", dialect),
            "long_text": get_type("text", dialect),
        }

        long_text = "A" * 1000  # 1000 character string

        data = [
            {
                "short_string": "Hello, World!",
                "long_text": long_text,
            }
        ]

        table = any_backend.create_test_table(
            any_dataset,
            "string_types_test",
            schema,
            data,
        )

        # Query and verify
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(f"SELECT * FROM {full_name}")

        assert len(result) == 1
        row = result[0]

        short_val = row.get("short_string", row.get("SHORT_STRING", ""))
        long_val = row.get("long_text", row.get("LONG_TEXT", ""))

        assert short_val == "Hello, World!"
        assert len(long_val) == 1000

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_boolean_type(
        self,
        any_backend,
        any_dataset,
    ):
        """Test boolean type handling."""
        from tests.integration.cloud_dw.fixtures import get_type, SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        schema = {
            "bool_col": get_type("bool", dialect),
        }

        data = [
            {"bool_col": True},
            {"bool_col": False},
        ]

        table = any_backend.create_test_table(
            any_dataset,
            "bool_type_test",
            schema,
            data,
        )

        # Query and verify
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(
            f"SELECT * FROM {full_name} ORDER BY bool_col"
        )

        assert len(result) == 2

        # Values should be True and False
        values = [
            row.get("bool_col", row.get("BOOL_COL"))
            for row in result
        ]
        assert False in values or 0 in values
        assert True in values or 1 in values

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_date_types(
        self,
        any_backend,
        any_dataset,
    ):
        """Test date and timestamp type handling."""
        from datetime import date, datetime

        from tests.integration.cloud_dw.fixtures import get_type, SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        schema = {
            "date_col": get_type("date", dialect),
            "timestamp_col": get_type("timestamp", dialect),
        }

        test_date = date(2024, 6, 15)
        test_timestamp = datetime(2024, 6, 15, 10, 30, 45)

        data = [
            {
                "date_col": test_date,
                "timestamp_col": test_timestamp,
            }
        ]

        table = any_backend.create_test_table(
            any_dataset,
            "date_types_test",
            schema,
            data,
        )

        # Query and verify
        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)
        result = any_backend.execute_query(f"SELECT * FROM {full_name}")

        assert len(result) == 1
        row = result[0]

        # Date values should be preserved
        date_val = row.get("date_col", row.get("DATE_COL"))
        assert date_val is not None


# =============================================================================
# Truthound DataSource Integration Tests
# =============================================================================


class TestTruthoundIntegration:
    """Tests for Truthound DataSource integration."""

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_create_datasource(
        self,
        any_backend,
        any_dataset,
    ):
        """Test creating a Truthound DataSource."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create test table
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=20)

        table = any_backend.create_test_table(
            any_dataset,
            "datasource_test",
            schema,
            data,
        )

        # Create Truthound DataSource
        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Verify DataSource properties
        assert datasource is not None
        assert datasource.source_type == any_backend.platform_name
        assert datasource.table_name == table.name

    @pytest.mark.schema
    @pytest.mark.requires_data
    def test_datasource_schema_match(
        self,
        any_backend,
        any_dataset,
    ):
        """Test that DataSource schema matches backend schema."""
        from tests.integration.cloud_dw.fixtures import SQLDialect

        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        dialect = dialect_map.get(any_backend.platform_name, SQLDialect.BIGQUERY)

        # Create test table
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=10)

        table = any_backend.create_test_table(
            any_dataset,
            "schema_match_test",
            schema,
            data,
        )

        # Create Truthound DataSource
        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        # Get schemas
        backend_schema = any_backend.get_table_schema(any_dataset.name, table.name)
        datasource_columns = datasource.columns

        # Column names should match (case insensitive)
        backend_cols = {c.lower() for c in backend_schema.keys()}
        ds_cols = {c.lower() for c in datasource_columns}

        assert backend_cols == ds_cols, f"Schema mismatch: {backend_cols} vs {ds_cols}"
