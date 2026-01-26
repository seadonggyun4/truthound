"""Tests for DuckDB data source.

This module tests DuckDBDataSource functionality.
Requires: pip install duckdb
"""

import pytest
import tempfile
from pathlib import Path

# Skip all tests if duckdb is not installed
duckdb = pytest.importorskip("duckdb")

from truthound.datasources.sql import (
    DuckDBDataSource,
    DuckDBDataSourceConfig,
)
from truthound.datasources import (
    ColumnType,
    DataSourceError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_duckdb():
    """Create a temporary DuckDB database with test data."""
    # Create temp directory and database path (don't pre-create the file)
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.duckdb"

    conn = duckdb.connect(str(db_path))

    # Create test table
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            age INTEGER,
            salary DOUBLE,
            active BOOLEAN
        )
    """)

    # Insert test data
    conn.execute("""
        INSERT INTO users (id, name, age, salary, active) VALUES
        (1, 'Alice', 25, 50000.0, true),
        (2, 'Bob', 30, 60000.0, false),
        (3, NULL, 35, 70000.0, true),
        (4, 'David', NULL, 80000.0, false),
        (5, 'Eve', 45, 90000.0, true)
    """)

    conn.close()

    yield str(db_path)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def duckdb_source(temp_duckdb):
    """Create a DuckDBDataSource for testing."""
    return DuckDBDataSource(table="users", database=temp_duckdb)


# =============================================================================
# DuckDBDataSource Tests
# =============================================================================


class TestDuckDBDataSource:
    """Tests for DuckDBDataSource."""

    def test_create_source(self, temp_duckdb):
        """Test creating DuckDB data source."""
        source = DuckDBDataSource(table="users", database=temp_duckdb)
        assert source.source_type == "duckdb"
        assert source.table_name == "users"

    def test_schema(self, duckdb_source):
        """Test schema retrieval."""
        schema = duckdb_source.schema
        assert "id" in schema
        assert "name" in schema
        assert "age" in schema
        assert schema["name"] == ColumnType.STRING

    def test_sql_schema(self, duckdb_source):
        """Test native SQL schema."""
        sql_schema = duckdb_source.sql_schema
        assert "id" in sql_schema
        assert "INTEGER" in sql_schema["id"]

    def test_columns(self, duckdb_source):
        """Test column list."""
        columns = duckdb_source.columns
        assert len(columns) == 5
        assert "id" in columns
        assert "name" in columns

    def test_row_count(self, duckdb_source):
        """Test row count."""
        assert duckdb_source.row_count == 5

    def test_validate_connection(self, duckdb_source):
        """Test connection validation."""
        assert duckdb_source.validate_connection() is True

    def test_execute_query(self, duckdb_source):
        """Test raw query execution."""
        result = duckdb_source.execute_query("SELECT COUNT(*) as cnt FROM users")
        assert result[0]["cnt"] == 5

    def test_execute_scalar(self, duckdb_source):
        """Test scalar query."""
        result = duckdb_source.execute_scalar("SELECT COUNT(*) FROM users")
        assert result == 5

    def test_to_polars_lazyframe(self, duckdb_source):
        """Test converting to Polars LazyFrame."""
        import polars as pl
        lf = duckdb_source.to_polars_lazyframe()
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == (5, 5)

    def test_sample(self, duckdb_source):
        """Test sampling."""
        sampled = duckdb_source.sample(n=3, seed=42)
        assert sampled.row_count <= 3

    def test_in_memory_database(self):
        """Test in-memory database."""
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER, value VARCHAR)")
        conn.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b')")
        conn.close()

        # Note: in-memory databases don't persist across connections
        # This test verifies the source can be created
        source = DuckDBDataSource(
            database=":memory:",
            query="SELECT 1 as id, 'test' as value",
        )
        assert source.row_count == 1


class TestDuckDBQueryMode:
    """Tests for DuckDB query mode."""

    def test_query_mode(self, temp_duckdb):
        """Test query mode."""
        source = DuckDBDataSource(
            database=temp_duckdb,
            query="SELECT id, name FROM users WHERE active = true",
        )
        assert source.row_count == 3

    def test_query_with_aggregation(self, temp_duckdb):
        """Test query with aggregation."""
        source = DuckDBDataSource(
            database=temp_duckdb,
            query="SELECT active, AVG(salary) as avg_salary FROM users GROUP BY active",
        )
        lf = source.to_polars_lazyframe()
        df = lf.collect()
        assert df.shape[0] == 2


class TestDuckDBHelperMethods:
    """Tests for DuckDB-specific helper methods."""

    def test_get_table_info(self, duckdb_source):
        """Test get_table_info method."""
        info = duckdb_source.get_table_info()
        assert len(info) == 5
        column_names = [row["column_name"] for row in info]
        assert "id" in column_names

    def test_get_tables(self, duckdb_source):
        """Test get_tables method."""
        tables = duckdb_source.get_tables()
        assert "users" in tables

    def test_explain(self, duckdb_source):
        """Test explain method."""
        plan = duckdb_source.explain()
        assert len(plan) > 0


class TestDuckDBFromDataFrame:
    """Tests for creating DuckDB from DataFrames."""

    def test_from_polars_dataframe(self):
        """Test creating from Polars DataFrame."""
        import polars as pl
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        })
        source = DuckDBDataSource.from_dataframe(df, "test_table")
        assert source.row_count == 3
        assert "id" in source.columns

    def test_from_polars_lazyframe(self):
        """Test creating from Polars LazyFrame."""
        import polars as pl
        lf = pl.LazyFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
        })
        source = DuckDBDataSource.from_dataframe(lf, "test_lazy")
        assert source.row_count == 3


class TestDuckDBFileReading:
    """Tests for DuckDB file reading capabilities."""

    def test_from_parquet(self, tmp_path):
        """Test reading from Parquet file."""
        import polars as pl

        # Create test Parquet file
        parquet_path = tmp_path / "test.parquet"
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
        })
        df.write_parquet(parquet_path)

        # Test reading
        source = DuckDBDataSource.from_parquet(str(parquet_path))
        assert source.row_count == 3

    def test_from_csv(self, tmp_path):
        """Test reading from CSV file."""
        import polars as pl

        # Create test CSV file
        csv_path = tmp_path / "test.csv"
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        df.write_csv(csv_path)

        # Test reading
        source = DuckDBDataSource.from_csv(str(csv_path))
        assert source.row_count == 3


class TestDuckDBWithTruthound:
    """Tests for DuckDB integration with Truthound API."""

    def test_th_check(self, duckdb_source):
        """Test th.check with DuckDB source."""
        import truthound as th
        report = th.check(source=duckdb_source, validators=["null"])
        assert report is not None

    def test_th_learn(self, duckdb_source):
        """Test th.learn with DuckDB source."""
        import truthound as th
        schema = th.learn(source=duckdb_source)
        assert "id" in schema.columns
        assert "name" in schema.columns
        assert schema.row_count == 5

    def test_th_profile(self, duckdb_source):
        """Test th.profile with DuckDB source."""
        import truthound as th
        profile = th.profile(source=duckdb_source)
        assert profile.row_count == 5
        assert profile.column_count == 5

    def test_th_scan(self, duckdb_source):
        """Test th.scan with DuckDB source."""
        import truthound as th
        pii_report = th.scan(source=duckdb_source)
        assert pii_report is not None
