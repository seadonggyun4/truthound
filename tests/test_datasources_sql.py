"""Tests for SQL data sources.

This module tests SQLite data source which doesn't require external dependencies.
PostgreSQL and MySQL tests are skipped without proper environment.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from truthound.datasources.sql import (
    SQLiteDataSource,
    SQLDataSourceConfig,
)
from truthound.datasources import (
    get_sql_datasource,
    ColumnType,
    DataSourceError,
)
from truthound.execution import SQLExecutionEngine


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create test table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            salary REAL,
            active INTEGER
        )
    """)

    # Insert test data
    test_data = [
        (1, "Alice", 25, 50000.0, 1),
        (2, "Bob", 30, 60000.0, 0),
        (3, None, 35, 70000.0, 1),
        (4, "David", None, 80000.0, 0),
        (5, "Eve", 45, 90000.0, 1),
    ]
    cursor.executemany(
        "INSERT INTO users (id, name, age, salary, active) VALUES (?, ?, ?, ?, ?)",
        test_data,
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sqlite_source(temp_sqlite_db):
    """Create a SQLiteDataSource for testing."""
    return SQLiteDataSource(table="users", database=temp_sqlite_db)


@pytest.fixture
def sql_engine(sqlite_source):
    """Create a SQLExecutionEngine for testing."""
    return SQLExecutionEngine(sqlite_source)


# =============================================================================
# SQLiteDataSource Tests
# =============================================================================


class TestSQLiteDataSource:
    """Tests for SQLiteDataSource."""

    def test_create_source(self, temp_sqlite_db):
        """Test creating SQLite data source."""
        source = SQLiteDataSource(table="users", database=temp_sqlite_db)
        assert source.source_type == "sqlite"
        assert source.table_name == "users"

    def test_schema(self, sqlite_source):
        """Test schema retrieval."""
        schema = sqlite_source.schema
        assert "id" in schema
        assert "name" in schema
        assert "age" in schema
        assert schema["name"] == ColumnType.STRING

    def test_sql_schema(self, sqlite_source):
        """Test native SQL schema."""
        sql_schema = sqlite_source.sql_schema
        assert "id" in sql_schema
        assert "INTEGER" in sql_schema["id"]

    def test_columns(self, sqlite_source):
        """Test column list."""
        columns = sqlite_source.columns
        assert len(columns) == 5
        assert "id" in columns
        assert "name" in columns

    def test_row_count(self, sqlite_source):
        """Test row count."""
        assert sqlite_source.row_count == 5

    def test_validate_connection(self, sqlite_source):
        """Test connection validation."""
        assert sqlite_source.validate_connection() is True

    def test_execute_query(self, sqlite_source):
        """Test raw query execution."""
        result = sqlite_source.execute_query("SELECT COUNT(*) as cnt FROM users")
        assert result[0]["cnt"] == 5

    def test_execute_scalar(self, sqlite_source):
        """Test scalar query."""
        result = sqlite_source.execute_scalar("SELECT COUNT(*) FROM users")
        assert result == 5

    def test_get_execution_engine(self, sqlite_source):
        """Test getting execution engine."""
        engine = sqlite_source.get_execution_engine()
        assert isinstance(engine, SQLExecutionEngine)
        assert engine.count_rows() == 5

    def test_to_polars_lazyframe(self, sqlite_source):
        """Test converting to Polars LazyFrame."""
        import polars as pl
        lf = sqlite_source.to_polars_lazyframe()
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == (5, 5)

    def test_sample(self, sqlite_source):
        """Test sampling."""
        sampled = sqlite_source.sample(n=2)
        assert sampled.row_count <= 2

    def test_file_not_found(self):
        """Test error when database file not found."""
        with pytest.raises(DataSourceError, match="Database file not found"):
            SQLiteDataSource(table="users", database="/nonexistent/path.db")

    def test_in_memory(self):
        """Test in-memory database."""
        # Create in-memory database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        cursor.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b')")
        conn.commit()

        # Note: In-memory databases don't persist across connections
        # This test is for API verification
        source = SQLiteDataSource(table="test", database=":memory:")
        # Schema will fail because table doesn't exist in new connection
        # This is expected behavior for :memory:

    def test_context_manager(self, temp_sqlite_db):
        """Test using as context manager."""
        with SQLiteDataSource(table="users", database=temp_sqlite_db) as source:
            assert source.row_count == 5

    def test_get_table_info(self, sqlite_source):
        """Test getting table info via PRAGMA."""
        info = sqlite_source.get_table_info()
        assert len(info) == 5
        assert any(col["name"] == "id" for col in info)


# =============================================================================
# SQLExecutionEngine Tests (with SQLite)
# =============================================================================


class TestSQLExecutionEngine:
    """Tests for SQLExecutionEngine using SQLite."""

    def test_engine_type(self, sql_engine):
        """Test engine type."""
        assert sql_engine.engine_type == "sql"
        assert sql_engine.supports_sql_pushdown is True

    def test_count_rows(self, sql_engine):
        """Test counting rows."""
        assert sql_engine.count_rows() == 5

    def test_get_columns(self, sql_engine):
        """Test getting columns."""
        columns = sql_engine.get_columns()
        assert len(columns) == 5
        assert "id" in columns

    def test_count_nulls(self, sql_engine):
        """Test counting nulls via SQL."""
        assert sql_engine.count_nulls("name") == 1
        assert sql_engine.count_nulls("age") == 1
        assert sql_engine.count_nulls("id") == 0

    def test_count_nulls_all(self, sql_engine):
        """Test counting nulls for all columns."""
        null_counts = sql_engine.count_nulls_all()
        assert null_counts["name"] == 1
        assert null_counts["age"] == 1

    def test_count_distinct(self, sql_engine):
        """Test counting distinct values."""
        # All IDs are unique
        assert sql_engine.count_distinct("id") == 5
        # Active has 2 distinct values (0 and 1)
        assert sql_engine.count_distinct("active") == 2

    def test_get_stats(self, sql_engine):
        """Test getting statistics via SQL."""
        stats = sql_engine.get_stats("salary")

        assert stats["count"] == 5
        assert stats["null_count"] == 0
        assert stats["mean"] == 70000.0
        assert stats["min"] == 50000.0
        assert stats["max"] == 90000.0
        assert stats["sum"] == 350000.0

    def test_get_value_counts(self, sql_engine):
        """Test getting value counts."""
        counts = sql_engine.get_value_counts("active")
        assert counts[1] == 3
        assert counts[0] == 2

    def test_aggregate(self, sql_engine):
        """Test SQL aggregations."""
        from truthound.execution import AggregationType

        result = sql_engine.aggregate({
            "salary": AggregationType.SUM,
            "id": AggregationType.COUNT,
            "age": AggregationType.MAX,
        })

        assert result["salary_sum"] == 350000.0
        assert result["id_count"] == 5
        assert result["age_max"] == 45

    def test_get_distinct_values(self, sql_engine):
        """Test getting distinct values."""
        values = sql_engine.get_distinct_values("active")
        assert set(values) == {0, 1}

    def test_get_column_values(self, sql_engine):
        """Test getting column values."""
        values = sql_engine.get_column_values("id", limit=3)
        assert len(values) == 3

    def test_count_matching(self, sql_engine):
        """Test counting with SQL condition."""
        count = sql_engine.count_matching("age > 30")
        assert count == 2  # 35 and 45

    def test_count_in_range(self, sql_engine):
        """Test counting values in range."""
        count = sql_engine.count_in_range("salary", min_value=60000, max_value=80000)
        assert count == 3

    def test_count_in_set(self, sql_engine):
        """Test counting values in set."""
        count = sql_engine.count_in_set("id", {1, 2, 3})
        assert count == 3

    def test_count_duplicates(self, sql_engine):
        """Test counting duplicates."""
        # No duplicates on unique column
        assert sql_engine.count_duplicates(["id"]) == 0

        # Duplicates on active column
        assert sql_engine.count_duplicates(["active"]) == 3

    def test_get_duplicate_values(self, sql_engine):
        """Test getting duplicate values."""
        dupes = sql_engine.get_duplicate_values(["active"])
        assert len(dupes) == 2  # Both 0 and 1 have duplicates

    def test_sample(self, sql_engine):
        """Test sampling."""
        sampled = sql_engine.sample(n=3)
        assert sampled.count_rows() <= 3

    def test_execute_sql(self, sql_engine):
        """Test executing raw SQL."""
        result = sql_engine.execute_sql("SELECT name FROM users WHERE age > 30")
        assert len(result) == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestSQLFactory:
    """Tests for SQL factory functions."""

    def test_get_sql_datasource_sqlite(self, temp_sqlite_db):
        """Test factory with SQLite path."""
        source = get_sql_datasource(temp_sqlite_db, table="users")
        assert source.source_type == "sqlite"
        assert source.row_count == 5

    def test_get_sql_datasource_memory(self):
        """Test factory with :memory:."""
        source = get_sql_datasource(":memory:", table="test")
        # Will fail on schema but creates source
        assert source.source_type == "sqlite"


# =============================================================================
# Connection Pool Tests
# =============================================================================


class TestConnectionPool:
    """Tests for SQL connection pooling."""

    def test_multiple_queries(self, sqlite_source):
        """Test multiple queries use connection pool."""
        # Execute multiple queries
        for _ in range(10):
            sqlite_source.execute_scalar("SELECT COUNT(*) FROM users")

        # Pool should have reused connections

    def test_concurrent_access(self, temp_sqlite_db):
        """Test concurrent access to database."""
        import threading
        results = []

        def query_db():
            source = SQLiteDataSource(table="users", database=temp_sqlite_db)
            count = source.row_count
            results.append(count)

        threads = [threading.Thread(target=query_db) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == 5 for r in results)


# =============================================================================
# Query Builder Tests
# =============================================================================


class TestQueryBuilders:
    """Tests for SQL query builder methods."""

    def test_build_count_query(self, sqlite_source):
        """Test building count query."""
        query = sqlite_source.build_count_query()
        assert "COUNT(*)" in query
        assert "users" in query

    def test_build_count_query_with_condition(self, sqlite_source):
        """Test building count query with condition."""
        query = sqlite_source.build_count_query("age > 30")
        assert "WHERE age > 30" in query

    def test_build_distinct_count_query(self, sqlite_source):
        """Test building distinct count query."""
        query = sqlite_source.build_distinct_count_query("name")
        assert "COUNT(DISTINCT" in query
        assert "name" in query

    def test_build_null_count_query(self, sqlite_source):
        """Test building null count query."""
        query = sqlite_source.build_null_count_query("name")
        assert "IS NULL" in query

    def test_build_stats_query(self, sqlite_source):
        """Test building statistics query."""
        query = sqlite_source.build_stats_query("salary")
        assert "AVG" in query
        assert "MIN" in query
        assert "MAX" in query


# =============================================================================
# From DataFrame Tests
# =============================================================================


class TestFromDataFrame:
    """Tests for creating SQLite from DataFrames."""

    def test_from_polars_dataframe(self):
        """Test creating SQLite from Polars DataFrame."""
        import polars as pl

        df = pl.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
        })

        source = SQLiteDataSource.from_dataframe(df, "test_table")
        assert source.row_count == 3
        assert "id" in source.columns

    def test_from_pandas_dataframe(self):
        """Test creating SQLite from Pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
        })

        source = SQLiteDataSource.from_dataframe(df, "test_table")
        assert source.row_count == 3
