"""Tests for data sources module.

This module tests the core data source implementations:
- PolarsDataSource
- PandasDataSource
- FileDataSource
- DictDataSource
"""

import pytest
import polars as pl
import tempfile
from pathlib import Path

from truthound.datasources import (
    get_datasource,
    detect_datasource_type,
    from_polars,
    from_dict,
    from_file,
    PolarsDataSource,
    FileDataSource,
    DictDataSource,
    DataSourceError,
    DataSourceSizeError,
    ColumnType,
    DataSourceCapability,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_polars_df():
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "active": [True, False, True, False, True],
    })


@pytest.fixture
def sample_polars_lf(sample_polars_df):
    """Create a sample Polars LazyFrame for testing."""
    return sample_polars_df.lazy()


@pytest.fixture
def sample_dict():
    """Create a sample dictionary for testing."""
    return {
        "id": [1, 2, 3],
        "value": ["a", "b", "c"],
    }


@pytest.fixture
def temp_csv_file(sample_polars_df):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_polars_df.write_csv(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_parquet_file(sample_polars_df):
    """Create a temporary Parquet file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_polars_df.write_parquet(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_json_file(sample_polars_df):
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        sample_polars_df.write_json(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# PolarsDataSource Tests
# =============================================================================


class TestPolarsDataSource:
    """Tests for PolarsDataSource."""

    def test_create_from_dataframe(self, sample_polars_df):
        """Test creating data source from DataFrame."""
        source = PolarsDataSource(sample_polars_df)
        assert source.source_type == "polars"
        assert source.row_count == 5
        assert len(source.columns) == 5

    def test_create_from_lazyframe(self, sample_polars_lf):
        """Test creating data source from LazyFrame."""
        source = PolarsDataSource(sample_polars_lf)
        assert source.source_type == "polars"
        assert len(source.columns) == 5

    def test_schema_mapping(self, sample_polars_df):
        """Test schema type mapping."""
        source = PolarsDataSource(sample_polars_df)
        schema = source.schema

        assert schema["id"] == ColumnType.INTEGER
        assert schema["name"] == ColumnType.STRING
        assert schema["age"] == ColumnType.INTEGER
        assert schema["salary"] == ColumnType.FLOAT
        assert schema["active"] == ColumnType.BOOLEAN

    def test_capabilities(self, sample_polars_df):
        """Test data source capabilities."""
        source = PolarsDataSource(sample_polars_df)
        caps = source.capabilities

        assert DataSourceCapability.LAZY_EVALUATION in caps
        assert DataSourceCapability.SAMPLING in caps
        assert DataSourceCapability.SCHEMA_INFERENCE in caps

    def test_get_execution_engine(self, sample_polars_df):
        """Test getting execution engine."""
        source = PolarsDataSource(sample_polars_df)
        engine = source.get_execution_engine()

        assert engine.engine_type == "polars"
        assert engine.count_rows() == 5

    def test_sample(self, sample_polars_df):
        """Test sampling."""
        source = PolarsDataSource(sample_polars_df)
        sampled = source.sample(n=2)

        # With small dataset, sample returns same data
        assert sampled.row_count <= source.row_count

    def test_to_polars_lazyframe(self, sample_polars_df):
        """Test converting to LazyFrame."""
        source = PolarsDataSource(sample_polars_df)
        lf = source.to_polars_lazyframe()

        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == (5, 5)

    def test_validate_connection(self, sample_polars_df):
        """Test connection validation."""
        source = PolarsDataSource(sample_polars_df)
        assert source.validate_connection() is True

    def test_get_numeric_columns(self, sample_polars_df):
        """Test getting numeric columns."""
        source = PolarsDataSource(sample_polars_df)
        numeric = source.get_numeric_columns()

        assert "id" in numeric
        assert "age" in numeric
        assert "salary" in numeric
        assert "name" not in numeric

    def test_get_string_columns(self, sample_polars_df):
        """Test getting string columns."""
        source = PolarsDataSource(sample_polars_df)
        strings = source.get_string_columns()

        assert "name" in strings
        assert "id" not in strings


# =============================================================================
# FileDataSource Tests
# =============================================================================


class TestFileDataSource:
    """Tests for FileDataSource."""

    def test_load_csv(self, temp_csv_file):
        """Test loading CSV file."""
        source = FileDataSource(temp_csv_file)
        assert source.source_type == "file"
        assert source.file_type == "csv"
        assert source.row_count == 5

    def test_load_parquet(self, temp_parquet_file):
        """Test loading Parquet file."""
        source = FileDataSource(temp_parquet_file)
        assert source.file_type == "parquet"
        assert source.row_count == 5

    def test_load_json(self, temp_json_file):
        """Test loading JSON file."""
        source = FileDataSource(temp_json_file)
        assert source.file_type == "json"
        assert source.row_count == 5

    def test_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(DataSourceError, match="File not found"):
            FileDataSource("/nonexistent/path/file.csv")

    def test_unsupported_extension(self, tmp_path):
        """Test error for unsupported file type."""
        unsupported = tmp_path / "data.xyz"
        unsupported.write_text("data")

        with pytest.raises(DataSourceError, match="Unsupported file type"):
            FileDataSource(unsupported)

    def test_get_execution_engine(self, temp_csv_file):
        """Test getting execution engine from file source."""
        source = FileDataSource(temp_csv_file)
        engine = source.get_execution_engine()

        assert engine.count_rows() == 5


# =============================================================================
# DictDataSource Tests
# =============================================================================


class TestDictDataSource:
    """Tests for DictDataSource."""

    def test_create_from_dict(self, sample_dict):
        """Test creating from dictionary."""
        source = DictDataSource(sample_dict)
        assert source.source_type == "dict"
        assert source.row_count == 3
        assert source.columns == ["id", "value"]

    def test_get_execution_engine(self, sample_dict):
        """Test getting execution engine."""
        source = DictDataSource(sample_dict)
        engine = source.get_execution_engine()

        assert engine.count_rows() == 3
        assert engine.count_distinct("value") == 3


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetDatasource:
    """Tests for get_datasource factory function."""

    def test_detect_polars_dataframe(self, sample_polars_df):
        """Test detecting Polars DataFrame."""
        source = get_datasource(sample_polars_df)
        assert isinstance(source, PolarsDataSource)

    def test_detect_polars_lazyframe(self, sample_polars_lf):
        """Test detecting Polars LazyFrame."""
        source = get_datasource(sample_polars_lf)
        assert isinstance(source, PolarsDataSource)

    def test_detect_dict(self, sample_dict):
        """Test detecting dictionary."""
        source = get_datasource(sample_dict)
        assert isinstance(source, DictDataSource)

    def test_detect_file(self, temp_csv_file):
        """Test detecting file path."""
        source = get_datasource(temp_csv_file)
        assert isinstance(source, FileDataSource)

    def test_unsupported_type(self):
        """Test error for unsupported type."""
        with pytest.raises(DataSourceError, match="Unsupported data type"):
            get_datasource(12345)

    def test_with_name(self, sample_polars_df):
        """Test creating with custom name."""
        source = get_datasource(sample_polars_df, name="my_source")
        assert source.name == "my_source"


class TestDetectDatasourceType:
    """Tests for detect_datasource_type function."""

    def test_detect_polars(self, sample_polars_df):
        """Test detecting Polars DataFrame type."""
        assert detect_datasource_type(sample_polars_df) == "polars"

    def test_detect_polars_lazy(self, sample_polars_lf):
        """Test detecting Polars LazyFrame type."""
        assert detect_datasource_type(sample_polars_lf) == "polars_lazy"

    def test_detect_dict(self, sample_dict):
        """Test detecting dictionary type."""
        assert detect_datasource_type(sample_dict) == "dict"

    def test_detect_file_csv(self, temp_csv_file):
        """Test detecting CSV file type."""
        assert detect_datasource_type(temp_csv_file) == "file:csv"

    def test_detect_unknown(self):
        """Test detecting unknown type."""
        assert detect_datasource_type(12345) == "unknown"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""

    def test_from_polars(self, sample_polars_df):
        """Test from_polars function."""
        source = from_polars(sample_polars_df, name="test")
        assert isinstance(source, PolarsDataSource)

    def test_from_dict(self, sample_dict):
        """Test from_dict function."""
        source = from_dict(sample_dict)
        assert isinstance(source, DictDataSource)

    def test_from_file(self, temp_csv_file):
        """Test from_file function."""
        source = from_file(temp_csv_file)
        assert isinstance(source, FileDataSource)


# =============================================================================
# Size Limit Tests
# =============================================================================


class TestSizeLimits:
    """Tests for size limit handling."""

    def test_needs_sampling_small(self, sample_polars_df):
        """Test needs_sampling for small data."""
        source = PolarsDataSource(sample_polars_df)
        assert source.needs_sampling() is False

    def test_needs_sampling_large(self, sample_polars_df):
        """Test needs_sampling for data exceeding limit."""
        from truthound.datasources.polars_source import PolarsDataSourceConfig

        config = PolarsDataSourceConfig(max_rows=3)
        source = PolarsDataSource(sample_polars_df, config)
        assert source.needs_sampling() is True

    def test_get_safe_sample(self, sample_polars_df):
        """Test get_safe_sample."""
        from truthound.datasources.polars_source import PolarsDataSourceConfig

        config = PolarsDataSourceConfig(max_rows=3, sample_size=2)
        source = PolarsDataSource(sample_polars_df, config)
        safe = source.get_safe_sample()

        assert safe.row_count <= 2

    def test_check_size_limits(self, sample_polars_df):
        """Test check_size_limits raises error."""
        from truthound.datasources.polars_source import PolarsDataSourceConfig

        config = PolarsDataSourceConfig(max_rows=3)
        source = PolarsDataSource(sample_polars_df, config)

        with pytest.raises(DataSourceSizeError):
            source.check_size_limits()


# =============================================================================
# Column Type Tests
# =============================================================================


class TestColumnTypes:
    """Tests for column type utilities."""

    def test_get_column_type(self, sample_polars_df):
        """Test getting specific column type."""
        source = PolarsDataSource(sample_polars_df)

        assert source.get_column_type("id") == ColumnType.INTEGER
        assert source.get_column_type("name") == ColumnType.STRING
        assert source.get_column_type("nonexistent") is None

    def test_get_datetime_columns(self):
        """Test getting datetime columns."""
        from datetime import date
        df = pl.DataFrame({
            "date": [date(2021, 1, 1), date(2021, 1, 2)],
            "value": [1, 2],
        })
        source = PolarsDataSource(df)
        dt_cols = source.get_datetime_columns()

        assert "date" in dt_cols
        assert "value" not in dt_cols


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager usage."""

    def test_polars_context_manager(self, sample_polars_df):
        """Test using PolarsDataSource as context manager."""
        with PolarsDataSource(sample_polars_df) as source:
            assert source.row_count == 5

    def test_file_context_manager(self, temp_csv_file):
        """Test using FileDataSource as context manager."""
        with FileDataSource(temp_csv_file) as source:
            assert source.row_count == 5
