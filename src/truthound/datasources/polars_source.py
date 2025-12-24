"""Polars data source implementation.

This module provides data sources for Polars DataFrames and LazyFrames,
as well as file-based data sources (CSV, JSON, Parquet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceError,
    polars_to_column_type,
)
from truthound.execution.polars_engine import PolarsExecutionEngine


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PolarsDataSourceConfig(DataSourceConfig):
    """Configuration for Polars data sources.

    Attributes:
        rechunk: Whether to rechunk data for better performance.
        streaming: Whether to use streaming mode for large files.
    """

    rechunk: bool = False
    streaming: bool = False


@dataclass
class FileDataSourceConfig(PolarsDataSourceConfig):
    """Configuration for file-based data sources.

    Attributes:
        infer_schema_length: Number of rows to infer schema from.
        ignore_errors: Whether to ignore parsing errors.
        encoding: File encoding (for CSV).
        separator: Column separator (for CSV).
    """

    infer_schema_length: int = 10000
    ignore_errors: bool = False
    encoding: str = "utf8"  # Polars requires 'utf8' not 'utf-8'
    separator: str = ","


# =============================================================================
# Polars DataFrame/LazyFrame Data Source
# =============================================================================


class PolarsDataSource(BaseDataSource[PolarsDataSourceConfig]):
    """Data source for Polars DataFrame or LazyFrame.

    This is the primary data source for in-memory Polars data.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> source = PolarsDataSource(df)
        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
        3
    """

    source_type = "polars"

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: PolarsDataSourceConfig | None = None,
    ) -> None:
        """Initialize Polars data source.

        Args:
            data: Polars DataFrame or LazyFrame.
            config: Optional configuration.
        """
        super().__init__(config)

        if isinstance(data, pl.DataFrame):
            self._lf = data.lazy()
            self._cached_row_count = len(data)
        else:
            self._lf = data
            self._cached_row_count = None

        self._polars_schema = self._lf.collect_schema()

    @classmethod
    def _default_config(cls) -> PolarsDataSourceConfig:
        return PolarsDataSourceConfig()

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema as column name to type mapping."""
        if self._cached_schema is None:
            self._cached_schema = {
                col: polars_to_column_type(dtype)
                for col, dtype in self._polars_schema.items()
            }
        return self._cached_schema

    @property
    def polars_schema(self) -> pl.Schema:
        """Get the native Polars schema."""
        return self._polars_schema

    @property
    def row_count(self) -> int | None:
        """Get row count."""
        if self._cached_row_count is None:
            self._cached_row_count = self._lf.select(pl.len()).collect().item()
        return self._cached_row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.LAZY_EVALUATION,
            DataSourceCapability.SAMPLING,
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.ROW_COUNT,
        }

    def get_execution_engine(self) -> PolarsExecutionEngine:
        """Get a Polars execution engine."""
        # ExecutionEngine uses its own config type
        return PolarsExecutionEngine(self._lf)

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "PolarsDataSource":
        """Create a new data source with sampled data."""
        row_count = self.row_count or 0

        if row_count <= n:
            return self

        fraction = min(n / row_count, 1.0)
        sampled = self._lf.collect().sample(fraction=fraction, seed=seed).lazy()

        config = PolarsDataSourceConfig(
            name=f"{self.name}_sample",
            max_rows=self._config.max_rows,
            sample_size=n,
        )
        return PolarsDataSource(sampled, config)

    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Get the underlying LazyFrame."""
        return self._lf

    def validate_connection(self) -> bool:
        """Validate by checking schema access."""
        try:
            _ = self._lf.collect_schema()
            return True
        except Exception:
            return False


# =============================================================================
# File-Based Data Sources
# =============================================================================


class FileDataSource(BaseDataSource[FileDataSourceConfig]):
    """Data source for file-based data (CSV, JSON, Parquet).

    Supports lazy loading using Polars scan functions.

    Example:
        >>> source = FileDataSource("data.csv")
        >>> print(source.columns)
        ['id', 'name', 'value']

        >>> source = FileDataSource("data.parquet")
        >>> engine = source.get_execution_engine()
    """

    source_type = "file"

    # Supported file extensions and their loaders
    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".ndjson": "ndjson",
        ".jsonl": "ndjson",
    }

    def __init__(
        self,
        path: str | Path,
        config: FileDataSourceConfig | None = None,
    ) -> None:
        """Initialize file data source.

        Args:
            path: Path to the data file.
            config: Optional configuration.
        """
        super().__init__(config)

        self._path = Path(path)
        if not self._path.exists():
            raise DataSourceError(f"File not found: {self._path}")

        self._file_type = self._detect_file_type()
        self._lf = self._create_lazyframe()
        self._polars_schema = self._lf.collect_schema()

    @classmethod
    def _default_config(cls) -> FileDataSourceConfig:
        return FileDataSourceConfig()

    @property
    def name(self) -> str:
        """Get the data source name (file name)."""
        if self._config.name:
            return self._config.name
        return self._path.name

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._path

    @property
    def file_type(self) -> str:
        """Get the detected file type."""
        return self._file_type

    def _detect_file_type(self) -> str:
        """Detect file type from extension."""
        suffix = self._path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise DataSourceError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        return self.SUPPORTED_EXTENSIONS[suffix]

    def _create_lazyframe(self) -> pl.LazyFrame:
        """Create LazyFrame from file."""
        path_str = str(self._path)
        cfg = self._config

        if self._file_type == "csv":
            return pl.scan_csv(
                path_str,
                separator=cfg.separator,
                encoding=cfg.encoding,
                infer_schema_length=cfg.infer_schema_length,
                ignore_errors=cfg.ignore_errors,
                rechunk=cfg.rechunk,
            )
        elif self._file_type == "parquet":
            return pl.scan_parquet(path_str, rechunk=cfg.rechunk)
        elif self._file_type == "json":
            # JSON doesn't have a scan method, read eagerly
            return pl.read_json(path_str).lazy()
        elif self._file_type == "ndjson":
            return pl.scan_ndjson(
                path_str,
                infer_schema_length=cfg.infer_schema_length,
                ignore_errors=cfg.ignore_errors,
                rechunk=cfg.rechunk,
            )
        else:
            raise DataSourceError(f"Unknown file type: {self._file_type}")

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema."""
        if self._cached_schema is None:
            self._cached_schema = {
                col: polars_to_column_type(dtype)
                for col, dtype in self._polars_schema.items()
            }
        return self._cached_schema

    @property
    def row_count(self) -> int | None:
        """Get row count (requires scan for files)."""
        if self._cached_row_count is None:
            self._cached_row_count = self._lf.select(pl.len()).collect().item()
        return self._cached_row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        caps = {
            DataSourceCapability.LAZY_EVALUATION,
            DataSourceCapability.SAMPLING,
            DataSourceCapability.SCHEMA_INFERENCE,
        }
        # Parquet files support efficient row count
        if self._file_type == "parquet":
            caps.add(DataSourceCapability.ROW_COUNT)
        return caps

    def get_execution_engine(self) -> PolarsExecutionEngine:
        """Get a Polars execution engine."""
        # ExecutionEngine uses its own config type
        return PolarsExecutionEngine(self._lf)

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "FileDataSource":
        """Create a new data source with sampled data.

        Note: This loads data into memory for sampling.
        """
        row_count = self.row_count or 0

        if row_count <= n:
            return self

        # For sampling, we need to collect and re-wrap
        fraction = min(n / row_count, 1.0)
        sampled_df = self._lf.collect().sample(fraction=fraction, seed=seed)

        # Return a PolarsDataSource instead since we've loaded into memory
        config = PolarsDataSourceConfig(
            name=f"{self.name}_sample",
            max_rows=self._config.max_rows,
            sample_size=n,
        )
        return PolarsDataSource(sampled_df, config)  # type: ignore

    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Get the underlying LazyFrame."""
        return self._lf

    def validate_connection(self) -> bool:
        """Validate by checking file exists and is readable."""
        try:
            return self._path.exists() and self._path.is_file()
        except Exception:
            return False


# =============================================================================
# Dictionary Data Source
# =============================================================================


class DictDataSource(PolarsDataSource):
    """Data source for Python dictionaries.

    Converts a dictionary to a Polars DataFrame.

    Example:
        >>> data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        >>> source = DictDataSource(data)
    """

    source_type = "dict"

    def __init__(
        self,
        data: dict[str, list[Any]],
        config: PolarsDataSourceConfig | None = None,
    ) -> None:
        """Initialize dictionary data source.

        Args:
            data: Dictionary with column names as keys and lists as values.
            config: Optional configuration.
        """
        df = pl.DataFrame(data)
        super().__init__(df, config)
