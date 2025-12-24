"""Pandas data source implementation.

This module provides data sources for Pandas DataFrames,
enabling native Pandas operations while maintaining compatibility
with the Truthound validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    pandas_dtype_to_column_type,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from truthound.execution.pandas_engine import PandasExecutionEngine


def _check_pandas_available() -> None:
    """Check if pandas is available."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise ImportError(
            "pandas is required for PandasDataSource. "
            "Install with: pip install pandas"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PandasDataSourceConfig(DataSourceConfig):
    """Configuration for Pandas data sources.

    Attributes:
        copy_data: Whether to copy the DataFrame on initialization.
        use_polars_fallback: Whether to use Polars for unsupported operations.
    """

    copy_data: bool = False
    use_polars_fallback: bool = True


# =============================================================================
# Pandas DataFrame Data Source
# =============================================================================


class PandasDataSource(BaseDataSource[PandasDataSourceConfig]):
    """Data source for Pandas DataFrame.

    This data source enables validation on Pandas DataFrames with
    native Pandas operations where possible.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> source = PandasDataSource(df)
        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
        3
    """

    source_type = "pandas"

    def __init__(
        self,
        data: "pd.DataFrame",
        config: PandasDataSourceConfig | None = None,
    ) -> None:
        """Initialize Pandas data source.

        Args:
            data: Pandas DataFrame.
            config: Optional configuration.
        """
        _check_pandas_available()
        super().__init__(config)

        if self._config.copy_data:
            self._df = data.copy()
        else:
            self._df = data

        self._cached_row_count = len(self._df)

    @classmethod
    def _default_config(cls) -> PandasDataSourceConfig:
        return PandasDataSourceConfig()

    @property
    def dataframe(self) -> "pd.DataFrame":
        """Get the underlying DataFrame."""
        return self._df

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema as column name to type mapping."""
        if self._cached_schema is None:
            self._cached_schema = {
                col: pandas_dtype_to_column_type(dtype)
                for col, dtype in self._df.dtypes.items()
            }
        return self._cached_schema

    @property
    def pandas_dtypes(self) -> dict[str, Any]:
        """Get the native Pandas dtypes."""
        return self._df.dtypes.to_dict()

    @property
    def row_count(self) -> int | None:
        """Get row count."""
        return self._cached_row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.SAMPLING,
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.ROW_COUNT,
        }

    def get_execution_engine(self) -> "PandasExecutionEngine":
        """Get a Pandas execution engine."""
        from truthound.execution.pandas_engine import PandasExecutionEngine
        return PandasExecutionEngine(self._df)

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "PandasDataSource":
        """Create a new data source with sampled data."""
        row_count = self.row_count or 0

        if row_count <= n:
            return self

        sampled = self._df.sample(n=n, random_state=seed)

        config = PandasDataSourceConfig(
            name=f"{self.name}_sample",
            max_rows=self._config.max_rows,
            sample_size=n,
            copy_data=False,
        )
        return PandasDataSource(sampled, config)

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame."""
        import polars as pl

        # Check size limits
        self.check_size_limits()

        return pl.from_pandas(self._df).lazy()

    def to_pandas_dataframe(self) -> "pd.DataFrame":
        """Get the underlying DataFrame."""
        return self._df

    def validate_connection(self) -> bool:
        """Validate by checking DataFrame is accessible."""
        try:
            _ = len(self._df)
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Pandas-specific Methods
    # -------------------------------------------------------------------------

    def describe(self) -> "pd.DataFrame":
        """Get pandas describe() output."""
        return self._df.describe()

    def info(self) -> dict[str, Any]:
        """Get DataFrame info."""
        import io
        buffer = io.StringIO()
        self._df.info(buf=buffer)
        return {
            "shape": self._df.shape,
            "columns": list(self._df.columns),
            "dtypes": {str(k): str(v) for k, v in self._df.dtypes.items()},
            "memory_usage": int(self._df.memory_usage(deep=True).sum()),
        }

    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return int(self._df.memory_usage(deep=True).sum())

    def head(self, n: int = 5) -> "pd.DataFrame":
        """Get first n rows."""
        return self._df.head(n)

    def tail(self, n: int = 5) -> "pd.DataFrame":
        """Get last n rows."""
        return self._df.tail(n)
