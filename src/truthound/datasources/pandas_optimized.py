"""Optimized Pandas to Polars conversion.

This module provides memory-efficient and performance-optimized conversion
from large Pandas DataFrames to Polars LazyFrames.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceError,
    pandas_dtype_to_column_type,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from truthound.execution.polars_engine import PolarsExecutionEngine


# =============================================================================
# Exceptions
# =============================================================================


class PandasOptimizationError(DataSourceError):
    """Error during Pandas optimization."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OptimizedPandasConfig(DataSourceConfig):
    """Configuration for optimized Pandas data sources.

    Attributes:
        chunk_size: Rows per chunk for chunked conversion.
        optimize_dtypes: Whether to optimize dtypes before conversion.
        use_categorical: Convert low-cardinality strings to categorical.
        categorical_threshold: Maximum unique values for categorical conversion.
        downcast_int: Downcast integers to smallest possible type.
        downcast_float: Downcast floats (64-bit to 32-bit if lossless).
        gc_after_chunk: Run garbage collection after each chunk.
        parallel: Use parallel processing for conversion.
        n_threads: Number of threads for parallel processing.
        preserve_index: Whether to preserve DataFrame index.
        copy_data: Whether to copy data during optimization.
    """

    chunk_size: int = 100_000
    optimize_dtypes: bool = True
    use_categorical: bool = True
    categorical_threshold: int = 50
    downcast_int: bool = True
    downcast_float: bool = True
    gc_after_chunk: bool = True
    parallel: bool = False
    n_threads: int = 4
    preserve_index: bool = False
    copy_data: bool = False


# =============================================================================
# DataFrame Optimizer
# =============================================================================


class DataFrameOptimizer:
    """Optimize Pandas DataFrame memory usage.

    Applies dtype optimizations to reduce memory footprint before
    conversion to Polars.

    Example:
        >>> optimizer = DataFrameOptimizer()
        >>> optimized_df = optimizer.optimize(large_df)
        >>> print(f"Saved {optimizer.memory_saved_bytes} bytes")
    """

    def __init__(
        self,
        use_categorical: bool = True,
        categorical_threshold: int = 50,
        downcast_int: bool = True,
        downcast_float: bool = True,
    ) -> None:
        """Initialize optimizer.

        Args:
            use_categorical: Convert low-cardinality strings to categorical.
            categorical_threshold: Max unique values for categorical.
            downcast_int: Downcast integers.
            downcast_float: Downcast floats.
        """
        self._use_categorical = use_categorical
        self._categorical_threshold = categorical_threshold
        self._downcast_int = downcast_int
        self._downcast_float = downcast_float

        self._original_memory: int = 0
        self._optimized_memory: int = 0

    @property
    def memory_saved_bytes(self) -> int:
        """Get bytes saved by optimization."""
        return self._original_memory - self._optimized_memory

    @property
    def memory_reduction_pct(self) -> float:
        """Get memory reduction percentage."""
        if self._original_memory == 0:
            return 0.0
        return (self.memory_saved_bytes / self._original_memory) * 100

    def optimize(self, df: "pd.DataFrame", inplace: bool = False) -> "pd.DataFrame":
        """Optimize DataFrame memory usage.

        Args:
            df: DataFrame to optimize.
            inplace: Modify DataFrame in-place (not recommended).

        Returns:
            Optimized DataFrame.
        """
        import pandas as pd
        import numpy as np

        self._original_memory = int(df.memory_usage(deep=True).sum())

        if not inplace:
            df = df.copy()

        for col in df.columns:
            dtype = df[col].dtype

            # Optimize integers
            if self._downcast_int and np.issubdtype(dtype, np.integer):
                df[col] = pd.to_numeric(df[col], downcast="integer")

            # Optimize floats
            elif self._downcast_float and np.issubdtype(dtype, np.floating):
                # Check if we can downcast without losing precision
                if df[col].notna().any():
                    min_val = df[col].min()
                    max_val = df[col].max()

                    # Check if fits in float32
                    if (
                        min_val >= np.finfo(np.float32).min
                        and max_val <= np.finfo(np.float32).max
                    ):
                        # Check precision loss
                        float32_vals = df[col].astype(np.float32)
                        if np.allclose(df[col].fillna(0), float32_vals.fillna(0)):
                            df[col] = float32_vals

            # Convert low-cardinality strings to categorical
            elif self._use_categorical and dtype == object:
                nunique = df[col].nunique()
                if nunique <= self._categorical_threshold:
                    df[col] = df[col].astype("category")

        self._optimized_memory = int(df.memory_usage(deep=True).sum())
        return df

    def get_optimization_report(self, df: "pd.DataFrame") -> dict[str, Any]:
        """Analyze potential optimizations without applying them.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dict with optimization suggestions per column.
        """
        import numpy as np
        import pandas as pd

        report: dict[str, Any] = {
            "total_memory_bytes": int(df.memory_usage(deep=True).sum()),
            "columns": {},
        }

        for col in df.columns:
            dtype = df[col].dtype
            col_memory = int(df[col].memory_usage(deep=True))

            col_report = {
                "dtype": str(dtype),
                "memory_bytes": col_memory,
                "suggestions": [],
            }

            # Handle categorical dtype first (skip optimization check)
            if isinstance(dtype, pd.CategoricalDtype):
                col_report["suggestions"].append("Already optimized (categorical)")

            elif np.issubdtype(dtype, np.integer):
                col_report["suggestions"].append("Downcast integer")

            elif np.issubdtype(dtype, np.floating):
                if df[col].notna().any():
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if (
                        min_val >= np.finfo(np.float32).min
                        and max_val <= np.finfo(np.float32).max
                    ):
                        col_report["suggestions"].append("Downcast to float32")

            elif dtype == object:
                nunique = df[col].nunique()
                if nunique <= self._categorical_threshold:
                    col_report["suggestions"].append(
                        f"Convert to categorical (nunique={nunique})"
                    )

            report["columns"][col] = col_report

        return report


# =============================================================================
# Optimized Pandas Data Source
# =============================================================================


class OptimizedPandasDataSource(BaseDataSource[OptimizedPandasConfig]):
    """Memory-efficient Pandas data source with optimized Polars conversion.

    Provides chunked, memory-aware conversion from large Pandas DataFrames
    to Polars LazyFrames with optional dtype optimization.

    Example:
        >>> import pandas as pd
        >>> # Large DataFrame with 10M rows
        >>> df = pd.read_csv("large_file.csv")
        >>>
        >>> # Create optimized source
        >>> source = OptimizedPandasDataSource(df, OptimizedPandasConfig(
        ...     chunk_size=100_000,
        ...     optimize_dtypes=True,
        ...     gc_after_chunk=True,
        ... ))
        >>>
        >>> # Memory-efficient conversion
        >>> lf = source.to_polars_streaming()
        >>>
        >>> # Or iterate over chunks
        >>> for chunk_lf in source.iter_polars_chunks():
        ...     results = validate_chunk(chunk_lf)
    """

    source_type = "pandas_optimized"

    def __init__(
        self,
        data: "pd.DataFrame",
        config: OptimizedPandasConfig | None = None,
    ) -> None:
        """Initialize optimized Pandas data source.

        Args:
            data: Pandas DataFrame.
            config: Optional configuration.
        """
        try:
            import pandas  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is required for OptimizedPandasDataSource. "
                "Install with: pip install pandas"
            )

        super().__init__(config)

        self._optimizer = DataFrameOptimizer(
            use_categorical=self._config.use_categorical,
            categorical_threshold=self._config.categorical_threshold,
            downcast_int=self._config.downcast_int,
            downcast_float=self._config.downcast_float,
        )

        # Optimize if configured
        if self._config.optimize_dtypes:
            self._df = self._optimizer.optimize(data, inplace=False)
        elif self._config.copy_data:
            self._df = data.copy()
        else:
            self._df = data

        self._cached_row_count = len(self._df)

    @classmethod
    def _default_config(cls) -> OptimizedPandasConfig:
        return OptimizedPandasConfig()

    @property
    def dataframe(self) -> "pd.DataFrame":
        """Get the underlying DataFrame."""
        return self._df

    @property
    def optimizer(self) -> DataFrameOptimizer:
        """Get the optimizer instance."""
        return self._optimizer

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema."""
        if self._cached_schema is None:
            self._cached_schema = {
                col: pandas_dtype_to_column_type(dtype)
                for col, dtype in self._df.dtypes.items()
            }
        return self._cached_schema

    @property
    def row_count(self) -> int | None:
        """Get row count."""
        return self._cached_row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get capabilities."""
        return {
            DataSourceCapability.SAMPLING,
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.ROW_COUNT,
            DataSourceCapability.STREAMING,
        }

    # -------------------------------------------------------------------------
    # Execution Engine
    # -------------------------------------------------------------------------

    def get_execution_engine(self) -> "PolarsExecutionEngine":
        """Get a Polars execution engine.

        Uses streaming conversion for memory efficiency.
        """
        from truthound.execution.polars_engine import PolarsExecutionEngine

        lf = self.to_polars_streaming()
        return PolarsExecutionEngine(lf)

    # -------------------------------------------------------------------------
    # Chunked Conversion
    # -------------------------------------------------------------------------

    def iter_polars_chunks(
        self, chunk_size: int | None = None
    ) -> Iterator["pl.LazyFrame"]:
        """Iterate over DataFrame in Polars LazyFrame chunks.

        Memory-efficient iteration that converts chunks one at a time
        and optionally triggers garbage collection.

        Args:
            chunk_size: Rows per chunk (uses config default if None).

        Yields:
            Polars LazyFrame chunks.

        Example:
            >>> for chunk in source.iter_polars_chunks():
            ...     # Process each chunk
            ...     results = validate(chunk)
            ...     process(results)
        """
        import polars as pl

        chunk_size = chunk_size or self._config.chunk_size
        n_rows = len(self._df)

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = self._df.iloc[start:end]

            # Convert chunk to Polars
            polars_df = pl.from_pandas(chunk)

            # Optionally preserve index
            if self._config.preserve_index and chunk.index.name:
                polars_df = polars_df.with_columns(
                    pl.lit(chunk.index.tolist()).alias(str(chunk.index.name))
                )

            yield polars_df.lazy()

            # Optional garbage collection
            if self._config.gc_after_chunk:
                gc.collect()

    def to_polars_streaming(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame using streaming/chunked approach.

        More memory-efficient than direct conversion for large DataFrames.

        Returns:
            Polars LazyFrame.
        """
        import polars as pl

        # For small DataFrames, use direct conversion
        if len(self._df) <= self._config.chunk_size:
            return pl.from_pandas(self._df).lazy()

        # Collect chunks and concatenate
        chunks = list(self.iter_polars_chunks())
        return pl.concat(chunks)

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame.

        Uses streaming conversion for memory efficiency.
        """
        return self.to_polars_streaming()

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "OptimizedPandasDataSource":
        """Create sampled data source."""
        if self.row_count and self.row_count <= n:
            return self

        sampled = self._df.sample(n=n, random_state=seed)

        config = OptimizedPandasConfig(
            name=f"{self.name}_sample",
            chunk_size=self._config.chunk_size,
            optimize_dtypes=False,  # Already optimized
            copy_data=False,
        )
        return OptimizedPandasDataSource(sampled, config)

    def validate_connection(self) -> bool:
        """Validate DataFrame is accessible."""
        try:
            _ = len(self._df)
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Memory Information
    # -------------------------------------------------------------------------

    def get_memory_usage(self) -> dict[str, Any]:
        """Get detailed memory usage information.

        Returns:
            Dict with memory usage details.
        """
        return {
            "total_bytes": int(self._df.memory_usage(deep=True).sum()),
            "optimization_saved_bytes": self._optimizer.memory_saved_bytes,
            "optimization_reduction_pct": self._optimizer.memory_reduction_pct,
            "row_count": len(self._df),
            "column_count": len(self._df.columns),
            "per_column": {
                col: int(self._df[col].memory_usage(deep=True))
                for col in self._df.columns
            },
        }

    def get_optimization_report(self) -> dict[str, Any]:
        """Get optimization suggestions report.

        Returns:
            Optimization suggestions per column.
        """
        return self._optimizer.get_optimization_report(self._df)


# =============================================================================
# Utility Functions
# =============================================================================


def optimize_pandas_to_polars(
    df: "pd.DataFrame",
    chunk_size: int = 100_000,
    optimize_dtypes: bool = True,
    gc_after_chunk: bool = True,
) -> "pl.LazyFrame":
    """Utility function for optimized chunked Pandas to Polars conversion.

    Args:
        df: Pandas DataFrame to convert.
        chunk_size: Rows per chunk.
        optimize_dtypes: Whether to optimize dtypes first.
        gc_after_chunk: Run garbage collection after each chunk.

    Returns:
        Polars LazyFrame.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("large_file.csv")
        >>> lf = optimize_pandas_to_polars(df, chunk_size=50_000)
    """
    config = OptimizedPandasConfig(
        chunk_size=chunk_size,
        optimize_dtypes=optimize_dtypes,
        gc_after_chunk=gc_after_chunk,
    )
    source = OptimizedPandasDataSource(df, config)
    return source.to_polars_streaming()


def estimate_polars_memory(df: "pd.DataFrame") -> int:
    """Estimate memory usage after Polars conversion.

    Polars typically uses less memory than Pandas due to:
    - No Python object overhead for strings
    - More efficient categorical representation
    - Native Arrow memory format

    Args:
        df: Pandas DataFrame.

    Returns:
        Estimated bytes after Polars conversion.
    """
    import numpy as np

    pandas_memory = df.memory_usage(deep=True).sum()

    # Polars typically uses 40-60% less memory for string-heavy DataFrames
    string_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    string_ratio = len(string_cols) / max(len(df.columns), 1)
    numeric_ratio = len(numeric_cols) / max(len(df.columns), 1)

    # Estimate: strings save ~50%, numerics save ~10%
    estimated_savings = (string_ratio * 0.5) + (numeric_ratio * 0.1)
    estimated_memory = int(pandas_memory * (1 - estimated_savings))

    return estimated_memory


def get_optimal_chunk_size(
    df: "pd.DataFrame",
    target_memory_mb: int = 256,
) -> int:
    """Calculate optimal chunk size for memory-constrained conversion.

    Args:
        df: DataFrame to analyze.
        target_memory_mb: Target memory per chunk in MB.

    Returns:
        Recommended chunk size (rows).
    """
    memory_per_row = df.memory_usage(deep=True).sum() / max(len(df), 1)
    target_bytes = target_memory_mb * 1024 * 1024

    optimal_size = int(target_bytes / max(memory_per_row, 1))

    # Clamp to reasonable bounds
    return max(1000, min(optimal_size, 1_000_000))
