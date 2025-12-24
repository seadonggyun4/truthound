"""Pandas execution engine implementation.

This module provides an execution engine for Pandas DataFrames,
enabling validation on Pandas data without requiring conversion
to Polars for simple operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from truthound.execution.base import (
    BaseExecutionEngine,
    ExecutionConfig,
    ExecutionSizeError,
)
from truthound.execution._protocols import AggregationType

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl


def _check_pandas_available() -> None:
    """Check if pandas is available."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise ImportError(
            "pandas is required for PandasExecutionEngine. "
            "Install with: pip install pandas"
        )


class PandasExecutionEngine(BaseExecutionEngine[ExecutionConfig]):
    """Execution engine based on Pandas DataFrame.

    This engine provides native Pandas operations where possible,
    with fallback to Polars for complex operations.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> engine = PandasExecutionEngine(df)
        >>> null_count = engine.count_nulls("a")
    """

    engine_type = "pandas"

    def __init__(
        self,
        data: "pd.DataFrame",
        config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize Pandas execution engine.

        Args:
            data: Pandas DataFrame.
            config: Optional configuration.
        """
        _check_pandas_available()
        super().__init__(config)

        self._df = data
        self._cached_polars_lf: "pl.LazyFrame | None" = None

    @property
    def dataframe(self) -> "pd.DataFrame":
        """Get the underlying DataFrame."""
        return self._df

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count total number of rows."""
        return len(self._df)

    def get_columns(self) -> list[str]:
        """Get list of column names."""
        return list(self._df.columns)

    def count_nulls(self, column: str) -> int:
        """Count null values in a column."""
        return int(self._df[column].isna().sum())

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls for all columns."""
        return self._df.isna().sum().to_dict()

    def count_distinct(self, column: str) -> int:
        """Count distinct values in a column."""
        return int(self._df[column].nunique(dropna=False))

    # -------------------------------------------------------------------------
    # Statistical Operations
    # -------------------------------------------------------------------------

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get comprehensive statistics for a numeric column."""
        series = self._df[column]

        return {
            "count": int(series.count()),
            "null_count": int(series.isna().sum()),
            "mean": float(series.mean()) if series.count() > 0 else None,
            "std": float(series.std()) if series.count() > 1 else None,
            "min": float(series.min()) if series.count() > 0 else None,
            "max": float(series.max()) if series.count() > 0 else None,
            "median": float(series.median()) if series.count() > 0 else None,
            "q25": float(series.quantile(0.25)) if series.count() > 0 else None,
            "q75": float(series.quantile(0.75)) if series.count() > 0 else None,
            "sum": float(series.sum()) if series.count() > 0 else None,
        }

    def get_quantiles(
        self,
        column: str,
        quantiles: list[float],
    ) -> list[float]:
        """Get specific quantiles for a column."""
        return [float(self._df[column].quantile(q)) for q in quantiles]

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value frequency counts for a column."""
        counts = self._df[column].value_counts(dropna=False)
        if limit:
            counts = counts.head(limit)
        return counts.to_dict()

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        aggregations: dict[str, AggregationType],
    ) -> dict[str, Any]:
        """Perform multiple aggregations."""
        results = {}

        for col, agg_type in aggregations.items():
            key = f"{col}_{agg_type.value}"
            results[key] = self.aggregate_column(col, agg_type)

        return results

    def aggregate_column(
        self,
        column: str,
        agg_type: AggregationType,
    ) -> Any:
        """Perform a single aggregation on a column."""
        series = self._df[column]

        agg_map = {
            AggregationType.COUNT: lambda s: int(s.count()),
            AggregationType.SUM: lambda s: s.sum(),
            AggregationType.MEAN: lambda s: s.mean(),
            AggregationType.MEDIAN: lambda s: s.median(),
            AggregationType.MIN: lambda s: s.min(),
            AggregationType.MAX: lambda s: s.max(),
            AggregationType.STD: lambda s: s.std(),
            AggregationType.VAR: lambda s: s.var(),
            AggregationType.FIRST: lambda s: s.iloc[0] if len(s) > 0 else None,
            AggregationType.LAST: lambda s: s.iloc[-1] if len(s) > 0 else None,
            AggregationType.COUNT_DISTINCT: lambda s: int(s.nunique(dropna=False)),
            AggregationType.NULL_COUNT: lambda s: int(s.isna().sum()),
        }

        return agg_map[agg_type](series)

    # -------------------------------------------------------------------------
    # Value Operations
    # -------------------------------------------------------------------------

    def get_distinct_values(
        self,
        column: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get distinct values from a column."""
        unique = self._df[column].unique().tolist()
        if limit:
            return unique[:limit]
        return unique

    def get_column_values(
        self,
        column: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        """Get values from a column."""
        series = self._df[column]
        if offset > 0:
            series = series.iloc[offset:]
        if limit:
            series = series.head(limit)
        return series.tolist()

    def get_sample_values(self, column: str, n: int = 5) -> list[Any]:
        """Get sample non-null values from a column."""
        non_null = self._df[column].dropna()
        return non_null.head(n).tolist()

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching a regex pattern."""
        return int(
            self._df[column]
            .astype(str)
            .str.contains(pattern, regex=True, na=False)
            .sum()
        )

    def count_not_matching_regex(self, column: str, pattern: str) -> int:
        """Count values not matching a regex pattern."""
        return int(
            ~self._df[column]
            .astype(str)
            .str.contains(pattern, regex=True, na=True)
        ).sum()

    # -------------------------------------------------------------------------
    # Range Operations
    # -------------------------------------------------------------------------

    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values within a range."""
        series = self._df[column]
        mask = series.notna()

        if min_value is not None:
            if inclusive:
                mask = mask & (series >= min_value)
            else:
                mask = mask & (series > min_value)

        if max_value is not None:
            if inclusive:
                mask = mask & (series <= max_value)
            else:
                mask = mask & (series < max_value)

        return int(mask.sum())

    def count_outside_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
    ) -> int:
        """Count values outside a range."""
        total = self.count_rows()
        in_range = self.count_in_range(column, min_value, max_value)
        return total - in_range

    # -------------------------------------------------------------------------
    # Set Operations
    # -------------------------------------------------------------------------

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are in a set."""
        return int(self._df[column].isin(values).sum())

    def count_not_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are not in a set."""
        return int((~self._df[column].isin(values)).sum())

    # -------------------------------------------------------------------------
    # Duplicate Operations
    # -------------------------------------------------------------------------

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicate rows based on specified columns."""
        return int(self._df.duplicated(subset=columns).sum())

    def get_duplicate_values(
        self,
        columns: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sample duplicate values."""
        duplicated = self._df[self._df.duplicated(subset=columns, keep=False)]
        grouped = duplicated.groupby(columns).size().reset_index(name="_count")
        grouped = grouped.sort_values("_count", ascending=False).head(limit)
        return grouped.drop("_count", axis=1).to_dict("records")

    # -------------------------------------------------------------------------
    # Filter Operations
    # -------------------------------------------------------------------------

    def filter_by_condition(self, condition: str) -> "PandasExecutionEngine":
        """Create a new engine with filtered data using query syntax."""
        try:
            filtered = self._df.query(condition)
            return PandasExecutionEngine(filtered, self._config)
        except Exception as e:
            raise NotImplementedError(
                f"Condition '{condition}' cannot be parsed by pandas.query(): {e}"
            )

    def count_matching(self, condition: str) -> int:
        """Count rows matching a condition."""
        try:
            return len(self._df.query(condition))
        except Exception:
            raise NotImplementedError(
                f"Condition '{condition}' cannot be parsed by pandas.query()"
            )

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame."""
        if self._cached_polars_lf is None:
            import polars as pl
            self._cached_polars_lf = pl.from_pandas(self._df).lazy()
        return self._cached_polars_lf

    def to_pandas_dataframe(self) -> "pd.DataFrame":
        """Return the underlying DataFrame."""
        return self._df

    def to_numpy(self, columns: list[str] | None = None) -> "np.ndarray":
        """Convert to numpy array."""
        import numpy as np

        row_count = self.count_rows()
        if row_count > self._config.max_rows_for_numpy:
            raise ExecutionSizeError(
                operation="to_numpy",
                current_size=row_count,
                max_size=self._config.max_rows_for_numpy,
            )

        if columns is None:
            # Get numeric columns
            numeric_df = self._df.select_dtypes(include=[np.number])
            columns = list(numeric_df.columns)

        if not columns:
            return np.array([])

        return self._df[columns].to_numpy()

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "PandasExecutionEngine":
        """Create a new engine with sampled data."""
        row_count = self.count_rows()

        if row_count <= n:
            return PandasExecutionEngine(self._df, self._config)

        sampled = self._df.sample(n=n, random_state=seed)
        return PandasExecutionEngine(sampled, self._config)

    # -------------------------------------------------------------------------
    # Additional Pandas Operations
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
            "dtypes": self._df.dtypes.astype(str).to_dict(),
            "memory_usage": self._df.memory_usage(deep=True).sum(),
        }
