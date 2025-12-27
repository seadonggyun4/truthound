"""Mixins for distributed execution engines.

This module provides reusable mixins that add common functionality
to distributed execution engines. These mixins follow the composition
pattern to enable feature extension without deep inheritance hierarchies.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Engine Composition                            │
    │                                                                  │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │            BaseDistributedEngine                          │  │
    │   │                      +                                    │  │
    │   │    ┌─────────────────────────────────────────────┐       │  │
    │   │    │  StatisticalMixin  │  DataQualityMixin      │       │  │
    │   │    │  SamplingMixin     │  PartitioningMixin     │       │  │
    │   │    │  IOOperationsMixin │  ValidationMixin       │       │  │
    │   │    └─────────────────────────────────────────────┘       │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Example:
    >>> class MyEngine(BaseDistributedEngine, StatisticalMixin, DataQualityMixin):
    ...     pass
    >>>
    >>> engine = MyEngine(data)
    >>> engine.get_percentiles("price", [0.25, 0.5, 0.75])
    >>> engine.check_data_quality()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for Mixins
# =============================================================================


@runtime_checkable
class DistributedEngineProtocol(Protocol):
    """Protocol defining the interface mixins expect from engines."""

    def count_rows(self) -> int:
        """Count total rows."""
        ...

    def get_columns(self) -> list[str]:
        """Get column names."""
        ...

    def count_nulls(self, column: str) -> int:
        """Count nulls in a column."""
        ...

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls in all columns."""
        ...

    def count_distinct(self, column: str) -> int:
        """Count distinct values."""
        ...

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get column statistics."""
        ...

    def _cache_key(self, *args: Any) -> str:
        """Generate cache key."""
        ...

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value."""
        ...

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        ...


# =============================================================================
# Statistical Mixin
# =============================================================================


class StatisticalMixin:
    """Mixin providing advanced statistical operations.

    This mixin adds statistical analysis capabilities beyond
    the basic aggregations provided by the base engine.
    """

    def get_percentiles(
        self: DistributedEngineProtocol,
        column: str,
        percentiles: list[float] | None = None,
    ) -> dict[str, float]:
        """Calculate percentiles for a column.

        Args:
            column: Column name.
            percentiles: List of percentiles (0-1). Defaults to quartiles.

        Returns:
            Dictionary mapping percentile names to values.
        """
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]

        cache_key = self._cache_key("get_percentiles", column, tuple(percentiles))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Delegate to engine-specific quantile implementation
        if hasattr(self, "get_quantiles"):
            values = self.get_quantiles(column, percentiles)
        else:
            # Fallback: collect and compute locally
            values = self._compute_percentiles_fallback(column, percentiles)

        result = {f"p{int(p * 100)}": v for p, v in zip(percentiles, values)}
        self._set_cached(cache_key, result)
        return result

    def _compute_percentiles_fallback(
        self: DistributedEngineProtocol,
        column: str,
        percentiles: list[float],
    ) -> list[float]:
        """Fallback percentile computation via sampling."""
        # Sample data and compute locally
        if hasattr(self, "sample"):
            sampled = self.sample(n=100000)
            if hasattr(sampled, "to_polars_lazyframe"):
                import polars as pl

                lf = sampled.to_polars_lazyframe()
                df = lf.collect()
                series = df.get_column(column)
                return [series.quantile(p) for p in percentiles]

        return [0.0] * len(percentiles)

    def get_skewness(
        self: DistributedEngineProtocol,
        column: str,
    ) -> float:
        """Calculate skewness of a column.

        Args:
            column: Column name.

        Returns:
            Skewness value.
        """
        cache_key = self._cache_key("get_skewness", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Compute using third moment
        stats = self.get_stats(column)
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)

        if std == 0:
            return 0.0

        # Sample for skewness calculation
        if hasattr(self, "sample"):
            sampled = self.sample(n=50000)
            if hasattr(sampled, "to_polars_lazyframe"):
                import polars as pl

                lf = sampled.to_polars_lazyframe()
                df = lf.collect()
                series = df.get_column(column)

                # Compute third standardized moment
                n = len(series)
                if n < 3:
                    return 0.0

                z = (series - mean) / std
                skew = (z ** 3).sum() * n / ((n - 1) * (n - 2))
                self._set_cached(cache_key, float(skew))
                return float(skew)

        return 0.0

    def get_kurtosis(
        self: DistributedEngineProtocol,
        column: str,
    ) -> float:
        """Calculate excess kurtosis of a column.

        Args:
            column: Column name.

        Returns:
            Excess kurtosis value.
        """
        cache_key = self._cache_key("get_kurtosis", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        stats = self.get_stats(column)
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)

        if std == 0:
            return 0.0

        if hasattr(self, "sample"):
            sampled = self.sample(n=50000)
            if hasattr(sampled, "to_polars_lazyframe"):
                import polars as pl

                lf = sampled.to_polars_lazyframe()
                df = lf.collect()
                series = df.get_column(column)

                n = len(series)
                if n < 4:
                    return 0.0

                z = (series - mean) / std
                # Excess kurtosis (Fisher's definition)
                kurt = (z ** 4).mean() - 3
                self._set_cached(cache_key, float(kurt))
                return float(kurt)

        return 0.0

    def get_correlation(
        self: DistributedEngineProtocol,
        column1: str,
        column2: str,
    ) -> float:
        """Calculate Pearson correlation between two columns.

        Args:
            column1: First column name.
            column2: Second column name.

        Returns:
            Correlation coefficient (-1 to 1).
        """
        cache_key = self._cache_key("get_correlation", column1, column2)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if hasattr(self, "sample"):
            sampled = self.sample(n=50000)
            if hasattr(sampled, "to_polars_lazyframe"):
                import polars as pl

                lf = sampled.to_polars_lazyframe()
                df = lf.collect()

                # Polars pearson_corr
                corr = df.select(
                    pl.corr(column1, column2).alias("correlation")
                )[0, 0]

                self._set_cached(cache_key, float(corr) if corr is not None else 0.0)
                return float(corr) if corr is not None else 0.0

        return 0.0


# =============================================================================
# Data Quality Mixin
# =============================================================================


@dataclass
class DataQualityReport:
    """Report containing data quality metrics."""

    total_rows: int
    total_columns: int
    completeness: dict[str, float]  # Column -> % non-null
    uniqueness: dict[str, float]  # Column -> % unique
    overall_completeness: float
    overall_score: float


class DataQualityMixin:
    """Mixin providing data quality assessment capabilities."""

    def get_completeness(
        self: DistributedEngineProtocol,
        column: str | None = None,
    ) -> float | dict[str, float]:
        """Calculate completeness (% non-null values).

        Args:
            column: Column name or None for all columns.

        Returns:
            Completeness percentage(s).
        """
        total_rows = self.count_rows()

        if total_rows == 0:
            if column:
                return 1.0
            return {col: 1.0 for col in self.get_columns()}

        if column:
            null_count = self.count_nulls(column)
            return (total_rows - null_count) / total_rows
        else:
            null_counts = self.count_nulls_all()
            return {
                col: (total_rows - count) / total_rows
                for col, count in null_counts.items()
            }

    def get_uniqueness(
        self: DistributedEngineProtocol,
        column: str | None = None,
    ) -> float | dict[str, float]:
        """Calculate uniqueness (% unique values).

        Args:
            column: Column name or None for all columns.

        Returns:
            Uniqueness percentage(s).
        """
        total_rows = self.count_rows()

        if total_rows == 0:
            if column:
                return 1.0
            return {col: 1.0 for col in self.get_columns()}

        if column:
            distinct = self.count_distinct(column)
            return distinct / total_rows
        else:
            return {
                col: self.count_distinct(col) / total_rows
                for col in self.get_columns()
            }

    def get_data_quality_report(
        self: DistributedEngineProtocol,
    ) -> DataQualityReport:
        """Generate a comprehensive data quality report.

        Returns:
            DataQualityReport with quality metrics.
        """
        columns = self.get_columns()
        total_rows = self.count_rows()

        completeness = self.get_completeness()
        if not isinstance(completeness, dict):
            completeness = {}

        uniqueness = self.get_uniqueness()
        if not isinstance(uniqueness, dict):
            uniqueness = {}

        overall_completeness = (
            sum(completeness.values()) / len(completeness)
            if completeness
            else 1.0
        )

        # Simple quality score: average of completeness and uniqueness metrics
        avg_uniqueness = (
            sum(uniqueness.values()) / len(uniqueness)
            if uniqueness
            else 1.0
        )
        overall_score = (overall_completeness + avg_uniqueness) / 2

        return DataQualityReport(
            total_rows=total_rows,
            total_columns=len(columns),
            completeness=completeness,
            uniqueness=uniqueness,
            overall_completeness=overall_completeness,
            overall_score=overall_score,
        )

    def check_null_threshold(
        self: DistributedEngineProtocol,
        column: str,
        threshold: float = 0.1,
    ) -> bool:
        """Check if column's null rate is below threshold.

        Args:
            column: Column name.
            threshold: Maximum allowed null rate (0-1).

        Returns:
            True if null rate is acceptable.
        """
        completeness = self.get_completeness(column)
        if isinstance(completeness, float):
            return (1 - completeness) <= threshold
        return True

    def check_unique_threshold(
        self: DistributedEngineProtocol,
        column: str,
        threshold: float = 0.9,
    ) -> bool:
        """Check if column's uniqueness is above threshold.

        Args:
            column: Column name.
            threshold: Minimum required uniqueness (0-1).

        Returns:
            True if uniqueness is acceptable.
        """
        uniqueness = self.get_uniqueness(column)
        if isinstance(uniqueness, float):
            return uniqueness >= threshold
        return True


# =============================================================================
# Partitioning Mixin
# =============================================================================


class PartitioningMixin:
    """Mixin providing advanced partitioning operations."""

    def repartition_by_column(
        self: DistributedEngineProtocol,
        column: str,
        num_partitions: int | None = None,
    ) -> Any:
        """Repartition data by column values (hash partitioning).

        Args:
            column: Column to partition by.
            num_partitions: Number of partitions (None = auto).

        Returns:
            New engine with repartitioned data.
        """
        # Delegate to backend-specific implementation
        if hasattr(self, "_repartition_by_column"):
            return self._repartition_by_column(column, num_partitions)
        elif hasattr(self, "_repartition"):
            # Fallback: just repartition without column awareness
            n = num_partitions or self._get_partition_count()
            return self._repartition(n)
        raise NotImplementedError("Repartitioning not supported")

    def get_partition_sizes(self: DistributedEngineProtocol) -> list[int]:
        """Get the row count of each partition.

        Returns:
            List of row counts per partition.
        """
        if hasattr(self, "_get_partition_info"):
            infos = self._get_partition_info()
            # If we have row counts in partition info
            return [info.row_count for info in infos]

        # Fallback: estimate from total rows and partition count
        total = self.count_rows()
        n_parts = self._get_partition_count() if hasattr(self, "_get_partition_count") else 1
        avg = total // n_parts
        return [avg] * n_parts

    def check_partition_skew(
        self: DistributedEngineProtocol,
        threshold: float = 3.0,
    ) -> bool:
        """Check if partitions are skewed beyond threshold.

        Args:
            threshold: Skew ratio threshold (max/min size ratio).

        Returns:
            True if partitions are skewed.
        """
        sizes = self.get_partition_sizes()
        if not sizes or all(s == 0 for s in sizes):
            return False

        non_zero = [s for s in sizes if s > 0]
        if len(non_zero) < 2:
            return False

        max_size = max(non_zero)
        min_size = min(non_zero)

        if min_size == 0:
            return True

        return (max_size / min_size) > threshold


# =============================================================================
# IO Operations Mixin
# =============================================================================


class IOOperationsMixin:
    """Mixin providing common I/O operations."""

    def write_parquet(
        self: DistributedEngineProtocol,
        path: str,
        partition_by: list[str] | None = None,
        compression: str = "snappy",
        **kwargs: Any,
    ) -> None:
        """Write data to Parquet format.

        Args:
            path: Output path.
            partition_by: Columns to partition by.
            compression: Compression codec.
            **kwargs: Additional arguments.
        """
        # Delegate to backend-specific implementation
        if hasattr(self, "_write_parquet"):
            return self._write_parquet(path, partition_by, compression, **kwargs)

        # Fallback: convert to Polars and write
        if hasattr(self, "to_polars_lazyframe"):
            lf = self.to_polars_lazyframe()
            df = lf.collect()
            df.write_parquet(path, compression=compression)
        else:
            raise NotImplementedError("Parquet writing not supported")

    def write_csv(
        self: DistributedEngineProtocol,
        path: str,
        header: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write data to CSV format.

        Args:
            path: Output path.
            header: Include header row.
            **kwargs: Additional arguments.
        """
        if hasattr(self, "_write_csv"):
            return self._write_csv(path, header, **kwargs)

        # Fallback: convert to Polars and write
        if hasattr(self, "to_polars_lazyframe"):
            lf = self.to_polars_lazyframe()
            df = lf.collect()
            df.write_csv(path, include_header=header)
        else:
            raise NotImplementedError("CSV writing not supported")

    def write_json(
        self: DistributedEngineProtocol,
        path: str,
        row_oriented: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write data to JSON format.

        Args:
            path: Output path.
            row_oriented: Use row-oriented format (one object per line).
            **kwargs: Additional arguments.
        """
        if hasattr(self, "_write_json"):
            return self._write_json(path, row_oriented, **kwargs)

        # Fallback: convert to Polars and write
        if hasattr(self, "to_polars_lazyframe"):
            lf = self.to_polars_lazyframe()
            df = lf.collect()
            df.write_ndjson(path) if row_oriented else df.write_json(path)
        else:
            raise NotImplementedError("JSON writing not supported")


# =============================================================================
# Validation Mixin
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    message: str
    details: dict[str, Any]


class ValidationMixin:
    """Mixin providing common validation operations."""

    def validate_not_null(
        self: DistributedEngineProtocol,
        columns: list[str] | None = None,
        threshold: float = 0.0,
    ) -> ValidationResult:
        """Validate that columns have acceptable null rates.

        Args:
            columns: Columns to check (None = all).
            threshold: Maximum allowed null rate (0-1).

        Returns:
            ValidationResult.
        """
        if columns is None:
            columns = self.get_columns()

        total_rows = self.count_rows()
        if total_rows == 0:
            return ValidationResult(
                passed=True,
                message="No data to validate",
                details={},
            )

        null_counts = self.count_nulls_all()
        failed_columns = {}

        for col in columns:
            if col in null_counts:
                null_rate = null_counts[col] / total_rows
                if null_rate > threshold:
                    failed_columns[col] = {
                        "null_count": null_counts[col],
                        "null_rate": null_rate,
                    }

        return ValidationResult(
            passed=len(failed_columns) == 0,
            message=f"{len(failed_columns)} columns exceeded null threshold" if failed_columns else "All columns passed",
            details={"failed_columns": failed_columns},
        )

    def validate_unique(
        self: DistributedEngineProtocol,
        columns: list[str],
    ) -> ValidationResult:
        """Validate that column combination is unique.

        Args:
            columns: Columns that should be unique together.

        Returns:
            ValidationResult.
        """
        total_rows = self.count_rows()
        if total_rows == 0:
            return ValidationResult(
                passed=True,
                message="No data to validate",
                details={},
            )

        if hasattr(self, "count_duplicates"):
            duplicates = self.count_duplicates(columns)
        else:
            # Fallback: use distinct count
            distinct = self.count_distinct(columns[0]) if len(columns) == 1 else total_rows
            duplicates = total_rows - distinct

        return ValidationResult(
            passed=duplicates == 0,
            message=f"Found {duplicates} duplicate rows" if duplicates > 0 else "All rows are unique",
            details={
                "total_rows": total_rows,
                "duplicate_count": duplicates,
                "columns": columns,
            },
        )

    def validate_range(
        self: DistributedEngineProtocol,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
    ) -> ValidationResult:
        """Validate that column values are within range.

        Args:
            column: Column name.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.

        Returns:
            ValidationResult.
        """
        stats = self.get_stats(column)
        actual_min = stats.get("min")
        actual_max = stats.get("max")

        issues = []

        if min_value is not None and actual_min is not None:
            if actual_min < min_value:
                issues.append(f"Minimum value {actual_min} is below {min_value}")

        if max_value is not None and actual_max is not None:
            if actual_max > max_value:
                issues.append(f"Maximum value {actual_max} is above {max_value}")

        return ValidationResult(
            passed=len(issues) == 0,
            message="; ".join(issues) if issues else "All values in range",
            details={
                "column": column,
                "actual_min": actual_min,
                "actual_max": actual_max,
                "expected_min": min_value,
                "expected_max": max_value,
            },
        )


# =============================================================================
# Convenience Combined Mixins
# =============================================================================


class FullFeaturedMixin(
    StatisticalMixin,
    DataQualityMixin,
    PartitioningMixin,
    IOOperationsMixin,
    ValidationMixin,
):
    """Combined mixin with all features.

    Use this mixin to get all available functionality:
    - Statistical analysis (percentiles, skewness, correlation)
    - Data quality assessment (completeness, uniqueness)
    - Partitioning operations
    - I/O operations (parquet, csv, json)
    - Validation operations (null checks, uniqueness, range)
    """

    pass
