"""Base classes for data drift validators.

This module provides base classes and utilities for detecting data drift
between reference (baseline) and current datasets.

Key Features:
- Memory-efficient reference data handling with statistics caching
- LRU cache for reference statistics (no raw data retention)
- Thread-safe operations
- Automatic cache invalidation
"""

from abc import abstractmethod
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.cache import (
    ReferenceCache,
    CacheConfig,
    NumericStatistics,
    CategoricalStatistics,
    get_global_cache,
    make_cache_key,
)


class DriftValidator(Validator):
    """Base class for drift detection validators.

    Drift validators compare a current dataset against a reference (baseline)
    dataset to detect distribution shifts, statistical changes, or data quality
    degradation over time.

    Key Concepts:
        - Reference data: The baseline/expected distribution (e.g., training data)
        - Current data: The data being validated (e.g., production data)
        - Drift score: A metric quantifying the difference between distributions
        - Threshold: The acceptable level of drift before triggering an alert

    Memory Optimization:
        - By default, only statistical summaries are cached (not raw data)
        - Use `cache_reference=True` to enable caching (recommended)
        - Use `cache_raw_data=True` only when statistical summary is insufficient

    Usage Pattern:
        1. Initialize with reference data and thresholds
        2. Call validate() with current data
        3. Check returned issues for drift detection results

    Example:
        # Memory-efficient: cache statistics only
        validator = MyDriftValidator(
            reference_data=large_df,
            cache_reference=True,  # Cache statistics, release raw data
        )

        # Traditional: keep raw data (higher memory)
        validator = MyDriftValidator(
            reference_data=large_df,
            cache_reference=False,  # Keep raw LazyFrame
        )
    """

    name = "drift_base"
    category = "drift"

    def __init__(
        self,
        reference_data: pl.LazyFrame | pl.DataFrame,
        cache_reference: bool = True,
        cache_config: CacheConfig | None = None,
        cache_key: str | None = None,
        **kwargs: Any,
    ):
        """Initialize drift validator.

        Args:
            reference_data: Baseline data to compare against
            cache_reference: If True, cache statistics and optionally release raw data
            cache_config: Optional cache configuration
            cache_key: Optional custom cache key (auto-generated if None)
            **kwargs: Additional config passed to base Validator
        """
        super().__init__(**kwargs)

        self._cache_reference = cache_reference
        self._cache_key = cache_key
        self._cache: ReferenceCache | None = None

        if cache_config:
            self._cache = ReferenceCache(cache_config)

        # Ensure reference data is LazyFrame for consistent handling
        if isinstance(reference_data, pl.DataFrame):
            self._reference_data: pl.LazyFrame | None = reference_data.lazy()
        else:
            self._reference_data = reference_data

        # Flag to track if statistics have been cached
        self._stats_cached: bool = False

    @property
    def reference_data(self) -> pl.LazyFrame:
        """Get the reference data as LazyFrame.

        Note: If caching is enabled and raw data was released,
        this may return an empty LazyFrame.
        Use get_reference_statistics() for cached statistics.
        """
        if self._reference_data is None:
            # Return empty LazyFrame if raw data was released
            return pl.LazyFrame()
        return self._reference_data

    def get_cache(self) -> ReferenceCache:
        """Get the cache instance (local or global)."""
        if self._cache is not None:
            return self._cache
        return get_global_cache()

    def get_cache_key(self, suffix: str = "") -> str:
        """Generate cache key for this validator.

        Args:
            suffix: Optional suffix to append

        Returns:
            Cache key string
        """
        if self._cache_key:
            return f"{self._cache_key}:{suffix}" if suffix else self._cache_key

        return make_cache_key(
            validator_name=self.name,
            column=getattr(self, 'column', '_all'),
            version="v1",
            extra=suffix,
        )

    def release_reference_data(self) -> None:
        """Release raw reference data to free memory.

        Call this after caching statistics if memory is a concern.
        Subsequent calls to reference_data will return an empty LazyFrame.
        """
        self._reference_data = None
        self.logger.debug(f"Released raw reference data for {self.name}")

    def is_statistics_cached(self) -> bool:
        """Check if reference statistics are cached."""
        return self._stats_cached or self.get_cache_key() in self.get_cache()

    @abstractmethod
    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> float:
        """Calculate drift score between reference and current data.

        Args:
            reference: Reference/baseline data
            current: Current data to check for drift

        Returns:
            Drift score (interpretation depends on specific validator)
        """
        pass

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate current data against reference for drift.

        Args:
            lf: Current LazyFrame to validate

        Returns:
            List of validation issues if drift is detected
        """
        pass

    def _calculate_severity(self, drift_score: float, threshold: float) -> Severity:
        """Calculate severity based on drift magnitude.

        Args:
            drift_score: The calculated drift score
            threshold: The threshold for drift detection

        Returns:
            Severity level based on how much drift exceeds threshold
        """
        if drift_score <= threshold:
            return Severity.LOW
        elif drift_score <= threshold * 1.5:
            return Severity.MEDIUM
        elif drift_score <= threshold * 2:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


class ColumnDriftValidator(DriftValidator):
    """Base class for single-column drift validators.

    Validates drift for a specific column between reference and current data.
    Supports caching of reference column statistics for memory efficiency.

    Memory Optimization:
        When cache_reference=True (default), the validator will:
        1. Compute and cache column statistics on first use
        2. Optionally release raw data after caching (call release_reference_data())
        3. Use cached statistics for subsequent drift calculations
    """

    name = "column_drift_base"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        is_categorical: bool = False,
        n_histogram_bins: int = 50,
        **kwargs: Any,
    ):
        """Initialize column drift validator.

        Args:
            column: Column name to check for drift
            reference_data: Baseline data to compare against
            is_categorical: If True, treat column as categorical
            n_histogram_bins: Number of histogram bins for numeric columns
            **kwargs: Additional config (including cache_reference)
        """
        super().__init__(reference_data=reference_data, **kwargs)
        self.column = column
        self.is_categorical = is_categorical
        self.n_histogram_bins = n_histogram_bins

        # Cached reference statistics
        self._ref_stats: NumericStatistics | CategoricalStatistics | None = None

    def _get_column_values(
        self, lf: pl.LazyFrame, drop_nulls: bool = True
    ) -> pl.Series:
        """Extract column values as Series.

        Args:
            lf: LazyFrame to extract from
            drop_nulls: Whether to drop null values

        Returns:
            Series of column values
        """
        if drop_nulls:
            return lf.select(pl.col(self.column)).drop_nulls().collect().to_series()
        return lf.select(pl.col(self.column)).collect().to_series()

    def get_reference_statistics(
        self,
        force_recompute: bool = False,
    ) -> NumericStatistics | CategoricalStatistics:
        """Get cached reference statistics, computing if necessary.

        This method provides memory-efficient access to reference data
        statistics. Statistics are cached and can be retrieved without
        holding the raw reference data in memory.

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            NumericStatistics or CategoricalStatistics for the reference column
        """
        # Return cached stats if available
        if self._ref_stats is not None and not force_recompute:
            return self._ref_stats

        # Check global cache
        cache_key = self.get_cache_key()
        if not force_recompute:
            cached = self.get_cache().get(cache_key)
            if cached is not None:
                self._ref_stats = cached
                self._stats_cached = True
                return cached

        # Compute statistics from reference data
        if self._reference_data is None:
            raise ValueError(
                "Reference data has been released and statistics are not cached. "
                "Call get_reference_statistics() before release_reference_data()."
            )

        if self.is_categorical:
            stats = CategoricalStatistics.from_lazyframe(
                self._reference_data,
                self.column,
            )
        else:
            stats = NumericStatistics.from_lazyframe(
                self._reference_data,
                self.column,
                n_bins=self.n_histogram_bins,
            )

        # Cache the statistics
        if self._cache_reference:
            self.get_cache().put(cache_key, stats)
            self._stats_cached = True

        self._ref_stats = stats
        return stats

    def cache_and_release(self) -> None:
        """Cache reference statistics and release raw data.

        This is a convenience method that:
        1. Computes and caches reference statistics
        2. Releases raw reference data to free memory

        Use this for memory-constrained environments.
        """
        # Ensure statistics are cached
        self.get_reference_statistics()
        # Release raw data
        self.release_reference_data()
        self.logger.info(
            f"Cached statistics and released raw data for {self.name}:{self.column}"
        )


class NumericDriftMixin:
    """Mixin for numeric column drift detection utilities."""

    @staticmethod
    def compute_histogram(
        values: pl.Series, n_bins: int = 10, range_min: float | None = None, range_max: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Compute histogram for numeric values.

        Args:
            values: Series of numeric values
            n_bins: Number of histogram bins
            range_min: Minimum value for histogram range
            range_max: Maximum value for histogram range

        Returns:
            Tuple of (bin_edges, frequencies)
        """
        import numpy as np

        arr = values.to_numpy()
        arr = arr[~np.isnan(arr)]  # Remove NaN values

        if len(arr) == 0:
            return [], []

        # Determine range
        if range_min is None:
            range_min = float(arr.min())
        if range_max is None:
            range_max = float(arr.max())

        # Compute histogram
        counts, edges = np.histogram(arr, bins=n_bins, range=(range_min, range_max))

        # Normalize to frequencies
        total = counts.sum()
        if total > 0:
            frequencies = (counts / total).tolist()
        else:
            frequencies = [0.0] * n_bins

        return edges.tolist(), frequencies


class CategoricalDriftMixin:
    """Mixin for categorical column drift detection utilities."""

    @staticmethod
    def compute_category_frequencies(
        values: pl.Series,
    ) -> dict[str, float]:
        """Compute normalized frequencies for categorical values.

        Args:
            values: Series of categorical values

        Returns:
            Dictionary of category -> frequency
        """
        counts = values.value_counts()
        total = len(values)

        if total == 0:
            return {}

        result = {}
        for row in counts.iter_rows():
            category, count = row
            result[str(category)] = count / total

        return result

    @staticmethod
    def align_categories(
        ref_freq: dict[str, float], curr_freq: dict[str, float]
    ) -> tuple[list[float], list[float]]:
        """Align category frequencies between reference and current.

        Ensures both distributions have the same categories in the same order.

        Args:
            ref_freq: Reference category frequencies
            curr_freq: Current category frequencies

        Returns:
            Tuple of (aligned_ref_frequencies, aligned_curr_frequencies)
        """
        all_categories = sorted(set(ref_freq.keys()) | set(curr_freq.keys()))

        ref_aligned = [ref_freq.get(cat, 0.0) for cat in all_categories]
        curr_aligned = [curr_freq.get(cat, 0.0) for cat in all_categories]

        return ref_aligned, curr_aligned
