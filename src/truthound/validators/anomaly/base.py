"""Base classes for anomaly detection validators.

This module provides extensible base classes for implementing various
anomaly detection algorithms.

Key Features:
- Memory-efficient reference data handling with statistics caching
- Sampling support for large datasets
- Both statistical and ML-based anomaly detection
"""

from abc import abstractmethod
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    NumericValidatorMixin,
)
from truthound.validators.cache import (
    ReferenceCache,
    CacheConfig,
    MultiColumnStatistics,
    NumericStatistics,
    get_global_cache,
    make_cache_key,
)


class AnomalyValidator(Validator, NumericValidatorMixin):
    """Base class for table-wide anomaly detection.

    Anomaly validators detect unusual patterns or outliers in data.
    They can work on single columns or multiple columns simultaneously.

    Memory Optimization:
        For ML-based validators that need reference data (e.g., for
        normalization parameters), statistics can be cached to avoid
        keeping large datasets in memory.

        # Memory-efficient usage:
        validator = MyAnomalyValidator(
            columns=["col1", "col2"],
            cache_normalization=True,
        )
        validator.fit(reference_df)  # Computes and caches normalization stats
        validator.release_fit_data()  # Release raw data, keep stats

    Subclasses should implement:
        - detect_anomalies(): Returns indices or mask of anomalous rows
    """

    category = "anomaly"

    def __init__(
        self,
        columns: list[str] | None = None,
        max_anomaly_ratio: float = 0.1,
        sample_size: int | None = None,
        cache_normalization: bool = True,
        cache_config: CacheConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize anomaly validator.

        Args:
            columns: Specific columns to check. If None, uses all numeric columns.
            max_anomaly_ratio: Maximum acceptable ratio of anomalies (0.0-1.0)
            sample_size: If set, sample this many rows for analysis (memory optimization)
            cache_normalization: If True, cache normalization parameters
            cache_config: Optional cache configuration
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.columns = columns
        self.max_anomaly_ratio = max_anomaly_ratio
        self.sample_size = sample_size
        self._cache_normalization = cache_normalization
        self._cache: ReferenceCache | None = None

        if cache_config:
            self._cache = ReferenceCache(cache_config)

        # Cached normalization parameters
        self._normalization_stats: MultiColumnStatistics | None = None
        self._is_fitted: bool = False

    def _get_anomaly_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get columns to analyze for anomaly detection."""
        if self.columns:
            return self.columns
        return self._get_numeric_columns(lf)

    def get_cache(self) -> ReferenceCache:
        """Get the cache instance (local or global)."""
        if self._cache is not None:
            return self._cache
        return get_global_cache()

    def get_cache_key(self, suffix: str = "") -> str:
        """Generate cache key for this validator."""
        columns_str = ":".join(sorted(self.columns or []))
        return make_cache_key(
            validator_name=self.name,
            column=columns_str or "_all",
            version="v1",
            extra=suffix,
        )

    def _sample_data(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Sample data if sample_size is set.

        This helps reduce memory usage for large datasets.
        """
        if self.sample_size is None:
            return lf

        count = lf.select(pl.len()).collect().item()
        if count <= self.sample_size:
            return lf

        # Use reservoir sampling for unbiased sample
        fraction = self.sample_size / count
        return lf.filter(pl.lit(1).sample(fraction=min(1.0, fraction * 1.1)))

    def fit(self, reference_data: pl.LazyFrame | pl.DataFrame) -> "AnomalyValidator":
        """Fit normalization parameters from reference data.

        This pre-computes and caches statistics needed for normalization,
        allowing the reference data to be released afterward.

        Args:
            reference_data: Data to compute normalization from

        Returns:
            self (for method chaining)
        """
        if isinstance(reference_data, pl.DataFrame):
            reference_data = reference_data.lazy()

        columns = self._get_anomaly_columns(reference_data)
        if not columns:
            self._is_fitted = True
            return self

        # Compute and cache statistics
        stats = MultiColumnStatistics.from_lazyframe(reference_data, columns)

        if self._cache_normalization:
            self.get_cache().put(self.get_cache_key("normalization"), stats)

        self._normalization_stats = stats
        self._is_fitted = True

        self.logger.debug(f"Fitted normalization for {len(columns)} columns")
        return self

    def get_normalization_stats(self) -> MultiColumnStatistics | None:
        """Get cached normalization statistics.

        Returns:
            MultiColumnStatistics if fitted/cached, else None
        """
        if self._normalization_stats is not None:
            return self._normalization_stats

        # Try cache
        cached = self.get_cache().get(self.get_cache_key("normalization"))
        if cached is not None and isinstance(cached, MultiColumnStatistics):
            self._normalization_stats = cached
            return cached

        return None

    @abstractmethod
    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies in the data.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)
            column_names: Names of the columns

        Returns:
            Tuple of:
                - Boolean mask array where True indicates anomaly
                - Dictionary of additional info (e.g., scores, thresholds)
        """
        pass

    def _calculate_severity(self, anomaly_ratio: float) -> Severity:
        """Calculate severity based on anomaly ratio."""
        if anomaly_ratio < 0.01:
            return Severity.LOW
        elif anomaly_ratio < 0.05:
            return Severity.MEDIUM
        elif anomaly_ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


class ColumnAnomalyValidator(Validator, NumericValidatorMixin):
    """Base class for single-column anomaly detection.

    Use this for methods that detect anomalies in individual columns
    independently.
    """

    category = "anomaly"

    def __init__(
        self,
        column: str,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize column anomaly validator.

        Args:
            column: Column to check for anomalies
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.max_anomaly_ratio = max_anomaly_ratio

    @abstractmethod
    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies in a single column.

        Args:
            values: 1D numpy array of column values (nulls removed)

        Returns:
            Tuple of:
                - Boolean mask array where True indicates anomaly
                - Dictionary of additional info
        """
        pass

    def _calculate_severity(self, anomaly_ratio: float) -> Severity:
        """Calculate severity based on anomaly ratio."""
        if anomaly_ratio < 0.01:
            return Severity.LOW
        elif anomaly_ratio < 0.05:
            return Severity.MEDIUM
        elif anomaly_ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


class StatisticalAnomalyMixin:
    """Mixin providing statistical anomaly detection utilities."""

    @staticmethod
    def compute_iqr_bounds(
        values: np.ndarray, multiplier: float = 1.5
    ) -> tuple[float, float]:
        """Compute IQR-based bounds.

        Args:
            values: 1D numpy array
            multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return float(lower), float(upper)

    @staticmethod
    def compute_mad(values: np.ndarray) -> tuple[float, float]:
        """Compute Median Absolute Deviation.

        MAD is a robust measure of variability:
        MAD = median(|X - median(X)|)

        Args:
            values: 1D numpy array

        Returns:
            Tuple of (median, MAD)
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return float(median), float(mad)

    @staticmethod
    def compute_modified_zscore(values: np.ndarray) -> np.ndarray:
        """Compute modified Z-scores using MAD.

        Modified Z-score = 0.6745 * (x - median) / MAD

        The constant 0.6745 makes the MAD consistent with std for normal data.

        Args:
            values: 1D numpy array

        Returns:
            Array of modified Z-scores
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            # Fallback to mean absolute deviation
            mad = np.mean(np.abs(values - median))
            if mad == 0:
                return np.zeros_like(values)

        return 0.6745 * (values - median) / mad


class MLAnomalyMixin:
    """Mixin for machine learning based anomaly detection.

    Provides utilities for working with sklearn-based anomaly detectors.
    Supports using cached normalization statistics for memory efficiency.
    """

    @staticmethod
    def normalize_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize data using robust scaling.

        Uses median and IQR for robustness to outliers.

        Args:
            data: 2D numpy array (n_samples, n_features)

        Returns:
            Tuple of (normalized_data, medians, iqrs)
        """
        medians = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqrs = q3 - q1

        # Avoid division by zero
        iqrs = np.where(iqrs == 0, 1, iqrs)

        normalized = (data - medians) / iqrs
        return normalized, medians, iqrs

    def normalize_with_cached_stats(
        self,
        data: np.ndarray,
        column_names: list[str],
    ) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        """Normalize data using cached statistics.

        This method uses pre-computed statistics from fit() instead of
        computing them from the input data. This is useful when you want
        to normalize test/production data using training data statistics.

        Args:
            data: 2D numpy array (n_samples, n_features)
            column_names: Names of the columns corresponding to data columns

        Returns:
            Tuple of (normalized_data, medians_dict, iqrs_dict)
        """
        # Try to get cached stats
        stats = None
        if hasattr(self, 'get_normalization_stats'):
            stats = self.get_normalization_stats()

        if stats is not None and hasattr(stats, 'medians') and hasattr(stats, 'iqrs'):
            # Use cached statistics
            medians = np.array([stats.medians.get(col, 0.0) for col in column_names])
            iqrs = np.array([stats.iqrs.get(col, 1.0) for col in column_names])

            # Avoid division by zero
            iqrs = np.where(iqrs == 0, 1, iqrs)

            normalized = (data - medians) / iqrs
            return normalized, stats.medians, stats.iqrs

        # Fallback: compute from data
        normalized, medians, iqrs = self.normalize_data(data)
        medians_dict = {col: float(m) for col, m in zip(column_names, medians)}
        iqrs_dict = {col: float(i) for col, i in zip(column_names, iqrs)}
        return normalized, medians_dict, iqrs_dict

    @staticmethod
    def validate_sklearn_available() -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def sample_for_training(
        data: np.ndarray,
        max_samples: int = 100000,
        random_state: int = 42,
    ) -> np.ndarray:
        """Sample data for training to reduce memory usage.

        For large datasets, training on a representative sample can
        significantly reduce memory and computation time while
        maintaining detection quality.

        Args:
            data: 2D numpy array (n_samples, n_features)
            max_samples: Maximum number of samples for training
            random_state: Random seed for reproducibility

        Returns:
            Sampled data array
        """
        if len(data) <= max_samples:
            return data

        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(data), size=max_samples, replace=False)
        return data[indices]
