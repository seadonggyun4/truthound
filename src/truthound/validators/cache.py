"""Reference data caching for drift and anomaly validators.

This module provides memory-efficient caching mechanisms for reference data
used in drift detection and anomaly detection validators.

Key Features:
- LRU cache with configurable size limits
- Statistics summary storage (no raw data retention)
- Thread-safe operations
- Automatic cache invalidation
- Memory tracking and limits

Usage:
    from truthound.validators.cache import (
        ReferenceCache,
        ReferenceStatistics,
        CacheConfig,
    )

    # Configure cache
    config = CacheConfig(max_entries=100, max_memory_mb=512)
    cache = ReferenceCache(config)

    # Store reference statistics (not raw data)
    stats = ReferenceStatistics.from_lazyframe(lf, column="price")
    cache.put("model_v1:price", stats)

    # Retrieve cached statistics
    cached_stats = cache.get("model_v1:price")
"""

from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict
from threading import RLock
from functools import lru_cache
import hashlib
import time
import sys

import polars as pl
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class CacheConfig:
    """Configuration for reference data cache.

    Attributes:
        max_entries: Maximum number of cache entries
        max_memory_mb: Maximum memory usage in MB (approximate)
        ttl_seconds: Time-to-live for cache entries (None = no expiration)
        enable_statistics_summary: Store statistics summary instead of raw data
        n_histogram_bins: Number of bins for histogram caching
        quantiles: Quantiles to cache (for drift detection)
    """
    max_entries: int = 100
    max_memory_mb: float = 512.0
    ttl_seconds: float | None = 3600.0  # 1 hour default
    enable_statistics_summary: bool = True
    n_histogram_bins: int = 50
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


# ============================================================================
# Statistics Summary Classes
# ============================================================================

@dataclass
class NumericStatistics:
    """Cached statistics for a numeric column.

    This replaces storing raw reference data with a compact statistical summary.
    Memory usage: ~1KB per column vs. potentially GB for raw data.
    """
    # Basic statistics
    count: int
    null_count: int
    mean: float
    std: float
    variance: float
    min_value: float
    max_value: float
    sum_value: float

    # Quantiles (configurable)
    quantiles: dict[float, float]  # {0.5: 50.0, 0.95: 95.0, ...}

    # Histogram for PSI/distribution comparison
    histogram_edges: list[float]  # Bin edges
    histogram_counts: list[float]  # Normalized frequencies

    # Metadata
    created_at: float = field(default_factory=time.time)
    source_hash: str = ""  # Hash of source data for validation

    def __post_init__(self) -> None:
        """Validate statistics."""
        if self.count < 0:
            raise ValueError("count must be non-negative")

    def estimate_memory_bytes(self) -> int:
        """Estimate memory usage of this statistics object."""
        # Base dataclass fields
        base_size = sys.getsizeof(self)
        # Quantiles dict
        quantiles_size = sys.getsizeof(self.quantiles) + sum(
            sys.getsizeof(k) + sys.getsizeof(v)
            for k, v in self.quantiles.items()
        )
        # Histogram lists
        hist_size = (
            sys.getsizeof(self.histogram_edges) +
            len(self.histogram_edges) * 8 +  # float64
            sys.getsizeof(self.histogram_counts) +
            len(self.histogram_counts) * 8
        )
        return base_size + quantiles_size + hist_size

    @classmethod
    def from_series(
        cls,
        series: pl.Series,
        n_bins: int = 50,
        quantiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
    ) -> "NumericStatistics":
        """Create statistics summary from a Polars Series.

        Args:
            series: Numeric Polars Series
            n_bins: Number of histogram bins
            quantiles: Quantiles to compute

        Returns:
            NumericStatistics instance
        """
        # Drop nulls for calculations
        non_null = series.drop_nulls()
        arr = non_null.to_numpy()

        if len(arr) == 0:
            return cls(
                count=0,
                null_count=len(series),
                mean=0.0,
                std=0.0,
                variance=0.0,
                min_value=0.0,
                max_value=0.0,
                sum_value=0.0,
                quantiles={},
                histogram_edges=[],
                histogram_counts=[],
            )

        # Basic statistics
        count = len(non_null)
        null_count = len(series) - count
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        var_val = float(np.var(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        sum_val = float(np.sum(arr))

        # Quantiles
        quantile_values = {}
        for q in quantiles:
            quantile_values[q] = float(np.percentile(arr, q * 100))

        # Histogram (quantile-based bins for robustness)
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(arr, percentiles)
        # Ensure unique edges
        edges = np.unique(edges)

        if len(edges) >= 2:
            counts, _ = np.histogram(arr, bins=edges)
            total = counts.sum()
            frequencies = (counts / total).tolist() if total > 0 else [0.0] * (len(edges) - 1)
        else:
            edges = [min_val, max_val] if min_val != max_val else [min_val]
            frequencies = [1.0] if len(edges) == 2 else []

        # Create hash for validation
        source_hash = hashlib.md5(
            f"{count}:{mean_val:.6f}:{std_val:.6f}".encode()
        ).hexdigest()[:16]

        return cls(
            count=count,
            null_count=null_count,
            mean=mean_val,
            std=std_val,
            variance=var_val,
            min_value=min_val,
            max_value=max_val,
            sum_value=sum_val,
            quantiles=quantile_values,
            histogram_edges=edges.tolist(),
            histogram_counts=frequencies,
            source_hash=source_hash,
        )

    @classmethod
    def from_lazyframe(
        cls,
        lf: pl.LazyFrame,
        column: str,
        n_bins: int = 50,
        quantiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
    ) -> "NumericStatistics":
        """Create statistics summary from a LazyFrame column.

        Args:
            lf: Polars LazyFrame
            column: Column name
            n_bins: Number of histogram bins
            quantiles: Quantiles to compute

        Returns:
            NumericStatistics instance
        """
        series = lf.select(pl.col(column)).collect().to_series()
        return cls.from_series(series, n_bins, quantiles)


@dataclass
class CategoricalStatistics:
    """Cached statistics for a categorical column."""

    # Category frequencies
    frequencies: dict[str, float]  # {category: frequency}

    # Basic counts
    count: int
    null_count: int
    unique_count: int

    # Top categories (for quick access)
    top_categories: list[tuple[str, float]]  # [(category, frequency), ...]

    # Metadata
    created_at: float = field(default_factory=time.time)
    source_hash: str = ""

    def estimate_memory_bytes(self) -> int:
        """Estimate memory usage."""
        base_size = sys.getsizeof(self)
        freq_size = sys.getsizeof(self.frequencies) + sum(
            sys.getsizeof(k) + sys.getsizeof(v)
            for k, v in self.frequencies.items()
        )
        top_size = sys.getsizeof(self.top_categories) + sum(
            sys.getsizeof(t) for t in self.top_categories
        )
        return base_size + freq_size + top_size

    @classmethod
    def from_series(
        cls,
        series: pl.Series,
        top_n: int = 100,
    ) -> "CategoricalStatistics":
        """Create statistics summary from a Polars Series.

        Args:
            series: Categorical Polars Series
            top_n: Number of top categories to store

        Returns:
            CategoricalStatistics instance
        """
        non_null = series.drop_nulls()

        if len(non_null) == 0:
            return cls(
                frequencies={},
                count=0,
                null_count=len(series),
                unique_count=0,
                top_categories=[],
            )

        # Value counts
        value_counts = non_null.value_counts()
        total = len(non_null)

        frequencies = {}
        for row in value_counts.iter_rows():
            category, count = row
            frequencies[str(category)] = count / total

        # Sort by frequency for top categories
        sorted_cats = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        top_categories = sorted_cats[:top_n]

        # Create hash
        source_hash = hashlib.md5(
            f"{total}:{len(frequencies)}".encode()
        ).hexdigest()[:16]

        return cls(
            frequencies=frequencies,
            count=len(non_null),
            null_count=len(series) - len(non_null),
            unique_count=len(frequencies),
            top_categories=top_categories,
            source_hash=source_hash,
        )

    @classmethod
    def from_lazyframe(
        cls,
        lf: pl.LazyFrame,
        column: str,
        top_n: int = 100,
    ) -> "CategoricalStatistics":
        """Create statistics summary from a LazyFrame column."""
        series = lf.select(pl.col(column)).collect().to_series()
        return cls.from_series(series, top_n)


@dataclass
class MultiColumnStatistics:
    """Cached statistics for multiple columns (for multivariate analysis)."""

    # Per-column statistics
    column_stats: dict[str, NumericStatistics]

    # Correlation matrix (upper triangle only to save space)
    correlation_matrix: dict[tuple[str, str], float]

    # Covariance matrix summary
    covariance_matrix: dict[tuple[str, str], float]

    # Column medians and IQRs for normalization
    medians: dict[str, float]
    iqrs: dict[str, float]

    # Metadata
    columns: list[str]
    created_at: float = field(default_factory=time.time)

    def estimate_memory_bytes(self) -> int:
        """Estimate memory usage."""
        base_size = sys.getsizeof(self)
        stats_size = sum(s.estimate_memory_bytes() for s in self.column_stats.values())
        matrix_size = (
            sys.getsizeof(self.correlation_matrix) +
            len(self.correlation_matrix) * 24  # Approximate per entry
        )
        return base_size + stats_size + matrix_size

    @classmethod
    def from_lazyframe(
        cls,
        lf: pl.LazyFrame,
        columns: list[str],
        n_bins: int = 50,
    ) -> "MultiColumnStatistics":
        """Create multi-column statistics from LazyFrame.

        Args:
            lf: Polars LazyFrame
            columns: List of numeric column names
            n_bins: Number of histogram bins per column

        Returns:
            MultiColumnStatistics instance
        """
        # Collect data for selected columns
        df = lf.select([pl.col(c) for c in columns]).drop_nulls().collect()

        if len(df) == 0:
            return cls(
                column_stats={},
                correlation_matrix={},
                covariance_matrix={},
                medians={},
                iqrs={},
                columns=columns,
            )

        # Per-column statistics
        column_stats = {}
        for col in columns:
            column_stats[col] = NumericStatistics.from_series(df[col], n_bins)

        # Compute correlation and covariance matrices
        data = df.to_numpy()
        correlation_matrix = {}
        covariance_matrix = {}

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i <= j:  # Upper triangle only
                    corr = float(np.corrcoef(data[:, i], data[:, j])[0, 1])
                    cov = float(np.cov(data[:, i], data[:, j])[0, 1])
                    correlation_matrix[(col1, col2)] = corr
                    covariance_matrix[(col1, col2)] = cov

        # Medians and IQRs for normalization
        medians = {}
        iqrs = {}
        for i, col in enumerate(columns):
            medians[col] = float(np.median(data[:, i]))
            q1, q3 = np.percentile(data[:, i], [25, 75])
            iqrs[col] = float(q3 - q1) if q3 != q1 else 1.0

        return cls(
            column_stats=column_stats,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            medians=medians,
            iqrs=iqrs,
            columns=columns,
        )


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: NumericStatistics | CategoricalStatistics | MultiColumnStatistics
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def is_expired(self, ttl_seconds: float | None) -> bool:
        """Check if entry has expired."""
        if ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def estimate_memory_bytes(self) -> int:
        """Estimate memory usage."""
        base_size = sys.getsizeof(self) + sys.getsizeof(self.key)
        value_size = self.value.estimate_memory_bytes()
        return base_size + value_size


# ============================================================================
# LRU Cache Implementation
# ============================================================================

class ReferenceCache:
    """Thread-safe LRU cache for reference data statistics.

    This cache stores statistical summaries of reference data instead of
    raw data, dramatically reducing memory usage while preserving the
    information needed for drift and anomaly detection.

    Example:
        cache = ReferenceCache(CacheConfig(max_entries=100))

        # Cache numeric statistics
        stats = NumericStatistics.from_lazyframe(lf, "price")
        cache.put("model_v1:price", stats)

        # Retrieve later
        cached = cache.get("model_v1:price")
        if cached:
            print(f"Mean: {cached.mean}, Std: {cached.std}")
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._total_memory_bytes: int = 0

        # Statistics
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> NumericStatistics | CategoricalStatistics | MultiColumnStatistics | None:
        """Get cached statistics by key.

        Args:
            key: Cache key

        Returns:
            Cached statistics or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired(self.config.ttl_seconds):
                self._remove_entry(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            self._hits += 1
            return entry.value

    def put(
        self,
        key: str,
        value: NumericStatistics | CategoricalStatistics | MultiColumnStatistics,
    ) -> None:
        """Store statistics in cache.

        Args:
            key: Cache key
            value: Statistics to cache
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Create new entry
            entry = CacheEntry(key=key, value=value)
            entry_size = entry.estimate_memory_bytes()

            # Evict entries if necessary
            self._evict_if_needed(entry_size)

            # Add new entry
            self._cache[key] = entry
            self._total_memory_bytes += entry_size

    def remove(self, key: str) -> bool:
        """Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_memory_bytes = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "entries": len(self._cache),
                "max_entries": self.config.max_entries,
                "memory_bytes": self._total_memory_bytes,
                "memory_mb": self._total_memory_bytes / (1024 * 1024),
                "max_memory_mb": self.config.max_memory_mb,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def _remove_entry(self, key: str) -> None:
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_memory_bytes -= entry.estimate_memory_bytes()
            self._total_memory_bytes = max(0, self._total_memory_bytes)

    def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict entries if cache limits are exceeded."""
        max_memory_bytes = int(self.config.max_memory_mb * 1024 * 1024)

        # Evict while over limits
        while self._cache and (
            len(self._cache) >= self.config.max_entries or
            self._total_memory_bytes + new_entry_size > max_memory_bytes
        ):
            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (without updating access time)."""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            return not entry.is_expired(self.config.ttl_seconds)

    def __len__(self) -> int:
        """Return number of cache entries."""
        with self._lock:
            return len(self._cache)


# ============================================================================
# Global Cache Instance
# ============================================================================

# Default global cache instance
_global_cache: ReferenceCache | None = None
_global_cache_lock = RLock()


def get_global_cache(config: CacheConfig | None = None) -> ReferenceCache:
    """Get or create the global reference cache.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        Global ReferenceCache instance
    """
    global _global_cache

    with _global_cache_lock:
        if _global_cache is None:
            _global_cache = ReferenceCache(config or CacheConfig())
        return _global_cache


def clear_global_cache() -> None:
    """Clear the global cache."""
    global _global_cache

    with _global_cache_lock:
        if _global_cache is not None:
            _global_cache.clear()


def reset_global_cache(config: CacheConfig | None = None) -> ReferenceCache:
    """Reset the global cache with new configuration.

    Args:
        config: New configuration

    Returns:
        New global ReferenceCache instance
    """
    global _global_cache

    with _global_cache_lock:
        _global_cache = ReferenceCache(config or CacheConfig())
        return _global_cache


# ============================================================================
# Cache Key Utilities
# ============================================================================

def make_cache_key(
    validator_name: str,
    column: str | list[str],
    version: str = "v1",
    extra: str = "",
) -> str:
    """Create a standardized cache key.

    Args:
        validator_name: Name of the validator
        column: Column name or list of column names
        version: Version string for cache invalidation
        extra: Extra key component

    Returns:
        Cache key string
    """
    if isinstance(column, list):
        col_str = ":".join(sorted(column))
    else:
        col_str = column

    parts = [validator_name, col_str, version]
    if extra:
        parts.append(extra)

    return "|".join(parts)


def hash_dataframe(lf: pl.LazyFrame, sample_size: int = 1000) -> str:
    """Create a hash of a LazyFrame for cache key generation.

    Uses sampling to avoid full materialization.

    Args:
        lf: LazyFrame to hash
        sample_size: Number of rows to sample

    Returns:
        Hash string
    """
    # Get schema hash
    schema = lf.collect_schema()
    schema_str = str(sorted(schema.items()))

    # Sample data hash
    sample = lf.head(sample_size).collect()
    data_str = sample.to_pandas().to_json()

    combined = f"{schema_str}:{data_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]
