"""Streaming ECDF and statistical test implementations.

This module provides memory-efficient implementations of statistical tests
that traditionally require loading two full datasets into memory.

Key Algorithms:
    - StreamingECDF: Approximate ECDF using T-Digest
    - StreamingKSTest: Streaming Kolmogorov-Smirnov test
    - StreamingStatistics: Online mean, variance, quantiles

Memory Complexity:
    - Traditional KS test: O(n + m) for n, m samples
    - Streaming KS test: O(compression) constant memory

Usage:
    class StreamingKSValidator(DriftValidator, StreamingECDFMixin):
        def validate(self, lf):
            # Build reference ECDF from stream
            ref_ecdf = self.build_streaming_ecdf(reference_lf, column)

            # Compute KS statistic vs current data stream
            ks_stat = self.streaming_ks_statistic(ref_ecdf, current_lf, column)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, TYPE_CHECKING
import heapq
import math

import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass
class Centroid:
    """A centroid in T-Digest.

    Represents a cluster of values with a mean and count.
    """

    mean: float
    count: int

    def __lt__(self, other: "Centroid") -> bool:
        return self.mean < other.mean


class TDigest:
    """T-Digest data structure for streaming quantile estimation.

    T-Digest provides accurate quantile estimates with bounded memory,
    especially accurate at the tails of the distribution.

    Reference: Dunning & Ertl, "Computing Extremely Accurate Quantiles
    Using t-Digests" (2019)

    Memory: O(compression) regardless of data size
    Accuracy: ~0.1% error at p=0.99, better at extremes

    Example:
        digest = TDigest(compression=100)
        for batch in data_stream:
            digest.update(batch)
        median = digest.quantile(0.5)
        p99 = digest.quantile(0.99)
    """

    def __init__(self, compression: float = 100.0):
        """Initialize T-Digest.

        Args:
            compression: Compression factor (higher = more accurate, more memory)
                        Typical values: 100-500
        """
        self.compression = compression
        self._centroids: list[Centroid] = []
        self._total_count = 0
        self._min = float("inf")
        self._max = float("-inf")
        self._buffer: list[float] = []
        self._buffer_size = int(compression * 2)

    def update(self, values: np.ndarray | list | float) -> None:
        """Update digest with new values.

        Args:
            values: Single value, list, or numpy array
        """
        if isinstance(values, (int, float)):
            values = [values]
        elif isinstance(values, np.ndarray):
            values = values.flatten().tolist()

        # Remove NaN/inf values
        values = [v for v in values if math.isfinite(v)]

        if not values:
            return

        self._buffer.extend(values)
        self._min = min(self._min, min(values))
        self._max = max(self._max, max(values))

        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Merge buffer into centroids."""
        if not self._buffer:
            return

        # Sort buffer
        self._buffer.sort()

        # Merge each value
        for value in self._buffer:
            self._add_centroid(Centroid(mean=value, count=1))

        self._buffer = []

        # Compress if too many centroids
        if len(self._centroids) > 3 * self.compression:
            self._compress()

    def _add_centroid(self, new_centroid: Centroid) -> None:
        """Add a centroid, potentially merging with neighbors."""
        self._total_count += new_centroid.count

        if not self._centroids:
            self._centroids.append(new_centroid)
            return

        # Find insertion point
        idx = self._find_insertion_point(new_centroid.mean)

        # Try to merge with nearest centroid
        if idx < len(self._centroids):
            existing = self._centroids[idx]
            merged_count = existing.count + new_centroid.count

            # Check if merge is allowed (size limit based on position)
            q = self._quantile_at_count(self._count_before(idx) + merged_count / 2)
            max_size = self._max_size_at_quantile(q)

            if merged_count <= max_size:
                # Merge
                new_mean = (
                    existing.mean * existing.count + new_centroid.mean * new_centroid.count
                ) / merged_count
                self._centroids[idx] = Centroid(mean=new_mean, count=merged_count)
                return

        # Insert as new centroid
        self._centroids.insert(idx, new_centroid)

    def _find_insertion_point(self, value: float) -> int:
        """Binary search for insertion point."""
        lo, hi = 0, len(self._centroids)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._centroids[mid].mean < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _count_before(self, idx: int) -> int:
        """Count of all values before index."""
        return sum(c.count for c in self._centroids[:idx])

    def _quantile_at_count(self, count: float) -> float:
        """Get quantile at given count."""
        if self._total_count == 0:
            return 0.5
        return count / self._total_count

    def _max_size_at_quantile(self, q: float) -> float:
        """Maximum centroid size at given quantile.

        Uses the scale function k(q) = δ/2 * (arcsin(2q-1)/π + 1/2)
        which gives smaller centroids at the tails.
        """
        # Scale function derivative
        k_prime = (
            self.compression
            * math.pi
            / (2 * math.sqrt(q * (1 - q) + 1e-10))
        )
        return max(1, int(self._total_count / k_prime))

    def _compress(self) -> None:
        """Compress centroids to reduce memory."""
        if len(self._centroids) <= self.compression:
            return

        # Merge adjacent centroids greedily
        new_centroids = []
        i = 0

        while i < len(self._centroids):
            current = self._centroids[i]

            # Try to merge with next
            while i + 1 < len(self._centroids):
                next_c = self._centroids[i + 1]
                merged_count = current.count + next_c.count

                q = self._quantile_at_count(
                    sum(c.count for c in new_centroids) + merged_count / 2
                )
                max_size = self._max_size_at_quantile(q)

                if merged_count <= max_size:
                    # Merge
                    new_mean = (
                        current.mean * current.count + next_c.mean * next_c.count
                    ) / merged_count
                    current = Centroid(mean=new_mean, count=merged_count)
                    i += 1
                else:
                    break

            new_centroids.append(current)
            i += 1

        self._centroids = new_centroids

    def quantile(self, q: float) -> float:
        """Get quantile value.

        Args:
            q: Quantile (0 to 1)

        Returns:
            Value at quantile
        """
        self._flush_buffer()

        if not self._centroids:
            return float("nan")

        if q <= 0:
            return self._min
        if q >= 1:
            return self._max

        target_count = q * self._total_count
        cumulative = 0

        for i, centroid in enumerate(self._centroids):
            next_cumulative = cumulative + centroid.count

            if next_cumulative >= target_count:
                # Interpolate within centroid
                if i == 0:
                    left = self._min
                else:
                    left = (self._centroids[i - 1].mean + centroid.mean) / 2

                if i == len(self._centroids) - 1:
                    right = self._max
                else:
                    right = (centroid.mean + self._centroids[i + 1].mean) / 2

                # Linear interpolation
                frac = (target_count - cumulative) / centroid.count
                return left + frac * (right - left)

            cumulative = next_cumulative

        return self._max

    def cdf(self, value: float) -> float:
        """Get CDF value (proportion of values <= x).

        Args:
            value: Value to query

        Returns:
            Proportion of values <= value
        """
        self._flush_buffer()

        if not self._centroids:
            return 0.5
        if value <= self._min:
            return 0.0
        if value >= self._max:
            return 1.0

        cumulative = 0

        for i, centroid in enumerate(self._centroids):
            if value < centroid.mean:
                # Interpolate
                if i == 0:
                    left = self._min
                    left_count = 0
                else:
                    left = self._centroids[i - 1].mean
                    left_count = cumulative

                # Proportion within this region
                frac = (value - left) / (centroid.mean - left + 1e-10)
                return (left_count + frac * centroid.count / 2) / self._total_count

            cumulative += centroid.count

        return 1.0

    @property
    def count(self) -> int:
        """Total count of values."""
        return self._total_count + len(self._buffer)

    @property
    def mean(self) -> float:
        """Estimate mean from centroids."""
        self._flush_buffer()
        if self._total_count == 0:
            return float("nan")
        return sum(c.mean * c.count for c in self._centroids) / self._total_count

    def merge(self, other: "TDigest") -> "TDigest":
        """Merge with another T-Digest.

        Args:
            other: Another T-Digest

        Returns:
            New merged T-Digest
        """
        result = TDigest(compression=max(self.compression, other.compression))

        # Flush both buffers
        self._flush_buffer()
        other._flush_buffer()

        # Merge all centroids
        all_centroids = self._centroids + other._centroids
        all_centroids.sort()

        for c in all_centroids:
            result._add_centroid(Centroid(mean=c.mean, count=c.count))

        result._min = min(self._min, other._min)
        result._max = max(self._max, other._max)
        result._compress()

        return result


class StreamingECDF:
    """Streaming Empirical CDF using T-Digest.

    Provides memory-efficient ECDF computation for large datasets.

    Example:
        ecdf = StreamingECDF(compression=200)
        for batch in data_stream:
            ecdf.update(batch)

        # Query CDF at specific points
        cdf_values = ecdf.cdf(query_points)

        # Get quantiles
        median = ecdf.quantile(0.5)
    """

    def __init__(self, compression: float = 200.0):
        """Initialize streaming ECDF.

        Args:
            compression: T-Digest compression factor
        """
        self._digest = TDigest(compression=compression)

    def update(self, values: np.ndarray) -> None:
        """Update ECDF with new values.

        Args:
            values: Array of values
        """
        self._digest.update(values)

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """Compute CDF at given points.

        Args:
            x: Query points

        Returns:
            CDF values
        """
        if isinstance(x, (int, float)):
            return self._digest.cdf(x)

        return np.array([self._digest.cdf(xi) for xi in x])

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        """Compute quantile values.

        Args:
            q: Quantile(s) to compute

        Returns:
            Value(s) at quantile(s)
        """
        if isinstance(q, (int, float)):
            return self._digest.quantile(q)

        return np.array([self._digest.quantile(qi) for qi in q])

    @property
    def count(self) -> int:
        """Number of values seen."""
        return self._digest.count

    @property
    def min(self) -> float:
        """Minimum value."""
        return self._digest._min

    @property
    def max(self) -> float:
        """Maximum value."""
        return self._digest._max

    def merge(self, other: "StreamingECDF") -> "StreamingECDF":
        """Merge with another ECDF."""
        result = StreamingECDF()
        result._digest = self._digest.merge(other._digest)
        return result


@dataclass
class StreamingStatistics:
    """Streaming statistics for distribution comparison.

    Tracks running statistics that can be used for statistical tests
    without storing the full dataset.

    Tracks:
        - Count, mean, variance (Welford's algorithm)
        - Min, max
        - Quantiles via T-Digest
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences
    min_val: float = float("inf")
    max_val: float = float("-inf")
    _digest: TDigest = field(default_factory=lambda: TDigest(compression=100))

    def update(self, values: np.ndarray) -> None:
        """Update statistics with new values.

        Args:
            values: Array of values (NaN values are ignored)
        """
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return

        for x in values:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

        self.min_val = min(self.min_val, values.min())
        self.max_val = max(self.max_val, values.max())
        self._digest.update(values)

    def update_batch(self, values: np.ndarray) -> None:
        """Batch update for efficiency."""
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return

        n = len(values)
        batch_mean = values.mean()
        batch_var = values.var(ddof=0)

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_var * n
        else:
            total = self.count + n
            delta = batch_mean - self.mean

            self.mean = (self.count * self.mean + n * batch_mean) / total
            self.m2 += batch_var * n + delta**2 * self.count * n / total
            self.count = total

        self.min_val = min(self.min_val, values.min())
        self.max_val = max(self.max_val, values.max())
        self._digest.update(values)

    @property
    def variance(self) -> float:
        """Sample variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    def quantile(self, q: float) -> float:
        """Get quantile value."""
        return self._digest.quantile(q)

    def cdf(self, x: float) -> float:
        """Get CDF value at x."""
        return self._digest.cdf(x)


class StreamingECDFMixin:
    """Mixin providing streaming ECDF and statistical test capabilities.

    This mixin enables memory-efficient statistical tests like KS test
    that traditionally require loading both datasets into memory.

    Example:
        class StreamingKSValidator(DriftValidator, StreamingECDFMixin):
            def __init__(self, ...):
                ...
                # Build reference ECDF once
                self._ref_ecdf = None

            def validate(self, lf):
                # Build reference ECDF if not cached
                if self._ref_ecdf is None:
                    self._ref_ecdf = self.build_streaming_ecdf(
                        self.reference_data, self.column
                    )

                # Compute KS statistic against current data
                ks_stat, p_value = self.streaming_ks_test(
                    self._ref_ecdf, lf, self.column
                )
    """

    def build_streaming_ecdf(
        self,
        lf: "pl.LazyFrame",
        column: str,
        compression: float = 200.0,
        chunk_size: int = 100000,
    ) -> StreamingECDF:
        """Build ECDF from streaming data.

        Args:
            lf: Input LazyFrame
            column: Column to analyze
            compression: T-Digest compression factor
            chunk_size: Processing chunk size

        Returns:
            StreamingECDF instance
        """
        import polars as pl
        from truthound.validators.memory.base import DataChunker

        ecdf = StreamingECDF(compression=compression)

        chunker = DataChunker(
            chunk_size=chunk_size,
            columns=[column],
            drop_nulls=True,
        )

        for chunk_df in chunker.iterate(lf):
            values = chunk_df.to_series().to_numpy()
            ecdf.update(values)

        return ecdf

    def build_streaming_statistics(
        self,
        lf: "pl.LazyFrame",
        column: str,
        chunk_size: int = 100000,
    ) -> StreamingStatistics:
        """Build streaming statistics from data.

        Args:
            lf: Input LazyFrame
            column: Column to analyze
            chunk_size: Processing chunk size

        Returns:
            StreamingStatistics instance
        """
        import polars as pl
        from truthound.validators.memory.base import DataChunker

        stats = StreamingStatistics()

        chunker = DataChunker(
            chunk_size=chunk_size,
            columns=[column],
            drop_nulls=True,
        )

        for chunk_df in chunker.iterate(lf):
            values = chunk_df.to_series().to_numpy()
            stats.update_batch(values)

        return stats

    def streaming_ks_statistic(
        self,
        ref_ecdf: StreamingECDF,
        lf: "pl.LazyFrame",
        column: str,
        chunk_size: int = 100000,
    ) -> float:
        """Compute KS statistic against streaming current data.

        Uses the reference ECDF and computes the maximum deviation
        as current data streams through.

        Args:
            ref_ecdf: Reference ECDF
            lf: Current data LazyFrame
            column: Column to compare
            chunk_size: Processing chunk size

        Returns:
            KS statistic (max |F_ref(x) - F_curr(x)|)
        """
        from truthound.validators.memory.base import DataChunker

        # Build current ECDF
        curr_ecdf = self.build_streaming_ecdf(lf, column, chunk_size=chunk_size)

        # Compute max deviation at key quantile points
        # Using more points for better accuracy
        quantile_points = np.linspace(0.001, 0.999, 1000)

        max_deviation = 0.0

        for q in quantile_points:
            # Get values at this quantile from both distributions
            ref_val = ref_ecdf.quantile(q)
            curr_val = curr_ecdf.quantile(q)

            # Compute CDF difference at these points
            ref_cdf_at_ref = ref_ecdf.cdf(ref_val)
            curr_cdf_at_ref = curr_ecdf.cdf(ref_val)
            dev1 = abs(ref_cdf_at_ref - curr_cdf_at_ref)

            ref_cdf_at_curr = ref_ecdf.cdf(curr_val)
            curr_cdf_at_curr = curr_ecdf.cdf(curr_val)
            dev2 = abs(ref_cdf_at_curr - curr_cdf_at_curr)

            max_deviation = max(max_deviation, dev1, dev2)

        return max_deviation

    def streaming_ks_test(
        self,
        ref_ecdf: StreamingECDF,
        lf: "pl.LazyFrame",
        column: str,
        chunk_size: int = 100000,
    ) -> tuple[float, float]:
        """Perform streaming KS test.

        Args:
            ref_ecdf: Reference ECDF
            lf: Current data LazyFrame
            column: Column to compare
            chunk_size: Processing chunk size

        Returns:
            Tuple of (ks_statistic, approximate_p_value)
        """
        ks_stat = self.streaming_ks_statistic(ref_ecdf, lf, column, chunk_size)

        # Approximate p-value using asymptotic distribution
        # P(D_n > x) ≈ 2 * sum_{k=1}^inf (-1)^(k+1) * exp(-2 * k^2 * n * x^2)
        n_eff = min(ref_ecdf.count, lf.select(pl.len()).collect().item())
        if n_eff == 0:
            return ks_stat, 1.0

        # Simplified asymptotic formula
        lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * ks_stat

        # Kolmogorov distribution approximation
        p_value = 0.0
        for k in range(1, 100):
            term = 2 * ((-1) ** (k + 1)) * math.exp(-2 * k * k * lambda_val * lambda_val)
            p_value += term
            if abs(term) < 1e-10:
                break

        p_value = max(0.0, min(1.0, p_value))

        return ks_stat, p_value

    def streaming_wasserstein(
        self,
        ref_stats: StreamingStatistics,
        lf: "pl.LazyFrame",
        column: str,
        chunk_size: int = 100000,
    ) -> float:
        """Compute approximate Wasserstein distance using streaming statistics.

        Uses the quantile function approach for memory efficiency.

        Args:
            ref_stats: Reference statistics
            lf: Current data LazyFrame
            column: Column to compare
            chunk_size: Processing chunk size

        Returns:
            Approximate Wasserstein distance
        """
        # Build current statistics
        curr_stats = self.build_streaming_statistics(lf, column, chunk_size)

        # Approximate Wasserstein using quantile differences
        n_points = 100
        quantiles = np.linspace(0.01, 0.99, n_points)

        total_diff = 0.0
        for q in quantiles:
            ref_val = ref_stats.quantile(q)
            curr_val = curr_stats.quantile(q)
            total_diff += abs(ref_val - curr_val)

        return total_diff / n_points


# Import polars for type hints
try:
    import polars as pl
except ImportError:
    pl = None
