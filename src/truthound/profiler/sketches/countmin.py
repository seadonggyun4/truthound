"""Count-Min Sketch for frequency estimation.

Count-Min Sketch is a probabilistic data structure for estimating
frequencies of elements in a data stream with limited memory.

Properties:
    - Space: O(width × depth) = O(1/ε × log(1/δ))
    - Query time: O(depth)
    - Update time: O(depth)
    - Never underestimates true count
    - May overestimate by at most εN with probability 1-δ

Reference:
    Cormode, G., & Muthukrishnan, S. (2005). "An improved data stream summary:
    the count-min sketch and its applications."
"""

from __future__ import annotations

import hashlib
import struct
import threading
from typing import Any

from truthound.profiler.sketches.protocols import (
    CountMinSketchConfig,
    SketchMetrics,
)


class CountMinSketch:
    """Count-Min Sketch for frequency estimation.

    Provides sub-linear space frequency estimation with configurable accuracy.
    Useful for finding heavy hitters (frequent elements) in data streams.

    Example:
        cms = CountMinSketch(width=2000, depth=5)
        for item in stream:
            cms.add(item)

        # Check frequency of a specific item
        freq = cms.estimate_frequency("popular_item")

        # Find items that appear in >1% of stream
        heavy_hitters = cms.get_heavy_hitters(threshold=0.01)

    Attributes:
        config: CountMinSketch configuration
    """

    def __init__(self, config: CountMinSketchConfig | None = None) -> None:
        """Initialize CountMinSketch with configuration.

        Args:
            config: CountMinSketch configuration. If None, uses defaults.
        """
        self.config = config or CountMinSketchConfig()
        self._table: list[list[int]] = [
            [0] * self.config.width for _ in range(self.config.depth)
        ]
        self._lock = threading.Lock()
        self._total_count: int = 0
        self._tracked_items: dict[Any, int] = {}
        self._track_threshold: float = 0.001

    def _hash(self, value: Any, hash_idx: int) -> int:
        """Compute hash for a value using the specified hash function index.

        Uses double hashing technique: h_i(x) = h1(x) + i*h2(x)

        Args:
            value: Value to hash
            hash_idx: Index of hash function (0 to depth-1)

        Returns:
            Hash value in range [0, width)
        """
        value_bytes = str(value).encode("utf-8")

        try:
            import xxhash

            h1 = xxhash.xxh64(value_bytes, seed=self.config.seed).intdigest()
            h2 = xxhash.xxh64(value_bytes, seed=self.config.seed + 1).intdigest()
        except ImportError:
            # Fallback to MD5-based hashing
            h1_bytes = hashlib.md5(value_bytes + b"\x00").digest()
            h2_bytes = hashlib.md5(value_bytes + b"\x01").digest()
            h1 = struct.unpack("<Q", h1_bytes[:8])[0]
            h2 = struct.unpack("<Q", h2_bytes[:8])[0]

        # Double hashing: h_i(x) = (h1(x) + i * h2(x)) mod width
        return (h1 + hash_idx * h2) % self.config.width

    def add(self, value: Any, count: int = 1) -> None:
        """Add a value to the sketch.

        Args:
            value: Any hashable value
            count: Number of occurrences to add (default: 1)
        """
        if count <= 0:
            return

        with self._lock:
            min_count = float("inf")
            for i in range(self.config.depth):
                idx = self._hash(value, i)
                self._table[i][idx] += count
                min_count = min(min_count, self._table[i][idx])

            self._total_count += count

            # Track items that might be heavy hitters
            if min_count >= self._total_count * self._track_threshold:
                self._tracked_items[value] = int(min_count)

    def add_batch(self, values: list[Any]) -> None:
        """Add multiple values efficiently.

        Args:
            values: List of hashable values
        """
        if not values:
            return

        # Count occurrences first
        counts: dict[Any, int] = {}
        for value in values:
            if value is not None:
                counts[value] = counts.get(value, 0) + 1

        # Apply updates with single lock
        with self._lock:
            for value, count in counts.items():
                min_count = float("inf")
                for i in range(self.config.depth):
                    idx = self._hash(value, i)
                    self._table[i][idx] += count
                    min_count = min(min_count, self._table[i][idx])

                self._total_count += count

                # Track potential heavy hitters
                if min_count >= self._total_count * self._track_threshold:
                    self._tracked_items[value] = int(min_count)

    def _estimate_frequency_unlocked(self, value: Any) -> int:
        """Estimate frequency without locking (internal use).

        Must be called while holding self._lock.
        """
        return min(
            self._table[i][self._hash(value, i)]
            for i in range(self.config.depth)
        )

    def estimate_frequency(self, value: Any) -> int:
        """Estimate the frequency of a value.

        Returns the minimum count across all hash function positions,
        which is guaranteed to never underestimate the true count.

        Args:
            value: Value to estimate frequency for

        Returns:
            Estimated count (may overestimate, never underestimates)
        """
        with self._lock:
            return self._estimate_frequency_unlocked(value)

    def get_heavy_hitters(self, threshold: float) -> list[tuple[Any, int]]:
        """Get elements that appear frequently.

        Returns items whose estimated frequency exceeds threshold * total_count.

        Args:
            threshold: Minimum frequency ratio (0.0-1.0)

        Returns:
            List of (value, estimated_count) tuples, sorted by count descending
        """
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be (0, 1], got {threshold}")

        min_count = int(self._total_count * threshold)
        results = []

        with self._lock:
            for value in self._tracked_items:
                est = self._estimate_frequency_unlocked(value)
                if est >= min_count:
                    results.append((value, est))

        # Sort by estimated count descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Merge another CountMinSketch into a new instance.

        Args:
            other: Another CountMinSketch with same dimensions

        Returns:
            New CountMinSketch with merged data

        Raises:
            ValueError: If dimensions don't match
        """
        if (
            self.config.width != other.config.width
            or self.config.depth != other.config.depth
        ):
            raise ValueError(
                f"Cannot merge CountMinSketch with different dimensions: "
                f"({self.config.width}×{self.config.depth}) vs "
                f"({other.config.width}×{other.config.depth})"
            )

        merged = CountMinSketch(self.config)
        with self._lock, other._lock:
            for i in range(self.config.depth):
                for j in range(self.config.width):
                    merged._table[i][j] = self._table[i][j] + other._table[i][j]
            merged._total_count = self._total_count + other._total_count

            # Merge tracked items
            all_items = set(self._tracked_items.keys()) | set(other._tracked_items.keys())
            for item in all_items:
                merged._tracked_items[item] = merged.estimate_frequency(item)

        return merged

    def merge_inplace(self, other: "CountMinSketch") -> None:
        """Merge another CountMinSketch into this instance in-place.

        Args:
            other: Another CountMinSketch with same dimensions

        Raises:
            ValueError: If dimensions don't match
        """
        if (
            self.config.width != other.config.width
            or self.config.depth != other.config.depth
        ):
            raise ValueError(
                f"Cannot merge CountMinSketch with different dimensions: "
                f"({self.config.width}×{self.config.depth}) vs "
                f"({other.config.width}×{other.config.depth})"
            )

        with self._lock, other._lock:
            for i in range(self.config.depth):
                for j in range(self.config.width):
                    self._table[i][j] += other._table[i][j]
            self._total_count += other._total_count

            # Merge tracked items
            for item in other._tracked_items:
                self._tracked_items[item] = self.estimate_frequency(item)

    def memory_bytes(self) -> int:
        """Return memory usage in bytes.

        Returns:
            Approximate memory usage (assuming 8 bytes per counter)
        """
        # Each counter is typically 8 bytes (64-bit int)
        return self.config.width * self.config.depth * 8

    def clear(self) -> None:
        """Reset the CountMinSketch to initial state."""
        with self._lock:
            self._table = [
                [0] * self.config.width for _ in range(self.config.depth)
            ]
            self._total_count = 0
            self._tracked_items.clear()

    @property
    def total_count(self) -> int:
        """Return total number of items added."""
        return self._total_count

    @property
    def error_bound(self) -> float:
        """Return the error bound as fraction of total count.

        With high probability, estimates are within:
            true_count <= estimate <= true_count + error_bound * total_count
        """
        return self.config.expected_error

    @property
    def confidence(self) -> float:
        """Return the confidence level for error bounds."""
        return self.config.confidence

    def metrics(self) -> SketchMetrics:
        """Get current metrics about the sketch.

        Returns:
            SketchMetrics with current state
        """
        with self._lock:
            non_zero = sum(
                1 for row in self._table for cell in row if cell > 0
            )
            total_cells = self.config.width * self.config.depth
            fill_ratio = non_zero / total_cells

        return SketchMetrics(
            elements_added=self._total_count,
            memory_bytes=self.memory_bytes(),
            estimated_error=self.error_bound,
            fill_ratio=fill_ratio,
        )

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={self.config.width}, depth={self.config.depth}, "
            f"total={self._total_count:,}, error=ε={self.error_bound:.4f})"
        )


def create_countmin(
    width: int = 2000,
    depth: int = 5,
    epsilon: float | None = None,
    delta: float | None = None,
    seed: int = 42,
) -> CountMinSketch:
    """Factory function for creating CountMinSketch instances.

    Args:
        width: Number of counters per row
        depth: Number of rows (hash functions)
        epsilon: If provided with delta, calculates optimal dimensions
        delta: Probability of exceeding epsilon error
        seed: Random seed

    Returns:
        Configured CountMinSketch instance
    """
    if epsilon is not None and delta is not None:
        config = CountMinSketchConfig.for_error_and_confidence(
            epsilon=epsilon, delta=delta, seed=seed
        )
    else:
        config = CountMinSketchConfig(width=width, depth=depth, seed=seed)
    return CountMinSketch(config)
