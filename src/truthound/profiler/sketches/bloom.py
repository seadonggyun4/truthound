"""Bloom Filter for set membership testing.

A Bloom filter is a space-efficient probabilistic data structure for
testing whether an element is a member of a set. False positive matches
are possible, but false negatives are not.

Properties:
    - Space: O(n) bits for n expected elements
    - Query time: O(k) where k = number of hash functions
    - Insert time: O(k)
    - No false negatives
    - False positives occur with configurable probability

Reference:
    Bloom, B. H. (1970). "Space/time trade-offs in hash coding with
    allowable errors."
"""

from __future__ import annotations

import hashlib
import math
import struct
import threading
from typing import Any

from truthound.profiler.sketches.protocols import (
    BloomFilterConfig,
    SketchMetrics,
)


class BloomFilter:
    """Bloom Filter for probabilistic set membership testing.

    Space-efficient structure for testing if an element is possibly in a set.
    Never produces false negatives, but may produce false positives with
    configurable probability.

    Example:
        # Create filter for 1M items with 1% false positive rate
        bf = BloomFilter(capacity=1_000_000, error_rate=0.01)

        # Add items
        for item in items:
            bf.add(item)

        # Test membership
        if bf.contains(query_item):
            print("Item possibly in set")
        else:
            print("Item definitely not in set")

    Attributes:
        config: BloomFilter configuration
    """

    def __init__(self, config: BloomFilterConfig | None = None) -> None:
        """Initialize BloomFilter with configuration.

        Args:
            config: BloomFilter configuration. If None, uses defaults.
        """
        self.config = config or BloomFilterConfig()
        self._size: int = self.config.optimal_size
        self._hash_count: int = self.config.optimal_hash_count
        # Use bytearray for efficient bit storage
        self._bits: bytearray = bytearray((self._size + 7) // 8)
        self._lock = threading.Lock()
        self._elements_added: int = 0

    def _hash(self, value: Any, hash_idx: int) -> int:
        """Compute hash for a value using double hashing.

        Args:
            value: Value to hash
            hash_idx: Index of hash function (0 to hash_count-1)

        Returns:
            Hash value in range [0, size)
        """
        value_bytes = str(value).encode("utf-8")

        try:
            import xxhash

            h1 = xxhash.xxh64(value_bytes, seed=self.config.seed).intdigest()
            h2 = xxhash.xxh64(value_bytes, seed=self.config.seed + 1).intdigest()
        except ImportError:
            h1_bytes = hashlib.md5(value_bytes + b"\x00").digest()
            h2_bytes = hashlib.md5(value_bytes + b"\x01").digest()
            h1 = struct.unpack("<Q", h1_bytes[:8])[0]
            h2 = struct.unpack("<Q", h2_bytes[:8])[0]

        return (h1 + hash_idx * h2) % self._size

    def _set_bit(self, bit_idx: int) -> None:
        """Set a bit in the bit array."""
        byte_idx = bit_idx // 8
        bit_pos = bit_idx % 8
        self._bits[byte_idx] |= (1 << bit_pos)

    def _get_bit(self, bit_idx: int) -> bool:
        """Get a bit from the bit array."""
        byte_idx = bit_idx // 8
        bit_pos = bit_idx % 8
        return bool(self._bits[byte_idx] & (1 << bit_pos))

    def add(self, value: Any) -> None:
        """Add a value to the filter.

        Args:
            value: Any hashable value
        """
        with self._lock:
            for i in range(self._hash_count):
                bit_idx = self._hash(value, i)
                self._set_bit(bit_idx)
            self._elements_added += 1

    def add_batch(self, values: list[Any]) -> None:
        """Add multiple values efficiently.

        Args:
            values: List of hashable values
        """
        if not values:
            return

        with self._lock:
            for value in values:
                if value is not None:
                    for i in range(self._hash_count):
                        bit_idx = self._hash(value, i)
                        self._set_bit(bit_idx)
                    self._elements_added += 1

    def contains(self, value: Any) -> bool:
        """Test if a value might be in the set.

        Args:
            value: Value to test

        Returns:
            True if possibly present (may be false positive)
            False if definitely not present (no false negatives)
        """
        with self._lock:
            for i in range(self._hash_count):
                bit_idx = self._hash(value, i)
                if not self._get_bit(bit_idx):
                    return False
            return True

    def false_positive_rate(self) -> float:
        """Calculate the current false positive probability.

        The formula is: (1 - e^(-kn/m))^k
        where k = hash count, n = elements added, m = bit array size

        Returns:
            Estimated false positive rate
        """
        if self._elements_added == 0:
            return 0.0

        # (1 - e^(-kn/m))^k
        k = self._hash_count
        n = self._elements_added
        m = self._size

        fill_ratio = 1.0 - math.exp(-k * n / m)
        return fill_ratio**k

    def merge(self, other: "BloomFilter") -> "BloomFilter":
        """Merge another BloomFilter into a new instance.

        Args:
            other: Another BloomFilter with same configuration

        Returns:
            New BloomFilter with merged data

        Raises:
            ValueError: If configurations don't match
        """
        if self._size != other._size or self._hash_count != other._hash_count:
            raise ValueError(
                f"Cannot merge BloomFilters with different configurations: "
                f"({self._size}, {self._hash_count}) vs "
                f"({other._size}, {other._hash_count})"
            )

        merged = BloomFilter(self.config)
        with self._lock, other._lock:
            # OR the bit arrays
            for i in range(len(self._bits)):
                merged._bits[i] = self._bits[i] | other._bits[i]
            # Approximate element count (actual may be less due to overlap)
            merged._elements_added = self._elements_added + other._elements_added

        return merged

    def merge_inplace(self, other: "BloomFilter") -> None:
        """Merge another BloomFilter into this instance in-place.

        Args:
            other: Another BloomFilter with same configuration

        Raises:
            ValueError: If configurations don't match
        """
        if self._size != other._size or self._hash_count != other._hash_count:
            raise ValueError(
                f"Cannot merge BloomFilters with different configurations: "
                f"({self._size}, {self._hash_count}) vs "
                f"({other._size}, {other._hash_count})"
            )

        with self._lock, other._lock:
            for i in range(len(self._bits)):
                self._bits[i] |= other._bits[i]
            self._elements_added += other._elements_added

    def memory_bytes(self) -> int:
        """Return memory usage in bytes.

        Returns:
            Size of bit array in bytes
        """
        return len(self._bits)

    def clear(self) -> None:
        """Reset the BloomFilter to initial state."""
        with self._lock:
            self._bits = bytearray((self._size + 7) // 8)
            self._elements_added = 0

    @property
    def fill_ratio(self) -> float:
        """Return the ratio of set bits to total bits."""
        with self._lock:
            set_bits = sum(bin(byte).count("1") for byte in self._bits)
            return set_bits / self._size

    @property
    def remaining_capacity(self) -> int:
        """Estimate remaining capacity before error rate is exceeded."""
        # Solve for n in: (1 - e^(-kn/m))^k = target_error_rate
        current_fp = self.false_positive_rate()
        if current_fp >= self.config.error_rate:
            return 0

        # Binary search for remaining capacity
        low, high = 0, self.config.capacity
        while low < high:
            mid = (low + high + 1) // 2
            # Estimate FP rate if we add 'mid' more elements
            test_n = self._elements_added + mid
            test_fp = (1.0 - math.exp(-self._hash_count * test_n / self._size)) ** self._hash_count
            if test_fp <= self.config.error_rate:
                low = mid
            else:
                high = mid - 1

        return low

    def metrics(self) -> SketchMetrics:
        """Get current metrics about the filter.

        Returns:
            SketchMetrics with current state
        """
        return SketchMetrics(
            elements_added=self._elements_added,
            memory_bytes=self.memory_bytes(),
            estimated_error=self.false_positive_rate(),
            fill_ratio=self.fill_ratio,
        )

    def __repr__(self) -> str:
        return (
            f"BloomFilter(capacity={self.config.capacity:,}, "
            f"elements={self._elements_added:,}, "
            f"fp_rate={self.false_positive_rate():.2%})"
        )

    def __contains__(self, value: Any) -> bool:
        """Enable 'in' operator support."""
        return self.contains(value)


def create_bloom_filter(
    capacity: int = 1_000_000,
    error_rate: float = 0.01,
    seed: int = 42,
) -> BloomFilter:
    """Factory function for creating BloomFilter instances.

    Args:
        capacity: Expected number of elements
        error_rate: Target false positive rate (0.0-1.0)
        seed: Random seed

    Returns:
        Configured BloomFilter instance
    """
    config = BloomFilterConfig(capacity=capacity, error_rate=error_rate, seed=seed)
    return BloomFilter(config)
