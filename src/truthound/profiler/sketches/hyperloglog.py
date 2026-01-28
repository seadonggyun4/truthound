"""HyperLogLog cardinality estimator implementation.

HyperLogLog is a probabilistic algorithm for estimating the number of
distinct elements in a multiset with O(1) memory complexity.

Reference:
    Flajolet, P., et al. "HyperLogLog: the analysis of a near-optimal
    cardinality estimation algorithm." (2007)
"""

from __future__ import annotations

import hashlib
import struct
import threading
from typing import Any

from truthound.profiler.sketches.protocols import (
    HyperLogLogConfig,
    SketchMetrics,
)


class HyperLogLog:
    """HyperLogLog cardinality estimator.

    Provides O(1) memory cardinality estimation with configurable precision.
    Thread-safe for concurrent add operations.

    Precision vs Memory vs Accuracy:
        precision=10: ~1KB memory, ±1.04% error
        precision=12: ~4KB memory, ±0.65% error (default)
        precision=14: ~16KB memory, ±0.41% error
        precision=16: ~64KB memory, ±0.26% error
        precision=18: ~256KB memory, ±0.16% error

    Example:
        hll = HyperLogLog(precision=12)
        for user_id in user_ids:
            hll.add(user_id)
        print(f"Distinct users: ~{hll.estimate():,}")

    Attributes:
        config: HyperLogLog configuration
        registers: Internal register array
    """

    def __init__(self, config: HyperLogLogConfig | None = None) -> None:
        """Initialize HyperLogLog with configuration.

        Args:
            config: HyperLogLog configuration. If None, uses defaults.
        """
        self.config = config or HyperLogLogConfig()
        self._registers: list[int] = [0] * self.config.num_registers
        self._lock = threading.Lock()
        self._elements_added: int = 0
        self._alpha: float = self._compute_alpha()

    def _compute_alpha(self) -> float:
        """Compute bias correction constant alpha."""
        m = self.config.num_registers
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def _hash(self, value: Any) -> int:
        """Hash a value to a 64-bit integer.

        Uses xxhash if available for better performance,
        falls back to MD5 otherwise.
        """
        value_bytes = str(value).encode("utf-8")

        try:
            import xxhash

            return xxhash.xxh64(value_bytes, seed=self.config.seed).intdigest()
        except ImportError:
            # Fallback to MD5 (slower but always available)
            hash_bytes = hashlib.md5(value_bytes).digest()[:8]
            return struct.unpack("<Q", hash_bytes)[0]

    def _rho(self, value: int, max_bits: int) -> int:
        """Count leading zeros + 1 (position of first 1-bit from right).

        This function finds the position of the rightmost 1-bit.
        For value=0, returns max_bits + 1.

        Args:
            value: Integer to analyze
            max_bits: Maximum bit position to check

        Returns:
            Position of first 1-bit (1-indexed)
        """
        if value == 0:
            return max_bits + 1

        count = 1
        while (value & 1) == 0 and count <= max_bits:
            count += 1
            value >>= 1

        return count

    def add(self, value: Any) -> None:
        """Add a value to the estimator.

        Thread-safe operation using fine-grained locking.

        Args:
            value: Any hashable value
        """
        hash_value = self._hash(value)

        # Use first 'precision' bits for bucket index
        bucket_index = hash_value >> (64 - self.config.precision)
        bucket_index = bucket_index & (self.config.num_registers - 1)

        # Use remaining bits for rho calculation
        remaining_bits = 64 - self.config.precision
        remaining = hash_value & ((1 << remaining_bits) - 1)
        rho = self._rho(remaining, remaining_bits)

        # Thread-safe register update
        with self._lock:
            self._registers[bucket_index] = max(self._registers[bucket_index], rho)
            self._elements_added += 1

    def add_batch(self, values: list[Any]) -> None:
        """Add multiple values efficiently.

        Batches lock acquisitions for better performance.

        Args:
            values: List of hashable values
        """
        if not values:
            return

        # Pre-compute all hashes and updates
        updates: dict[int, int] = {}
        for value in values:
            if value is None:
                continue

            hash_value = self._hash(value)
            bucket_index = hash_value >> (64 - self.config.precision)
            bucket_index = bucket_index & (self.config.num_registers - 1)

            remaining_bits = 64 - self.config.precision
            remaining = hash_value & ((1 << remaining_bits) - 1)
            rho = self._rho(remaining, remaining_bits)

            # Track max rho per bucket
            if bucket_index not in updates or rho > updates[bucket_index]:
                updates[bucket_index] = rho

        # Apply all updates with single lock
        with self._lock:
            for bucket_index, rho in updates.items():
                self._registers[bucket_index] = max(self._registers[bucket_index], rho)
            self._elements_added += len(values)

    def estimate(self) -> int:
        """Estimate the cardinality (number of distinct elements).

        Uses harmonic mean with small range correction (Linear Counting)
        and large range correction for 32-bit hash overflow.

        Returns:
            Estimated number of distinct elements
        """
        import math

        with self._lock:
            registers = self._registers.copy()

        # Raw estimate using harmonic mean
        indicator = sum(2.0 ** (-r) for r in registers)
        raw_estimate = self._alpha * (self.config.num_registers**2) / indicator

        # Small range correction (Linear Counting)
        if raw_estimate <= 2.5 * self.config.num_registers:
            zeros = registers.count(0)
            if zeros > 0:
                linear_count = self.config.num_registers * math.log(
                    self.config.num_registers / zeros
                )
                return int(linear_count)

        # Large range correction (for 32-bit hashes)
        # Not needed for 64-bit hashes used here
        return int(raw_estimate)

    def standard_error(self) -> float:
        """Return the standard error rate.

        Returns:
            Standard error as a ratio (e.g., 0.0065 = 0.65% for precision=12)
        """
        return self.config.expected_error

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Merge another HyperLogLog into a new instance.

        Args:
            other: Another HyperLogLog with same precision

        Returns:
            New HyperLogLog with merged data

        Raises:
            ValueError: If precision values don't match
        """
        if self.config.precision != other.config.precision:
            raise ValueError(
                f"Cannot merge HyperLogLog with different precision: "
                f"{self.config.precision} vs {other.config.precision}"
            )

        merged = HyperLogLog(self.config)
        with self._lock, other._lock:
            merged._registers = [
                max(a, b) for a, b in zip(self._registers, other._registers)
            ]
            merged._elements_added = self._elements_added + other._elements_added
        return merged

    def merge_inplace(self, other: "HyperLogLog") -> None:
        """Merge another HyperLogLog into this instance in-place.

        Args:
            other: Another HyperLogLog with same precision

        Raises:
            ValueError: If precision values don't match
        """
        if self.config.precision != other.config.precision:
            raise ValueError(
                f"Cannot merge HyperLogLog with different precision: "
                f"{self.config.precision} vs {other.config.precision}"
            )

        with self._lock, other._lock:
            for i in range(self.config.num_registers):
                self._registers[i] = max(self._registers[i], other._registers[i])
            self._elements_added += other._elements_added

    def memory_bytes(self) -> int:
        """Return memory usage in bytes.

        Returns:
            Approximate memory usage
        """
        # Each register is typically 6 bits, stored in a byte
        return self.config.num_registers

    def clear(self) -> None:
        """Reset the HyperLogLog to initial state."""
        with self._lock:
            self._registers = [0] * self.config.num_registers
            self._elements_added = 0

    def metrics(self) -> SketchMetrics:
        """Get current metrics about the sketch.

        Returns:
            SketchMetrics with current state
        """
        with self._lock:
            non_zero = sum(1 for r in self._registers if r > 0)
            fill_ratio = non_zero / self.config.num_registers

        return SketchMetrics(
            elements_added=self._elements_added,
            memory_bytes=self.memory_bytes(),
            estimated_error=self.standard_error(),
            fill_ratio=fill_ratio,
        )

    def __repr__(self) -> str:
        return (
            f"HyperLogLog(precision={self.config.precision}, "
            f"estimate={self.estimate():,}, "
            f"error=±{self.standard_error():.2%})"
        )


def create_hyperloglog(
    precision: int = 12,
    target_error: float | None = None,
    seed: int = 42,
) -> HyperLogLog:
    """Factory function for creating HyperLogLog instances.

    Args:
        precision: Number of precision bits (4-18)
        target_error: If provided, calculates optimal precision for this error rate
        seed: Random seed for hash functions

    Returns:
        Configured HyperLogLog instance
    """
    if target_error is not None:
        config = HyperLogLogConfig.for_error_rate(target_error, seed=seed)
    else:
        config = HyperLogLogConfig(precision=precision, seed=seed)
    return HyperLogLog(config)
