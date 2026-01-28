"""Factory for creating probabilistic data structures.

Provides a unified interface for creating and managing sketch data structures
with appropriate configurations for different use cases.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from truthound.profiler.sketches.protocols import (
    BloomFilterConfig,
    CountMinSketchConfig,
    HyperLogLogConfig,
    Sketch,
    SketchConfig,
)
from truthound.profiler.sketches.hyperloglog import HyperLogLog
from truthound.profiler.sketches.countmin import CountMinSketch
from truthound.profiler.sketches.bloom import BloomFilter


class SketchType(Enum):
    """Available sketch types."""

    HYPERLOGLOG = auto()
    COUNTMIN = auto()
    BLOOM = auto()


class SketchPreset(Enum):
    """Preset configurations for common use cases."""

    # Low memory, lower accuracy
    MINIMAL = auto()
    # Balanced memory/accuracy
    STANDARD = auto()
    # Higher memory, better accuracy
    HIGH_ACCURACY = auto()
    # Maximum accuracy, most memory
    MAXIMUM = auto()


# Preset configurations for each sketch type
_PRESETS: dict[SketchType, dict[SketchPreset, SketchConfig]] = {
    SketchType.HYPERLOGLOG: {
        SketchPreset.MINIMAL: HyperLogLogConfig(precision=10),  # ~1KB, ±1.04%
        SketchPreset.STANDARD: HyperLogLogConfig(precision=12),  # ~4KB, ±0.65%
        SketchPreset.HIGH_ACCURACY: HyperLogLogConfig(precision=14),  # ~16KB, ±0.41%
        SketchPreset.MAXIMUM: HyperLogLogConfig(precision=16),  # ~64KB, ±0.26%
    },
    SketchType.COUNTMIN: {
        SketchPreset.MINIMAL: CountMinSketchConfig(width=500, depth=3),
        SketchPreset.STANDARD: CountMinSketchConfig(width=2000, depth=5),
        SketchPreset.HIGH_ACCURACY: CountMinSketchConfig(width=10000, depth=7),
        SketchPreset.MAXIMUM: CountMinSketchConfig(width=50000, depth=10),
    },
    SketchType.BLOOM: {
        SketchPreset.MINIMAL: BloomFilterConfig(capacity=100_000, error_rate=0.1),
        SketchPreset.STANDARD: BloomFilterConfig(capacity=1_000_000, error_rate=0.01),
        SketchPreset.HIGH_ACCURACY: BloomFilterConfig(capacity=10_000_000, error_rate=0.001),
        SketchPreset.MAXIMUM: BloomFilterConfig(capacity=100_000_000, error_rate=0.0001),
    },
}


class SketchFactory:
    """Factory for creating sketch data structures.

    Provides methods to create sketches with custom or preset configurations,
    and utilities for selecting appropriate sketch types for different use cases.

    Example:
        factory = SketchFactory()

        # Create with preset
        hll = factory.create(SketchType.HYPERLOGLOG, preset=SketchPreset.STANDARD)

        # Create with custom config
        cms = factory.create_countmin(width=5000, depth=7)

        # Auto-select sketch type for use case
        sketch = factory.for_cardinality(error_rate=0.01)
        sketch = factory.for_frequency(epsilon=0.001, delta=0.01)
        sketch = factory.for_membership(capacity=1_000_000)
    """

    def create(
        self,
        sketch_type: SketchType,
        preset: SketchPreset | None = None,
        config: SketchConfig | None = None,
        **kwargs: Any,
    ) -> Sketch:
        """Create a sketch of the specified type.

        Args:
            sketch_type: Type of sketch to create
            preset: Optional preset configuration
            config: Optional custom configuration
            **kwargs: Additional configuration options

        Returns:
            Configured sketch instance

        Raises:
            ValueError: If invalid configuration
        """
        if config is None and preset is not None:
            config = _PRESETS[sketch_type][preset]

        if sketch_type == SketchType.HYPERLOGLOG:
            return self._create_hyperloglog(config, **kwargs)
        elif sketch_type == SketchType.COUNTMIN:
            return self._create_countmin(config, **kwargs)
        elif sketch_type == SketchType.BLOOM:
            return self._create_bloom(config, **kwargs)
        else:
            raise ValueError(f"Unknown sketch type: {sketch_type}")

    def _create_hyperloglog(
        self,
        config: SketchConfig | None = None,
        precision: int = 12,
        seed: int = 42,
        **kwargs: Any,
    ) -> HyperLogLog:
        """Create a HyperLogLog instance."""
        if config is not None:
            if isinstance(config, HyperLogLogConfig):
                return HyperLogLog(config)
            raise ValueError(f"Expected HyperLogLogConfig, got {type(config)}")
        return HyperLogLog(HyperLogLogConfig(precision=precision, seed=seed))

    def _create_countmin(
        self,
        config: SketchConfig | None = None,
        width: int = 2000,
        depth: int = 5,
        seed: int = 42,
        **kwargs: Any,
    ) -> CountMinSketch:
        """Create a CountMinSketch instance."""
        if config is not None:
            if isinstance(config, CountMinSketchConfig):
                return CountMinSketch(config)
            raise ValueError(f"Expected CountMinSketchConfig, got {type(config)}")
        return CountMinSketch(CountMinSketchConfig(width=width, depth=depth, seed=seed))

    def _create_bloom(
        self,
        config: SketchConfig | None = None,
        capacity: int = 1_000_000,
        error_rate: float = 0.01,
        seed: int = 42,
        **kwargs: Any,
    ) -> BloomFilter:
        """Create a BloomFilter instance."""
        if config is not None:
            if isinstance(config, BloomFilterConfig):
                return BloomFilter(config)
            raise ValueError(f"Expected BloomFilterConfig, got {type(config)}")
        return BloomFilter(BloomFilterConfig(capacity=capacity, error_rate=error_rate, seed=seed))

    # Convenience methods for specific use cases

    def create_hyperloglog(
        self,
        precision: int = 12,
        target_error: float | None = None,
        seed: int = 42,
    ) -> HyperLogLog:
        """Create a HyperLogLog for cardinality estimation.

        Args:
            precision: Number of precision bits (4-18)
            target_error: If provided, calculates optimal precision
            seed: Random seed

        Returns:
            Configured HyperLogLog instance
        """
        if target_error is not None:
            config = HyperLogLogConfig.for_error_rate(target_error, seed=seed)
        else:
            config = HyperLogLogConfig(precision=precision, seed=seed)
        return HyperLogLog(config)

    def create_countmin(
        self,
        width: int = 2000,
        depth: int = 5,
        epsilon: float | None = None,
        delta: float | None = None,
        seed: int = 42,
    ) -> CountMinSketch:
        """Create a CountMinSketch for frequency estimation.

        Args:
            width: Number of counters per row
            depth: Number of rows (hash functions)
            epsilon: Error bound (as fraction of total count)
            delta: Failure probability
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

    def create_bloom(
        self,
        capacity: int = 1_000_000,
        error_rate: float = 0.01,
        seed: int = 42,
    ) -> BloomFilter:
        """Create a BloomFilter for membership testing.

        Args:
            capacity: Expected number of elements
            error_rate: Target false positive rate
            seed: Random seed

        Returns:
            Configured BloomFilter instance
        """
        config = BloomFilterConfig(capacity=capacity, error_rate=error_rate, seed=seed)
        return BloomFilter(config)

    # Use-case oriented factory methods

    def for_cardinality(
        self,
        error_rate: float = 0.01,
        seed: int = 42,
    ) -> HyperLogLog:
        """Create a sketch optimized for cardinality estimation.

        Args:
            error_rate: Target standard error rate
            seed: Random seed

        Returns:
            HyperLogLog configured for target error rate
        """
        return self.create_hyperloglog(target_error=error_rate, seed=seed)

    def for_frequency(
        self,
        epsilon: float = 0.001,
        delta: float = 0.01,
        seed: int = 42,
    ) -> CountMinSketch:
        """Create a sketch optimized for frequency estimation.

        Args:
            epsilon: Error bound (as fraction of total count)
            delta: Failure probability
            seed: Random seed

        Returns:
            CountMinSketch configured for target accuracy
        """
        return self.create_countmin(epsilon=epsilon, delta=delta, seed=seed)

    def for_membership(
        self,
        capacity: int,
        error_rate: float = 0.01,
        seed: int = 42,
    ) -> BloomFilter:
        """Create a sketch optimized for membership testing.

        Args:
            capacity: Expected number of elements
            error_rate: Target false positive rate
            seed: Random seed

        Returns:
            BloomFilter configured for target accuracy
        """
        return self.create_bloom(capacity=capacity, error_rate=error_rate, seed=seed)


# Module-level factory instance for convenience
_factory = SketchFactory()


def create_sketch(
    sketch_type: SketchType | str,
    preset: SketchPreset | str | None = None,
    **kwargs: Any,
) -> Sketch:
    """Create a sketch data structure.

    Convenience function for creating sketches without instantiating the factory.

    Args:
        sketch_type: Type of sketch ("hyperloglog", "countmin", "bloom")
        preset: Preset configuration ("minimal", "standard", "high_accuracy", "maximum")
        **kwargs: Additional configuration options

    Returns:
        Configured sketch instance

    Example:
        hll = create_sketch("hyperloglog", precision=14)
        cms = create_sketch("countmin", epsilon=0.001, delta=0.01)
        bf = create_sketch("bloom", capacity=1_000_000, error_rate=0.01)
    """
    # Convert string to enum if needed
    if isinstance(sketch_type, str):
        sketch_type = SketchType[sketch_type.upper()]

    if isinstance(preset, str):
        preset = SketchPreset[preset.upper()]

    return _factory.create(sketch_type, preset=preset, **kwargs)
