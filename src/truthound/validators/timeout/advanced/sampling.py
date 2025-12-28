"""Adaptive sampling based on data characteristics.

This module provides intelligent sampling strategies that adjust
sample sizes based on data characteristics and time constraints.

Sampling strategies:
- Uniform: Random sampling with fixed ratio
- Stratified: Preserves distribution of categorical columns
- Reservoir: Streaming-friendly sampling for large datasets
- Adaptive: Automatically adjusts based on time budget

Example:
    from truthound.validators.timeout.advanced.sampling import (
        AdaptiveSampler,
        calculate_sample_size,
    )

    sampler = AdaptiveSampler()

    # Calculate optimal sample size for time budget
    sample_size = sampler.calculate_size(
        total_rows=1_000_000,
        time_budget_seconds=10.0,
        min_sample=1000,
    )

    # Sample data
    result = sampler.sample(data, sample_size)
"""

from __future__ import annotations

import math
import random
import statistics
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Sequence, TypeVar

T = TypeVar("T")


class SamplingMethod(str, Enum):
    """Sampling methods."""

    UNIFORM = "uniform"
    STRATIFIED = "stratified"
    RESERVOIR = "reservoir"
    SYSTEMATIC = "systematic"
    ADAPTIVE = "adaptive"


@dataclass
class DataCharacteristics:
    """Characteristics of a dataset.

    Attributes:
        row_count: Number of rows
        column_count: Number of columns
        estimated_bytes: Estimated size in bytes
        null_ratio: Ratio of null values
        unique_ratio: Average ratio of unique values per column
        has_categorical: Whether dataset has categorical columns
        categorical_columns: List of categorical column names
        estimated_processing_time_per_row_ms: Estimated ms per row
    """

    row_count: int
    column_count: int = 0
    estimated_bytes: int = 0
    null_ratio: float = 0.0
    unique_ratio: float = 0.5
    has_categorical: bool = False
    categorical_columns: list[str] = field(default_factory=list)
    estimated_processing_time_per_row_ms: float = 0.01

    @classmethod
    def from_data(cls, data: Any) -> "DataCharacteristics":
        """Infer characteristics from data.

        Args:
            data: Dataset (list, DataFrame, etc.)

        Returns:
            DataCharacteristics
        """
        # Try to get row count
        if hasattr(data, "__len__"):
            row_count = len(data)
        elif hasattr(data, "shape"):
            row_count = data.shape[0]
        else:
            row_count = 0

        # Try to get column count
        column_count = 0
        if hasattr(data, "shape") and len(data.shape) > 1:
            column_count = data.shape[1]
        elif hasattr(data, "columns"):
            column_count = len(data.columns)

        return cls(
            row_count=row_count,
            column_count=column_count,
        )


@dataclass
class SamplingResult(Generic[T]):
    """Result of sampling operation.

    Attributes:
        data: Sampled data
        original_size: Original dataset size
        sample_size: Actual sample size
        sampling_ratio: Ratio of original sampled
        method: Sampling method used
        indices: Indices of sampled items (if available)
        metadata: Additional metadata
    """

    data: T
    original_size: int
    sample_size: int
    sampling_ratio: float
    method: SamplingMethod
    indices: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_full(self) -> bool:
        """Check if full dataset (no sampling)."""
        return self.sample_size >= self.original_size

    @property
    def confidence_multiplier(self) -> float:
        """Get multiplier for confidence based on sampling ratio.

        Returns:
            Multiplier (0.0-1.0) for adjusting confidence
        """
        if self.is_full:
            return 1.0
        # Confidence decreases with smaller samples
        return min(1.0, math.sqrt(self.sampling_ratio))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_size": self.original_size,
            "sample_size": self.sample_size,
            "sampling_ratio": self.sampling_ratio,
            "method": self.method.value,
            "is_full": self.is_full,
            "confidence_multiplier": self.confidence_multiplier,
            "metadata": self.metadata,
        }


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def sample(
        self,
        data: Sequence[T],
        sample_size: int,
    ) -> SamplingResult[list[T]]:
        """Sample data.

        Args:
            data: Data to sample
            sample_size: Desired sample size

        Returns:
            SamplingResult with sampled data
        """
        pass


class UniformSampling(SamplingStrategy):
    """Uniform random sampling.

    Samples items with equal probability.
    """

    def __init__(self, seed: int | None = None):
        """Initialize uniform sampling.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "uniform"

    def sample(
        self,
        data: Sequence[T],
        sample_size: int,
    ) -> SamplingResult[list[T]]:
        """Sample uniformly at random."""
        n = len(data)
        if sample_size >= n:
            return SamplingResult(
                data=list(data),
                original_size=n,
                sample_size=n,
                sampling_ratio=1.0,
                method=SamplingMethod.UNIFORM,
                indices=list(range(n)),
            )

        indices = self._rng.sample(range(n), sample_size)
        indices.sort()  # Preserve order
        sampled = [data[i] for i in indices]

        return SamplingResult(
            data=sampled,
            original_size=n,
            sample_size=sample_size,
            sampling_ratio=sample_size / n,
            method=SamplingMethod.UNIFORM,
            indices=indices,
        )


class StratifiedSampling(SamplingStrategy):
    """Stratified sampling based on a key function.

    Preserves the distribution of strata in the sample.
    """

    def __init__(
        self,
        key_fn: Any | None = None,
        seed: int | None = None,
    ):
        """Initialize stratified sampling.

        Args:
            key_fn: Function to extract stratum key
            seed: Random seed
        """
        self.key_fn = key_fn or (lambda x: x)
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "stratified"

    def sample(
        self,
        data: Sequence[T],
        sample_size: int,
    ) -> SamplingResult[list[T]]:
        """Sample with stratification."""
        n = len(data)
        if sample_size >= n:
            return SamplingResult(
                data=list(data),
                original_size=n,
                sample_size=n,
                sampling_ratio=1.0,
                method=SamplingMethod.STRATIFIED,
                indices=list(range(n)),
            )

        # Group by stratum
        strata: dict[Any, list[int]] = {}
        for i, item in enumerate(data):
            key = self.key_fn(item)
            if key not in strata:
                strata[key] = []
            strata[key].append(i)

        # Calculate samples per stratum
        sampling_ratio = sample_size / n
        indices = []

        for stratum_indices in strata.values():
            stratum_size = len(stratum_indices)
            stratum_sample_size = max(1, int(stratum_size * sampling_ratio))
            stratum_sample_size = min(stratum_sample_size, stratum_size)

            sampled_indices = self._rng.sample(stratum_indices, stratum_sample_size)
            indices.extend(sampled_indices)

        # Trim if needed
        if len(indices) > sample_size:
            indices = self._rng.sample(indices, sample_size)

        indices.sort()
        sampled = [data[i] for i in indices]

        return SamplingResult(
            data=sampled,
            original_size=n,
            sample_size=len(sampled),
            sampling_ratio=len(sampled) / n,
            method=SamplingMethod.STRATIFIED,
            indices=indices,
            metadata={"strata_count": len(strata)},
        )


class ReservoirSampling(SamplingStrategy):
    """Reservoir sampling for streaming data.

    Maintains a fixed-size sample as data streams in.
    Uses Algorithm R for uniform random sampling.
    """

    def __init__(self, seed: int | None = None):
        """Initialize reservoir sampling.

        Args:
            seed: Random seed
        """
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "reservoir"

    def sample(
        self,
        data: Sequence[T],
        sample_size: int,
    ) -> SamplingResult[list[T]]:
        """Sample using reservoir algorithm."""
        n = len(data)
        if sample_size >= n:
            return SamplingResult(
                data=list(data),
                original_size=n,
                sample_size=n,
                sampling_ratio=1.0,
                method=SamplingMethod.RESERVOIR,
                indices=list(range(n)),
            )

        # Algorithm R
        reservoir = list(data[:sample_size])
        indices = list(range(sample_size))

        for i in range(sample_size, n):
            j = self._rng.randint(0, i)
            if j < sample_size:
                reservoir[j] = data[i]
                indices[j] = i

        return SamplingResult(
            data=reservoir,
            original_size=n,
            sample_size=sample_size,
            sampling_ratio=sample_size / n,
            method=SamplingMethod.RESERVOIR,
            indices=sorted(indices),
        )

    def create_stream_sampler(self, sample_size: int) -> "StreamReservoir[T]":
        """Create a streaming reservoir sampler.

        Args:
            sample_size: Size of reservoir

        Returns:
            StreamReservoir for incremental sampling
        """
        return StreamReservoir(sample_size, self.seed)


class StreamReservoir(Generic[T]):
    """Streaming reservoir for incremental sampling.

    Use this when data arrives incrementally and you need to
    maintain a fixed-size sample.

    Example:
        reservoir = StreamReservoir(100)
        for item in streaming_data():
            reservoir.add(item)
        sample = reservoir.get_sample()
    """

    def __init__(self, sample_size: int, seed: int | None = None):
        """Initialize stream reservoir.

        Args:
            sample_size: Size of reservoir
            seed: Random seed
        """
        self.sample_size = sample_size
        self._reservoir: list[T] = []
        self._count = 0
        self._rng = random.Random(seed)

    def add(self, item: T) -> None:
        """Add an item to the reservoir.

        Args:
            item: Item to add
        """
        self._count += 1

        if len(self._reservoir) < self.sample_size:
            self._reservoir.append(item)
        else:
            j = self._rng.randint(0, self._count - 1)
            if j < self.sample_size:
                self._reservoir[j] = item

    def get_sample(self) -> list[T]:
        """Get current sample.

        Returns:
            List of sampled items
        """
        return list(self._reservoir)

    @property
    def total_seen(self) -> int:
        """Get total items seen."""
        return self._count

    @property
    def sampling_ratio(self) -> float:
        """Get current sampling ratio."""
        if self._count == 0:
            return 1.0
        return min(1.0, len(self._reservoir) / self._count)


class SystematicSampling(SamplingStrategy):
    """Systematic sampling with fixed interval.

    Selects every k-th item after a random start.
    """

    def __init__(self, seed: int | None = None):
        """Initialize systematic sampling.

        Args:
            seed: Random seed for start position
        """
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "systematic"

    def sample(
        self,
        data: Sequence[T],
        sample_size: int,
    ) -> SamplingResult[list[T]]:
        """Sample systematically."""
        n = len(data)
        if sample_size >= n:
            return SamplingResult(
                data=list(data),
                original_size=n,
                sample_size=n,
                sampling_ratio=1.0,
                method=SamplingMethod.SYSTEMATIC,
                indices=list(range(n)),
            )

        # Calculate interval
        interval = n / sample_size
        start = self._rng.random() * interval

        indices = []
        position = start
        while position < n and len(indices) < sample_size:
            indices.append(int(position))
            position += interval

        sampled = [data[i] for i in indices]

        return SamplingResult(
            data=sampled,
            original_size=n,
            sample_size=len(sampled),
            sampling_ratio=len(sampled) / n,
            method=SamplingMethod.SYSTEMATIC,
            indices=indices,
            metadata={"interval": interval, "start": start},
        )


@dataclass
class SamplingConfig:
    """Configuration for adaptive sampling.

    Attributes:
        min_sample_size: Minimum sample size
        max_sample_size: Maximum sample size
        target_confidence: Target confidence level
        time_weight: Weight for time vs accuracy tradeoff
        prefer_stratified: Prefer stratified sampling when possible
    """

    min_sample_size: int = 100
    max_sample_size: int = 100000
    target_confidence: float = 0.95
    time_weight: float = 0.5
    prefer_stratified: bool = True


class AdaptiveSampler:
    """Adaptive sampler that adjusts strategy based on conditions.

    This sampler automatically selects the best sampling strategy
    and sample size based on:
    - Time budget available
    - Data characteristics
    - Target confidence level

    Example:
        sampler = AdaptiveSampler()

        # Calculate optimal sample size
        size = sampler.calculate_size(
            total_rows=1_000_000,
            time_budget_seconds=5.0,
        )

        # Sample data
        result = sampler.sample(data, size)
    """

    def __init__(
        self,
        config: SamplingConfig | None = None,
        strategies: dict[SamplingMethod, SamplingStrategy] | None = None,
    ):
        """Initialize adaptive sampler.

        Args:
            config: Sampling configuration
            strategies: Available sampling strategies
        """
        self.config = config or SamplingConfig()
        self.strategies = strategies or {
            SamplingMethod.UNIFORM: UniformSampling(),
            SamplingMethod.STRATIFIED: StratifiedSampling(),
            SamplingMethod.RESERVOIR: ReservoirSampling(),
            SamplingMethod.SYSTEMATIC: SystematicSampling(),
        }
        self._execution_history: list[tuple[int, float]] = []  # (size, time_ms)
        self._lock = threading.Lock()

    def calculate_size(
        self,
        total_rows: int,
        time_budget_seconds: float | None = None,
        characteristics: DataCharacteristics | None = None,
    ) -> int:
        """Calculate optimal sample size.

        Args:
            total_rows: Total number of rows
            time_budget_seconds: Available time budget
            characteristics: Data characteristics

        Returns:
            Recommended sample size
        """
        # Start with full dataset
        sample_size = total_rows

        # Apply minimum
        sample_size = max(self.config.min_sample_size, sample_size)

        # Apply maximum
        sample_size = min(self.config.max_sample_size, sample_size)

        # Adjust for time budget
        if time_budget_seconds is not None and characteristics is not None:
            estimated_time_per_row_ms = characteristics.estimated_processing_time_per_row_ms
            available_ms = time_budget_seconds * 1000

            # How many rows can we process?
            max_processable = int(available_ms / max(estimated_time_per_row_ms, 0.001))
            sample_size = min(sample_size, max_processable)

        # Adjust for confidence
        # Using Cochran's formula for sample size
        z = 1.96  # 95% confidence
        if self.config.target_confidence >= 0.99:
            z = 2.576
        elif self.config.target_confidence >= 0.95:
            z = 1.96
        elif self.config.target_confidence >= 0.90:
            z = 1.645

        # Standard sample size formula (assuming p=0.5 for maximum variability)
        p = 0.5
        e = 0.05  # 5% margin of error
        min_for_confidence = int((z ** 2 * p * (1 - p)) / (e ** 2))

        sample_size = max(sample_size, min(min_for_confidence, total_rows))

        # Final bounds check
        sample_size = max(self.config.min_sample_size, sample_size)
        sample_size = min(self.config.max_sample_size, min(sample_size, total_rows))

        return sample_size

    def select_strategy(
        self,
        characteristics: DataCharacteristics | None = None,
        sample_size: int | None = None,
    ) -> SamplingMethod:
        """Select best sampling strategy.

        Args:
            characteristics: Data characteristics
            sample_size: Target sample size

        Returns:
            Recommended sampling method
        """
        # Default to uniform
        if characteristics is None:
            return SamplingMethod.UNIFORM

        # Use stratified for categorical data
        if characteristics.has_categorical and self.config.prefer_stratified:
            return SamplingMethod.STRATIFIED

        # Use reservoir for very large datasets
        if characteristics.row_count > 10_000_000:
            return SamplingMethod.RESERVOIR

        # Use systematic for moderate datasets
        if characteristics.row_count > 1_000_000:
            return SamplingMethod.SYSTEMATIC

        return SamplingMethod.UNIFORM

    def sample(
        self,
        data: Sequence[T],
        sample_size: int | None = None,
        method: SamplingMethod | None = None,
        characteristics: DataCharacteristics | None = None,
    ) -> SamplingResult[list[T]]:
        """Sample data.

        Args:
            data: Data to sample
            sample_size: Desired sample size (None = auto)
            method: Sampling method (None = auto)
            characteristics: Data characteristics

        Returns:
            SamplingResult
        """
        total_rows = len(data)

        # Auto-calculate sample size
        if sample_size is None:
            if characteristics is None:
                characteristics = DataCharacteristics.from_data(data)
            sample_size = self.calculate_size(total_rows, characteristics=characteristics)

        # Auto-select method
        if method is None:
            method = self.select_strategy(characteristics, sample_size)

        # Get strategy
        strategy = self.strategies.get(method, UniformSampling())

        # Sample
        import time
        start = time.time()
        result = strategy.sample(data, sample_size)
        elapsed_ms = (time.time() - start) * 1000

        # Record for learning
        with self._lock:
            self._execution_history.append((result.sample_size, elapsed_ms))
            # Keep last 100
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]

        return result

    def get_estimated_time_per_row(self) -> float:
        """Get estimated processing time per row.

        Returns:
            Estimated milliseconds per row
        """
        with self._lock:
            if not self._execution_history:
                return 0.01  # Default

            # Linear regression
            total_rows = sum(h[0] for h in self._execution_history)
            total_time = sum(h[1] for h in self._execution_history)

            if total_rows == 0:
                return 0.01

            return total_time / total_rows


# Module-level sampler
_default_sampler: AdaptiveSampler | None = None


def calculate_sample_size(
    total_rows: int,
    time_budget_seconds: float | None = None,
    min_sample: int = 100,
    max_sample: int = 100000,
) -> int:
    """Calculate optimal sample size.

    Args:
        total_rows: Total number of rows
        time_budget_seconds: Available time budget
        min_sample: Minimum sample size
        max_sample: Maximum sample size

    Returns:
        Recommended sample size
    """
    global _default_sampler
    if _default_sampler is None:
        _default_sampler = AdaptiveSampler(SamplingConfig(
            min_sample_size=min_sample,
            max_sample_size=max_sample,
        ))

    characteristics = DataCharacteristics(row_count=total_rows)
    return _default_sampler.calculate_size(
        total_rows,
        time_budget_seconds,
        characteristics,
    )


def auto_sample(
    data: Sequence[T],
    time_budget_seconds: float | None = None,
) -> SamplingResult[list[T]]:
    """Automatically sample data.

    Args:
        data: Data to sample
        time_budget_seconds: Available time budget

    Returns:
        SamplingResult
    """
    global _default_sampler
    if _default_sampler is None:
        _default_sampler = AdaptiveSampler()

    characteristics = DataCharacteristics.from_data(data)
    sample_size = _default_sampler.calculate_size(
        len(data),
        time_budget_seconds,
        characteristics,
    )

    return _default_sampler.sample(data, sample_size, characteristics=characteristics)
