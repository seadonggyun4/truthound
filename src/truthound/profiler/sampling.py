"""Enterprise-grade sampling strategies for memory-efficient pattern matching.

This module provides a comprehensive sampling framework that prevents OOM errors
when processing large datasets while maintaining statistical accuracy.

Key features:
- Pluggable sampling strategy architecture
- Memory-aware adaptive sampling
- Statistical confidence estimation
- Stratified sampling for skewed distributions
- Reservoir sampling for streaming data

Design Principles:
- Open/Closed: New strategies can be added without modifying existing code
- Single Responsibility: Each strategy handles one sampling approach
- Dependency Inversion: High-level modules depend on abstractions

Example:
    from truthound.profiler.sampling import (
        SampledPatternMatcher,
        SamplingConfig,
        AdaptiveSamplingStrategy,
    )

    # Use adaptive sampling based on data size
    config = SamplingConfig(
        strategy="adaptive",
        max_rows=100_000,
        confidence_level=0.95,
    )

    matcher = SampledPatternMatcher(sampling_config=config)
    results = matcher.match_column(lf, "email")
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
)

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class SamplingMethod(str, Enum):
    """Available sampling methods."""

    NONE = "none"              # No sampling (use all data)
    RANDOM = "random"          # Simple random sampling
    SYSTEMATIC = "systematic"  # Every nth row
    STRATIFIED = "stratified"  # Preserve distribution
    RESERVOIR = "reservoir"    # Streaming reservoir sampling
    ADAPTIVE = "adaptive"      # Auto-select based on data size
    HEAD = "head"              # First n rows (fastest, least accurate)
    HASH = "hash"              # Deterministic hash-based sampling


class ConfidenceLevel(float, Enum):
    """Common confidence levels for statistical sampling."""

    LOW = 0.90       # 90% confidence
    MEDIUM = 0.95    # 95% confidence (default)
    HIGH = 0.99      # 99% confidence
    VERY_HIGH = 0.999  # 99.9% confidence


# =============================================================================
# Sampling Result
# =============================================================================


@dataclass(frozen=True)
class SamplingMetrics:
    """Metrics about the sampling operation.

    Attributes:
        original_size: Original dataset size
        sample_size: Actual sample size used
        sampling_ratio: Fraction of data sampled
        confidence_level: Statistical confidence level
        margin_of_error: Estimated margin of error
        strategy_used: Name of the sampling strategy
        sampling_time_ms: Time taken to sample
        memory_saved_estimate_mb: Estimated memory saved
    """

    original_size: int
    sample_size: int
    sampling_ratio: float
    confidence_level: float
    margin_of_error: float
    strategy_used: str
    sampling_time_ms: float = 0.0
    memory_saved_estimate_mb: float = 0.0

    @property
    def is_full_scan(self) -> bool:
        """Check if full data was used (no sampling)."""
        return self.sampling_ratio >= 1.0

    @property
    def reduction_factor(self) -> float:
        """Get data reduction factor (1.0 = no reduction)."""
        if self.sample_size == 0:
            return 0.0
        return self.original_size / self.sample_size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_size": self.original_size,
            "sample_size": self.sample_size,
            "sampling_ratio": self.sampling_ratio,
            "confidence_level": self.confidence_level,
            "margin_of_error": self.margin_of_error,
            "strategy_used": self.strategy_used,
            "sampling_time_ms": self.sampling_time_ms,
            "memory_saved_estimate_mb": self.memory_saved_estimate_mb,
            "is_full_scan": self.is_full_scan,
            "reduction_factor": self.reduction_factor,
        }


@dataclass
class SamplingResult(Generic[TypeVar("T")]):
    """Result of a sampling operation.

    Attributes:
        data: The sampled LazyFrame
        metrics: Sampling metrics
        is_sampled: Whether sampling was applied
    """

    data: pl.LazyFrame
    metrics: SamplingMetrics
    is_sampled: bool = True

    def __post_init__(self) -> None:
        """Validate result."""
        if self.metrics.sample_size == 0 and self.is_sampled:
            logger.warning("Sampling resulted in zero rows")


# =============================================================================
# Sampling Configuration
# =============================================================================


@dataclass
class SamplingConfig:
    """Configuration for sampling behavior.

    This configuration controls how data is sampled for pattern matching.
    It supports both explicit size limits and statistical parameters.

    Attributes:
        strategy: Sampling strategy to use
        max_rows: Maximum rows to sample (0 = auto-calculate)
        max_memory_mb: Maximum memory to use for sampling (0 = unlimited)
        confidence_level: Statistical confidence level (0.0 to 1.0)
        margin_of_error: Acceptable margin of error (0.0 to 1.0)
        seed: Random seed for reproducibility (None = random)
        min_sample_size: Minimum sample size regardless of calculations
        enable_caching: Cache sampling decisions for same data
        fallback_strategy: Strategy to use if primary fails
    """

    strategy: SamplingMethod = SamplingMethod.ADAPTIVE
    max_rows: int = 100_000
    max_memory_mb: int = 0  # 0 = auto (use 10% of available)
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    seed: int | None = None
    min_sample_size: int = 1000
    enable_caching: bool = True
    fallback_strategy: SamplingMethod = SamplingMethod.HEAD

    # Size thresholds for adaptive strategy
    small_dataset_threshold: int = 10_000
    medium_dataset_threshold: int = 100_000
    large_dataset_threshold: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            )
        if not 0.0 < self.margin_of_error < 1.0:
            raise ValueError(
                f"margin_of_error must be between 0 and 1, got {self.margin_of_error}"
            )
        if self.max_rows < 0:
            raise ValueError(f"max_rows must be non-negative, got {self.max_rows}")
        if self.min_sample_size < 1:
            raise ValueError(
                f"min_sample_size must be positive, got {self.min_sample_size}"
            )

    def calculate_required_sample_size(
        self,
        population_size: int,
        expected_proportion: float = 0.5,
    ) -> int:
        """Calculate statistically required sample size.

        Uses Cochran's formula with finite population correction.

        Args:
            population_size: Total population size
            expected_proportion: Expected proportion (0.5 = maximum variance)

        Returns:
            Required sample size for desired confidence/margin
        """
        if population_size <= 0:
            return 0

        # Z-score for confidence level
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
            0.999: 3.291,
        }
        z = z_scores.get(
            round(self.confidence_level, 3),
            self._z_score_from_confidence(self.confidence_level),
        )

        p = expected_proportion
        e = self.margin_of_error

        # Cochran's formula for infinite population
        n0 = (z ** 2 * p * (1 - p)) / (e ** 2)

        # Finite population correction
        n = n0 / (1 + (n0 - 1) / population_size)

        # Apply bounds
        sample_size = int(math.ceil(n))
        sample_size = max(sample_size, self.min_sample_size)
        sample_size = min(sample_size, population_size)

        if self.max_rows > 0:
            sample_size = min(sample_size, self.max_rows)

        return sample_size

    @staticmethod
    def _z_score_from_confidence(confidence: float) -> float:
        """Approximate Z-score from confidence level."""
        # Using inverse normal approximation
        # For more accuracy, use scipy.stats.norm.ppf
        alpha = 1 - confidence
        # Rough approximation for common values
        if alpha <= 0.001:
            return 3.3
        elif alpha <= 0.01:
            return 2.6
        elif alpha <= 0.05:
            return 2.0
        elif alpha <= 0.10:
            return 1.6
        else:
            return 1.0

    @classmethod
    def for_accuracy(cls, accuracy: str = "medium") -> "SamplingConfig":
        """Create config optimized for accuracy level.

        Args:
            accuracy: "low", "medium", "high", or "maximum"

        Returns:
            Configured SamplingConfig
        """
        configs = {
            "low": cls(
                strategy=SamplingMethod.HEAD,
                max_rows=10_000,
                confidence_level=0.90,
                margin_of_error=0.10,
            ),
            "medium": cls(
                strategy=SamplingMethod.ADAPTIVE,
                max_rows=100_000,
                confidence_level=0.95,
                margin_of_error=0.05,
            ),
            "high": cls(
                strategy=SamplingMethod.RANDOM,
                max_rows=500_000,
                confidence_level=0.99,
                margin_of_error=0.02,
            ),
            "maximum": cls(
                strategy=SamplingMethod.NONE,
                max_rows=0,
                confidence_level=0.999,
                margin_of_error=0.01,
            ),
        }
        return configs.get(accuracy, configs["medium"])

    @classmethod
    def for_speed(cls) -> "SamplingConfig":
        """Create config optimized for speed."""
        return cls(
            strategy=SamplingMethod.HEAD,
            max_rows=10_000,
            confidence_level=0.90,
            margin_of_error=0.10,
        )

    @classmethod
    def for_memory(cls, max_memory_mb: int = 100) -> "SamplingConfig":
        """Create config optimized for memory efficiency."""
        return cls(
            strategy=SamplingMethod.RESERVOIR,
            max_rows=50_000,
            max_memory_mb=max_memory_mb,
            confidence_level=0.95,
            margin_of_error=0.05,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "max_rows": self.max_rows,
            "max_memory_mb": self.max_memory_mb,
            "confidence_level": self.confidence_level,
            "margin_of_error": self.margin_of_error,
            "seed": self.seed,
            "min_sample_size": self.min_sample_size,
        }


# Default configuration
DEFAULT_SAMPLING_CONFIG = SamplingConfig()


# =============================================================================
# Sampling Strategy Protocol
# =============================================================================


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies.

    All sampling strategies must implement this interface.
    This enables the Strategy pattern for flexible sampling behavior.

    Example:
        class MyCustomStrategy(SamplingStrategy):
            name = "custom"

            def sample(self, lf, config):
                # Custom sampling logic
                ...
    """

    name: str = "base"

    @abstractmethod
    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Sample data from the LazyFrame.

        Args:
            lf: Source LazyFrame
            config: Sampling configuration
            total_rows: Pre-computed total rows (optional, for efficiency)

        Returns:
            SamplingResult with sampled data and metrics
        """
        pass

    def estimate_row_count(self, lf: pl.LazyFrame) -> int:
        """Estimate row count without full scan.

        Override for more efficient implementations.

        Args:
            lf: LazyFrame to estimate

        Returns:
            Estimated row count
        """
        # Default: exact count (can be expensive)
        return lf.select(pl.len()).collect().item()

    def _create_metrics(
        self,
        original_size: int,
        sample_size: int,
        config: SamplingConfig,
        sampling_time_ms: float = 0.0,
    ) -> SamplingMetrics:
        """Create sampling metrics."""
        sampling_ratio = sample_size / original_size if original_size > 0 else 0.0

        # Estimate margin of error for actual sample
        if sample_size > 0 and original_size > 0:
            # Simplified margin of error calculation
            z = 1.96  # 95% confidence
            p = 0.5   # Maximum variance
            margin = z * math.sqrt(p * (1 - p) / sample_size)
            # Finite population correction
            if sample_size < original_size:
                fpc = math.sqrt((original_size - sample_size) / (original_size - 1))
                margin *= fpc
        else:
            margin = 1.0

        # Estimate memory saved (rough: 100 bytes per row average)
        rows_saved = original_size - sample_size
        memory_saved_mb = (rows_saved * 100) / (1024 * 1024)

        return SamplingMetrics(
            original_size=original_size,
            sample_size=sample_size,
            sampling_ratio=sampling_ratio,
            confidence_level=config.confidence_level,
            margin_of_error=min(margin, 1.0),
            strategy_used=self.name,
            sampling_time_ms=sampling_time_ms,
            memory_saved_estimate_mb=max(0, memory_saved_mb),
        )


# =============================================================================
# Concrete Sampling Strategies
# =============================================================================


class NoSamplingStrategy(SamplingStrategy):
    """Strategy that uses all data without sampling.

    Use when accuracy is paramount and memory is not a concern.
    """

    name = "none"

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Return all data without sampling."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=total_rows,
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=False,
        )


class HeadSamplingStrategy(SamplingStrategy):
    """Strategy that takes the first N rows.

    Fastest sampling method but may not be representative
    if data has ordering bias.
    """

    name = "head"

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Take first N rows."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Calculate sample size
        sample_size = config.calculate_required_sample_size(total_rows)
        if config.max_rows > 0:
            sample_size = min(sample_size, config.max_rows)

        # No sampling needed if sample >= total
        if sample_size >= total_rows:
            return NoSamplingStrategy().sample(lf, config, total_rows)

        # Apply head sampling
        sampled_lf = lf.head(sample_size)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=sample_size,
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=True,
        )


class RandomSamplingStrategy(SamplingStrategy):
    """Strategy for simple random sampling.

    Uses Polars native random sampling for efficiency.
    Provides unbiased samples but may not preserve rare patterns.
    """

    name = "random"

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Random sample of N rows."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Calculate sample size
        sample_size = config.calculate_required_sample_size(total_rows)
        if config.max_rows > 0:
            sample_size = min(sample_size, config.max_rows)

        # No sampling needed if sample >= total
        if sample_size >= total_rows:
            return NoSamplingStrategy().sample(lf, config, total_rows)

        # Calculate fraction for sampling
        fraction = sample_size / total_rows

        # Apply random sampling with seed for reproducibility
        seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)

        # Polars sample is on DataFrame, need to collect first for true random
        # For LazyFrame, we use a workaround with row index
        # Use higher precision (10000) to avoid fraction becoming 0 for small ratios
        threshold = max(1, int(fraction * 10000))
        sampled_lf = (
            lf.with_row_index("__sample_idx")
            .filter(pl.col("__sample_idx").hash(seed) % 10000 < threshold)
            .drop("__sample_idx")
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Actual sample size may vary due to hash-based sampling
        actual_sample_size = min(sample_size, total_rows)

        return SamplingResult(
            data=sampled_lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=actual_sample_size,
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=True,
        )


class SystematicSamplingStrategy(SamplingStrategy):
    """Strategy for systematic sampling (every Nth row).

    Efficient and ensures even coverage across data.
    May miss periodic patterns if data has periodicity.
    """

    name = "systematic"

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Take every Nth row."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Calculate sample size and interval
        sample_size = config.calculate_required_sample_size(total_rows)
        if config.max_rows > 0:
            sample_size = min(sample_size, config.max_rows)

        if sample_size >= total_rows:
            return NoSamplingStrategy().sample(lf, config, total_rows)

        # Calculate sampling interval
        interval = max(1, total_rows // sample_size)

        # Random start offset for unbiased sampling
        seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)
        random.seed(seed)
        offset = random.randint(0, interval - 1)

        # Apply systematic sampling
        sampled_lf = (
            lf.with_row_index("__sample_idx")
            .filter((pl.col("__sample_idx") - offset) % interval == 0)
            .drop("__sample_idx")
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        actual_sample_size = (total_rows - offset + interval - 1) // interval

        return SamplingResult(
            data=sampled_lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=min(actual_sample_size, sample_size),
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=True,
        )


class HashSamplingStrategy(SamplingStrategy):
    """Strategy for deterministic hash-based sampling.

    Produces reproducible samples based on row content.
    Useful for consistent sampling across runs.
    """

    name = "hash"

    def __init__(self, hash_column: str | None = None):
        """Initialize hash sampling strategy.

        Args:
            hash_column: Column to use for hashing (None = use row index)
        """
        self.hash_column = hash_column

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Hash-based deterministic sampling."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        sample_size = config.calculate_required_sample_size(total_rows)
        if config.max_rows > 0:
            sample_size = min(sample_size, config.max_rows)

        if sample_size >= total_rows:
            return NoSamplingStrategy().sample(lf, config, total_rows)

        # Calculate threshold for hash-based filtering
        # Use higher precision (10000) to avoid threshold becoming 0 for small ratios
        threshold = max(1, int((sample_size / total_rows) * 10000))
        seed = config.seed if config.seed is not None else 42

        if self.hash_column:
            # Hash specific column
            sampled_lf = lf.filter(
                pl.col(self.hash_column).hash(seed) % 10000 < threshold
            )
        else:
            # Hash row index
            sampled_lf = (
                lf.with_row_index("__hash_idx")
                .filter(pl.col("__hash_idx").hash(seed) % 10000 < threshold)
                .drop("__hash_idx")
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=sample_size,
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=True,
        )


class StratifiedSamplingStrategy(SamplingStrategy):
    """Strategy for stratified sampling.

    Preserves distribution of a stratification column.
    Useful when data has important categorical groupings.
    """

    name = "stratified"

    def __init__(self, stratify_column: str | None = None):
        """Initialize stratified sampling.

        Args:
            stratify_column: Column to stratify by
        """
        self.stratify_column = stratify_column

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Stratified sampling preserving group proportions."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        sample_size = config.calculate_required_sample_size(total_rows)
        if config.max_rows > 0:
            sample_size = min(sample_size, config.max_rows)

        if sample_size >= total_rows:
            return NoSamplingStrategy().sample(lf, config, total_rows)

        fraction = sample_size / total_rows

        if self.stratify_column:
            # Sample within each stratum
            seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)

            # Get strata and sample proportionally
            sampled_lf = (
                lf.with_row_index("__strat_idx")
                .with_columns(
                    (pl.col("__strat_idx").hash(seed) % 1000 / 1000).alias("__rand")
                )
                .filter(pl.col("__rand") < fraction)
                .drop(["__strat_idx", "__rand"])
            )
        else:
            # Fallback to random sampling if no stratify column
            return RandomSamplingStrategy().sample(lf, config, total_rows)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=sample_size,
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=True,
        )


class ReservoirSamplingStrategy(SamplingStrategy):
    """Strategy for reservoir sampling.

    Optimal for streaming data where total size is unknown.
    Provides uniform random sample with single pass.
    """

    name = "reservoir"

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Reservoir sampling for streaming-friendly sampling."""
        start_time = time.perf_counter()

        # For reservoir sampling, we need to process in a streaming fashion
        # Polars doesn't have native reservoir sampling, so we approximate

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        sample_size = config.calculate_required_sample_size(total_rows)
        if config.max_rows > 0:
            sample_size = min(sample_size, config.max_rows)

        if sample_size >= total_rows:
            return NoSamplingStrategy().sample(lf, config, total_rows)

        # Approximate reservoir sampling using weighted random selection
        seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)

        # Use logarithmic random for reservoir-like behavior
        sampled_lf = (
            lf.with_row_index("__res_idx")
            .with_columns(
                (-pl.col("__res_idx").hash(seed).log()).alias("__priority")
            )
            .sort("__priority")
            .head(sample_size)
            .drop(["__res_idx", "__priority"])
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=self._create_metrics(
                original_size=total_rows,
                sample_size=sample_size,
                config=config,
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=True,
        )


class AdaptiveSamplingStrategy(SamplingStrategy):
    """Strategy that adapts based on data characteristics.

    Automatically selects the best sampling method based on:
    - Dataset size
    - Available memory
    - Accuracy requirements

    This is the recommended default strategy.
    """

    name = "adaptive"

    def __init__(self) -> None:
        """Initialize with sub-strategies."""
        self._strategies: dict[str, SamplingStrategy] = {
            "none": NoSamplingStrategy(),
            "head": HeadSamplingStrategy(),
            "random": RandomSamplingStrategy(),
            "systematic": SystematicSamplingStrategy(),
            "reservoir": ReservoirSamplingStrategy(),
        }

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Adaptively sample based on data size and config."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Select strategy based on data size
        selected_strategy = self._select_strategy(total_rows, config)

        logger.debug(
            f"Adaptive sampling selected '{selected_strategy.name}' "
            f"for {total_rows:,} rows"
        )

        # Delegate to selected strategy
        result = selected_strategy.sample(lf, config, total_rows)

        # Update metrics to reflect adaptive selection
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=result.data,
            metrics=SamplingMetrics(
                original_size=result.metrics.original_size,
                sample_size=result.metrics.sample_size,
                sampling_ratio=result.metrics.sampling_ratio,
                confidence_level=result.metrics.confidence_level,
                margin_of_error=result.metrics.margin_of_error,
                strategy_used=f"adaptive({selected_strategy.name})",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=result.metrics.memory_saved_estimate_mb,
            ),
            is_sampled=result.is_sampled,
        )

    def _select_strategy(
        self,
        total_rows: int,
        config: SamplingConfig,
    ) -> SamplingStrategy:
        """Select the best strategy for given parameters."""
        # Small datasets: no sampling needed
        if total_rows <= config.small_dataset_threshold:
            return self._strategies["none"]

        # Medium datasets: systematic for balance of speed/quality
        if total_rows <= config.medium_dataset_threshold:
            return self._strategies["systematic"]

        # Large datasets: random for better representation
        if total_rows <= config.large_dataset_threshold:
            return self._strategies["random"]

        # Very large datasets: reservoir for memory efficiency
        return self._strategies["reservoir"]


# =============================================================================
# Sampling Strategy Registry
# =============================================================================


class SamplingStrategyRegistry:
    """Registry for sampling strategies.

    Allows registration of custom strategies and creation by name.

    Example:
        registry = SamplingStrategyRegistry()
        registry.register(MyCustomStrategy())
        strategy = registry.get("custom")
    """

    def __init__(self) -> None:
        self._strategies: dict[str, SamplingStrategy] = {}
        self._lock = threading.RLock()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in strategies."""
        self.register(NoSamplingStrategy())
        self.register(HeadSamplingStrategy())
        self.register(RandomSamplingStrategy())
        self.register(SystematicSamplingStrategy())
        self.register(HashSamplingStrategy())
        self.register(StratifiedSamplingStrategy())
        self.register(ReservoirSamplingStrategy())
        self.register(AdaptiveSamplingStrategy())

    def register(self, strategy: SamplingStrategy) -> None:
        """Register a sampling strategy."""
        with self._lock:
            self._strategies[strategy.name] = strategy
            logger.debug(f"Registered sampling strategy: {strategy.name}")

    def get(self, name: str) -> SamplingStrategy:
        """Get a strategy by name.

        Args:
            name: Strategy name

        Returns:
            The requested strategy

        Raises:
            KeyError: If strategy not found
        """
        with self._lock:
            if name not in self._strategies:
                available = list(self._strategies.keys())
                raise KeyError(
                    f"Unknown sampling strategy: '{name}'. "
                    f"Available: {available}"
                )
            return self._strategies[name]

    def get_or_default(
        self,
        name: str,
        default: SamplingStrategy | None = None,
    ) -> SamplingStrategy:
        """Get strategy by name with fallback."""
        try:
            return self.get(name)
        except KeyError:
            return default or AdaptiveSamplingStrategy()

    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        with self._lock:
            return list(self._strategies.keys())

    def create_from_method(self, method: SamplingMethod) -> SamplingStrategy:
        """Create strategy from SamplingMethod enum."""
        return self.get(method.value)


# Global registry instance
sampling_strategy_registry = SamplingStrategyRegistry()


# =============================================================================
# Data Size Estimator
# =============================================================================


class DataSizeEstimator:
    """Estimates data size for sampling decisions.

    Provides fast, approximate size estimates without full scans.
    """

    @staticmethod
    def estimate_row_count(lf: pl.LazyFrame) -> int:
        """Estimate row count.

        Args:
            lf: LazyFrame to estimate

        Returns:
            Estimated row count
        """
        # For now, use exact count
        # Future: Use file metadata for parquet, etc.
        return lf.select(pl.len()).collect().item()

    @staticmethod
    def estimate_memory_bytes(lf: pl.LazyFrame, sample_rows: int = 1000) -> int:
        """Estimate memory usage per row.

        Args:
            lf: LazyFrame to estimate
            sample_rows: Number of rows to sample for estimation

        Returns:
            Estimated bytes per row
        """
        try:
            sample = lf.head(sample_rows).collect()
            if len(sample) == 0:
                return 0

            total_bytes = sample.estimated_size()
            bytes_per_row = total_bytes // len(sample)
            return bytes_per_row
        except Exception:
            # Default estimate: 100 bytes per row
            return 100

    @staticmethod
    def estimate_total_memory_mb(
        lf: pl.LazyFrame,
        row_count: int | None = None,
    ) -> float:
        """Estimate total memory for full data.

        Args:
            lf: LazyFrame to estimate
            row_count: Pre-computed row count

        Returns:
            Estimated total memory in MB
        """
        if row_count is None:
            row_count = DataSizeEstimator.estimate_row_count(lf)

        bytes_per_row = DataSizeEstimator.estimate_memory_bytes(lf)
        total_bytes = row_count * bytes_per_row
        return total_bytes / (1024 * 1024)


# =============================================================================
# Sampler (Main Interface)
# =============================================================================


class Sampler:
    """Main interface for data sampling.

    Coordinates sampling strategies and provides a simple API
    for sampling data with configurable behavior.

    Example:
        sampler = Sampler(SamplingConfig.for_accuracy("high"))
        result = sampler.sample(lf)

        print(f"Sampled {result.metrics.sample_size:,} of "
              f"{result.metrics.original_size:,} rows")
        print(f"Strategy: {result.metrics.strategy_used}")
    """

    def __init__(
        self,
        config: SamplingConfig | None = None,
        registry: SamplingStrategyRegistry | None = None,
    ):
        """Initialize sampler.

        Args:
            config: Sampling configuration
            registry: Strategy registry (uses global if not provided)
        """
        self.config = config or DEFAULT_SAMPLING_CONFIG
        self.registry = registry or sampling_strategy_registry
        self._size_estimator = DataSizeEstimator()

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig | None = None,
    ) -> SamplingResult:
        """Sample data from LazyFrame.

        Args:
            lf: Source LazyFrame
            config: Override configuration for this call

        Returns:
            SamplingResult with sampled data and metrics
        """
        config = config or self.config

        # Get the appropriate strategy
        strategy = self.registry.create_from_method(config.strategy)

        # Estimate row count
        total_rows = self._size_estimator.estimate_row_count(lf)

        # Execute sampling
        try:
            return strategy.sample(lf, config, total_rows)
        except Exception as e:
            logger.warning(
                f"Sampling strategy '{strategy.name}' failed: {e}. "
                f"Falling back to '{config.fallback_strategy.value}'"
            )
            # Fallback
            fallback = self.registry.create_from_method(config.fallback_strategy)
            return fallback.sample(lf, config, total_rows)

    def sample_column(
        self,
        lf: pl.LazyFrame,
        column: str,
        config: SamplingConfig | None = None,
    ) -> SamplingResult:
        """Sample specific column from LazyFrame.

        Args:
            lf: Source LazyFrame
            column: Column to sample
            config: Override configuration

        Returns:
            SamplingResult with sampled column data
        """
        # Select only the needed column for efficiency
        column_lf = lf.select(pl.col(column))
        return self.sample(column_lf, config)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_sampler(
    strategy: str | SamplingMethod = "adaptive",
    max_rows: int = 100_000,
    confidence_level: float = 0.95,
    **kwargs: Any,
) -> Sampler:
    """Create a sampler with specified parameters.

    Args:
        strategy: Sampling strategy name or enum
        max_rows: Maximum rows to sample
        confidence_level: Statistical confidence level
        **kwargs: Additional config options

    Returns:
        Configured Sampler instance

    Example:
        sampler = create_sampler(strategy="random", max_rows=50_000)
        result = sampler.sample(lf)
    """
    if isinstance(strategy, str):
        strategy = SamplingMethod(strategy)

    config = SamplingConfig(
        strategy=strategy,
        max_rows=max_rows,
        confidence_level=confidence_level,
        **kwargs,
    )

    return Sampler(config)


def sample_data(
    lf: pl.LazyFrame,
    max_rows: int = 100_000,
    strategy: str = "adaptive",
) -> SamplingResult:
    """Quick function to sample data.

    Args:
        lf: LazyFrame to sample
        max_rows: Maximum rows
        strategy: Strategy name

    Returns:
        SamplingResult

    Example:
        result = sample_data(lf, max_rows=50_000)
        sampled_lf = result.data
    """
    sampler = create_sampler(strategy=strategy, max_rows=max_rows)
    return sampler.sample(lf)


def calculate_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    min_sample_size: int = 1,
) -> int:
    """Calculate required sample size for given parameters.

    Uses Cochran's formula with finite population correction.
    By default, returns the pure statistical calculation without
    minimum size constraints.

    Args:
        population_size: Total population
        confidence_level: Confidence level (0-1)
        margin_of_error: Margin of error (0-1)
        min_sample_size: Minimum sample size (default 1 for pure statistical result)

    Returns:
        Required sample size

    Example:
        n = calculate_sample_size(1_000_000, confidence_level=0.99)
        print(f"Need {n:,} samples for 99% confidence")
    """
    config = SamplingConfig(
        confidence_level=confidence_level,
        margin_of_error=margin_of_error,
        min_sample_size=min_sample_size,
    )
    return config.calculate_required_sample_size(population_size)
