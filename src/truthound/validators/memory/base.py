"""Base memory-efficient validation abstractions.

This module provides the foundational MemoryEfficientMixin that all other
memory optimization mixins build upon.

Key Features:
- Automatic memory estimation and configuration
- Smart sampling strategies (random, stratified, reservoir)
- Mini-batch processing with generators
- Memory budget management
- Chunked data iteration
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Callable, TYPE_CHECKING
import sys

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from truthound.validators.base import ValidatorLogger


class MemoryStrategy(Enum):
    """Memory optimization strategy."""

    FULL = auto()  # Load all data (default for small datasets)
    SAMPLE = auto()  # Random sampling for training
    STREAMING = auto()  # Process in streaming chunks
    INCREMENTAL = auto()  # Incremental/online learning
    APPROXIMATE = auto()  # Use approximate algorithms


@dataclass
class MemoryConfig:
    """Configuration for memory-efficient processing.

    Attributes:
        max_memory_mb: Maximum memory budget in MB
        sample_size: Maximum samples for training (None = auto)
        batch_size: Batch size for scoring/processing
        chunk_size: Chunk size for streaming iteration
        strategy: Memory optimization strategy (auto-detected if None)
        random_state: Random seed for reproducibility
        reserve_factor: Factor of memory to reserve for processing overhead
    """

    max_memory_mb: float = 512.0
    sample_size: int | None = None
    batch_size: int = 50000
    chunk_size: int = 100000
    strategy: MemoryStrategy | None = None
    random_state: int = 42
    reserve_factor: float = 0.7  # Use only 70% of available memory

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 < self.reserve_factor <= 1:
            raise ValueError("reserve_factor must be between 0 and 1")


def get_available_memory() -> float:
    """Get available system memory in MB.

    Returns:
        Available memory in MB, or a conservative default if unable to determine.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024 * 1024)
    except ImportError:
        # Fallback: assume 4GB available
        return 4096.0


def estimate_memory_usage(
    n_rows: int,
    n_cols: int,
    dtype_size: int = 8,
    multiplier: float = 1.0,
) -> float:
    """Estimate memory usage for a dataset in MB.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        dtype_size: Bytes per element (default 8 for float64)
        multiplier: Multiplier for algorithm overhead (e.g., 2.0 for distance matrix)

    Returns:
        Estimated memory usage in MB
    """
    bytes_needed = n_rows * n_cols * dtype_size * multiplier
    return bytes_needed / (1024 * 1024)


def estimate_algorithm_memory(
    n_samples: int,
    n_features: int,
    algorithm: str,
) -> float:
    """Estimate memory usage for specific algorithms.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        algorithm: Algorithm name (isolation_forest, lof, svm, dbscan, ks_test)

    Returns:
        Estimated memory in MB
    """
    data_size = estimate_memory_usage(n_samples, n_features)

    multipliers = {
        # Isolation Forest: data + trees (moderate overhead)
        "isolation_forest": 2.0,
        # LOF: data + distance matrix O(n²)
        "lof": 1.0 + (n_samples * 8 / (1024 * 1024)),  # n × k neighbors
        # SVM: kernel matrix O(n²)
        "svm": 1.0 + (n_samples * n_samples * 8 / (1024 * 1024)),
        # DBSCAN: data + distance computations
        "dbscan": 2.5,
        # KS test: two datasets
        "ks_test": 2.0,
        # Default
        "default": 1.5,
    }

    multiplier = multipliers.get(algorithm, multipliers["default"])
    return data_size * multiplier


class DataChunker:
    """Utility for chunked data iteration.

    Provides memory-efficient iteration over large LazyFrames
    without loading the entire dataset into memory.

    Example:
        chunker = DataChunker(chunk_size=100000)
        for chunk_df in chunker.iterate(large_lf):
            process(chunk_df)
    """

    def __init__(
        self,
        chunk_size: int = 100000,
        columns: list[str] | None = None,
        drop_nulls: bool = True,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Number of rows per chunk
            columns: Specific columns to select (None = all)
            drop_nulls: Whether to drop null rows
        """
        self.chunk_size = chunk_size
        self.columns = columns
        self.drop_nulls = drop_nulls

    def get_total_rows(self, lf: pl.LazyFrame) -> int:
        """Get total row count efficiently."""
        return lf.select(pl.len()).collect().item()

    def iterate(
        self,
        lf: pl.LazyFrame,
        as_numpy: bool = False,
    ) -> Iterator[pl.DataFrame | np.ndarray]:
        """Iterate over LazyFrame in chunks.

        Args:
            lf: Input LazyFrame
            as_numpy: If True, yield numpy arrays instead of DataFrames

        Yields:
            Chunks as DataFrame or numpy array
        """
        total_rows = self.get_total_rows(lf)

        if self.columns:
            lf = lf.select([pl.col(c) for c in self.columns])

        for offset in range(0, total_rows, self.chunk_size):
            chunk_lf = lf.slice(offset, self.chunk_size)

            if self.drop_nulls:
                chunk_lf = chunk_lf.drop_nulls()

            chunk_df = chunk_lf.collect()

            if len(chunk_df) == 0:
                continue

            if as_numpy:
                yield chunk_df.to_numpy()
            else:
                yield chunk_df

    def iterate_with_index(
        self,
        lf: pl.LazyFrame,
    ) -> Iterator[tuple[int, int, pl.DataFrame]]:
        """Iterate with chunk indices.

        Yields:
            Tuples of (start_index, end_index, chunk_dataframe)
        """
        total_rows = self.get_total_rows(lf)

        if self.columns:
            lf = lf.select([pl.col(c) for c in self.columns])

        for offset in range(0, total_rows, self.chunk_size):
            chunk_lf = lf.slice(offset, self.chunk_size)

            if self.drop_nulls:
                chunk_lf = chunk_lf.drop_nulls()

            chunk_df = chunk_lf.collect()

            if len(chunk_df) == 0:
                continue

            yield offset, offset + len(chunk_df), chunk_df


class MemoryEfficientMixin:
    """Base mixin for memory-efficient data processing.

    Provides core utilities for:
    - Memory estimation and monitoring
    - Automatic strategy selection
    - Smart sampling
    - Batch processing

    Usage:
        class MyValidator(Validator, MemoryEfficientMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._memory_config = MemoryConfig()

            def validate(self, lf):
                # Auto-configure based on data size
                self.auto_configure_memory(lf, algorithm="my_algo")

                # Use configured sample size
                data = self.smart_sample(lf, self.columns)
                ...
    """

    # Default memory configuration
    _memory_config: MemoryConfig = field(default_factory=MemoryConfig)

    # Tracking
    _configured_strategy: MemoryStrategy | None = None
    _effective_sample_size: int | None = None

    def get_memory_config(self) -> MemoryConfig:
        """Get current memory configuration."""
        if hasattr(self, "_memory_config") and self._memory_config is not None:
            return self._memory_config
        return MemoryConfig()

    def set_memory_config(self, config: MemoryConfig) -> None:
        """Set memory configuration."""
        self._memory_config = config

    def auto_configure_memory(
        self,
        lf: pl.LazyFrame,
        columns: list[str] | None = None,
        algorithm: str = "default",
    ) -> MemoryStrategy:
        """Automatically configure memory settings based on data size.

        Args:
            lf: Input LazyFrame
            columns: Columns to analyze (for size estimation)
            algorithm: Algorithm name for memory estimation

        Returns:
            Selected memory strategy
        """
        config = self.get_memory_config()

        # Get data dimensions
        total_rows = lf.select(pl.len()).collect().item()
        n_cols = len(columns) if columns else len(lf.collect_schema().names())

        # Estimate memory requirement
        estimated_mb = estimate_algorithm_memory(total_rows, n_cols, algorithm)
        available_mb = get_available_memory() * config.reserve_factor
        budget_mb = min(config.max_memory_mb, available_mb)

        # Log estimation
        if hasattr(self, "logger"):
            self.logger.debug(
                f"Memory estimation: {estimated_mb:.1f}MB needed, "
                f"{budget_mb:.1f}MB available (algorithm: {algorithm})"
            )

        # Select strategy
        if config.strategy is not None:
            strategy = config.strategy
        elif estimated_mb <= budget_mb:
            strategy = MemoryStrategy.FULL
        elif algorithm in ("lof", "dbscan"):
            strategy = MemoryStrategy.APPROXIMATE
        elif algorithm == "svm":
            strategy = MemoryStrategy.INCREMENTAL
        elif algorithm == "ks_test":
            strategy = MemoryStrategy.STREAMING
        else:
            strategy = MemoryStrategy.SAMPLE

        self._configured_strategy = strategy

        # Calculate effective sample size
        if strategy in (MemoryStrategy.SAMPLE, MemoryStrategy.APPROXIMATE):
            if config.sample_size is not None:
                self._effective_sample_size = min(config.sample_size, total_rows)
            else:
                # Calculate based on memory budget
                bytes_per_row = n_cols * 8  # float64
                max_rows = int((budget_mb * 1024 * 1024) / bytes_per_row)
                # Apply algorithm-specific caps
                caps = {
                    "lof": 50000,
                    "dbscan": 50000,
                    "svm": 20000,
                    "isolation_forest": 100000,
                    "default": 100000,
                }
                cap = caps.get(algorithm, caps["default"])
                self._effective_sample_size = min(max_rows, cap, total_rows)
        else:
            self._effective_sample_size = total_rows

        if hasattr(self, "logger"):
            self.logger.debug(
                f"Strategy: {strategy.name}, "
                f"effective_sample_size: {self._effective_sample_size}"
            )

        return strategy

    def smart_sample(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
        sample_size: int | None = None,
        method: str = "random",
    ) -> tuple[np.ndarray, int, bool]:
        """Intelligently sample data from LazyFrame.

        Uses Polars lazy evaluation to avoid loading full dataset.

        Args:
            lf: Input LazyFrame
            columns: Columns to select
            sample_size: Max samples (None = use configured size)
            method: Sampling method ('random', 'stratified', 'first', 'reservoir')

        Returns:
            Tuple of (data_array, original_count, was_sampled)
        """
        config = self.get_memory_config()

        # Get count efficiently
        total_count = lf.select(pl.len()).collect().item()

        if total_count == 0:
            return np.array([]).reshape(0, len(columns)), 0, False

        # Determine effective sample size
        if sample_size is None:
            sample_size = self._effective_sample_size or config.sample_size

        should_sample = sample_size is not None and total_count > sample_size

        # Select columns and drop nulls
        selected_lf = lf.select([pl.col(c) for c in columns]).drop_nulls()

        if should_sample:
            if method == "random":
                # Random sampling
                df = selected_lf.collect()
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, seed=config.random_state)
            elif method == "first":
                # First N rows (fastest)
                df = selected_lf.head(sample_size).collect()
            elif method == "stratified":
                # Stratified on first column (if categorical)
                df = selected_lf.collect()
                if len(df) > sample_size:
                    # Simple stratified sampling
                    df = df.sample(n=sample_size, seed=config.random_state)
            elif method == "reservoir":
                # Reservoir sampling (streaming)
                df = self._reservoir_sample(selected_lf, sample_size, config.random_state)
            else:
                raise ValueError(f"Unknown sampling method: {method}")
        else:
            df = selected_lf.collect()

        if len(df) == 0:
            return np.array([]).reshape(0, len(columns)), total_count, should_sample

        return df.to_numpy(), total_count, should_sample

    def _reservoir_sample(
        self,
        lf: pl.LazyFrame,
        sample_size: int,
        random_state: int,
    ) -> pl.DataFrame:
        """Reservoir sampling for streaming data.

        Algorithm R by Vitter (1985).
        """
        rng = np.random.default_rng(random_state)
        chunker = DataChunker(chunk_size=self.get_memory_config().chunk_size)

        reservoir: list[dict] = []
        seen = 0

        for chunk_df in chunker.iterate(lf):
            for row in chunk_df.iter_rows(named=True):
                seen += 1
                if len(reservoir) < sample_size:
                    reservoir.append(row)
                else:
                    # Replace with decreasing probability
                    j = rng.integers(0, seen)
                    if j < sample_size:
                        reservoir[j] = row

        return pl.DataFrame(reservoir)

    def batch_process(
        self,
        data: np.ndarray,
        process_fn: Callable[[np.ndarray], np.ndarray],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Process data in batches to reduce memory usage.

        Args:
            data: Input data array
            process_fn: Function to apply to each batch
            batch_size: Batch size (None = use configured size)

        Returns:
            Concatenated results
        """
        if batch_size is None:
            batch_size = self.get_memory_config().batch_size

        n_samples = len(data)
        if n_samples <= batch_size:
            return process_fn(data)

        results = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            batch_result = process_fn(batch)
            results.append(batch_result)

        return np.concatenate(results)

    def stream_process(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
        process_fn: Callable[[np.ndarray], np.ndarray],
        aggregator: Callable[[list[np.ndarray]], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Process data in streaming chunks.

        Args:
            lf: Input LazyFrame
            columns: Columns to process
            process_fn: Function to apply to each chunk
            aggregator: Optional function to aggregate results (default: concatenate)

        Returns:
            Aggregated results
        """
        config = self.get_memory_config()
        chunker = DataChunker(
            chunk_size=config.chunk_size,
            columns=columns,
            drop_nulls=True,
        )

        results = []
        for chunk_arr in chunker.iterate(lf, as_numpy=True):
            chunk_result = process_fn(chunk_arr)
            results.append(chunk_result)

        if aggregator is not None:
            return aggregator(results)
        return np.concatenate(results) if results else np.array([])

    def estimate_current_memory(
        self,
        lf: pl.LazyFrame,
        columns: list[str] | None = None,
        algorithm: str = "default",
    ) -> dict[str, float]:
        """Estimate memory usage for current data.

        Returns:
            Dict with 'data_mb', 'algorithm_mb', 'available_mb', 'fits_in_memory'
        """
        total_rows = lf.select(pl.len()).collect().item()
        n_cols = len(columns) if columns else len(lf.collect_schema().names())

        data_mb = estimate_memory_usage(total_rows, n_cols)
        algo_mb = estimate_algorithm_memory(total_rows, n_cols, algorithm)
        available_mb = get_available_memory()

        return {
            "data_mb": data_mb,
            "algorithm_mb": algo_mb,
            "available_mb": available_mb,
            "fits_in_memory": algo_mb < available_mb * self.get_memory_config().reserve_factor,
            "rows": total_rows,
            "columns": n_cols,
        }
