"""Streaming/chunked profiling for memory-efficient processing.

This module provides streaming profiling capabilities that allow
profiling of datasets larger than available memory.

Key features:
- Chunked processing with configurable chunk sizes
- Incremental statistics aggregation
- Memory-aware processing
- Progress tracking with callbacks
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence, TypeVar

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    PatternMatch,
    ProfilerConfig,
    TableProfile,
    ValueFrequency,
)
from truthound.profiler.errors import (
    AnalysisError,
    ErrorCollector,
    ErrorSeverity,
    MemoryError as ProfilerMemoryError,
)


# =============================================================================
# Incremental Statistics
# =============================================================================


@dataclass
class IncrementalStats:
    """Maintains incremental statistics across chunks.

    Uses Welford's algorithm for numerically stable variance calculation.
    """

    count: int = 0
    null_count: int = 0
    distinct_values: set = field(default_factory=set)

    # Numeric stats (Welford's algorithm)
    _mean: float = 0.0
    _m2: float = 0.0  # Sum of squared differences from mean
    _min: float | None = None
    _max: float | None = None
    _numeric_count: int = 0  # Count of numeric values processed

    # String stats
    min_length: int | None = None
    max_length: int | None = None
    _length_sum: float = 0.0
    _length_count: int = 0
    empty_string_count: int = 0

    # Value frequency (with size limit)
    _value_counts: dict = field(default_factory=dict)
    _max_distinct_tracked: int = 10000

    def update_from_chunk(
        self,
        chunk: pl.DataFrame,
        column: str,
        is_numeric: bool = False,
        is_string: bool = False,
    ) -> None:
        """Update statistics from a new chunk.

        Args:
            chunk: DataFrame chunk
            column: Column name
            is_numeric: Whether column is numeric
            is_string: Whether column is string
        """
        self.count += len(chunk)
        self.null_count += chunk[column].null_count()

        # Get non-null values
        non_null = chunk.filter(pl.col(column).is_not_null())
        if len(non_null) == 0:
            return

        values = non_null[column]

        # Update distinct values (with limit)
        if len(self.distinct_values) < self._max_distinct_tracked:
            for v in values.to_list():
                if len(self.distinct_values) >= self._max_distinct_tracked:
                    break
                self.distinct_values.add(v)

        # Update value counts (for top/bottom values)
        self._update_value_counts(values)

        # Update numeric stats
        if is_numeric:
            self._update_numeric_stats(values)

        # Update string stats
        if is_string:
            self._update_string_stats(values)

    def _update_numeric_stats(self, values: pl.Series) -> None:
        """Update numeric statistics using Welford's algorithm."""
        for value in values.to_list():
            if value is None:
                continue

            val = float(value)

            # Update min/max
            if self._min is None or val < self._min:
                self._min = val
            if self._max is None or val > self._max:
                self._max = val

            # Welford's online algorithm for mean and variance
            self._numeric_count += 1
            delta = val - self._mean
            self._mean += delta / self._numeric_count
            delta2 = val - self._mean
            self._m2 += delta * delta2

    def _update_string_stats(self, values: pl.Series) -> None:
        """Update string length statistics."""
        lengths = values.str.len_chars()

        for length in lengths.to_list():
            if length is None:
                continue

            if length == 0:
                self.empty_string_count += 1

            if self.min_length is None or length < self.min_length:
                self.min_length = length
            if self.max_length is None or length > self.max_length:
                self.max_length = length

            self._length_sum += length
            self._length_count += 1

    def _update_value_counts(self, values: pl.Series) -> None:
        """Update value frequency counts."""
        for value in values.to_list():
            if value is None:
                continue
            key = str(value)
            self._value_counts[key] = self._value_counts.get(key, 0) + 1

    @property
    def mean(self) -> float | None:
        """Get current mean."""
        n = self.count - self.null_count
        return self._mean if n > 0 else None

    @property
    def variance(self) -> float | None:
        """Get sample variance."""
        if self._numeric_count < 2:
            return None
        return self._m2 / (self._numeric_count - 1)

    @property
    def std(self) -> float | None:
        """Get standard deviation."""
        var = self.variance
        return var ** 0.5 if var is not None else None

    @property
    def avg_length(self) -> float | None:
        """Get average string length."""
        if self._length_count == 0:
            return None
        return self._length_sum / self._length_count

    @property
    def null_ratio(self) -> float:
        """Get null ratio."""
        return self.null_count / self.count if self.count > 0 else 0.0

    @property
    def distinct_count(self) -> int:
        """Get distinct value count (may be estimate if > max_distinct_tracked)."""
        return len(self.distinct_values)

    @property
    def unique_ratio(self) -> float:
        """Get unique ratio."""
        non_null = self.count - self.null_count
        return self.distinct_count / non_null if non_null > 0 else 0.0

    def get_top_values(self, n: int = 10) -> tuple[ValueFrequency, ...]:
        """Get top N most frequent values."""
        total = sum(self._value_counts.values())
        sorted_items = sorted(
            self._value_counts.items(),
            key=lambda x: -x[1],
        )[:n]

        return tuple(
            ValueFrequency(value=k, count=v, ratio=v / total if total > 0 else 0.0)
            for k, v in sorted_items
        )

    def get_bottom_values(self, n: int = 10) -> tuple[ValueFrequency, ...]:
        """Get bottom N least frequent values."""
        total = sum(self._value_counts.values())
        sorted_items = sorted(
            self._value_counts.items(),
            key=lambda x: x[1],
        )[:n]

        return tuple(
            ValueFrequency(value=k, count=v, ratio=v / total if total > 0 else 0.0)
            for k, v in sorted_items
        )

    def to_distribution_stats(self) -> DistributionStats | None:
        """Convert to DistributionStats (for numeric columns)."""
        if self._min is None:
            return None

        return DistributionStats(
            mean=self.mean,
            std=self.std,
            min=self._min,
            max=self._max,
            # Note: median, q1, q3, skewness, kurtosis require full data or approximation
            median=None,
            q1=None,
            q3=None,
            skewness=None,
            kurtosis=None,
        )


# =============================================================================
# Chunk Iterator Protocol
# =============================================================================


class ChunkIterator(Protocol):
    """Protocol for iterating over data chunks."""

    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Iterate over chunks."""
        ...

    @property
    def total_chunks(self) -> int | None:
        """Total number of chunks (None if unknown)."""
        ...


@dataclass
class FileChunkIterator:
    """Iterate over chunks from a file.

    Supports: .parquet, .csv, .ndjson
    """

    path: Path
    chunk_size: int = 100_000
    _total_rows: int | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Iterate over file chunks."""
        suffix = self.path.suffix.lower()

        if suffix == ".parquet":
            yield from self._iter_parquet()
        elif suffix == ".csv":
            yield from self._iter_csv()
        elif suffix in {".json", ".ndjson"}:
            yield from self._iter_ndjson()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _iter_parquet(self) -> Iterator[pl.DataFrame]:
        """Iterate over parquet file chunks."""
        # Parquet files can be read in row groups
        lf = pl.scan_parquet(self.path)
        total = lf.select(pl.len()).collect().item()
        self._total_rows = total

        offset = 0
        while offset < total:
            chunk = lf.slice(offset, self.chunk_size).collect()
            yield chunk
            offset += len(chunk)

    def _iter_csv(self) -> Iterator[pl.DataFrame]:
        """Iterate over CSV file chunks using batched reader."""
        reader = pl.read_csv_batched(self.path, batch_size=self.chunk_size)
        if reader is None:
            return

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            yield batches[0]

    def _iter_ndjson(self) -> Iterator[pl.DataFrame]:
        """Iterate over NDJSON file chunks."""
        lf = pl.scan_ndjson(self.path)
        total = lf.select(pl.len()).collect().item()
        self._total_rows = total

        offset = 0
        while offset < total:
            chunk = lf.slice(offset, self.chunk_size).collect()
            yield chunk
            offset += len(chunk)

    @property
    def total_chunks(self) -> int | None:
        """Estimate total chunks."""
        if self._total_rows is None:
            return None
        return (self._total_rows + self.chunk_size - 1) // self.chunk_size


@dataclass
class DataFrameChunkIterator:
    """Iterate over chunks from an in-memory DataFrame."""

    df: pl.DataFrame
    chunk_size: int = 100_000

    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Iterate over DataFrame chunks."""
        total = len(self.df)
        offset = 0
        while offset < total:
            yield self.df.slice(offset, self.chunk_size)
            offset += self.chunk_size

    @property
    def total_chunks(self) -> int | None:
        """Get total number of chunks."""
        return (len(self.df) + self.chunk_size - 1) // self.chunk_size


# =============================================================================
# Progress Tracking
# =============================================================================


@dataclass
class StreamingProgress:
    """Progress information for streaming profiling."""

    chunks_processed: int = 0
    total_chunks: int | None = None
    rows_processed: int = 0
    current_column: str = ""
    elapsed_seconds: float = 0.0
    memory_usage_mb: float = 0.0

    @property
    def progress_ratio(self) -> float | None:
        """Get progress ratio (0.0 to 1.0)."""
        if self.total_chunks is None or self.total_chunks == 0:
            return None
        return self.chunks_processed / self.total_chunks

    @property
    def progress_percent(self) -> float | None:
        """Get progress percentage."""
        ratio = self.progress_ratio
        return ratio * 100 if ratio is not None else None


ProgressCallback = Callable[[StreamingProgress], None]


# =============================================================================
# Streaming Profiler
# =============================================================================


class StreamingProfiler:
    """Memory-efficient profiler using chunked processing.

    This profiler processes data in chunks, maintaining incremental
    statistics to profile datasets larger than available memory.

    Example:
        profiler = StreamingProfiler(chunk_size=50_000)

        # From file
        profile = profiler.profile_file("large_data.parquet")

        # With progress callback
        def on_progress(p: StreamingProgress):
            print(f"Progress: {p.progress_percent:.1f}%")

        profile = profiler.profile_file(
            "data.csv",
            progress_callback=on_progress
        )

        # From DataFrame
        profile = profiler.profile_dataframe(large_df, name="my_data")
    """

    def __init__(
        self,
        *,
        chunk_size: int = 100_000,
        config: ProfilerConfig | None = None,
        error_collector: ErrorCollector | None = None,
        pattern_sample_limit: int = 10_000,
    ):
        """Initialize streaming profiler.

        Args:
            chunk_size: Number of rows per chunk
            config: Profiler configuration
            error_collector: Error collector for graceful error handling
            pattern_sample_limit: Max rows for pattern detection
        """
        self.chunk_size = chunk_size
        self.config = config or ProfilerConfig()
        self.error_collector = error_collector or ErrorCollector()
        self.pattern_sample_limit = pattern_sample_limit

    def profile_file(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> TableProfile:
        """Profile a file using streaming.

        Args:
            path: Path to the data file
            name: Optional name for the profile
            progress_callback: Optional callback for progress updates

        Returns:
            Complete table profile
        """
        path = Path(path)
        iterator = FileChunkIterator(path, chunk_size=self.chunk_size)

        return self._profile_chunks(
            iterator,
            name=name or path.stem,
            source=str(path),
            progress_callback=progress_callback,
        )

    def profile_dataframe(
        self,
        df: pl.DataFrame,
        *,
        name: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> TableProfile:
        """Profile a DataFrame using streaming.

        Args:
            df: DataFrame to profile
            name: Optional name for the profile
            progress_callback: Optional callback for progress updates

        Returns:
            Complete table profile
        """
        iterator = DataFrameChunkIterator(df, chunk_size=self.chunk_size)

        return self._profile_chunks(
            iterator,
            name=name,
            source="in_memory",
            progress_callback=progress_callback,
        )

    def _profile_chunks(
        self,
        chunks: ChunkIterator,
        name: str,
        source: str,
        progress_callback: ProgressCallback | None,
    ) -> TableProfile:
        """Profile data from chunk iterator.

        Args:
            chunks: Iterator yielding DataFrame chunks
            name: Profile name
            source: Data source identifier
            progress_callback: Progress callback

        Returns:
            Complete table profile
        """
        start_time = time.perf_counter()
        progress = StreamingProgress(total_chunks=chunks.total_chunks)

        # Initialize per-column stats
        column_stats: dict[str, IncrementalStats] = {}
        schema: pl.Schema | None = None
        total_rows = 0

        # Process chunks
        for chunk in chunks:
            if schema is None:
                schema = chunk.schema
                # Initialize stats for each column
                for col_name in schema.names():
                    column_stats[col_name] = IncrementalStats()

            total_rows += len(chunk)
            progress.chunks_processed += 1
            progress.rows_processed = total_rows
            progress.elapsed_seconds = time.perf_counter() - start_time

            # Update each column's stats
            for col_name, dtype in schema.items():
                is_numeric = self._is_numeric(dtype)
                is_string = self._is_string(dtype)

                progress.current_column = col_name

                with self.error_collector.catch(
                    column=col_name,
                    analyzer="streaming",
                    severity=ErrorSeverity.WARNING,
                ):
                    column_stats[col_name].update_from_chunk(
                        chunk, col_name, is_numeric=is_numeric, is_string=is_string
                    )

            # Call progress callback
            if progress_callback:
                progress_callback(progress)

        if schema is None:
            # Empty data
            return TableProfile(
                name=name,
                source=source,
                profiled_at=datetime.now(),
                profile_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Build column profiles
        column_profiles = []
        for col_name, dtype in schema.items():
            stats = column_stats[col_name]

            profile = ColumnProfile(
                name=col_name,
                physical_type=str(dtype),
                row_count=stats.count,
                null_count=stats.null_count,
                null_ratio=stats.null_ratio,
                distinct_count=stats.distinct_count,
                unique_ratio=stats.unique_ratio,
                is_unique=stats.distinct_count == (stats.count - stats.null_count),
                is_constant=stats.distinct_count <= 1,
                distribution=stats.to_distribution_stats() if self._is_numeric(dtype) else None,
                top_values=stats.get_top_values(self.config.top_n_values),
                bottom_values=stats.get_bottom_values(self.config.top_n_values),
                min_length=stats.min_length if self._is_string(dtype) else None,
                max_length=stats.max_length if self._is_string(dtype) else None,
                avg_length=stats.avg_length if self._is_string(dtype) else None,
                empty_string_count=stats.empty_string_count if self._is_string(dtype) else 0,
                profiled_at=datetime.now(),
            )
            column_profiles.append(profile)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return TableProfile(
            name=name,
            row_count=total_rows,
            column_count=len(column_profiles),
            columns=tuple(column_profiles),
            source=source,
            profiled_at=datetime.now(),
            profile_duration_ms=duration_ms,
        )

    def _is_numeric(self, dtype: pl.DataType) -> bool:
        """Check if dtype is numeric."""
        return type(dtype) in {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

    def _is_string(self, dtype: pl.DataType) -> bool:
        """Check if dtype is string."""
        return type(dtype) in {pl.String, pl.Utf8}


# =============================================================================
# Convenience Functions
# =============================================================================


def stream_profile_file(
    path: str | Path,
    *,
    chunk_size: int = 100_000,
    name: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TableProfile:
    """Convenience function to profile a file with streaming.

    Args:
        path: Path to the data file
        chunk_size: Rows per chunk
        name: Optional profile name
        progress_callback: Progress callback

    Returns:
        Table profile

    Example:
        from truthound.profiler.streaming import stream_profile_file

        # Basic usage
        profile = stream_profile_file("large_data.parquet")

        # With progress
        def show_progress(p):
            if p.progress_percent:
                print(f"\\rProcessing: {p.progress_percent:.1f}%", end="")

        profile = stream_profile_file(
            "very_large_data.csv",
            chunk_size=50_000,
            progress_callback=show_progress
        )
    """
    profiler = StreamingProfiler(chunk_size=chunk_size)
    return profiler.profile_file(path, name=name, progress_callback=progress_callback)


def stream_profile_dataframe(
    df: pl.DataFrame,
    *,
    chunk_size: int = 100_000,
    name: str = "",
    progress_callback: ProgressCallback | None = None,
) -> TableProfile:
    """Convenience function to profile a DataFrame with streaming.

    Args:
        df: DataFrame to profile
        chunk_size: Rows per chunk
        name: Optional profile name
        progress_callback: Optional callback for progress updates

    Returns:
        Table profile
    """
    profiler = StreamingProfiler(chunk_size=chunk_size)
    return profiler.profile_dataframe(df, name=name, progress_callback=progress_callback)
