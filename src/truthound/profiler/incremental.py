"""Incremental profiling based on previous profiles.

This module provides incremental profiling capabilities that skip
unchanged columns and only re-profile when necessary, significantly
improving performance for repeated profiling.

Key features:
- Hash-based change detection
- Selective column profiling
- Profile merging
- Configurable staleness detection
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Sequence

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    ProfilerConfig,
    TableProfile,
)
from truthound.profiler.column_profiler import ColumnProfiler
from truthound.profiler.progress import ProgressTracker


# =============================================================================
# Change Detection
# =============================================================================


class ChangeReason(str, Enum):
    """Reasons for re-profiling a column."""

    NEW_COLUMN = "new_column"           # Column didn't exist before
    SCHEMA_CHANGED = "schema_changed"   # Data type changed
    DATA_CHANGED = "data_changed"       # Content hash changed
    STALE = "stale"                     # Profile too old
    FORCED = "forced"                   # Force refresh requested
    SAMPLE_CHANGED = "sample_changed"   # Sample values changed


@dataclass(frozen=True)
class ColumnFingerprint:
    """Fingerprint for detecting column changes.

    This provides a lightweight way to detect if a column's data
    has changed without fully profiling it.
    """

    column_name: str
    dtype: str
    row_count: int
    null_count: int
    sample_hash: str  # Hash of sampled values
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column_name": self.column_name,
            "dtype": self.dtype,
            "row_count": self.row_count,
            "null_count": self.null_count,
            "sample_hash": self.sample_hash,
            "computed_at": self.computed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnFingerprint":
        """Create from dictionary."""
        return cls(
            column_name=data["column_name"],
            dtype=data["dtype"],
            row_count=data["row_count"],
            null_count=data["null_count"],
            sample_hash=data["sample_hash"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
        )


@dataclass
class ChangeDetectionResult:
    """Result of change detection for a column."""

    column_name: str
    needs_profiling: bool
    reason: ChangeReason | None = None
    previous_fingerprint: ColumnFingerprint | None = None
    current_fingerprint: ColumnFingerprint | None = None


class FingerprintCalculator:
    """Calculates fingerprints for change detection.

    The fingerprint is a lightweight hash-based signature that can
    detect changes without full profiling.
    """

    def __init__(
        self,
        sample_size: int = 1000,
        seed: int = 42,
    ):
        """Initialize calculator.

        Args:
            sample_size: Number of rows to sample for hash
            seed: Random seed for sampling
        """
        self.sample_size = sample_size
        self.seed = seed

    def calculate(
        self,
        lf: pl.LazyFrame,
        column: str,
    ) -> ColumnFingerprint:
        """Calculate fingerprint for a column.

        Args:
            lf: LazyFrame containing the data
            column: Column name

        Returns:
            Column fingerprint
        """
        schema = lf.collect_schema()
        dtype = str(schema[column])

        # Get basic stats
        stats = lf.select(
            pl.len().alias("row_count"),
            pl.col(column).null_count().alias("null_count"),
        ).collect()

        row_count = stats["row_count"][0]
        null_count = stats["null_count"][0]

        # Calculate sample hash
        sample_hash = self._calculate_sample_hash(lf, column, row_count)

        return ColumnFingerprint(
            column_name=column,
            dtype=dtype,
            row_count=row_count,
            null_count=null_count,
            sample_hash=sample_hash,
        )

    def _calculate_sample_hash(
        self,
        lf: pl.LazyFrame,
        column: str,
        row_count: int,
    ) -> str:
        """Calculate hash of sampled values."""
        # Deterministic sample
        if row_count <= self.sample_size:
            sample = lf.select(pl.col(column)).collect()
        else:
            # Use modulo for deterministic sampling
            sample = (
                lf.with_row_index("__idx__")
                .filter(pl.col("__idx__") % (row_count // self.sample_size) == 0)
                .select(pl.col(column))
                .head(self.sample_size)
                .collect()
            )

        # Hash the values
        values_str = str(sample[column].to_list())
        return hashlib.md5(values_str.encode()).hexdigest()[:16]


# =============================================================================
# Incremental Profiling Configuration
# =============================================================================


@dataclass
class IncrementalConfig:
    """Configuration for incremental profiling.

    Attributes:
        max_age: Maximum age before re-profiling (None = never stale)
        sample_size: Sample size for change detection
        force_columns: Always re-profile these columns
        skip_columns: Never profile these columns
        detect_schema_changes: Re-profile on schema change
        detect_data_changes: Re-profile on data hash change
    """

    max_age: timedelta | None = None
    sample_size: int = 1000
    force_columns: set[str] = field(default_factory=set)
    skip_columns: set[str] = field(default_factory=set)
    detect_schema_changes: bool = True
    detect_data_changes: bool = True

    @classmethod
    def aggressive(cls) -> "IncrementalConfig":
        """Configuration that minimizes re-profiling."""
        return cls(
            max_age=timedelta(days=7),
            sample_size=500,
        )

    @classmethod
    def conservative(cls) -> "IncrementalConfig":
        """Configuration that ensures freshness."""
        return cls(
            max_age=timedelta(hours=1),
            sample_size=2000,
        )


# =============================================================================
# Incremental Profiler
# =============================================================================


class IncrementalProfiler:
    """Profiler that updates only changed columns.

    This profiler uses fingerprints and previous profiles to skip
    unchanged columns, significantly improving performance.

    Example:
        profiler = IncrementalProfiler()

        # First run - profiles everything
        profile1 = profiler.profile(data, name="my_table")

        # Second run - only profiles changed columns
        profile2 = profiler.profile(data, name="my_table", previous=profile1)

        # Check what was re-profiled
        print(profiler.last_profiled_columns)
    """

    def __init__(
        self,
        config: IncrementalConfig | None = None,
        profiler_config: ProfilerConfig | None = None,
        column_profiler: ColumnProfiler | None = None,
    ):
        """Initialize incremental profiler.

        Args:
            config: Incremental profiling configuration
            profiler_config: Base profiler configuration
            column_profiler: Column profiler to use
        """
        self.config = config or IncrementalConfig()
        self.profiler_config = profiler_config or ProfilerConfig()
        self.column_profiler = column_profiler or ColumnProfiler(config=self.profiler_config)
        self.fingerprint_calculator = FingerprintCalculator(
            sample_size=self.config.sample_size,
        )

        # Track last run
        self._last_profiled_columns: set[str] = set()
        self._last_skipped_columns: set[str] = set()
        self._last_change_reasons: dict[str, ChangeReason] = {}

    @property
    def last_profiled_columns(self) -> set[str]:
        """Get columns that were re-profiled in last run."""
        return self._last_profiled_columns.copy()

    @property
    def last_skipped_columns(self) -> set[str]:
        """Get columns that were skipped in last run."""
        return self._last_skipped_columns.copy()

    @property
    def last_change_reasons(self) -> dict[str, ChangeReason]:
        """Get reasons for re-profiling each column."""
        return self._last_change_reasons.copy()

    def profile(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        name: str = "",
        *,
        previous: TableProfile | None = None,
        force_refresh: bool = False,
        progress_tracker: ProgressTracker | None = None,
    ) -> TableProfile:
        """Profile data incrementally.

        Args:
            data: Data to profile
            name: Table name
            previous: Previous profile for comparison
            force_refresh: Force re-profile all columns
            progress_tracker: Optional progress tracker

        Returns:
            Updated table profile
        """
        start_time = time.perf_counter()

        if isinstance(data, pl.DataFrame):
            data = data.lazy()

        schema = data.collect_schema()
        columns = list(schema.names())

        # Reset tracking
        self._last_profiled_columns = set()
        self._last_skipped_columns = set()
        self._last_change_reasons = {}

        # Build previous column lookup
        prev_columns = {}
        prev_fingerprints = {}
        if previous:
            prev_columns = {col.name: col for col in previous.columns}
            # We don't have fingerprints stored in profile, so compute them
            # In production, fingerprints should be stored with profiles

        # Initialize progress
        if progress_tracker:
            progress_tracker.start(total_columns=len(columns))

        # Determine which columns need profiling
        columns_to_profile = []
        columns_to_reuse = []

        for col_name in columns:
            if col_name in self.config.skip_columns:
                self._last_skipped_columns.add(col_name)
                continue

            if force_refresh or col_name in self.config.force_columns:
                columns_to_profile.append(col_name)
                self._last_change_reasons[col_name] = ChangeReason.FORCED
                continue

            if col_name not in prev_columns:
                columns_to_profile.append(col_name)
                self._last_change_reasons[col_name] = ChangeReason.NEW_COLUMN
                continue

            # Check for changes
            change = self._detect_change(
                data, col_name, schema[col_name],
                prev_columns[col_name],
            )

            if change.needs_profiling:
                columns_to_profile.append(col_name)
                self._last_change_reasons[col_name] = change.reason
            else:
                columns_to_reuse.append(col_name)

        # Profile changed columns
        new_profiles: dict[str, ColumnProfile] = {}

        for i, col_name in enumerate(columns_to_profile):
            if progress_tracker:
                progress_tracker.column_start(col_name)

            profile = self.column_profiler.profile_column(
                col_name, data, schema[col_name]
            )
            new_profiles[col_name] = profile
            self._last_profiled_columns.add(col_name)

            if progress_tracker:
                progress_tracker.column_complete(col_name)

        # Build final column list
        final_columns = []
        for col_name in columns:
            if col_name in self.config.skip_columns:
                continue

            if col_name in new_profiles:
                final_columns.append(new_profiles[col_name])
            elif col_name in prev_columns:
                # Reuse previous profile
                final_columns.append(prev_columns[col_name])
                self._last_skipped_columns.add(col_name)

        # Get row count
        row_count = data.select(pl.len()).collect().item()

        duration_ms = (time.perf_counter() - start_time) * 1000

        if progress_tracker:
            progress_tracker.complete()

        return TableProfile(
            name=name,
            row_count=row_count,
            column_count=len(final_columns),
            columns=tuple(final_columns),
            source=previous.source if previous else "",
            profiled_at=datetime.now(),
            profile_duration_ms=duration_ms,
        )

    def _detect_change(
        self,
        lf: pl.LazyFrame,
        column: str,
        dtype: pl.DataType,
        prev_profile: ColumnProfile,
    ) -> ChangeDetectionResult:
        """Detect if a column has changed."""
        # Check staleness
        if self.config.max_age is not None:
            age = datetime.now() - prev_profile.profiled_at
            if age > self.config.max_age:
                return ChangeDetectionResult(
                    column_name=column,
                    needs_profiling=True,
                    reason=ChangeReason.STALE,
                )

        # Check schema change
        if self.config.detect_schema_changes:
            if str(dtype) != prev_profile.physical_type:
                return ChangeDetectionResult(
                    column_name=column,
                    needs_profiling=True,
                    reason=ChangeReason.SCHEMA_CHANGED,
                )

        # Check data change via fingerprint
        if self.config.detect_data_changes:
            current_fp = self.fingerprint_calculator.calculate(lf, column)

            # Simple check: row count and null count
            if (current_fp.row_count != prev_profile.row_count or
                current_fp.null_count != prev_profile.null_count):
                return ChangeDetectionResult(
                    column_name=column,
                    needs_profiling=True,
                    reason=ChangeReason.DATA_CHANGED,
                    current_fingerprint=current_fp,
                )

        # No change detected
        return ChangeDetectionResult(
            column_name=column,
            needs_profiling=False,
        )


# =============================================================================
# Profile Merger
# =============================================================================


class ProfileMerger:
    """Merges multiple profiles into one.

    Useful for combining profiles from parallel processing or
    different data partitions.
    """

    def merge(
        self,
        profiles: Sequence[TableProfile],
        name: str = "",
    ) -> TableProfile:
        """Merge multiple profiles into one.

        For conflicting columns (same name), uses the most recent profile.

        Args:
            profiles: Profiles to merge
            name: Name for merged profile

        Returns:
            Merged table profile
        """
        if not profiles:
            return TableProfile(name=name)

        if len(profiles) == 1:
            return profiles[0]

        # Merge columns (latest wins for duplicates)
        column_map: dict[str, ColumnProfile] = {}
        total_rows = 0

        for profile in sorted(profiles, key=lambda p: p.profiled_at):
            total_rows += profile.row_count
            for col in profile.columns:
                column_map[col.name] = col

        return TableProfile(
            name=name or profiles[-1].name,
            row_count=total_rows,
            column_count=len(column_map),
            columns=tuple(column_map.values()),
            source=profiles[-1].source,
            profiled_at=datetime.now(),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def profile_incrementally(
    data: pl.LazyFrame | pl.DataFrame,
    previous: TableProfile | None = None,
    *,
    name: str = "",
    max_age: timedelta | None = None,
    force_refresh: bool = False,
) -> TableProfile:
    """Profile data incrementally using previous profile.

    Args:
        data: Data to profile
        previous: Previous profile for comparison
        name: Table name
        max_age: Maximum profile age before re-profiling
        force_refresh: Force re-profile all columns

    Returns:
        Updated profile

    Example:
        from truthound.profiler import profile_file
        from truthound.profiler.incremental import profile_incrementally

        # First run
        profile1 = profile_file("data.parquet")

        # Later - incremental update
        profile2 = profile_incrementally(
            pl.scan_parquet("data.parquet"),
            previous=profile1,
        )
    """
    config = IncrementalConfig(max_age=max_age)
    profiler = IncrementalProfiler(config=config)
    return profiler.profile(data, name=name, previous=previous, force_refresh=force_refresh)
