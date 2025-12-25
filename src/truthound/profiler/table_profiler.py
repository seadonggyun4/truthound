"""Table-level profiling implementation.

This module provides the main DataProfiler class that orchestrates
column profiling and adds table-level analysis including:
- Duplicate row detection
- Column correlation analysis
- Memory estimation
- Holistic table metrics
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    Profiler,
    ProfilerConfig,
    TableProfile,
    register_profiler,
)
from truthound.profiler.column_profiler import ColumnProfiler


# =============================================================================
# Table Analyzers (Strategy Pattern)
# =============================================================================


class TableAnalyzer:
    """Base class for table-level analyzers."""

    name: str = "base"

    def analyze(
        self,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        """Analyze the table and return metrics."""
        raise NotImplementedError


class DuplicateRowAnalyzer(TableAnalyzer):
    """Analyzes duplicate rows in the table."""

    name = "duplicate_rows"

    def analyze(
        self,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        # Count total rows
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return {
                "duplicate_row_count": 0,
                "duplicate_row_ratio": 0.0,
            }

        # Count unique rows
        unique_rows = lf.unique().select(pl.len()).collect().item()
        duplicate_count = total_rows - unique_rows

        return {
            "duplicate_row_count": duplicate_count,
            "duplicate_row_ratio": duplicate_count / total_rows,
        }


class MemoryEstimator(TableAnalyzer):
    """Estimates memory usage of the table."""

    name = "memory"

    # Approximate bytes per element for each type
    _TYPE_SIZES: dict[type[pl.DataType], int] = {
        pl.Int8: 1,
        pl.Int16: 2,
        pl.Int32: 4,
        pl.Int64: 8,
        pl.UInt8: 1,
        pl.UInt16: 2,
        pl.UInt32: 4,
        pl.UInt64: 8,
        pl.Float32: 4,
        pl.Float64: 8,
        pl.Boolean: 1,
        pl.Date: 4,
        pl.Datetime: 8,
        pl.Time: 8,
        pl.Duration: 8,
    }

    def analyze(
        self,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        schema = lf.collect_schema()
        row_count = lf.select(pl.len()).collect().item()

        estimated_bytes = 0
        for col_name, dtype in schema.items():
            dtype_type = type(dtype)
            if dtype_type in self._TYPE_SIZES:
                estimated_bytes += row_count * self._TYPE_SIZES[dtype_type]
            elif dtype_type in {pl.String, pl.Utf8}:
                # Estimate average string length
                avg_len = lf.select(
                    pl.col(col_name).str.len_bytes().mean()
                ).collect().item()
                avg_len = avg_len or 10  # Default assumption
                estimated_bytes += row_count * int(avg_len)
            else:
                # Default estimate
                estimated_bytes += row_count * 8

        return {"estimated_memory_bytes": estimated_bytes}


class CorrelationAnalyzer(TableAnalyzer):
    """Analyzes correlations between numeric columns."""

    name = "correlation"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def analyze(
        self,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        schema = lf.collect_schema()

        # Get numeric columns
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }
        numeric_cols = [
            name for name, dtype in schema.items()
            if type(dtype) in numeric_types
        ]

        if len(numeric_cols) < 2:
            return {"correlations": ()}

        correlations: list[tuple[str, str, float]] = []

        # Compute pairwise correlations
        df = lf.select(numeric_cols).collect()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                try:
                    corr = df.select(
                        pl.corr(col1, col2).alias("corr")
                    )["corr"][0]

                    if corr is not None and abs(corr) >= config.correlation_threshold:
                        correlations.append((col1, col2, float(corr)))
                except Exception:
                    pass

        return {"correlations": tuple(correlations)}


# =============================================================================
# Main DataProfiler
# =============================================================================


@register_profiler("default")
class DataProfiler(Profiler):
    """Main profiler that orchestrates table and column profiling.

    This is the primary entry point for profiling data. It combines
    column-level and table-level analysis into a comprehensive profile.

    Example:
        profiler = DataProfiler()
        profile = profiler.profile(lazy_frame)

        # Access results
        print(f"Rows: {profile.row_count}")
        for col in profile:
            print(f"{col.name}: {col.inferred_type}")

        # Export to JSON
        profile_dict = profile.to_dict()
    """

    name = "default"
    description = "Standard data profiler with comprehensive analysis"

    def __init__(
        self,
        *,
        config: ProfilerConfig | None = None,
        column_profiler: ColumnProfiler | None = None,
        table_analyzers: Sequence[TableAnalyzer] | None = None,
        parallel: bool = False,
        **kwargs: Any,
    ):
        """Initialize the data profiler.

        Args:
            config: Profiler configuration
            column_profiler: Custom column profiler (uses default if None)
            table_analyzers: Custom table analyzers (uses defaults if None)
            parallel: Whether to profile columns in parallel
            **kwargs: Additional arguments passed to base Profiler
        """
        super().__init__(**kwargs)
        self.config = config or ProfilerConfig()
        self.column_profiler = column_profiler or ColumnProfiler(config=self.config)
        self.parallel = parallel

        # Default table analyzers
        self.table_analyzers = list(table_analyzers) if table_analyzers else [
            DuplicateRowAnalyzer(),
            MemoryEstimator(),
        ]

        # Add correlation analyzer if configured
        if self.include_correlations:
            self.table_analyzers.append(
                CorrelationAnalyzer(threshold=self.config.correlation_threshold)
            )

    def add_table_analyzer(self, analyzer: TableAnalyzer) -> None:
        """Add a custom table analyzer."""
        self.table_analyzers.append(analyzer)

    def profile(
        self,
        data: pl.LazyFrame,
        name: str = "",
        source: str = "",
    ) -> TableProfile:
        """Profile the given data.

        Args:
            data: LazyFrame to profile
            name: Optional name for the table
            source: Optional source identifier (e.g., file path)

        Returns:
            Complete table profile
        """
        start_time = time.perf_counter()

        # Apply sampling if configured
        lf = self._maybe_sample(data)

        # Get schema
        schema = lf.collect_schema()
        columns = list(schema.names())

        # Get row count
        row_count = lf.select(pl.len()).collect().item()

        # Profile columns
        if self.parallel and len(columns) > 1:
            column_profiles = self._profile_columns_parallel(lf, schema)
        else:
            column_profiles = self._profile_columns_sequential(lf, schema)

        # Run table-level analyzers
        table_metrics = self._analyze_table(lf)

        # Build final profile
        duration_ms = (time.perf_counter() - start_time) * 1000

        return TableProfile(
            name=name,
            row_count=row_count,
            column_count=len(columns),
            estimated_memory_bytes=table_metrics.get("estimated_memory_bytes", 0),
            columns=tuple(column_profiles),
            duplicate_row_count=table_metrics.get("duplicate_row_count", 0),
            duplicate_row_ratio=table_metrics.get("duplicate_row_ratio", 0.0),
            correlations=table_metrics.get("correlations", ()),
            source=source,
            profiled_at=datetime.now(),
            profile_duration_ms=duration_ms,
        )

    def _profile_columns_sequential(
        self,
        lf: pl.LazyFrame,
        schema: pl.Schema,
    ) -> list[ColumnProfile]:
        """Profile columns sequentially."""
        profiles = []
        for col_name, dtype in schema.items():
            profile = self.column_profiler.profile_column(col_name, lf, dtype)
            profiles.append(profile)
        return profiles

    def _profile_columns_parallel(
        self,
        lf: pl.LazyFrame,
        schema: pl.Schema,
    ) -> list[ColumnProfile]:
        """Profile columns in parallel."""
        profiles: list[ColumnProfile] = []
        columns = list(schema.items())

        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = {
                executor.submit(
                    self.column_profiler.profile_column, col_name, lf, dtype
                ): col_name
                for col_name, dtype in columns
            }

            for future in as_completed(futures):
                try:
                    profile = future.result()
                    profiles.append(profile)
                except Exception:
                    # Skip failed columns
                    pass

        # Sort by original column order
        col_order = {name: i for i, (name, _) in enumerate(columns)}
        profiles.sort(key=lambda p: col_order.get(p.name, len(col_order)))

        return profiles

    def _analyze_table(self, lf: pl.LazyFrame) -> dict[str, Any]:
        """Run all table-level analyzers."""
        metrics: dict[str, Any] = {}
        for analyzer in self.table_analyzers:
            try:
                result = analyzer.analyze(lf, self.config)
                metrics.update(result)
            except Exception:
                pass
        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================


def profile_dataframe(
    data: pl.LazyFrame | pl.DataFrame,
    *,
    name: str = "",
    config: ProfilerConfig | None = None,
    **kwargs: Any,
) -> TableProfile:
    """Convenience function to profile a DataFrame.

    Args:
        data: DataFrame or LazyFrame to profile
        name: Optional name for the table
        config: Profiler configuration
        **kwargs: Additional arguments passed to DataProfiler

    Returns:
        Complete table profile

    Example:
        import polars as pl
        from truthound.profiler import profile_dataframe

        df = pl.read_parquet("data.parquet")
        profile = profile_dataframe(df, name="my_data")
    """
    if isinstance(data, pl.DataFrame):
        data = data.lazy()

    profiler = DataProfiler(config=config, **kwargs)
    return profiler.profile(data, name=name)


def profile_file(
    path: str | Path,
    *,
    name: str | None = None,
    config: ProfilerConfig | None = None,
    **kwargs: Any,
) -> TableProfile:
    """Profile data from a file.

    Supports: .parquet, .csv, .json, .ndjson

    Args:
        path: Path to the data file
        name: Optional name (defaults to filename)
        config: Profiler configuration
        **kwargs: Additional arguments passed to DataProfiler

    Returns:
        Complete table profile

    Example:
        from truthound.profiler import profile_file

        profile = profile_file("data.parquet")
        print(profile.to_dict())
    """
    path = Path(path)
    suffix = path.suffix.lower()

    readers = {
        ".parquet": pl.scan_parquet,
        ".csv": pl.scan_csv,
        ".json": pl.scan_ndjson,
        ".ndjson": pl.scan_ndjson,
    }

    if suffix not in readers:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {list(readers.keys())}"
        )

    lf = readers[suffix](path)
    table_name = name or path.stem

    profiler = DataProfiler(config=config, **kwargs)
    return profiler.profile(lf, name=table_name, source=str(path))


def _json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_profile(
    profile: TableProfile,
    path: str | Path,
    indent: int = 2,
) -> None:
    """Save a profile to a JSON file.

    Args:
        profile: Profile to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=indent, ensure_ascii=False, default=_json_serializer)


def load_profile(path: str | Path) -> TableProfile:
    """Load a profile from a JSON file.

    Args:
        path: Path to the profile JSON file

    Returns:
        Reconstructed TableProfile
    """
    from truthound.profiler.base import DistributionStats, PatternMatch, ValueFrequency

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct column profiles
    columns = []
    for col_data in data.get("columns", []):
        # Reconstruct nested objects
        if "distribution" in col_data:
            col_data["distribution"] = DistributionStats(**col_data["distribution"])

        if "top_values" in col_data:
            col_data["top_values"] = tuple(
                ValueFrequency(**v) for v in col_data["top_values"]
            )

        if "bottom_values" in col_data:
            col_data["bottom_values"] = tuple(
                ValueFrequency(**v) for v in col_data["bottom_values"]
            )

        if "detected_patterns" in col_data:
            col_data["detected_patterns"] = tuple(
                PatternMatch(
                    pattern=p["pattern"],
                    regex=p["regex"],
                    match_ratio=p["match_ratio"],
                    sample_matches=tuple(p.get("sample_matches", [])),
                )
                for p in col_data["detected_patterns"]
            )

        if "suggested_validators" in col_data:
            col_data["suggested_validators"] = tuple(col_data["suggested_validators"])

        # Parse datetime strings
        if "profiled_at" in col_data:
            col_data["profiled_at"] = datetime.fromisoformat(col_data["profiled_at"])

        if "min_date" in col_data and col_data["min_date"]:
            col_data["min_date"] = datetime.fromisoformat(col_data["min_date"])
        if "max_date" in col_data and col_data["max_date"]:
            col_data["max_date"] = datetime.fromisoformat(col_data["max_date"])

        # Handle inferred_type
        from truthound.profiler.base import DataType
        if "inferred_type" in col_data:
            col_data["inferred_type"] = DataType(col_data["inferred_type"])

        columns.append(ColumnProfile(**col_data))

    # Reconstruct correlations
    correlations = tuple(
        (c["column1"], c["column2"], c["correlation"])
        for c in data.get("correlations", [])
    )

    return TableProfile(
        name=data.get("name", ""),
        row_count=data.get("row_count", 0),
        column_count=data.get("column_count", 0),
        estimated_memory_bytes=data.get("estimated_memory_bytes", 0),
        columns=tuple(columns),
        duplicate_row_count=data.get("duplicate_row_count", 0),
        duplicate_row_ratio=data.get("duplicate_row_ratio", 0.0),
        correlations=correlations,
        source=data.get("source", ""),
        profiled_at=datetime.fromisoformat(data["profiled_at"]) if "profiled_at" in data else datetime.now(),
        profile_duration_ms=data.get("profile_duration_ms", 0.0),
    )
