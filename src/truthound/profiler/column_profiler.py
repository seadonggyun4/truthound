"""Column-level profiling implementation.

This module provides detailed per-column profiling including:
- Basic statistics (null ratio, distinct count, etc.)
- Distribution analysis for numeric columns
- Pattern detection for string columns
- Temporal analysis for datetime columns
- Semantic type inference
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    PatternMatch,
    ProfilerConfig,
    TypeInferrer,
    ValueFrequency,
    register_type_inferrer,
)


# =============================================================================
# Pattern Definitions
# =============================================================================


@dataclass(frozen=True)
class PatternDefinition:
    """Definition of a pattern to detect."""

    name: str
    regex: str
    data_type: DataType
    priority: int = 0  # Higher priority patterns are checked first

    def compile(self) -> re.Pattern:
        return re.compile(self.regex)


# Built-in patterns ordered by specificity (more specific first)
BUILTIN_PATTERNS: tuple[PatternDefinition, ...] = (
    # Korean specific (highest priority)
    PatternDefinition(
        "korean_rrn",
        r"^\d{6}-[1-4]\d{6}$",
        DataType.KOREAN_RRN,
        priority=100,
    ),
    PatternDefinition(
        "korean_phone",
        r"^01[0-9]-\d{3,4}-\d{4}$",
        DataType.KOREAN_PHONE,
        priority=100,
    ),
    PatternDefinition(
        "korean_business_number",
        r"^\d{3}-\d{2}-\d{5}$",
        DataType.KOREAN_BUSINESS_NUMBER,
        priority=100,
    ),
    # Standard patterns
    PatternDefinition(
        "uuid",
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        DataType.UUID,
        priority=90,
    ),
    PatternDefinition(
        "email",
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        DataType.EMAIL,
        priority=80,
    ),
    PatternDefinition(
        "url",
        r"^https?://[^\s/$.?#].[^\s]*$",
        DataType.URL,
        priority=80,
    ),
    PatternDefinition(
        "ip_address",
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
        DataType.IP_ADDRESS,
        priority=70,
    ),
    PatternDefinition(
        "phone",
        r"^\+?[1-9]\d{6,14}$",  # E.164: min 7 digits total (country + subscriber)
        DataType.PHONE,
        priority=60,
    ),
    PatternDefinition(
        "json",
        r'^[\[\{].*[\]\}]$',
        DataType.JSON,
        priority=50,
    ),
    # Numeric string patterns (lower priority - fallback detection)
    PatternDefinition(
        "currency_string",
        r"^-?\d{1,3}(,\d{3})+(\.\d{2})?$",  # 1,234.56 (requires comma separator)
        DataType.CURRENCY,
        priority=45,
    ),
    PatternDefinition(
        "percentage_string",
        r"^-?\d+(\.\d+)?%$",  # 85.5% or 100%
        DataType.PERCENTAGE,
        priority=45,
    ),
    PatternDefinition(
        "float_string",
        r"^-?\d+\.\d+$",  # 69000.00, -15.3 (decimal numbers)
        DataType.FLOAT,
        priority=40,
    ),
    PatternDefinition(
        "integer_string",
        r"^-?\d{3,}$",  # 123, -456 (min 3 digits to avoid age confusion)
        DataType.INTEGER,
        priority=35,
    ),
)


# =============================================================================
# Analysis Strategies (Strategy Pattern)
# =============================================================================


class ColumnAnalyzer(ABC):
    """Abstract base for column analysis strategies.

    Each analyzer focuses on one aspect of column profiling.
    """

    name: str = "base"
    applicable_types: set[type[pl.DataType]] | None = None  # None = all types

    @abstractmethod
    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        """Analyze the column and return metrics.

        Returns:
            Dictionary of metric name to value
        """
        pass

    def is_applicable(self, dtype: pl.DataType) -> bool:
        """Check if this analyzer is applicable to the given dtype."""
        if self.applicable_types is None:
            return True
        return type(dtype) in self.applicable_types


class BasicStatsAnalyzer(ColumnAnalyzer):
    """Analyzes basic column statistics."""

    name = "basic_stats"

    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        stats = lf.select(
            pl.len().alias("row_count"),
            pl.col(column).null_count().alias("null_count"),
            pl.col(column).n_unique().alias("distinct_count"),
        ).collect()

        row_count = stats["row_count"][0]
        null_count = stats["null_count"][0]
        distinct_count = stats["distinct_count"][0]

        # Handle nulls in distinct count
        non_null_count = row_count - null_count
        unique_ratio = distinct_count / non_null_count if non_null_count > 0 else 0.0

        return {
            "row_count": row_count,
            "null_count": null_count,
            "null_ratio": null_count / row_count if row_count > 0 else 0.0,
            "distinct_count": distinct_count,
            "unique_ratio": unique_ratio,
            "is_unique": distinct_count == non_null_count and non_null_count > 0,
            "is_constant": distinct_count <= 1,
        }


class NumericAnalyzer(ColumnAnalyzer):
    """Analyzes numeric column distributions."""

    name = "numeric"
    applicable_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }

    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        stats = lf.select(
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.25).alias("q1"),
            pl.col(column).quantile(0.75).alias("q3"),
            pl.col(column).skew().alias("skewness"),
            pl.col(column).kurtosis().alias("kurtosis"),
        ).collect()

        return {
            "distribution": DistributionStats(
                mean=stats["mean"][0],
                std=stats["std"][0],
                min=stats["min"][0],
                max=stats["max"][0],
                median=stats["median"][0],
                q1=stats["q1"][0],
                q3=stats["q3"][0],
                skewness=stats["skewness"][0],
                kurtosis=stats["kurtosis"][0],
            )
        }


class StringAnalyzer(ColumnAnalyzer):
    """Analyzes string column properties."""

    name = "string"
    applicable_types = {pl.String, pl.Utf8}

    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        stats = lf.select(
            pl.col(column).str.len_chars().min().alias("min_length"),
            pl.col(column).str.len_chars().max().alias("max_length"),
            pl.col(column).str.len_chars().mean().alias("avg_length"),
            (pl.col(column) == "").sum().alias("empty_string_count"),
        ).collect()

        return {
            "min_length": stats["min_length"][0],
            "max_length": stats["max_length"][0],
            "avg_length": stats["avg_length"][0],
            "empty_string_count": stats["empty_string_count"][0],
        }


class DatetimeAnalyzer(ColumnAnalyzer):
    """Analyzes datetime column properties."""

    name = "datetime"
    applicable_types = {pl.Date, pl.Datetime}

    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        stats = lf.select(
            pl.col(column).min().alias("min_date"),
            pl.col(column).max().alias("max_date"),
        ).collect()

        min_date = stats["min_date"][0]
        max_date = stats["max_date"][0]

        # Convert to datetime if needed
        if isinstance(min_date, (int, float)):
            min_date = None
        if isinstance(max_date, (int, float)):
            max_date = None

        return {
            "min_date": min_date,
            "max_date": max_date,
        }


class ValueFrequencyAnalyzer(ColumnAnalyzer):
    """Analyzes value frequencies (top/bottom values)."""

    name = "value_frequency"

    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        n = config.top_n_values

        # Get value counts
        value_counts = (
            lf.select(pl.col(column))
            .filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(pl.len().alias("count"))
            .collect()
        )

        if len(value_counts) == 0:
            return {"top_values": (), "bottom_values": ()}

        total = value_counts["count"].sum()

        # Top values
        top = value_counts.sort("count", descending=True).head(n)
        top_values = tuple(
            ValueFrequency(
                value=row[column],
                count=row["count"],
                ratio=row["count"] / total if total > 0 else 0.0,
            )
            for row in top.iter_rows(named=True)
        )

        # Bottom values (least frequent)
        bottom = value_counts.sort("count").head(n)
        bottom_values = tuple(
            ValueFrequency(
                value=row[column],
                count=row["count"],
                ratio=row["count"] / total if total > 0 else 0.0,
            )
            for row in bottom.iter_rows(named=True)
        )

        return {
            "top_values": top_values,
            "bottom_values": bottom_values,
        }


class PatternAnalyzer(ColumnAnalyzer):
    """Detects patterns in string columns."""

    name = "pattern"
    applicable_types = {pl.String, pl.Utf8}

    def __init__(
        self,
        patterns: Sequence[PatternDefinition] | None = None,
        min_match_ratio: float = 0.8,
    ):
        self.patterns = patterns or BUILTIN_PATTERNS
        self.min_match_ratio = min_match_ratio

    def analyze(
        self,
        column: str,
        lf: pl.LazyFrame,
        config: ProfilerConfig,
    ) -> dict[str, Any]:
        # Sample for pattern detection
        sample_size = config.pattern_sample_size
        sample = (
            lf.select(pl.col(column))
            .filter(pl.col(column).is_not_null())
            .head(sample_size)
            .collect()
        )

        if len(sample) == 0:
            return {"detected_patterns": ()}

        values = sample[column].to_list()
        total = len(values)
        detected: list[PatternMatch] = []

        # Sort patterns by priority (highest first)
        sorted_patterns = sorted(self.patterns, key=lambda p: -p.priority)

        for pattern_def in sorted_patterns:
            compiled = pattern_def.compile()
            matches = [v for v in values if v and compiled.match(str(v))]
            match_ratio = len(matches) / total if total > 0 else 0.0

            if match_ratio >= config.min_pattern_match_ratio:
                detected.append(
                    PatternMatch(
                        pattern=pattern_def.name,
                        regex=pattern_def.regex,
                        match_ratio=match_ratio,
                        sample_matches=tuple(matches[:5]),
                    )
                )

        return {"detected_patterns": tuple(detected)}


# =============================================================================
# Type Inferrers
# =============================================================================


@register_type_inferrer("physical")
class PhysicalTypeInferrer(TypeInferrer):
    """Infers type based on physical Polars type."""

    name = "physical"
    priority = 0  # Lowest priority, fallback

    _TYPE_MAPPING: dict[type[pl.DataType], DataType] = {
        pl.Int8: DataType.INTEGER,
        pl.Int16: DataType.INTEGER,
        pl.Int32: DataType.INTEGER,
        pl.Int64: DataType.INTEGER,
        pl.UInt8: DataType.INTEGER,
        pl.UInt16: DataType.INTEGER,
        pl.UInt32: DataType.INTEGER,
        pl.UInt64: DataType.INTEGER,
        pl.Float32: DataType.FLOAT,
        pl.Float64: DataType.FLOAT,
        pl.Boolean: DataType.BOOLEAN,
        pl.String: DataType.STRING,
        pl.Utf8: DataType.STRING,
        pl.Date: DataType.DATE,
        pl.Datetime: DataType.DATETIME,
        pl.Time: DataType.TIME,
        pl.Duration: DataType.DURATION,
    }

    def infer(
        self,
        column: str,
        lf: pl.LazyFrame,
        physical_type: pl.DataType,
    ) -> DataType | None:
        return self._TYPE_MAPPING.get(type(physical_type), DataType.UNKNOWN)


@register_type_inferrer("pattern")
class PatternBasedTypeInferrer(TypeInferrer):
    """Infers semantic type based on detected patterns."""

    name = "pattern"
    priority = 50  # Higher priority than physical

    def __init__(self, min_match_ratio: float = 0.9):
        self.min_match_ratio = min_match_ratio
        self._patterns = BUILTIN_PATTERNS

    def infer(
        self,
        column: str,
        lf: pl.LazyFrame,
        physical_type: pl.DataType,
    ) -> DataType | None:
        # Only applicable to string types
        if type(physical_type) not in {pl.String, pl.Utf8}:
            return None

        # Sample data
        sample = (
            lf.select(pl.col(column))
            .filter(pl.col(column).is_not_null())
            .head(1000)
            .collect()
        )

        if len(sample) == 0:
            return None

        values = sample[column].to_list()
        total = len(values)

        # Check patterns in priority order
        sorted_patterns = sorted(self._patterns, key=lambda p: -p.priority)

        for pattern_def in sorted_patterns:
            compiled = pattern_def.compile()
            matches = sum(1 for v in values if v and compiled.match(str(v)))
            match_ratio = matches / total if total > 0 else 0.0

            if match_ratio >= self.min_match_ratio:
                return pattern_def.data_type

        return None


@register_type_inferrer("cardinality")
class CardinalityTypeInferrer(TypeInferrer):
    """Infers categorical/identifier types based on cardinality."""

    name = "cardinality"
    priority = 30

    def __init__(
        self,
        categorical_threshold: float = 0.05,
        identifier_threshold: float = 0.95,
    ):
        self.categorical_threshold = categorical_threshold
        self.identifier_threshold = identifier_threshold

    def infer(
        self,
        column: str,
        lf: pl.LazyFrame,
        physical_type: pl.DataType,
    ) -> DataType | None:
        stats = lf.select(
            pl.len().alias("total"),
            pl.col(column).n_unique().alias("unique"),
            pl.col(column).null_count().alias("nulls"),
        ).collect()

        total = stats["total"][0]
        unique = stats["unique"][0]
        nulls = stats["nulls"][0]

        non_null = total - nulls
        if non_null == 0:
            return None

        unique_ratio = unique / non_null

        # Low cardinality = categorical
        if unique_ratio <= self.categorical_threshold:
            return DataType.CATEGORICAL

        # Very high cardinality (and unique) = identifier
        if unique_ratio >= self.identifier_threshold and unique == non_null:
            return DataType.IDENTIFIER

        return None


# =============================================================================
# Column Profiler
# =============================================================================


class ColumnProfiler:
    """Profiles individual columns with configurable analyzers.

    This class orchestrates multiple analyzers to build a complete
    column profile. It uses a strategy pattern allowing easy extension.

    Example:
        profiler = ColumnProfiler()
        profile = profiler.profile_column("email", lazy_frame, pl.String)

        # With custom analyzers
        profiler = ColumnProfiler(
            analyzers=[BasicStatsAnalyzer(), CustomAnalyzer()]
        )
    """

    def __init__(
        self,
        analyzers: Sequence[ColumnAnalyzer] | None = None,
        type_inferrers: Sequence[TypeInferrer] | None = None,
        config: ProfilerConfig | None = None,
    ):
        """Initialize column profiler.

        Args:
            analyzers: List of analyzers to use. If None, uses defaults.
            type_inferrers: List of type inferrers. If None, uses defaults.
            config: Profiler configuration.
        """
        self.config = config or ProfilerConfig()

        # Default analyzers
        if analyzers:
            self.analyzers = list(analyzers)
        else:
            self.analyzers = [
                BasicStatsAnalyzer(),
                NumericAnalyzer(),
                StringAnalyzer(),
                DatetimeAnalyzer(),
                ValueFrequencyAnalyzer(),
            ]
            # Only add PatternAnalyzer if include_patterns is True
            if self.config.include_patterns:
                self.analyzers.append(PatternAnalyzer())

        # Default type inferrers (sorted by priority, highest first)
        if type_inferrers:
            self.type_inferrers = list(type_inferrers)
        else:
            self.type_inferrers = [
                CardinalityTypeInferrer(),
                PhysicalTypeInferrer(),
            ]
            # Only add PatternBasedTypeInferrer if include_patterns is True
            if self.config.include_patterns:
                self.type_inferrers.append(PatternBasedTypeInferrer())
        self.type_inferrers.sort(key=lambda x: -x.priority)

    def add_analyzer(self, analyzer: ColumnAnalyzer) -> None:
        """Add a custom analyzer."""
        self.analyzers.append(analyzer)

    def add_type_inferrer(self, inferrer: TypeInferrer) -> None:
        """Add a custom type inferrer."""
        self.type_inferrers.append(inferrer)
        self.type_inferrers.sort(key=lambda x: -x.priority)

    def profile_column(
        self,
        column: str,
        lf: pl.LazyFrame,
        dtype: pl.DataType,
    ) -> ColumnProfile:
        """Profile a single column.

        Args:
            column: Column name
            lf: LazyFrame containing the data
            dtype: Polars data type of the column

        Returns:
            Complete column profile
        """
        start_time = time.perf_counter()

        # Run applicable analyzers
        metrics: dict[str, Any] = {
            "name": column,
            "physical_type": str(dtype),
        }

        for analyzer in self.analyzers:
            if analyzer.is_applicable(dtype):
                try:
                    result = analyzer.analyze(column, lf, self.config)
                    metrics.update(result)
                except Exception:
                    # Skip failed analyzers
                    pass

        # Infer semantic type
        inferred_type = self._infer_type(column, lf, dtype)
        metrics["inferred_type"] = inferred_type

        # Generate suggested validators
        metrics["suggested_validators"] = self._suggest_validators(metrics)

        # Calculate profiling duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics["profile_duration_ms"] = duration_ms
        metrics["profiled_at"] = datetime.now()

        return ColumnProfile(**metrics)

    def _infer_type(
        self,
        column: str,
        lf: pl.LazyFrame,
        dtype: pl.DataType,
    ) -> DataType:
        """Infer semantic type using registered inferrers."""
        for inferrer in self.type_inferrers:
            try:
                result = inferrer.infer(column, lf, dtype)
                if result is not None:
                    return result
            except Exception:
                continue
        return DataType.UNKNOWN

    def _suggest_validators(self, metrics: dict[str, Any]) -> tuple[str, ...]:
        """Suggest validators based on profile metrics."""
        suggestions: list[str] = []

        # Null handling
        null_ratio = metrics.get("null_ratio", 0)
        if null_ratio > 0:
            suggestions.append(f"CompletenessRatioValidator(min_ratio={1 - null_ratio:.2f})")
        else:
            suggestions.append("NotNullValidator()")

        # Uniqueness
        if metrics.get("is_unique"):
            suggestions.append("UniqueValidator()")
        elif metrics.get("unique_ratio", 0) > 0.9:
            suggestions.append(
                f"UniqueRatioValidator(min_ratio={metrics['unique_ratio']:.2f})"
            )

        # Pattern-based suggestions
        inferred_type = metrics.get("inferred_type", DataType.UNKNOWN)
        type_validators = {
            DataType.EMAIL: "EmailValidator()",
            DataType.URL: "UrlValidator()",
            DataType.UUID: "UuidValidator()",
            DataType.IP_ADDRESS: "IpAddressValidator()",
            DataType.PHONE: "PhoneValidator()",
            DataType.KOREAN_RRN: "KoreanRRNValidator()",
            DataType.KOREAN_PHONE: "KoreanPhoneValidator()",
            DataType.KOREAN_BUSINESS_NUMBER: "KoreanBusinessNumberValidator()",
        }
        if inferred_type in type_validators:
            suggestions.append(type_validators[inferred_type])

        # Distribution-based suggestions (for numeric)
        dist = metrics.get("distribution")
        if dist and isinstance(dist, DistributionStats):
            if dist.min is not None and dist.max is not None:
                suggestions.append(
                    f"RangeValidator(min_value={dist.min}, max_value={dist.max})"
                )

        # String length suggestions
        if metrics.get("min_length") is not None:
            suggestions.append(
                f"LengthValidator(min_length={metrics['min_length']}, "
                f"max_length={metrics['max_length']})"
            )

        return tuple(suggestions)
