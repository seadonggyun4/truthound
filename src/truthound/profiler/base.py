"""Base classes and core data structures for data profiling.

This module provides the foundational abstractions for the profiling system:
- ProfileResult: Immutable data structures for profile results
- Profiler: Abstract base class for all profilers
- ProfilerRegistry: Dynamic registration of custom profilers
"""

from __future__ import annotations

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
    TypeVar,
    runtime_checkable,
)

import polars as pl

if TYPE_CHECKING:
    from truthound.validators.base import Validator


# =============================================================================
# Enums
# =============================================================================


class DataType(str, Enum):
    """Inferred logical data types for profiling."""

    # Basic types
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"

    # Semantic types (detected from patterns)
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UUID = "uuid"
    IP_ADDRESS = "ip_address"
    JSON = "json"

    # Identifiers
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"

    # Numeric subtypes
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

    # Korean specific
    KOREAN_RRN = "korean_rrn"
    KOREAN_PHONE = "korean_phone"
    KOREAN_BUSINESS_NUMBER = "korean_business_number"

    # Unknown
    UNKNOWN = "unknown"


class Strictness(str, Enum):
    """Strictness level for rule generation."""

    LOOSE = "loose"      # Permissive rules, fewer false positives
    MEDIUM = "medium"    # Balanced approach
    STRICT = "strict"    # Strict rules, comprehensive validation


class ProfileCategory(str, Enum):
    """Categories of profile metrics."""

    SCHEMA = "schema"
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    DISTRIBUTION = "distribution"
    FORMAT = "format"
    PATTERN = "pattern"
    TEMPORAL = "temporal"
    CORRELATION = "correlation"


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass(frozen=True)
class PatternMatch:
    """Represents a detected pattern in data."""

    pattern: str
    regex: str
    match_ratio: float
    sample_matches: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "regex": self.regex,
            "match_ratio": self.match_ratio,
            "sample_matches": list(self.sample_matches),
        }


@dataclass(frozen=True)
class DistributionStats:
    """Statistical distribution metrics."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q1: float | None = None  # 25th percentile
    q3: float | None = None  # 75th percentile
    skewness: float | None = None
    kurtosis: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class ValueFrequency:
    """Frequency of a specific value."""

    value: Any
    count: int
    ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "count": self.count, "ratio": self.ratio}


@dataclass(frozen=True)
class ColumnProfile:
    """Complete profile for a single column.

    This is the central data structure containing all profiling metrics
    for one column. It's immutable to ensure thread-safety and consistency.
    """

    # Basic info
    name: str
    physical_type: str  # Polars dtype as string
    inferred_type: DataType = DataType.UNKNOWN

    # Completeness
    row_count: int = 0
    null_count: int = 0
    null_ratio: float = 0.0
    empty_string_count: int = 0

    # Uniqueness
    distinct_count: int = 0
    unique_ratio: float = 0.0
    is_unique: bool = False
    is_constant: bool = False

    # Distribution (for numeric)
    distribution: DistributionStats | None = None

    # Value analysis
    top_values: tuple[ValueFrequency, ...] = field(default_factory=tuple)
    bottom_values: tuple[ValueFrequency, ...] = field(default_factory=tuple)

    # String analysis
    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None

    # Pattern analysis
    detected_patterns: tuple[PatternMatch, ...] = field(default_factory=tuple)

    # Temporal analysis (for datetime)
    min_date: datetime | None = None
    max_date: datetime | None = None
    date_gaps: int = 0

    # Suggested validators
    suggested_validators: tuple[str, ...] = field(default_factory=tuple)

    # Metadata
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "physical_type": self.physical_type,
            "inferred_type": self.inferred_type.value,
            "row_count": self.row_count,
            "null_count": self.null_count,
            "null_ratio": self.null_ratio,
            "empty_string_count": self.empty_string_count,
            "distinct_count": self.distinct_count,
            "unique_ratio": self.unique_ratio,
            "is_unique": self.is_unique,
            "is_constant": self.is_constant,
        }

        if self.distribution:
            result["distribution"] = self.distribution.to_dict()

        if self.top_values:
            result["top_values"] = [v.to_dict() for v in self.top_values]

        if self.bottom_values:
            result["bottom_values"] = [v.to_dict() for v in self.bottom_values]

        if self.min_length is not None:
            result["min_length"] = self.min_length
            result["max_length"] = self.max_length
            result["avg_length"] = self.avg_length

        if self.detected_patterns:
            result["detected_patterns"] = [p.to_dict() for p in self.detected_patterns]

        if self.min_date:
            result["min_date"] = self.min_date.isoformat()
            result["max_date"] = self.max_date.isoformat() if self.max_date else None
            result["date_gaps"] = self.date_gaps

        if self.suggested_validators:
            result["suggested_validators"] = list(self.suggested_validators)

        result["profiled_at"] = self.profiled_at.isoformat()
        result["profile_duration_ms"] = self.profile_duration_ms

        return result


@dataclass(frozen=True)
class TableProfile:
    """Complete profile for a table/dataset.

    Contains both table-level metrics and column profiles.
    """

    # Table info
    name: str = ""
    row_count: int = 0
    column_count: int = 0

    # Memory estimation
    estimated_memory_bytes: int = 0

    # Column profiles
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)

    # Table-level metrics
    duplicate_row_count: int = 0
    duplicate_row_ratio: float = 0.0

    # Correlation matrix (column pairs with high correlation)
    correlations: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)

    # Metadata
    source: str = ""
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0

    def __iter__(self) -> Iterator[ColumnProfile]:
        """Iterate over column profiles."""
        return iter(self.columns)

    def __getitem__(self, key: str | int) -> ColumnProfile:
        """Get column profile by name or index."""
        if isinstance(key, int):
            return self.columns[key]
        for col in self.columns:
            if col.name == key:
                return col
        raise KeyError(f"Column '{key}' not found in profile")

    def get(self, column_name: str) -> ColumnProfile | None:
        """Get column profile by name, returns None if not found."""
        for col in self.columns:
            if col.name == column_name:
                return col
        return None

    @property
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_ratio": self.duplicate_row_ratio,
            "columns": [col.to_dict() for col in self.columns],
            "correlations": [
                {"column1": c1, "column2": c2, "correlation": corr}
                for c1, c2, corr in self.correlations
            ],
            "source": self.source,
            "profiled_at": self.profiled_at.isoformat(),
            "profile_duration_ms": self.profile_duration_ms,
        }


# =============================================================================
# Profiler Protocols and Base Classes
# =============================================================================


@runtime_checkable
class ProfilerProtocol(Protocol):
    """Protocol defining the profiler interface.

    Use this for type hints when you need duck typing.
    """

    def profile(self, data: pl.LazyFrame) -> TableProfile:
        """Profile the given data."""
        ...


class Profiler(ABC):
    """Abstract base class for all profilers.

    Subclass this to create custom profilers. The profiler system
    uses a pipeline of analysis steps that can be customized.

    Example:
        class CustomProfiler(Profiler):
            def profile(self, data: pl.LazyFrame) -> TableProfile:
                # Custom profiling logic
                ...
    """

    name: str = "base"
    description: str = "Base profiler"

    def __init__(
        self,
        *,
        sample_size: int | None = None,
        include_patterns: bool = True,
        include_correlations: bool = False,
        top_n_values: int = 10,
        pattern_sample_size: int = 1000,
    ):
        """Initialize profiler with configuration.

        Args:
            sample_size: If set, profile only a sample of rows
            include_patterns: Whether to detect patterns in string columns
            include_correlations: Whether to compute column correlations
            top_n_values: Number of top/bottom values to include
            pattern_sample_size: Sample size for pattern detection
        """
        self.sample_size = sample_size
        self.include_patterns = include_patterns
        self.include_correlations = include_correlations
        self.top_n_values = top_n_values
        self.pattern_sample_size = pattern_sample_size

    @abstractmethod
    def profile(self, data: pl.LazyFrame) -> TableProfile:
        """Profile the given data.

        Args:
            data: LazyFrame to profile

        Returns:
            Complete table profile
        """
        pass

    def _maybe_sample(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply sampling if configured."""
        if self.sample_size is not None:
            return lf.head(self.sample_size)
        return lf


# =============================================================================
# Type Inference Protocol
# =============================================================================


class TypeInferrer(ABC):
    """Abstract base class for type inference.

    Type inferrers analyze column data to determine semantic types
    beyond the physical storage type.
    """

    name: str = "base"
    priority: int = 0  # Higher priority inferrers run first

    @abstractmethod
    def infer(
        self,
        column: str,
        lf: pl.LazyFrame,
        physical_type: pl.DataType,
    ) -> DataType | None:
        """Infer the semantic type of a column.

        Args:
            column: Column name
            lf: LazyFrame containing the data
            physical_type: Polars physical type

        Returns:
            Inferred DataType or None if cannot infer
        """
        pass


# =============================================================================
# Registry for Extensibility
# =============================================================================


class ProfilerRegistry:
    """Registry for dynamic profiler registration.

    Allows users to register custom profilers that will be
    available throughout the system.

    Example:
        registry = ProfilerRegistry()
        registry.register(CustomProfiler)

        # Later
        profiler = registry.get("custom")
    """

    def __init__(self) -> None:
        self._profilers: dict[str, type[Profiler]] = {}
        self._type_inferrers: dict[str, type[TypeInferrer]] = {}

    def register_profiler(
        self,
        profiler_class: type[Profiler],
        name: str | None = None,
    ) -> None:
        """Register a profiler class.

        Args:
            profiler_class: Profiler class to register
            name: Optional name override (uses class.name if not provided)
        """
        key = name or profiler_class.name
        self._profilers[key] = profiler_class

    def register_type_inferrer(
        self,
        inferrer_class: type[TypeInferrer],
        name: str | None = None,
    ) -> None:
        """Register a type inferrer class.

        Args:
            inferrer_class: TypeInferrer class to register
            name: Optional name override
        """
        key = name or inferrer_class.name
        self._type_inferrers[key] = inferrer_class

    def get_profiler(self, name: str) -> type[Profiler]:
        """Get a registered profiler by name.

        Raises:
            KeyError: If profiler not found
        """
        if name not in self._profilers:
            raise KeyError(
                f"Profiler '{name}' not found. "
                f"Available: {list(self._profilers.keys())}"
            )
        return self._profilers[name]

    def get_type_inferrer(self, name: str) -> type[TypeInferrer]:
        """Get a registered type inferrer by name."""
        if name not in self._type_inferrers:
            raise KeyError(
                f"Type inferrer '{name}' not found. "
                f"Available: {list(self._type_inferrers.keys())}"
            )
        return self._type_inferrers[name]

    def list_profilers(self) -> list[str]:
        """List all registered profiler names."""
        return list(self._profilers.keys())

    def list_type_inferrers(self) -> list[str]:
        """List all registered type inferrer names."""
        return list(self._type_inferrers.keys())


# Global registry instance
profiler_registry = ProfilerRegistry()


def register_profiler(
    name: str | None = None,
) -> Callable[[type[Profiler]], type[Profiler]]:
    """Decorator to register a profiler class.

    Example:
        @register_profiler("custom")
        class CustomProfiler(Profiler):
            ...
    """
    def decorator(cls: type[Profiler]) -> type[Profiler]:
        profiler_registry.register_profiler(cls, name)
        return cls
    return decorator


def register_type_inferrer(
    name: str | None = None,
) -> Callable[[type[TypeInferrer]], type[TypeInferrer]]:
    """Decorator to register a type inferrer class."""
    def decorator(cls: type[TypeInferrer]) -> type[TypeInferrer]:
        profiler_registry.register_type_inferrer(cls, name)
        return cls
    return decorator


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ProfilerConfig:
    """Configuration for profiling operations.

    This provides a centralized way to configure profiling behavior.
    """

    # Sampling
    sample_size: int | None = None
    random_seed: int = 42

    # Analysis options
    include_patterns: bool = True
    include_correlations: bool = False
    include_distributions: bool = True

    # Performance tuning
    top_n_values: int = 10
    pattern_sample_size: int = 1000
    correlation_threshold: float = 0.7

    # Pattern detection
    min_pattern_match_ratio: float = 0.8

    # Parallel processing
    n_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
