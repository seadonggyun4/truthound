"""Utility functions and shared components for validators.

This module addresses:
- #21: Complete documentation with comprehensive docstrings
- #23: Strict type hints (reduced Any usage)
- #24: Centralized severity calculation (eliminates code duplication)

Design Principles:
    1. Strong typing: Explicit types for all parameters and returns
    2. Pure functions: No side effects, deterministic outputs
    3. Documentation: Every function has complete docstrings
    4. Reusability: Common patterns extracted for DRY code
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Final,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    overload,
)

import polars as pl

from truthound.types import Severity


# =============================================================================
# Type Definitions (#23: Explicit Type Aliases)
# =============================================================================

# Column name type
ColumnName: TypeAlias = str

# Column list type
ColumnList: TypeAlias = Sequence[ColumnName]

# Polars data types for filtering
PolarsDataType: TypeAlias = type[pl.DataType]

# Threshold tuple type (critical, high, medium)
SeverityThresholds: TypeAlias = tuple[float, float, float]

# Sample value type (mixed types possible)
SampleValue: TypeAlias = str | int | float | bool | None

# Sample list type
SampleList: TypeAlias = list[SampleValue]

# Validation ratio (0.0 to 1.0)
Ratio: TypeAlias = float

# Count type (non-negative integer)
Count: TypeAlias = int


# =============================================================================
# Constants (#21: Well-documented constants)
# =============================================================================

# Default severity thresholds (critical, high, medium)
# - > 50% issues: CRITICAL
# - > 20% issues: HIGH
# - > 5% issues: MEDIUM
# - <= 5% issues: LOW
DEFAULT_SEVERITY_THRESHOLDS: Final[SeverityThresholds] = (0.5, 0.2, 0.05)

# Strict thresholds for critical validations
# - > 10% issues: CRITICAL
# - > 5% issues: HIGH
# - > 1% issues: MEDIUM
STRICT_SEVERITY_THRESHOLDS: Final[SeverityThresholds] = (0.1, 0.05, 0.01)

# Lenient thresholds for informational validations
# - > 80% issues: CRITICAL
# - > 50% issues: HIGH
# - > 20% issues: MEDIUM
LENIENT_SEVERITY_THRESHOLDS: Final[SeverityThresholds] = (0.8, 0.5, 0.2)


# =============================================================================
# Severity Calculation (#24: Centralized, Reusable Logic)
# =============================================================================


class SeverityCalculator:
    """Centralized severity calculation logic.

    This class eliminates code duplication across validators by providing
    a single, well-tested implementation of severity calculation.

    Attributes:
        thresholds: Tuple of (critical, high, medium) thresholds.
        override: Optional severity to always return.

    Example:
        >>> calc = SeverityCalculator()
        >>> calc.from_ratio(0.6)  # > 50%
        Severity.CRITICAL

        >>> calc = SeverityCalculator(thresholds=STRICT_SEVERITY_THRESHOLDS)
        >>> calc.from_ratio(0.08)  # > 5% with strict thresholds
        Severity.HIGH

        >>> calc = SeverityCalculator(override=Severity.LOW)
        >>> calc.from_ratio(0.99)  # Override always returns LOW
        Severity.LOW
    """

    __slots__ = ("thresholds", "override")

    def __init__(
        self,
        thresholds: SeverityThresholds = DEFAULT_SEVERITY_THRESHOLDS,
        override: Severity | None = None,
    ) -> None:
        """Initialize the severity calculator.

        Args:
            thresholds: Tuple of (critical_threshold, high_threshold, medium_threshold).
                        Values should be in descending order (critical > high > medium).
            override: If set, always return this severity regardless of ratio.

        Raises:
            ValueError: If thresholds are not in valid range or order.
        """
        self._validate_thresholds(thresholds)
        self.thresholds = thresholds
        self.override = override

    @staticmethod
    def _validate_thresholds(thresholds: SeverityThresholds) -> None:
        """Validate that thresholds are in correct order and range.

        Args:
            thresholds: The thresholds to validate.

        Raises:
            ValueError: If thresholds are invalid.
        """
        critical, high, medium = thresholds

        # Check range
        for name, value in [("critical", critical), ("high", high), ("medium", medium)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} threshold must be in [0.0, 1.0], got {value}"
                )

        # Check order
        if not (critical >= high >= medium):
            raise ValueError(
                f"Thresholds must be in descending order: "
                f"critical({critical}) >= high({high}) >= medium({medium})"
            )

    def from_ratio(self, ratio: Ratio) -> Severity:
        """Calculate severity from a ratio of issues.

        Args:
            ratio: The ratio of problematic values (0.0 to 1.0).
                   For example, 0.15 means 15% of values have issues.

        Returns:
            Severity level based on thresholds.

        Example:
            >>> calc = SeverityCalculator()
            >>> calc.from_ratio(0.0)
            Severity.LOW
            >>> calc.from_ratio(0.1)
            Severity.MEDIUM
            >>> calc.from_ratio(0.3)
            Severity.HIGH
            >>> calc.from_ratio(0.6)
            Severity.CRITICAL
        """
        if self.override is not None:
            return self.override

        critical_th, high_th, medium_th = self.thresholds

        if ratio > critical_th:
            return Severity.CRITICAL
        elif ratio > high_th:
            return Severity.HIGH
        elif ratio > medium_th:
            return Severity.MEDIUM
        return Severity.LOW

    def from_counts(self, failure_count: Count, total_count: Count) -> Severity:
        """Calculate severity from failure and total counts.

        Args:
            failure_count: Number of failing/problematic values.
            total_count: Total number of values checked.

        Returns:
            Severity level based on failure ratio.

        Raises:
            ValueError: If counts are negative or failure > total.

        Example:
            >>> calc = SeverityCalculator()
            >>> calc.from_counts(10, 100)  # 10%
            Severity.MEDIUM
            >>> calc.from_counts(0, 100)  # 0%
            Severity.LOW
        """
        if failure_count < 0:
            raise ValueError(f"failure_count cannot be negative: {failure_count}")
        if total_count < 0:
            raise ValueError(f"total_count cannot be negative: {total_count}")
        if failure_count > total_count:
            raise ValueError(
                f"failure_count ({failure_count}) cannot exceed "
                f"total_count ({total_count})"
            )

        if total_count == 0:
            return Severity.LOW

        ratio = failure_count / total_count
        return self.from_ratio(ratio)

    def with_override(self, override: Severity) -> "SeverityCalculator":
        """Create a new calculator with the specified override.

        Args:
            override: Severity to always return.

        Returns:
            New SeverityCalculator instance with override set.
        """
        return SeverityCalculator(thresholds=self.thresholds, override=override)

    def with_thresholds(self, thresholds: SeverityThresholds) -> "SeverityCalculator":
        """Create a new calculator with different thresholds.

        Args:
            thresholds: New thresholds to use.

        Returns:
            New SeverityCalculator instance with new thresholds.
        """
        return SeverityCalculator(thresholds=thresholds, override=self.override)


# Global default calculator instance
_default_calculator = SeverityCalculator()


def calculate_severity(
    ratio: Ratio,
    thresholds: SeverityThresholds = DEFAULT_SEVERITY_THRESHOLDS,
    override: Severity | None = None,
) -> Severity:
    """Calculate severity from a ratio (convenience function).

    This is a module-level function for simple use cases.
    For repeated calculations, use SeverityCalculator class.

    Args:
        ratio: The ratio of problematic values (0.0 to 1.0).
        thresholds: Optional custom thresholds.
        override: Optional severity override.

    Returns:
        Calculated severity level.

    Example:
        >>> calculate_severity(0.15)
        Severity.MEDIUM
        >>> calculate_severity(0.15, thresholds=STRICT_SEVERITY_THRESHOLDS)
        Severity.CRITICAL
    """
    if override is not None:
        return override

    if thresholds == DEFAULT_SEVERITY_THRESHOLDS:
        return _default_calculator.from_ratio(ratio)

    return SeverityCalculator(thresholds=thresholds).from_ratio(ratio)


def calculate_severity_from_counts(
    failure_count: Count,
    total_count: Count,
    thresholds: SeverityThresholds = DEFAULT_SEVERITY_THRESHOLDS,
    override: Severity | None = None,
) -> Severity:
    """Calculate severity from counts (convenience function).

    Args:
        failure_count: Number of failing values.
        total_count: Total number of values.
        thresholds: Optional custom thresholds.
        override: Optional severity override.

    Returns:
        Calculated severity level.
    """
    if override is not None:
        return override

    if total_count == 0:
        return Severity.LOW

    ratio = failure_count / total_count
    return calculate_severity(ratio, thresholds, override)


# =============================================================================
# Mostly Threshold Logic (#24: Centralized)
# =============================================================================


@dataclass(frozen=True, slots=True)
class MostlyResult:
    """Result of a mostly threshold check.

    Attributes:
        passes: Whether the validation passes the mostly threshold.
        pass_ratio: The actual pass ratio (1.0 - failure_ratio).
        required_ratio: The required pass ratio (mostly threshold).
        failure_count: Number of failing values.
        total_count: Total number of values.
    """

    passes: bool
    pass_ratio: float
    required_ratio: float
    failure_count: int
    total_count: int

    @property
    def failure_ratio(self) -> float:
        """Get the failure ratio."""
        return 1.0 - self.pass_ratio

    @property
    def margin(self) -> float:
        """Get margin above/below threshold (positive = passing)."""
        return self.pass_ratio - self.required_ratio


def check_mostly_threshold(
    failure_count: Count,
    total_count: Count,
    mostly: float | None,
) -> MostlyResult:
    """Check if validation passes the mostly threshold.

    The 'mostly' parameter allows a certain percentage of values to fail
    while still considering the validation as passed.

    Args:
        failure_count: Number of failing values.
        total_count: Total number of values.
        mostly: Required pass ratio (0.0 to 1.0), or None for no threshold.

    Returns:
        MostlyResult with detailed information.

    Example:
        >>> result = check_mostly_threshold(5, 100, 0.95)
        >>> result.passes  # 95% pass rate required, 95% achieved
        True
        >>> result.pass_ratio
        0.95

        >>> result = check_mostly_threshold(10, 100, 0.95)
        >>> result.passes  # 95% required, only 90% achieved
        False
    """
    if mostly is None:
        # No threshold means it never passes automatically
        pass_ratio = (total_count - failure_count) / max(total_count, 1)
        return MostlyResult(
            passes=False,
            pass_ratio=pass_ratio,
            required_ratio=1.0,
            failure_count=failure_count,
            total_count=total_count,
        )

    if total_count == 0:
        return MostlyResult(
            passes=True,
            pass_ratio=1.0,
            required_ratio=mostly,
            failure_count=0,
            total_count=0,
        )

    pass_ratio = 1.0 - (failure_count / total_count)
    passes = pass_ratio >= mostly

    return MostlyResult(
        passes=passes,
        pass_ratio=pass_ratio,
        required_ratio=mostly,
        failure_count=failure_count,
        total_count=total_count,
    )


# =============================================================================
# Column Type Utilities (#23: Type-safe column filtering)
# =============================================================================


# Pre-defined type sets with explicit typing
NUMERIC_DTYPES: Final[frozenset[PolarsDataType]] = frozenset({
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
})

STRING_DTYPES: Final[frozenset[PolarsDataType]] = frozenset({
    pl.String,
    pl.Utf8,
})

DATETIME_DTYPES: Final[frozenset[PolarsDataType]] = frozenset({
    pl.Date,
    pl.Datetime,
    pl.Time,
    pl.Duration,
})

FLOAT_DTYPES: Final[frozenset[PolarsDataType]] = frozenset({
    pl.Float32,
    pl.Float64,
})

INTEGER_DTYPES: Final[frozenset[PolarsDataType]] = frozenset({
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
})


def get_columns_by_dtype(
    lf: pl.LazyFrame,
    dtypes: frozenset[PolarsDataType] | set[PolarsDataType],
) -> list[ColumnName]:
    """Get column names matching the specified data types.

    Args:
        lf: LazyFrame to inspect.
        dtypes: Set of Polars data types to match.

    Returns:
        List of column names with matching types.

    Example:
        >>> lf = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).lazy()
        >>> get_columns_by_dtype(lf, NUMERIC_DTYPES)
        ['a']
        >>> get_columns_by_dtype(lf, STRING_DTYPES)
        ['b']
    """
    schema = lf.collect_schema()
    return [
        name for name, dtype in schema.items()
        if type(dtype) in dtypes
    ]


def filter_existing_columns(
    lf: pl.LazyFrame,
    columns: ColumnList,
) -> list[ColumnName]:
    """Filter column list to only include existing columns.

    Args:
        lf: LazyFrame to check against.
        columns: List of column names to filter.

    Returns:
        List of columns that exist in the LazyFrame.

    Example:
        >>> lf = pl.DataFrame({"a": [1], "b": [2]}).lazy()
        >>> filter_existing_columns(lf, ["a", "c", "b"])
        ['a', 'b']
    """
    schema = lf.collect_schema()
    existing = set(schema.names())
    return [c for c in columns if c in existing]


def validate_columns_exist(
    lf: pl.LazyFrame,
    columns: ColumnList,
) -> tuple[bool, list[ColumnName]]:
    """Validate that all specified columns exist.

    Args:
        lf: LazyFrame to check.
        columns: Required column names.

    Returns:
        Tuple of (all_exist, missing_columns).

    Example:
        >>> lf = pl.DataFrame({"a": [1], "b": [2]}).lazy()
        >>> validate_columns_exist(lf, ["a", "b"])
        (True, [])
        >>> validate_columns_exist(lf, ["a", "c"])
        (False, ['c'])
    """
    schema = lf.collect_schema()
    existing = set(schema.names())
    missing = [c for c in columns if c not in existing]
    return len(missing) == 0, missing


# =============================================================================
# Sample Value Utilities (#23: Type-safe sampling)
# =============================================================================


def safe_to_sample_list(
    values: Sequence[SampleValue],
    max_length: int = 5,
    stringify: bool = True,
) -> SampleList:
    """Safely convert values to a sample list.

    Args:
        values: Values to convert.
        max_length: Maximum number of samples.
        stringify: If True, convert all values to strings.

    Returns:
        List of sample values.
    """
    samples = list(values)[:max_length]
    if stringify:
        return [str(v) if v is not None else None for v in samples]
    return samples


def truncate_string(
    value: str,
    max_length: int = 50,
    suffix: str = "...",
) -> str:
    """Truncate a string to maximum length.

    Args:
        value: String to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated string.

    Example:
        >>> truncate_string("Hello, World!", 10)
        'Hello, ...'
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - len(suffix)] + suffix


# =============================================================================
# Issue Message Formatting (#21: Consistent message format)
# =============================================================================


def format_issue_count(
    count: Count,
    total: Count | None = None,
    noun: str = "value",
    noun_plural: str | None = None,
) -> str:
    """Format an issue count as a human-readable string.

    Args:
        count: Number of issues.
        total: Optional total count for percentage.
        noun: Singular noun for the count.
        noun_plural: Plural noun (default: noun + "s").

    Returns:
        Formatted string.

    Example:
        >>> format_issue_count(5)
        '5 values'
        >>> format_issue_count(1)
        '1 value'
        >>> format_issue_count(5, 100)
        '5 values (5.0%)'
        >>> format_issue_count(3, noun="row")
        '3 rows'
    """
    plural = noun_plural or f"{noun}s"
    word = noun if count == 1 else plural

    if total is not None and total > 0:
        pct = (count / total) * 100
        return f"{count:,} {word} ({pct:.1f}%)"
    return f"{count:,} {word}"


def format_range(
    min_val: float | int | None,
    max_val: float | int | None,
    inclusive: bool = True,
) -> str:
    """Format a numeric range as a string.

    Args:
        min_val: Minimum value (None for unbounded).
        max_val: Maximum value (None for unbounded).
        inclusive: Whether bounds are inclusive.

    Returns:
        Formatted range string.

    Example:
        >>> format_range(0, 100)
        '[0, 100]'
        >>> format_range(0, None)
        '>= 0'
        >>> format_range(None, 100)
        '<= 100'
        >>> format_range(0, 100, inclusive=False)
        '(0, 100)'
    """
    if min_val is not None and max_val is not None:
        brackets = ("(", ")") if not inclusive else ("[", "]")
        return f"{brackets[0]}{min_val}, {max_val}{brackets[1]}"
    elif min_val is not None:
        op = ">=" if inclusive else ">"
        return f"{op} {min_val}"
    elif max_val is not None:
        op = "<=" if inclusive else "<"
        return f"{op} {max_val}"
    return "any value"


# =============================================================================
# Validation Helpers
# =============================================================================


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(
    value: float,
    min_val: float,
    max_val: float,
) -> float:
    """Clamp a value to a range.

    Args:
        value: Value to clamp.
        min_val: Minimum value.
        max_val: Maximum value.

    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type aliases
    "ColumnName",
    "ColumnList",
    "PolarsDataType",
    "SeverityThresholds",
    "SampleValue",
    "SampleList",
    "Ratio",
    "Count",
    # Constants
    "DEFAULT_SEVERITY_THRESHOLDS",
    "STRICT_SEVERITY_THRESHOLDS",
    "LENIENT_SEVERITY_THRESHOLDS",
    # Severity calculation
    "SeverityCalculator",
    "calculate_severity",
    "calculate_severity_from_counts",
    # Mostly threshold
    "MostlyResult",
    "check_mostly_threshold",
    # Column utilities
    "NUMERIC_DTYPES",
    "STRING_DTYPES",
    "DATETIME_DTYPES",
    "FLOAT_DTYPES",
    "INTEGER_DTYPES",
    "get_columns_by_dtype",
    "filter_existing_columns",
    "validate_columns_exist",
    # Sample utilities
    "safe_to_sample_list",
    "truncate_string",
    # Formatting
    "format_issue_count",
    "format_range",
    # Helpers
    "safe_divide",
    "clamp",
]
