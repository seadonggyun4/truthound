"""Base classes for validators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import polars as pl

from truthound.types import Severity


@dataclass
class ValidationIssue:
    """Represents a single data quality issue found during validation."""

    column: str
    issue_type: str
    count: int
    severity: Severity
    details: str | None = None
    expected: Any | None = None
    actual: Any | None = None
    sample_values: list[Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "column": self.column,
            "issue_type": self.issue_type,
            "count": self.count,
            "severity": self.severity.value,
            "details": self.details,
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.actual is not None:
            result["actual"] = self.actual
        if self.sample_values is not None:
            result["sample_values"] = self.sample_values
        return result


@dataclass
class ValidatorConfig:
    """Common configuration for validators."""

    columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    severity_override: Severity | None = None
    sample_size: int = 5
    # Mostly parameter: fraction of rows that must pass (0.0 to 1.0)
    # e.g., mostly=0.95 means 95% of rows must pass for the validation to succeed
    mostly: float | None = None


# Type sets for mixins
NUMERIC_TYPES: set[type[pl.DataType]] = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

STRING_TYPES: set[type[pl.DataType]] = {pl.String, pl.Utf8}

DATETIME_TYPES: set[type[pl.DataType]] = {pl.Date, pl.Datetime, pl.Time, pl.Duration}


class Validator(ABC):
    """Abstract base class for all validators."""

    name: str = "base"
    category: str = "general"

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        self.config = config or ValidatorConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation on the given LazyFrame."""
        pass

    def _get_target_columns(
        self,
        lf: pl.LazyFrame,
        dtype_filter: set[type[pl.DataType]] | None = None,
    ) -> list[str]:
        """Get columns to validate based on config and dtype filter."""
        schema = lf.collect_schema()
        columns = list(schema.names())

        if self.config.columns:
            columns = [c for c in columns if c in self.config.columns]

        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]

        if dtype_filter:
            columns = [c for c in columns if type(schema[c]) in dtype_filter]

        return columns

    def _calculate_severity(
        self,
        ratio: float,
        thresholds: tuple[float, float, float] = (0.5, 0.2, 0.05),
    ) -> Severity:
        """Calculate severity based on ratio and thresholds."""
        if self.config.severity_override:
            return self.config.severity_override

        critical_th, high_th, medium_th = thresholds
        if ratio > critical_th:
            return Severity.CRITICAL
        elif ratio > high_th:
            return Severity.HIGH
        elif ratio > medium_th:
            return Severity.MEDIUM
        return Severity.LOW

    def _passes_mostly(self, failure_count: int, total_count: int) -> bool:
        """Check if validation passes based on mostly threshold.

        Returns True if the validation should be considered passed
        (i.e., failure ratio is within acceptable limits).
        """
        if self.config.mostly is None:
            return False  # No mostly threshold, report all failures

        if total_count == 0:
            return True

        pass_ratio = 1 - (failure_count / total_count)
        return pass_ratio >= self.config.mostly

    def _get_mostly_adjusted_severity(
        self,
        failure_count: int,
        total_count: int,
        base_severity: Severity,
    ) -> Severity | None:
        """Get severity adjusted for mostly threshold.

        Returns None if validation passes due to mostly threshold.
        """
        if self._passes_mostly(failure_count, total_count):
            return None
        return base_severity


class NumericValidatorMixin:
    """Mixin for validators that work with numeric columns."""

    def _get_numeric_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=NUMERIC_TYPES)  # type: ignore


class StringValidatorMixin:
    """Mixin for validators that work with string columns."""

    def _get_string_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=STRING_TYPES)  # type: ignore


class DatetimeValidatorMixin:
    """Mixin for validators that work with datetime columns."""

    def _get_datetime_columns(self, lf: pl.LazyFrame) -> list[str]:
        return self._get_target_columns(lf, dtype_filter=DATETIME_TYPES)  # type: ignore


class ColumnValidator(Validator):
    """Template for column-level validation."""

    @abstractmethod
    def check_column(
        self,
        lf: pl.LazyFrame,
        col: str,
        total_rows: int,
    ) -> ValidationIssue | None:
        """Check a single column. Implement in subclass."""
        pass

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        columns = self._get_target_columns(lf)

        # Get total rows with single query
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        for col in columns:
            issue = self.check_column(lf, col, total_rows)
            if issue:
                issues.append(issue)

        return issues


class AggregateValidator(Validator, NumericValidatorMixin):
    """Template for aggregate statistics validation."""

    @abstractmethod
    def check_aggregate(
        self,
        col: str,
        stats: dict[str, Any],
        total_rows: int,
    ) -> ValidationIssue | None:
        """Check aggregate stats for a column. Implement in subclass."""
        pass

    def _compute_stats(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> tuple[int, dict[str, dict[str, Any]]]:
        """Compute statistics for all columns in single query."""
        exprs: list[pl.Expr] = [pl.len().alias("_total")]

        for col in columns:
            exprs.extend([
                pl.col(col).mean().alias(f"_mean_{col}"),
                pl.col(col).std().alias(f"_std_{col}"),
                pl.col(col).min().alias(f"_min_{col}"),
                pl.col(col).max().alias(f"_max_{col}"),
                pl.col(col).sum().alias(f"_sum_{col}"),
                pl.col(col).median().alias(f"_median_{col}"),
                pl.col(col).count().alias(f"_count_{col}"),
            ])

        result = lf.select(exprs).collect()
        total = result["_total"][0]

        stats: dict[str, dict[str, Any]] = {}
        for col in columns:
            stats[col] = {
                "mean": result[f"_mean_{col}"][0],
                "std": result[f"_std_{col}"][0],
                "min": result[f"_min_{col}"][0],
                "max": result[f"_max_{col}"][0],
                "sum": result[f"_sum_{col}"][0],
                "median": result[f"_median_{col}"][0],
                "count": result[f"_count_{col}"][0],
            }

        return total, stats

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        total_rows, all_stats = self._compute_stats(lf, columns)

        if total_rows == 0:
            return issues

        for col in columns:
            issue = self.check_aggregate(col, all_stats[col], total_rows)
            if issue:
                issues.append(issue)

        return issues
