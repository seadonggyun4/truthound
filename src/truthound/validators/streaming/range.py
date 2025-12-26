"""Streaming range and outlier validators for large datasets."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, NUMERIC_TYPES
from truthound.validators.streaming.base import StreamingValidator, StreamingState
from truthound.validators.registry import register_validator


@register_validator
class StreamingRangeValidator(StreamingValidator):
    """Streaming version of BetweenValidator for very large datasets.

    Processes data in chunks and aggregates out-of-range counts.

    Example:
        validator = StreamingRangeValidator(
            min_value=0,
            max_value=100,
            chunk_size=100_000,
        )
    """

    name = "streaming_range"
    category = "streaming"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def _get_numeric_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get numeric columns matching config."""
        schema = lf.collect_schema()
        columns = list(schema.names())

        if self.config.columns:
            columns = [c for c in columns if c in self.config.columns]

        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]

        return [c for c in columns if type(schema[c]) in NUMERIC_TYPES]

    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Count out-of-range values in this chunk."""
        columns = self._get_numeric_columns(chunk_df.lazy())

        for col in columns:
            col_data = chunk_df.get_column(col)

            # Build condition
            if self.inclusive:
                below = col_data < self.min_value if self.min_value is not None else pl.lit(False)
                above = col_data > self.max_value if self.max_value is not None else pl.lit(False)
            else:
                below = col_data <= self.min_value if self.min_value is not None else pl.lit(False)
                above = col_data >= self.max_value if self.max_value is not None else pl.lit(False)

            out_count = ((below | above) & col_data.is_not_null()).sum()
            total_count = len(chunk_df)

            state.update_column_stat(col, "out_count", out_count, "sum")
            state.update_column_stat(col, "total_count", total_count, "sum")

    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Generate issues from aggregated out-of-range counts."""
        issues: list[ValidationIssue] = []

        for col, stats in state.column_stats.items():
            out_count = stats.get("out_count", 0)

            if out_count > 0:
                ratio = out_count / total_rows if total_rows > 0 else 0
                range_str = f"[{self.min_value}, {self.max_value}]"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="out_of_range",
                        count=out_count,
                        severity=self._calculate_severity(ratio, (0.1, 0.05, 0.01)),
                        details=f"{out_count} values outside {range_str} (streaming)",
                        expected=range_str,
                    )
                )

        return issues


@register_validator
class StreamingOutlierValidator(StreamingValidator):
    """Streaming IQR-based outlier detector.

    Uses two-pass streaming:
    1. First pass: Calculate Q1, Q3 for each column
    2. Second pass: Count outliers based on IQR bounds

    Example:
        validator = StreamingOutlierValidator(
            iqr_multiplier=1.5,
            chunk_size=100_000,
        )
    """

    name = "streaming_outlier"
    category = "streaming"

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.iqr_multiplier = iqr_multiplier
        self._bounds: dict[str, tuple[float, float]] | None = None

    def _get_numeric_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get numeric columns matching config."""
        schema = lf.collect_schema()
        columns = list(schema.names())

        if self.config.columns:
            columns = [c for c in columns if c in self.config.columns]

        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]

        return [c for c in columns if type(schema[c]) in NUMERIC_TYPES]

    def _calculate_bounds(self, lf: pl.LazyFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
        """Calculate IQR bounds for all columns in single query."""
        exprs = []
        for col in columns:
            exprs.extend([
                pl.col(col).quantile(0.25).alias(f"_q1_{col}"),
                pl.col(col).quantile(0.75).alias(f"_q3_{col}"),
            ])

        result = lf.select(exprs).collect()

        bounds = {}
        for col in columns:
            q1 = result[f"_q1_{col}"][0]
            q3 = result[f"_q3_{col}"][0]

            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - self.iqr_multiplier * iqr
                    upper = q3 + self.iqr_multiplier * iqr
                    bounds[col] = (lower, upper)

        return bounds

    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Count outliers in this chunk using pre-calculated bounds."""
        if self._bounds is None:
            return

        for col, (lower, upper) in self._bounds.items():
            col_data = chunk_df.get_column(col)
            outlier_count = ((col_data < lower) | (col_data > upper)).sum()
            non_null_count = col_data.drop_nulls().len()

            state.update_column_stat(col, "outlier_count", outlier_count, "sum")
            state.update_column_stat(col, "non_null_count", non_null_count, "sum")
            state.update_column_stat(col, "lower", lower, "last")
            state.update_column_stat(col, "upper", upper, "last")

    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Generate issues from aggregated outlier counts."""
        issues: list[ValidationIssue] = []

        for col, stats in state.column_stats.items():
            outlier_count = stats.get("outlier_count", 0)
            non_null_count = stats.get("non_null_count", 0)
            lower = stats.get("lower", 0)
            upper = stats.get("upper", 0)

            if outlier_count > 0:
                ratio = outlier_count / non_null_count if non_null_count > 0 else 0
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="outlier",
                        count=outlier_count,
                        severity=Severity.MEDIUM if ratio > 0.1 else Severity.LOW,
                        details=f"IQR bounds: [{lower:.2f}, {upper:.2f}] (streaming)",
                        expected=f"[{lower:.2f}, {upper:.2f}]",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Override to add bound calculation pass."""
        columns = self._get_numeric_columns(lf)

        if not columns:
            return []

        # First pass: calculate bounds
        self._bounds = self._calculate_bounds(lf, columns)

        if not self._bounds:
            return []

        # Second pass: count outliers using parent's streaming
        return super().validate(lf)


@register_validator
class StreamingPositiveValidator(StreamingValidator):
    """Streaming validator for positive values.

    Example:
        validator = StreamingPositiveValidator(chunk_size=100_000)
    """

    name = "streaming_positive"
    category = "streaming"

    def _get_numeric_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get numeric columns matching config."""
        schema = lf.collect_schema()
        columns = list(schema.names())

        if self.config.columns:
            columns = [c for c in columns if c in self.config.columns]

        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]

        return [c for c in columns if type(schema[c]) in NUMERIC_TYPES]

    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Count non-positive values in this chunk."""
        columns = self._get_numeric_columns(chunk_df.lazy())

        for col in columns:
            col_data = chunk_df.get_column(col)
            neg_count = ((col_data <= 0) & col_data.is_not_null()).sum()
            total_count = len(chunk_df)

            state.update_column_stat(col, "neg_count", neg_count, "sum")
            state.update_column_stat(col, "total_count", total_count, "sum")

    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Generate issues from aggregated counts."""
        issues: list[ValidationIssue] = []

        for col, stats in state.column_stats.items():
            neg_count = stats.get("neg_count", 0)

            if neg_count > 0:
                ratio = neg_count / total_rows if total_rows > 0 else 0
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_positive",
                        count=neg_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{neg_count} non-positive values (streaming)",
                        expected="> 0",
                    )
                )

        return issues
