"""Streaming completeness validators for large datasets."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.streaming.base import StreamingValidator, StreamingState
from truthound.validators.registry import register_validator


@register_validator
class StreamingNullValidator(StreamingValidator):
    """Streaming version of NullValidator for very large datasets.

    Processes data in chunks and aggregates null counts across chunks.

    Example:
        validator = StreamingNullValidator(chunk_size=100_000)
        issues = validator.validate(large_lazyframe)
    """

    name = "streaming_null"
    category = "streaming"

    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Count nulls in this chunk."""
        columns = self._get_target_columns(chunk_df.lazy())

        for col in columns:
            null_count = chunk_df.get_column(col).null_count()
            total_count = len(chunk_df)

            state.update_column_stat(col, "null_count", null_count, "sum")
            state.update_column_stat(col, "total_count", total_count, "sum")

    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Generate issues from aggregated null counts."""
        issues: list[ValidationIssue] = []

        for col, stats in state.column_stats.items():
            null_count = stats.get("null_count", 0)

            if null_count > 0:
                null_pct = null_count / total_rows if total_rows > 0 else 0
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="null",
                        count=null_count,
                        severity=self._calculate_severity(null_pct),
                        details=f"{null_pct:.1%} of values are null (streaming)",
                    )
                )

        return issues


@register_validator
class StreamingCompletenessValidator(StreamingValidator):
    """Streaming completeness ratio validator.

    Validates that columns meet minimum completeness ratio.

    Example:
        validator = StreamingCompletenessValidator(
            min_ratio=0.95,
            chunk_size=50_000,
        )
    """

    name = "streaming_completeness"
    category = "streaming"

    def __init__(
        self,
        min_ratio: float = 0.95,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_ratio = min_ratio

    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Count non-null values in this chunk."""
        columns = self._get_target_columns(chunk_df.lazy())

        for col in columns:
            non_null_count = chunk_df.get_column(col).drop_nulls().len()
            total_count = len(chunk_df)

            state.update_column_stat(col, "non_null_count", non_null_count, "sum")
            state.update_column_stat(col, "total_count", total_count, "sum")

    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Check completeness ratio against threshold."""
        issues: list[ValidationIssue] = []

        for col, stats in state.column_stats.items():
            non_null_count = stats.get("non_null_count", 0)
            total_count = stats.get("total_count", total_rows)

            if total_count == 0:
                continue

            ratio = non_null_count / total_count

            if ratio < self.min_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="completeness_ratio",
                        count=total_count - non_null_count,
                        severity=self._calculate_severity(
                            1 - ratio, thresholds=(0.5, 0.2, 0.05)
                        ),
                        details=f"Completeness {ratio:.1%} < {self.min_ratio:.1%} (streaming)",
                        expected=self.min_ratio,
                        actual=ratio,
                    )
                )

        return issues


@register_validator
class StreamingNaNValidator(StreamingValidator):
    """Streaming NaN detector for large float datasets.

    Processes data in chunks to detect NaN values with bounded memory.

    Example:
        validator = StreamingNaNValidator(chunk_size=100_000)
        issues = validator.validate(large_float_data)
    """

    name = "streaming_nan"
    category = "streaming"

    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Count NaN values in float columns."""
        schema = chunk_df.schema

        for col, dtype in schema.items():
            if dtype in (pl.Float32, pl.Float64):
                # Check config filters
                if self.config.columns and col not in self.config.columns:
                    continue
                if self.config.exclude_columns and col in self.config.exclude_columns:
                    continue

                nan_count = chunk_df.get_column(col).is_nan().sum()
                non_null_count = chunk_df.get_column(col).drop_nulls().len()

                state.update_column_stat(col, "nan_count", nan_count, "sum")
                state.update_column_stat(col, "non_null_count", non_null_count, "sum")

    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Generate issues from aggregated NaN counts."""
        issues: list[ValidationIssue] = []

        for col, stats in state.column_stats.items():
            nan_count = stats.get("nan_count", 0)
            non_null_count = stats.get("non_null_count", 0)

            if nan_count > 0:
                ratio = nan_count / non_null_count if non_null_count > 0 else 1.0
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="nan",
                        count=nan_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{ratio:.1%} of non-null values are NaN (streaming)",
                        expected="No NaN values",
                        actual=f"{nan_count} NaN values",
                    )
                )

        return issues
