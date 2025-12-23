"""Table size and memory validators.

Validators for checking table size and memory usage.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.table.base import TableValidator
from truthound.validators.registry import register_validator


@register_validator
class TableMemorySizeValidator(TableValidator):
    """Validates that table memory size is within bounds.

    Example:
        # Table should not exceed 1GB in memory
        validator = TableMemorySizeValidator(
            max_size_mb=1024,
        )

        # Table should be at least 100MB (sanity check)
        validator = TableMemorySizeValidator(
            min_size_mb=100,
            max_size_mb=2048,
        )
    """

    name = "table_memory_size"
    category = "table"

    def __init__(
        self,
        max_size_mb: float | None = None,
        min_size_mb: float | None = None,
        max_size_bytes: int | None = None,
        min_size_bytes: int | None = None,
        **kwargs: Any,
    ):
        """Initialize memory size validator.

        Args:
            max_size_mb: Maximum size in megabytes
            min_size_mb: Minimum size in megabytes
            max_size_bytes: Maximum size in bytes (overrides max_size_mb)
            min_size_bytes: Minimum size in bytes (overrides min_size_mb)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)

        # Convert MB to bytes
        self.max_size_bytes = max_size_bytes
        self.min_size_bytes = min_size_bytes

        if max_size_mb is not None and max_size_bytes is None:
            self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        if min_size_mb is not None and min_size_bytes is None:
            self.min_size_bytes = int(min_size_mb * 1024 * 1024)

        if self.max_size_bytes is None and self.min_size_bytes is None:
            raise ValueError(
                "At least one of 'max_size_mb', 'min_size_mb', 'max_size_bytes', or 'min_size_bytes' required"
            )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Collect to get actual size (estimated_size works on DataFrame)
        df = lf.collect()
        actual_size = df.estimated_size()

        if self.min_size_bytes is not None and actual_size < self.min_size_bytes:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="table_size_below_minimum",
                    count=actual_size,
                    severity=Severity.MEDIUM,
                    details=f"Table size {self._format_size(actual_size)} below minimum {self._format_size(self.min_size_bytes)}",
                    expected=f">= {self._format_size(self.min_size_bytes)}",
                )
            )

        if self.max_size_bytes is not None and actual_size > self.max_size_bytes:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="table_size_above_maximum",
                    count=actual_size,
                    severity=Severity.HIGH,
                    details=f"Table size {self._format_size(actual_size)} exceeds maximum {self._format_size(self.max_size_bytes)}",
                    expected=f"<= {self._format_size(self.max_size_bytes)}",
                )
            )

        return issues

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        if size_bytes >= 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
        elif size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes} bytes"


@register_validator
class TableRowToColumnRatioValidator(TableValidator):
    """Validates the ratio of rows to columns.

    Useful for detecting wide tables or very tall narrow tables.

    Example:
        # Table should not be too wide (max 100 columns per 1000 rows)
        validator = TableRowToColumnRatioValidator(
            min_ratio=10,  # At least 10 rows per column
        )
    """

    name = "table_row_column_ratio"
    category = "table"

    def __init__(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
        **kwargs: Any,
    ):
        """Initialize row to column ratio validator.

        Args:
            min_ratio: Minimum rows per column ratio
            max_ratio: Maximum rows per column ratio
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if min_ratio is None and max_ratio is None:
            raise ValueError("At least one of 'min_ratio' or 'max_ratio' required")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        col_count = len(schema)
        row_count = lf.select(pl.len()).collect().item()

        if col_count == 0:
            return issues

        ratio = row_count / col_count

        if self.min_ratio is not None and ratio < self.min_ratio:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_column_ratio_too_low",
                    count=int(ratio),
                    severity=Severity.MEDIUM,
                    details=f"Row/column ratio {ratio:.2f} below minimum {self.min_ratio}. Table may be too wide.",
                    expected=f"Ratio >= {self.min_ratio}",
                )
            )

        if self.max_ratio is not None and ratio > self.max_ratio:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_column_ratio_too_high",
                    count=int(ratio),
                    severity=Severity.MEDIUM,
                    details=f"Row/column ratio {ratio:.2f} exceeds maximum {self.max_ratio}. Table may be too narrow.",
                    expected=f"Ratio <= {self.max_ratio}",
                )
            )

        return issues


@register_validator
class TableDimensionsValidator(TableValidator):
    """Validates table dimensions (rows and columns) together.

    Example:
        validator = TableDimensionsValidator(
            min_rows=100,
            max_rows=1000000,
            min_cols=5,
            max_cols=100,
        )
    """

    name = "table_dimensions"
    category = "table"

    def __init__(
        self,
        min_rows: int | None = None,
        max_rows: int | None = None,
        min_cols: int | None = None,
        max_cols: int | None = None,
        **kwargs: Any,
    ):
        """Initialize dimensions validator.

        Args:
            min_rows: Minimum row count
            max_rows: Maximum row count
            min_cols: Minimum column count
            max_cols: Maximum column count
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_cols = min_cols
        self.max_cols = max_cols

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        col_count = len(schema)
        row_count = lf.select(pl.len()).collect().item()

        # Row checks
        if self.min_rows is not None and row_count < self.min_rows:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_count_below_minimum",
                    count=row_count,
                    severity=Severity.HIGH,
                    details=f"Table has {row_count} rows, minimum is {self.min_rows}",
                    expected=f">= {self.min_rows} rows",
                )
            )

        if self.max_rows is not None and row_count > self.max_rows:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_count_above_maximum",
                    count=row_count,
                    severity=Severity.HIGH,
                    details=f"Table has {row_count} rows, maximum is {self.max_rows}",
                    expected=f"<= {self.max_rows} rows",
                )
            )

        # Column checks
        if self.min_cols is not None and col_count < self.min_cols:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="column_count_below_minimum",
                    count=col_count,
                    severity=Severity.HIGH,
                    details=f"Table has {col_count} columns, minimum is {self.min_cols}",
                    expected=f">= {self.min_cols} columns",
                )
            )

        if self.max_cols is not None and col_count > self.max_cols:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="column_count_above_maximum",
                    count=col_count,
                    severity=Severity.HIGH,
                    details=f"Table has {col_count} columns, maximum is {self.max_cols}",
                    expected=f"<= {self.max_cols} columns",
                )
            )

        return issues
