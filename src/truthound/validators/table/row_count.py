"""Table row count validators.

Validators for checking row counts and data volume.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.table.base import TableValidator
from truthound.validators.registry import register_validator


@register_validator
class TableRowCountRangeValidator(TableValidator):
    """Validates that table row count is within expected range.

    Example:
        # Table should have between 1000 and 10000 rows
        validator = TableRowCountRangeValidator(
            min_rows=1000,
            max_rows=10000,
        )

        # Table should have at least 100 rows
        validator = TableRowCountRangeValidator(
            min_rows=100,
        )
    """

    name = "table_row_count_range"
    category = "table"

    def __init__(
        self,
        min_rows: int | None = None,
        max_rows: int | None = None,
        **kwargs: Any,
    ):
        """Initialize row count range validator.

        Args:
            min_rows: Minimum expected rows (inclusive)
            max_rows: Maximum expected rows (inclusive)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.min_rows = min_rows
        self.max_rows = max_rows

        if min_rows is None and max_rows is None:
            raise ValueError("At least one of 'min_rows' or 'max_rows' required")

        if min_rows is not None and max_rows is not None and min_rows > max_rows:
            raise ValueError("'min_rows' cannot be greater than 'max_rows'")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        row_count = lf.select(pl.len()).collect().item()

        if self.min_rows is not None and row_count < self.min_rows:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_count_below_minimum",
                    count=row_count,
                    severity=Severity.HIGH,
                    details=f"Table has {row_count} rows, expected at least {self.min_rows}",
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
                    details=f"Table has {row_count} rows, expected at most {self.max_rows}",
                    expected=f"<= {self.max_rows} rows",
                )
            )

        return issues


@register_validator
class TableRowCountExactValidator(TableValidator):
    """Validates that table has exactly the expected row count.

    Example:
        # Table should have exactly 1000 rows
        validator = TableRowCountExactValidator(expected_rows=1000)
    """

    name = "table_row_count_exact"
    category = "table"

    def __init__(
        self,
        expected_rows: int,
        tolerance: int = 0,
        **kwargs: Any,
    ):
        """Initialize exact row count validator.

        Args:
            expected_rows: Expected number of rows
            tolerance: Allowed deviation from expected (default 0)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.expected_rows = expected_rows
        self.tolerance = tolerance

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        row_count = lf.select(pl.len()).collect().item()
        diff = abs(row_count - self.expected_rows)

        if diff > self.tolerance:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_count_mismatch",
                    count=row_count,
                    severity=Severity.HIGH,
                    details=f"Table has {row_count} rows, expected {self.expected_rows} (±{self.tolerance})",
                    expected=f"{self.expected_rows} ± {self.tolerance} rows",
                )
            )

        return issues


@register_validator
class TableRowCountCompareValidator(TableValidator):
    """Validates table row count compared to another table.

    Example:
        # Main table should have same row count as reference
        validator = TableRowCountCompareValidator(
            reference_table=reference_lf,
            comparison="equal",
        )

        # Main table should have more rows than reference
        validator = TableRowCountCompareValidator(
            reference_table=reference_lf,
            comparison="greater",
        )
    """

    name = "table_row_count_compare"
    category = "table"

    COMPARISONS = {
        "equal": lambda a, b: a == b,
        "greater": lambda a, b: a > b,
        "greater_equal": lambda a, b: a >= b,
        "less": lambda a, b: a < b,
        "less_equal": lambda a, b: a <= b,
    }

    def __init__(
        self,
        reference_table: pl.LazyFrame,
        comparison: str = "equal",
        tolerance_ratio: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize row count compare validator.

        Args:
            reference_table: Reference table to compare against
            comparison: Comparison type ('equal', 'greater', 'greater_equal', 'less', 'less_equal')
            tolerance_ratio: Allowed ratio deviation for 'equal' comparison
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.reference_table = reference_table
        self.comparison = comparison
        self.tolerance_ratio = tolerance_ratio

        if comparison not in self.COMPARISONS:
            raise ValueError(f"Invalid comparison: {comparison}. Use one of {list(self.COMPARISONS.keys())}")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        main_count = lf.select(pl.len()).collect().item()
        ref_count = self.reference_table.select(pl.len()).collect().item()

        comparator = self.COMPARISONS[self.comparison]

        # For 'equal' with tolerance
        if self.comparison == "equal" and self.tolerance_ratio > 0:
            tolerance = int(ref_count * self.tolerance_ratio)
            is_valid = abs(main_count - ref_count) <= tolerance
        else:
            is_valid = comparator(main_count, ref_count)

        if not is_valid:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="row_count_comparison_failed",
                    count=main_count,
                    severity=Severity.HIGH,
                    details=f"Row count {main_count} is not {self.comparison} to reference {ref_count}",
                    expected=f"Row count {self.comparison} {ref_count}",
                )
            )

        return issues


@register_validator
class TableNotEmptyValidator(TableValidator):
    """Validates that table is not empty.

    Example:
        validator = TableNotEmptyValidator()
    """

    name = "table_not_empty"
    category = "table"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        row_count = lf.select(pl.len()).collect().item()

        if row_count == 0:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="table_is_empty",
                    count=0,
                    severity=Severity.CRITICAL,
                    details="Table has no rows",
                    expected="At least 1 row",
                )
            )

        return issues
