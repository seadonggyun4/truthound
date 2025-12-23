"""Query-based row count validators.

Validators for checking row counts with flexible query conditions.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.query.base import QueryValidator
from truthound.validators.registry import register_validator


@register_validator
class QueryRowCountValidator(QueryValidator):
    """Validates row count from a query result.

    Equivalent to expect_query_to_count in Great Expectations.

    Example:
        # Exact count
        validator = QueryRowCountValidator(
            query="SELECT * FROM data WHERE status = 'active'",
            expected_count=100,
        )

        # Range
        validator = QueryRowCountValidator(
            query="SELECT * FROM data WHERE created_at > '2024-01-01'",
            min_count=50,
            max_count=200,
        )
    """

    name = "query_row_count"
    category = "query"

    def __init__(
        self,
        query: str,
        expected_count: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.expected_count = expected_count
        self.min_count = min_count
        self.max_count = max_count

        if expected_count is None and min_count is None and max_count is None:
            raise ValueError(
                "At least one of 'expected_count', 'min_count', or 'max_count' must be provided"
            )

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        row_count = len(result)

        if self.expected_count is not None and row_count != self.expected_count:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_row_count_mismatch",
                    count=abs(row_count - self.expected_count),
                    severity=Severity.HIGH,
                    details=f"Query returned {row_count} rows, expected exactly {self.expected_count}",
                    expected=f"Exactly {self.expected_count} rows",
                )
            )

        if self.min_count is not None and row_count < self.min_count:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_row_count_below_min",
                    count=self.min_count - row_count,
                    severity=Severity.HIGH,
                    details=f"Query returned {row_count} rows, expected at least {self.min_count}",
                    expected=f">= {self.min_count} rows",
                )
            )

        if self.max_count is not None and row_count > self.max_count:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_row_count_above_max",
                    count=row_count - self.max_count,
                    severity=Severity.MEDIUM,
                    details=f"Query returned {row_count} rows, expected at most {self.max_count}",
                    expected=f"<= {self.max_count} rows",
                )
            )

        return issues


@register_validator
class QueryRowCountRatioValidator(QueryValidator):
    """Validates that query result row count is within a ratio of total.

    Example:
        # At least 90% of orders should be completed
        validator = QueryRowCountRatioValidator(
            query="SELECT * FROM data WHERE status = 'completed'",
            min_ratio=0.9,
        )

        # Less than 5% should be errors
        validator = QueryRowCountRatioValidator(
            query="SELECT * FROM data WHERE status = 'error'",
            max_ratio=0.05,
        )
    """

    name = "query_row_count_ratio"
    category = "query"

    def __init__(
        self,
        query: str,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if min_ratio is None and max_ratio is None:
            raise ValueError("At least one of 'min_ratio' or 'max_ratio' must be provided")

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        query_count = len(result)
        total_count = original_lf.select(pl.len()).collect().item()

        if total_count == 0:
            return issues

        ratio = query_count / total_count

        if self.min_ratio is not None and ratio < self.min_ratio:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_ratio_below_min",
                    count=query_count,
                    severity=Severity.HIGH,
                    details=f"Query returned {ratio:.2%} of rows ({query_count}/{total_count}), expected at least {self.min_ratio:.2%}",
                    expected=f">= {self.min_ratio:.2%}",
                )
            )

        if self.max_ratio is not None and ratio > self.max_ratio:
            severity = Severity.HIGH if ratio > self.max_ratio * 2 else Severity.MEDIUM
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_ratio_above_max",
                    count=query_count,
                    severity=severity,
                    details=f"Query returned {ratio:.2%} of rows ({query_count}/{total_count}), expected at most {self.max_ratio:.2%}",
                    expected=f"<= {self.max_ratio:.2%}",
                )
            )

        return issues


@register_validator
class QueryRowCountCompareValidator(QueryValidator):
    """Compares row counts between two queries.

    Useful for checking data consistency across different conditions.

    Example:
        # Orders count should equal order_items count (1:1)
        validator = QueryRowCountCompareValidator(
            query="SELECT * FROM data WHERE type = 'order'",
            compare_query="SELECT * FROM data WHERE type = 'order_item'",
            relationship="equal",
        )

        # Active users should be less than total users
        validator = QueryRowCountCompareValidator(
            query="SELECT * FROM data WHERE status = 'active'",
            compare_query="SELECT * FROM data",
            relationship="less_than",
        )
    """

    name = "query_row_count_compare"
    category = "query"

    RELATIONSHIPS = {
        "equal": lambda a, b: a == b,
        "not_equal": lambda a, b: a != b,
        "greater_than": lambda a, b: a > b,
        "greater_than_or_equal": lambda a, b: a >= b,
        "less_than": lambda a, b: a < b,
        "less_than_or_equal": lambda a, b: a <= b,
    }

    def __init__(
        self,
        query: str,
        compare_query: str,
        relationship: str = "equal",
        tolerance: int = 0,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.compare_query = compare_query
        self.relationship = relationship
        self.tolerance = tolerance

        if relationship not in self.RELATIONSHIPS:
            raise ValueError(
                f"Invalid relationship: {relationship}. "
                f"Valid options: {list(self.RELATIONSHIPS.keys())}"
            )

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        count_a = len(result)

        # Execute compare query
        ctx = pl.SQLContext()
        ctx.register(self.table_name, original_lf)
        compare_result = ctx.execute(self.compare_query).collect()
        count_b = len(compare_result)

        # Apply tolerance for equal comparison
        if self.relationship == "equal" and self.tolerance > 0:
            is_valid = abs(count_a - count_b) <= self.tolerance
        else:
            comparator = self.RELATIONSHIPS[self.relationship]
            is_valid = comparator(count_a, count_b)

        if not is_valid:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_count_comparison_failed",
                    count=abs(count_a - count_b),
                    severity=Severity.HIGH,
                    details=f"Query A returned {count_a} rows, Query B returned {count_b} rows. Expected A {self.relationship} B",
                    expected=f"Query A {self.relationship} Query B",
                )
            )

        return issues
