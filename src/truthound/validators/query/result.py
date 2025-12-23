"""Query result validators.

Validators for checking query results against expected values.
Equivalent to Great Expectations' expect_query_to_* expectations.
"""

from typing import Any, Callable

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.query.base import QueryValidator
from truthound.validators.registry import register_validator


@register_validator
class QueryReturnsSingleValueValidator(QueryValidator):
    """Validates that a query returns a single expected value.

    Equivalent to expect_query_to_return_value in Great Expectations.

    Example:
        # Check if count of active users is exactly 100
        validator = QueryReturnsSingleValueValidator(
            query="SELECT COUNT(*) as cnt FROM data WHERE status = 'active'",
            expected_value=100,
        )

        # Check max value
        validator = QueryReturnsSingleValueValidator(
            query="SELECT MAX(price) as max_price FROM data",
            expected_value=999.99,
            tolerance=0.01,  # Allow small floating point differences
        )
    """

    name = "query_returns_single_value"
    category = "query"

    def __init__(
        self,
        query: str,
        expected_value: Any,
        tolerance: float | None = None,
        value_column: str | None = None,
        **kwargs: Any,
    ):
        """Initialize validator.

        Args:
            query: SQL query that returns a single value
            expected_value: Expected value from the query
            tolerance: Tolerance for numeric comparisons
            value_column: Column name containing the value (auto-detected if None)
            **kwargs: Additional validator config
        """
        super().__init__(query=query, **kwargs)
        self.expected_value = expected_value
        self.tolerance = tolerance
        self.value_column = value_column

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if len(result) == 0 or len(result.columns) == 0:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_empty_result",
                    count=1,
                    severity=Severity.HIGH,
                    details="Query returned empty result",
                    expected=f"Single value: {self.expected_value}",
                )
            )
            return issues

        # Get the value from result
        col_name = self.value_column or result.columns[0]
        actual_value = result[col_name][0]

        # Compare values
        is_match = False
        if self.tolerance is not None and isinstance(actual_value, (int, float)):
            is_match = abs(actual_value - self.expected_value) <= self.tolerance
        else:
            is_match = actual_value == self.expected_value

        if not is_match:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_value_mismatch",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Query returned {actual_value}, expected {self.expected_value}",
                    expected=str(self.expected_value),
                    sample_values=[str(actual_value)],
                )
            )

        return issues


@register_validator
class QueryReturnsNoRowsValidator(QueryValidator):
    """Validates that a query returns no rows.

    Useful for checking that certain conditions never occur.

    Example:
        # No orders with negative amounts
        validator = QueryReturnsNoRowsValidator(
            query="SELECT * FROM data WHERE amount < 0",
        )

        # No orphaned records
        validator = QueryReturnsNoRowsValidator(
            query="SELECT * FROM data WHERE parent_id IS NOT NULL AND parent_id NOT IN (SELECT id FROM data)",
        )
    """

    name = "query_returns_no_rows"
    category = "query"

    def __init__(self, query: str, **kwargs: Any):
        super().__init__(query=query, **kwargs)

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        row_count = len(result)
        if row_count > 0:
            total = original_lf.select(pl.len()).collect().item()

            if self._passes_mostly(row_count, total):
                return issues

            ratio = row_count / total if total > 0 else 1

            # Get sample of failing rows
            sample_values = []
            for i in range(min(self.config.sample_size, row_count)):
                row_str = ", ".join(str(result[col][i]) for col in result.columns[:3])
                sample_values.append(f"({row_str})")

            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_unexpected_rows",
                    count=row_count,
                    severity=self._calculate_severity(ratio),
                    details=f"Query returned {row_count} rows, expected 0",
                    expected="No rows",
                    sample_values=sample_values,
                )
            )

        return issues


@register_validator
class QueryReturnsRowsValidator(QueryValidator):
    """Validates that a query returns at least one row.

    Useful for checking that required data exists.

    Example:
        # At least one admin user exists
        validator = QueryReturnsRowsValidator(
            query="SELECT * FROM data WHERE role = 'admin'",
            min_rows=1,
        )
    """

    name = "query_returns_rows"
    category = "query"

    def __init__(
        self,
        query: str,
        min_rows: int = 1,
        max_rows: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.min_rows = min_rows
        self.max_rows = max_rows

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        row_count = len(result)

        if row_count < self.min_rows:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_insufficient_rows",
                    count=row_count,
                    severity=Severity.HIGH,
                    details=f"Query returned {row_count} rows, expected at least {self.min_rows}",
                    expected=f">= {self.min_rows} rows",
                )
            )

        if self.max_rows is not None and row_count > self.max_rows:
            issues.append(
                ValidationIssue(
                    column="_query",
                    issue_type="query_too_many_rows",
                    count=row_count,
                    severity=Severity.MEDIUM,
                    details=f"Query returned {row_count} rows, expected at most {self.max_rows}",
                    expected=f"<= {self.max_rows} rows",
                )
            )

        return issues


@register_validator
class QueryResultMatchesValidator(QueryValidator):
    """Validates that query result matches expected DataFrame or condition.

    Example:
        # Check aggregation results
        validator = QueryResultMatchesValidator(
            query="SELECT category, COUNT(*) as cnt FROM data GROUP BY category",
            expected_df=pl.DataFrame({
                "category": ["A", "B", "C"],
                "cnt": [10, 20, 30],
            }),
        )

        # Check with custom matcher
        validator = QueryResultMatchesValidator(
            query="SELECT AVG(price) as avg_price FROM data",
            matcher=lambda df: df["avg_price"][0] > 100,
        )
    """

    name = "query_result_matches"
    category = "query"

    def __init__(
        self,
        query: str,
        expected_df: pl.DataFrame | None = None,
        matcher: Callable[[pl.DataFrame], bool] | None = None,
        check_order: bool = False,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.expected_df = expected_df
        self.matcher = matcher
        self.check_order = check_order

        if expected_df is None and matcher is None:
            raise ValueError("Either 'expected_df' or 'matcher' must be provided")

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.matcher:
            try:
                if not self.matcher(result):
                    issues.append(
                        ValidationIssue(
                            column="_query",
                            issue_type="query_matcher_failed",
                            count=1,
                            severity=Severity.HIGH,
                            details="Query result did not match custom condition",
                            expected="Custom matcher to return True",
                        )
                    )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        column="_query",
                        issue_type="query_matcher_error",
                        count=1,
                        severity=Severity.CRITICAL,
                        details=f"Matcher raised exception: {e}",
                        expected="Matcher to execute successfully",
                    )
                )
        elif self.expected_df is not None:
            # Compare DataFrames
            if not self.check_order:
                # Sort both for comparison
                sort_cols = list(result.columns)
                try:
                    result = result.sort(sort_cols)
                    expected = self.expected_df.sort(sort_cols)
                except Exception:
                    expected = self.expected_df
            else:
                expected = self.expected_df

            if not result.equals(expected):
                issues.append(
                    ValidationIssue(
                        column="_query",
                        issue_type="query_result_mismatch",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Query result does not match expected DataFrame. Got {len(result)} rows, expected {len(expected)}",
                        expected=f"DataFrame with {len(expected)} rows",
                    )
                )

        return issues
