"""Query-based aggregate validators.

Validators for checking aggregate query results.
"""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.query.base import QueryValidator
from truthound.validators.registry import register_validator


@register_validator
class QueryAggregateValidator(QueryValidator):
    """Validates aggregate values from a query.

    Flexible validator for checking any aggregate result.

    Example:
        # Check average is within range
        validator = QueryAggregateValidator(
            query="SELECT AVG(price) as avg_price FROM data",
            column="avg_price",
            min_value=10.0,
            max_value=100.0,
        )

        # Check sum equals expected
        validator = QueryAggregateValidator(
            query="SELECT SUM(quantity) as total FROM data",
            column="total",
            expected_value=1000,
        )
    """

    name = "query_aggregate"
    category = "query"

    def __init__(
        self,
        query: str,
        column: str,
        expected_value: float | int | None = None,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.column = column
        self.expected_value = expected_value
        self.min_value = min_value
        self.max_value = max_value
        self.tolerance = tolerance

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if len(result) == 0 or self.column not in result.columns:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_missing",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Aggregate column '{self.column}' not found or empty result",
                    expected=f"Column '{self.column}' with value",
                )
            )
            return issues

        actual_value = result[self.column][0]

        if actual_value is None:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_null",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Aggregate value is NULL",
                    expected="Non-null value",
                )
            )
            return issues

        # Check expected value
        if self.expected_value is not None:
            if self.tolerance is not None:
                is_match = abs(actual_value - self.expected_value) <= self.tolerance
            else:
                is_match = actual_value == self.expected_value

            if not is_match:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="query_aggregate_mismatch",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Aggregate value {actual_value} != expected {self.expected_value}",
                        expected=str(self.expected_value),
                        sample_values=[str(actual_value)],
                    )
                )

        # Check min value
        if self.min_value is not None and actual_value < self.min_value:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_below_min",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Aggregate value {actual_value} < min {self.min_value}",
                    expected=f">= {self.min_value}",
                    sample_values=[str(actual_value)],
                )
            )

        # Check max value
        if self.max_value is not None and actual_value > self.max_value:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_above_max",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=f"Aggregate value {actual_value} > max {self.max_value}",
                    expected=f"<= {self.max_value}",
                    sample_values=[str(actual_value)],
                )
            )

        return issues


@register_validator
class QueryGroupAggregateValidator(QueryValidator):
    """Validates aggregate values for each group in a GROUP BY query.

    Example:
        # Each category's average should be positive
        validator = QueryGroupAggregateValidator(
            query="SELECT category, AVG(price) as avg_price FROM data GROUP BY category",
            group_column="category",
            aggregate_column="avg_price",
            min_value=0,
        )
    """

    name = "query_group_aggregate"
    category = "query"

    def __init__(
        self,
        query: str,
        group_column: str,
        aggregate_column: str,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        expected_groups: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.group_column = group_column
        self.aggregate_column = aggregate_column
        self.min_value = min_value
        self.max_value = max_value
        self.expected_groups = set(expected_groups) if expected_groups else None

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check required columns exist
        for col in [self.group_column, self.aggregate_column]:
            if col not in result.columns:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="query_column_not_found",
                        count=1,
                        severity=Severity.CRITICAL,
                        details=f"Column '{col}' not found in query result",
                        expected=f"Column '{col}' to exist",
                    )
                )
                return issues

        # Check expected groups
        if self.expected_groups:
            actual_groups = set(result[self.group_column].to_list())
            missing = self.expected_groups - actual_groups
            if missing:
                issues.append(
                    ValidationIssue(
                        column=self.group_column,
                        issue_type="query_missing_groups",
                        count=len(missing),
                        severity=Severity.HIGH,
                        details=f"Missing expected groups: {list(missing)[:5]}",
                        expected=f"Groups: {list(self.expected_groups)[:5]}",
                    )
                )

        # Check aggregate bounds for each group
        failing_groups = []
        for i in range(len(result)):
            group = result[self.group_column][i]
            value = result[self.aggregate_column][i]

            if value is None:
                continue

            failed = False
            if self.min_value is not None and value < self.min_value:
                failed = True
            if self.max_value is not None and value > self.max_value:
                failed = True

            if failed:
                failing_groups.append(f"{group}={value}")

        if failing_groups:
            issues.append(
                ValidationIssue(
                    column=self.aggregate_column,
                    issue_type="query_group_aggregate_out_of_bounds",
                    count=len(failing_groups),
                    severity=Severity.HIGH,
                    details=f"{len(failing_groups)} groups have out-of-bounds aggregate values",
                    expected=f"Values in [{self.min_value}, {self.max_value}]",
                    sample_values=failing_groups[: self.config.sample_size],
                )
            )

        return issues


@register_validator
class QueryAggregateCompareValidator(QueryValidator):
    """Compares aggregate values between two queries.

    Example:
        # Sum of debits should equal sum of credits
        validator = QueryAggregateCompareValidator(
            query="SELECT SUM(amount) as total FROM data WHERE type = 'debit'",
            compare_query="SELECT SUM(amount) as total FROM data WHERE type = 'credit'",
            column="total",
            relationship="equal",
            tolerance=0.01,
        )
    """

    name = "query_aggregate_compare"
    category = "query"

    RELATIONSHIPS = {
        "equal": lambda a, b, t: abs(a - b) <= t if t else a == b,
        "greater_than": lambda a, b, t: a > b,
        "greater_than_or_equal": lambda a, b, t: a >= b,
        "less_than": lambda a, b, t: a < b,
        "less_than_or_equal": lambda a, b, t: a <= b,
    }

    def __init__(
        self,
        query: str,
        compare_query: str,
        column: str,
        relationship: Literal[
            "equal", "greater_than", "greater_than_or_equal", "less_than", "less_than_or_equal"
        ] = "equal",
        tolerance: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.compare_query = compare_query
        self.column = column
        self.relationship = relationship
        self.tolerance = tolerance

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if len(result) == 0 or self.column not in result.columns:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_missing",
                    count=1,
                    severity=Severity.HIGH,
                    details="First query returned no result",
                    expected="Aggregate value",
                )
            )
            return issues

        value_a = result[self.column][0]

        # Execute compare query
        ctx = pl.SQLContext()
        ctx.register(self.table_name, original_lf)
        compare_result = ctx.execute(self.compare_query).collect()

        if len(compare_result) == 0 or self.column not in compare_result.columns:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_missing",
                    count=1,
                    severity=Severity.HIGH,
                    details="Compare query returned no result",
                    expected="Aggregate value",
                )
            )
            return issues

        value_b = compare_result[self.column][0]

        comparator = self.RELATIONSHIPS[self.relationship]
        is_valid = comparator(value_a, value_b, self.tolerance)

        if not is_valid:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_aggregate_comparison_failed",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Query A: {value_a}, Query B: {value_b}. Expected A {self.relationship} B",
                    expected=f"A {self.relationship} B",
                    sample_values=[f"A={value_a}", f"B={value_b}"],
                )
            )

        return issues
