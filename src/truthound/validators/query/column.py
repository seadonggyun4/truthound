"""Query-based column validators.

Validators for checking column values using SQL queries.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.query.base import QueryValidator
from truthound.validators.registry import register_validator


@register_validator
class QueryColumnValuesValidator(QueryValidator):
    """Validates that query returns expected column values.

    Example:
        # Check distinct status values
        validator = QueryColumnValuesValidator(
            query="SELECT DISTINCT status FROM data",
            column="status",
            expected_values=["active", "inactive", "pending"],
        )
    """

    name = "query_column_values"
    category = "query"

    def __init__(
        self,
        query: str,
        column: str,
        expected_values: list[Any] | None = None,
        forbidden_values: list[Any] | None = None,
        allow_subset: bool = True,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.column = column
        self.expected_values = set(expected_values) if expected_values else None
        self.forbidden_values = set(forbidden_values) if forbidden_values else None
        self.allow_subset = allow_subset

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.column not in result.columns:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_column_not_found",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Column '{self.column}' not found in query result",
                    expected=f"Column '{self.column}' to exist",
                )
            )
            return issues

        actual_values = set(result[self.column].drop_nulls().to_list())

        if self.expected_values:
            if self.allow_subset:
                # All actual values should be in expected
                unexpected = actual_values - self.expected_values
                if unexpected:
                    issues.append(
                        ValidationIssue(
                            column=self.column,
                            issue_type="query_unexpected_values",
                            count=len(unexpected),
                            severity=Severity.HIGH,
                            details=f"Found {len(unexpected)} unexpected values",
                            expected=f"Values in: {list(self.expected_values)[:5]}",
                            sample_values=[str(v) for v in list(unexpected)[:5]],
                        )
                    )
            else:
                # Actual should exactly match expected
                if actual_values != self.expected_values:
                    missing = self.expected_values - actual_values
                    extra = actual_values - self.expected_values
                    issues.append(
                        ValidationIssue(
                            column=self.column,
                            issue_type="query_values_mismatch",
                            count=len(missing) + len(extra),
                            severity=Severity.HIGH,
                            details=f"Missing: {list(missing)[:3]}, Extra: {list(extra)[:3]}",
                            expected=f"Exactly: {list(self.expected_values)[:5]}",
                        )
                    )

        if self.forbidden_values:
            forbidden_found = actual_values & self.forbidden_values
            if forbidden_found:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="query_forbidden_values",
                        count=len(forbidden_found),
                        severity=Severity.HIGH,
                        details=f"Found {len(forbidden_found)} forbidden values",
                        expected=f"No values from: {list(self.forbidden_values)[:5]}",
                        sample_values=[str(v) for v in list(forbidden_found)[:5]],
                    )
                )

        return issues


@register_validator
class QueryColumnUniqueValidator(QueryValidator):
    """Validates that a column in query result has unique values.

    Example:
        # Check that aggregation keys are unique
        validator = QueryColumnUniqueValidator(
            query="SELECT user_id, SUM(amount) as total FROM data GROUP BY user_id",
            column="user_id",
        )
    """

    name = "query_column_unique"
    category = "query"

    def __init__(
        self,
        query: str,
        column: str,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.column = column

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.column not in result.columns:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_column_not_found",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Column '{self.column}' not found in query result",
                    expected=f"Column '{self.column}' to exist",
                )
            )
            return issues

        total = len(result)
        unique_count = result[self.column].n_unique()
        duplicates = total - unique_count

        if duplicates > 0:
            # Find duplicate values
            value_counts = result.group_by(self.column).len()
            dup_values = value_counts.filter(pl.col("len") > 1)

            sample_values = [
                str(v) for v in dup_values[self.column].head(self.config.sample_size).to_list()
            ]

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_column_not_unique",
                    count=duplicates,
                    severity=Severity.HIGH,
                    details=f"Column has {duplicates} duplicate entries ({unique_count} unique out of {total})",
                    expected="All unique values",
                    sample_values=sample_values,
                )
            )

        return issues


@register_validator
class QueryColumnNotNullValidator(QueryValidator):
    """Validates that a column in query result has no null values.

    Example:
        # Check that aggregation results are not null
        validator = QueryColumnNotNullValidator(
            query="SELECT category, AVG(price) as avg_price FROM data GROUP BY category",
            column="avg_price",
        )
    """

    name = "query_column_not_null"
    category = "query"

    def __init__(
        self,
        query: str,
        column: str,
        **kwargs: Any,
    ):
        super().__init__(query=query, **kwargs)
        self.column = column

    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.column not in result.columns:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_column_not_found",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Column '{self.column}' not found in query result",
                    expected=f"Column '{self.column}' to exist",
                )
            )
            return issues

        null_count = result[self.column].null_count()
        total = len(result)

        if null_count > 0:
            if self._passes_mostly(null_count, total):
                return issues

            ratio = null_count / total if total > 0 else 0

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="query_column_has_nulls",
                    count=null_count,
                    severity=self._calculate_severity(ratio),
                    details=f"Column has {null_count}/{total} null values ({ratio:.2%})",
                    expected="No null values",
                )
            )

        return issues
