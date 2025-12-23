"""Base classes for multi-column compound validators.

Multi-column validators check relationships and computations across multiple columns,
providing powerful business rule validation capabilities.

Design Principles:
1. Flexible column specification (list of column names)
2. Support for computed expressions
3. Row-level and aggregate-level validation
4. Clear error messages showing which columns failed
"""

from abc import abstractmethod
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, ValidatorConfig
from truthound.validators.registry import register_validator


class MultiColumnValidator(Validator):
    """Base class for multi-column validators.

    Provides infrastructure for validating relationships between multiple columns.
    Subclasses implement specific validation logic.

    Example:
        class MyMultiColumnValidator(MultiColumnValidator):
            def get_validation_expression(self) -> pl.Expr:
                # Return expression that evaluates to True for valid rows
                pass
    """

    name = "multi_column_base"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        **kwargs: Any,
    ):
        """Initialize multi-column validator.

        Args:
            columns: List of column names to validate
            **kwargs: Additional validator config
        """
        super().__init__(**kwargs)
        self.columns = columns

        if len(columns) < 2:
            raise ValueError("Multi-column validators require at least 2 columns")

    @abstractmethod
    def get_validation_expression(self) -> pl.Expr:
        """Return Polars expression that evaluates to True for valid rows.

        Returns:
            Polars expression returning boolean series
        """
        pass

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using the expression."""
        issues: list[ValidationIssue] = []

        # Verify columns exist
        schema = lf.collect_schema()
        missing_cols = [c for c in self.columns if c not in schema]
        if missing_cols:
            issues.append(
                ValidationIssue(
                    column=", ".join(missing_cols),
                    issue_type="columns_not_found",
                    count=len(missing_cols),
                    severity=Severity.CRITICAL,
                    details=f"Columns not found: {missing_cols}",
                    expected=f"Columns to exist: {self.columns}",
                )
            )
            return issues

        try:
            validation_expr = self.get_validation_expression()

            result = lf.select([
                pl.len().alias("_total"),
                (~validation_expr).sum().alias("_failures"),
            ]).collect()

            total = result["_total"][0]
            failures = result["_failures"][0]

            if failures > 0:
                if self._passes_mostly(failures, total):
                    return issues

                ratio = failures / total if total > 0 else 0

                # Get sample of failing rows
                sample_df = (
                    lf.filter(~validation_expr)
                    .select(self.columns)
                    .head(self.config.sample_size)
                    .collect()
                )

                sample_values = []
                for i in range(len(sample_df)):
                    row_str = ", ".join(
                        f"{col}={sample_df[col][i]}" for col in self.columns
                    )
                    sample_values.append(f"({row_str})")

                issues.append(
                    ValidationIssue(
                        column=", ".join(self.columns),
                        issue_type=self._get_issue_type(),
                        count=failures,
                        severity=self._calculate_severity(ratio),
                        details=self._get_details(failures, total),
                        expected=self._get_expected(),
                        sample_values=sample_values,
                    )
                )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    column=", ".join(self.columns),
                    issue_type="multi_column_validation_error",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Validation failed: {e}",
                    expected="Successful validation",
                )
            )

        return issues

    def _get_issue_type(self) -> str:
        """Return the issue type for failures."""
        return "multi_column_validation_failed"

    def _get_details(self, failures: int, total: int) -> str:
        """Return detailed message for failures."""
        return f"{failures}/{total} rows failed multi-column validation"

    def _get_expected(self) -> str:
        """Return expected condition description."""
        return f"All rows to pass validation on columns: {self.columns}"


class ColumnArithmeticValidator(MultiColumnValidator):
    """Base class for column arithmetic validators (sum, product, difference, etc.)."""

    name = "column_arithmetic_base"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        result_column: str | None = None,
        expected_value: float | int | None = None,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        tolerance: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize arithmetic validator.

        Args:
            columns: Columns to perform arithmetic on
            result_column: Column containing expected result (mutually exclusive with expected_value)
            expected_value: Expected constant value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            tolerance: Tolerance for float comparisons
            **kwargs: Additional config
        """
        super().__init__(columns=columns, **kwargs)
        self.result_column = result_column
        self.expected_value = expected_value
        self.min_value = min_value
        self.max_value = max_value
        self.tolerance = tolerance

        if result_column is None and expected_value is None and min_value is None and max_value is None:
            raise ValueError(
                "At least one of 'result_column', 'expected_value', 'min_value', or 'max_value' must be provided"
            )

    @abstractmethod
    def get_computed_expression(self) -> pl.Expr:
        """Return expression computing the arithmetic result."""
        pass

    def get_validation_expression(self) -> pl.Expr:
        """Return validation expression."""
        computed = self.get_computed_expression()

        conditions = []

        if self.result_column:
            if self.tolerance > 0:
                conditions.append(
                    (computed - pl.col(self.result_column)).abs() <= self.tolerance
                )
            else:
                conditions.append(computed == pl.col(self.result_column))

        if self.expected_value is not None:
            if self.tolerance > 0:
                conditions.append((computed - self.expected_value).abs() <= self.tolerance)
            else:
                conditions.append(computed == self.expected_value)

        if self.min_value is not None:
            conditions.append(computed >= self.min_value)

        if self.max_value is not None:
            conditions.append(computed <= self.max_value)

        # Combine all conditions with AND
        result = conditions[0]
        for cond in conditions[1:]:
            result = result & cond

        return result
