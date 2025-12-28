"""Fluent builder API for creating validators without subclassing.

This module provides a builder pattern for creating simple validators
without having to write a full class. Ideal for one-off validations
or when embedding validation logic in configuration.

Example:
    # Create a simple column validator
    validator = (
        ValidatorBuilder("positive_values")
        .category("numeric")
        .description("Checks that numeric values are positive")
        .for_numeric_columns()
        .check(lambda col, df: df.filter(pl.col(col) < 0).height)
        .with_issue_type("negative_value")
        .with_severity(Severity.MEDIUM)
        .build()
    )

    # Use like any other validator
    issues = validator.validate(lf)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorConfig,
    NUMERIC_TYPES,
    STRING_TYPES,
    DATETIME_TYPES,
    FLOAT_TYPES,
)


@dataclass
class ColumnCheck:
    """Definition of a column-level check."""

    check_fn: Callable[[str, pl.LazyFrame], int]  # Returns violation count
    issue_type: str = "check_failed"
    severity: Severity = Severity.MEDIUM
    message_template: str = "Column '{column}' has {count} violations"
    sample_fn: Callable[[str, pl.LazyFrame], list[Any]] | None = None


@dataclass
class AggregateCheck:
    """Definition of an aggregate-level check."""

    check_fn: Callable[[str, dict[str, Any]], bool]  # Returns True if check passes
    issue_type: str = "aggregate_check_failed"
    severity: Severity = Severity.MEDIUM
    message_template: str = "Column '{column}' failed aggregate check"


class ValidatorBuilder:
    """Fluent builder for creating validators.

    Provides a step-by-step API for building validators without
    needing to subclass Validator. Suitable for simple validations.

    Example:
        validator = (
            ValidatorBuilder("email_domain")
            .category("string")
            .description("Validates email domain is allowed")
            .for_string_columns()
            .check_column(
                lambda col, lf: lf.filter(
                    ~pl.col(col).str.contains("@company.com$")
                ).select(pl.len()).collect().item()
            )
            .with_issue_type("invalid_email_domain")
            .with_severity(Severity.HIGH)
            .with_message("Column '{column}' has {count} emails with non-company domains")
            .build()
        )

    Attributes:
        _name: Validator name
        _category: Validator category
        _description: Human-readable description
        _dtype_filter: Set of allowed data types
        _column_checks: List of column-level checks
        _aggregate_checks: List of aggregate-level checks
        _config: Validator configuration
    """

    def __init__(self, name: str):
        """Initialize the builder.

        Args:
            name: Unique validator name
        """
        self._name = name
        self._category = "custom"
        self._description = ""
        self._dtype_filter: set[type[pl.DataType]] | None = None
        self._column_checks: list[ColumnCheck] = []
        self._aggregate_checks: list[AggregateCheck] = []
        self._config: ValidatorConfig | None = None

        # Current check being built
        self._current_check_fn: Callable | None = None
        self._current_issue_type = "validation_failed"
        self._current_severity = Severity.MEDIUM
        self._current_message = "Validation failed for column '{column}'"
        self._current_sample_fn: Callable | None = None

    def category(self, category: str) -> "ValidatorBuilder":
        """Set the validator category.

        Args:
            category: Category name (e.g., "string", "numeric", "custom")

        Returns:
            Self for chaining
        """
        self._category = category
        return self

    def description(self, description: str) -> "ValidatorBuilder":
        """Set the validator description.

        Args:
            description: Human-readable description

        Returns:
            Self for chaining
        """
        self._description = description
        return self

    def for_columns(
        self, dtype_filter: set[type[pl.DataType]] | None = None
    ) -> "ValidatorBuilder":
        """Set the data type filter for columns.

        Args:
            dtype_filter: Set of allowed Polars data types

        Returns:
            Self for chaining
        """
        self._dtype_filter = dtype_filter
        return self

    def for_numeric_columns(self) -> "ValidatorBuilder":
        """Filter to only numeric columns.

        Returns:
            Self for chaining
        """
        self._dtype_filter = NUMERIC_TYPES
        return self

    def for_string_columns(self) -> "ValidatorBuilder":
        """Filter to only string columns.

        Returns:
            Self for chaining
        """
        self._dtype_filter = STRING_TYPES
        return self

    def for_datetime_columns(self) -> "ValidatorBuilder":
        """Filter to only datetime columns.

        Returns:
            Self for chaining
        """
        self._dtype_filter = DATETIME_TYPES
        return self

    def for_float_columns(self) -> "ValidatorBuilder":
        """Filter to only float columns.

        Returns:
            Self for chaining
        """
        self._dtype_filter = FLOAT_TYPES
        return self

    def check_column(
        self, check_fn: Callable[[str, pl.LazyFrame], int]
    ) -> "ValidatorBuilder":
        """Add a column check function.

        The check function receives (column_name, lazyframe) and should
        return the count of violations found.

        Args:
            check_fn: Function (column, lf) -> violation_count

        Returns:
            Self for chaining

        Example:
            .check_column(
                lambda col, lf: lf.filter(
                    pl.col(col).is_null()
                ).select(pl.len()).collect().item()
            )
        """
        # Finalize previous check if exists
        self._finalize_current_check()

        self._current_check_fn = check_fn
        return self

    def check(
        self, check_fn: Callable[[str, pl.LazyFrame], int]
    ) -> "ValidatorBuilder":
        """Alias for check_column.

        Args:
            check_fn: Function (column, lf) -> violation_count

        Returns:
            Self for chaining
        """
        return self.check_column(check_fn)

    def with_issue_type(self, issue_type: str) -> "ValidatorBuilder":
        """Set the issue type for the current check.

        Args:
            issue_type: Issue type identifier

        Returns:
            Self for chaining
        """
        self._current_issue_type = issue_type
        return self

    def with_severity(self, severity: Severity) -> "ValidatorBuilder":
        """Set the severity for the current check.

        Args:
            severity: Severity level

        Returns:
            Self for chaining
        """
        self._current_severity = severity
        return self

    def with_message(self, template: str) -> "ValidatorBuilder":
        """Set the message template for the current check.

        Available placeholders:
        - {column}: Column name
        - {count}: Violation count

        Args:
            template: Message template string

        Returns:
            Self for chaining
        """
        self._current_message = template
        return self

    def with_samples(
        self, sample_fn: Callable[[str, pl.LazyFrame], list[Any]]
    ) -> "ValidatorBuilder":
        """Set a function to collect sample violations.

        Args:
            sample_fn: Function (column, lf) -> list of sample values

        Returns:
            Self for chaining
        """
        self._current_sample_fn = sample_fn
        return self

    def with_config(self, config: ValidatorConfig) -> "ValidatorBuilder":
        """Set the validator configuration.

        Args:
            config: Validator configuration

        Returns:
            Self for chaining
        """
        self._config = config
        return self

    def _finalize_current_check(self) -> None:
        """Finalize the current check and add to list."""
        if self._current_check_fn is not None:
            self._column_checks.append(
                ColumnCheck(
                    check_fn=self._current_check_fn,
                    issue_type=self._current_issue_type,
                    severity=self._current_severity,
                    message_template=self._current_message,
                    sample_fn=self._current_sample_fn,
                )
            )
            # Reset for next check
            self._current_check_fn = None
            self._current_issue_type = "validation_failed"
            self._current_severity = Severity.MEDIUM
            self._current_message = "Validation failed for column '{column}'"
            self._current_sample_fn = None

    def build(self) -> Validator:
        """Build and return the validator instance.

        Returns:
            Configured Validator instance

        Raises:
            ValueError: If no checks have been defined
        """
        # Finalize any pending check
        self._finalize_current_check()

        if not self._column_checks and not self._aggregate_checks:
            raise ValueError("At least one check must be defined")

        # Create the validator class dynamically
        builder = self

        class BuiltValidator(Validator):
            name = builder._name
            category = builder._category

            def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
                super().__init__(config or builder._config, **kwargs)
                self._column_checks = builder._column_checks
                self._dtype_filter = builder._dtype_filter

            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                issues: list[ValidationIssue] = []
                columns = self._get_target_columns(lf, self._dtype_filter)

                for col in columns:
                    for check in self._column_checks:
                        try:
                            count = check.check_fn(col, lf)
                            if count > 0:
                                samples = None
                                if check.sample_fn:
                                    try:
                                        samples = check.sample_fn(col, lf)
                                    except Exception:
                                        pass

                                issues.append(
                                    ValidationIssue(
                                        column=col,
                                        issue_type=check.issue_type,
                                        count=count,
                                        severity=check.severity,
                                        details=check.message_template.format(
                                            column=col, count=count
                                        ),
                                        sample_values=samples,
                                    )
                                )
                        except Exception as e:
                            if self.config.graceful_degradation:
                                self.logger.warning(
                                    f"Check failed for column {col}: {e}"
                                )
                            else:
                                raise

                return issues

        # Set docstring
        BuiltValidator.__doc__ = builder._description

        return BuiltValidator(self._config)


class ColumnCheckBuilder:
    """Builder for individual column checks.

    Example:
        check = (
            ColumnCheckBuilder()
            .violation_filter(pl.col("value") < 0)
            .issue_type("negative_value")
            .severity(Severity.HIGH)
            .message("Found {count} negative values in '{column}'")
            .build()
        )
    """

    def __init__(self) -> None:
        self._filter_expr: pl.Expr | None = None
        self._issue_type = "validation_failed"
        self._severity = Severity.MEDIUM
        self._message = "Validation failed for column '{column}'"

    def violation_filter(self, expr: pl.Expr) -> "ColumnCheckBuilder":
        """Set the filter expression for violations.

        Args:
            expr: Polars expression that evaluates to True for violations

        Returns:
            Self for chaining
        """
        self._filter_expr = expr
        return self

    def issue_type(self, issue_type: str) -> "ColumnCheckBuilder":
        """Set the issue type.

        Args:
            issue_type: Issue type identifier

        Returns:
            Self for chaining
        """
        self._issue_type = issue_type
        return self

    def severity(self, severity: Severity) -> "ColumnCheckBuilder":
        """Set the severity.

        Args:
            severity: Severity level

        Returns:
            Self for chaining
        """
        self._severity = severity
        return self

    def message(self, template: str) -> "ColumnCheckBuilder":
        """Set the message template.

        Args:
            template: Message template with {column} and {count} placeholders

        Returns:
            Self for chaining
        """
        self._message = template
        return self

    def build(self) -> ColumnCheck:
        """Build the column check.

        Returns:
            ColumnCheck instance

        Raises:
            ValueError: If no filter expression was set
        """
        if self._filter_expr is None:
            raise ValueError("violation_filter must be set")

        # Capture for closure
        expr = self._filter_expr

        def check_fn(col: str, lf: pl.LazyFrame) -> int:
            # Replace generic column reference with specific column
            col_expr = expr.meta.pop().alias(col) if hasattr(expr, "meta") else expr
            try:
                # Try to use the expression directly
                count = lf.filter(expr).select(pl.len()).collect().item()
            except Exception:
                # Fallback: assume expr uses pl.col() generically
                count = lf.filter(expr).select(pl.len()).collect().item()
            return count

        return ColumnCheck(
            check_fn=check_fn,
            issue_type=self._issue_type,
            severity=self._severity,
            message_template=self._message,
        )


class AggregateCheckBuilder:
    """Builder for aggregate-level checks.

    Example:
        check = (
            AggregateCheckBuilder()
            .check(lambda col, stats: stats["mean"] > 0)
            .issue_type("non_positive_mean")
            .severity(Severity.MEDIUM)
            .message("Column '{column}' has non-positive mean")
            .build()
        )
    """

    def __init__(self) -> None:
        self._check_fn: Callable[[str, dict[str, Any]], bool] | None = None
        self._issue_type = "aggregate_check_failed"
        self._severity = Severity.MEDIUM
        self._message = "Aggregate check failed for column '{column}'"

    def check(
        self, check_fn: Callable[[str, dict[str, Any]], bool]
    ) -> "AggregateCheckBuilder":
        """Set the check function.

        The function receives (column_name, stats_dict) where stats_dict
        contains: mean, std, min, max, sum, median, count.

        Should return True if check passes, False if it fails.

        Args:
            check_fn: Function (column, stats) -> passes

        Returns:
            Self for chaining
        """
        self._check_fn = check_fn
        return self

    def issue_type(self, issue_type: str) -> "AggregateCheckBuilder":
        """Set the issue type.

        Args:
            issue_type: Issue type identifier

        Returns:
            Self for chaining
        """
        self._issue_type = issue_type
        return self

    def severity(self, severity: Severity) -> "AggregateCheckBuilder":
        """Set the severity.

        Args:
            severity: Severity level

        Returns:
            Self for chaining
        """
        self._severity = severity
        return self

    def message(self, template: str) -> "AggregateCheckBuilder":
        """Set the message template.

        Args:
            template: Message template with {column} placeholder

        Returns:
            Self for chaining
        """
        self._message = template
        return self

    def build(self) -> AggregateCheck:
        """Build the aggregate check.

        Returns:
            AggregateCheck instance

        Raises:
            ValueError: If no check function was set
        """
        if self._check_fn is None:
            raise ValueError("check function must be set")

        return AggregateCheck(
            check_fn=self._check_fn,
            issue_type=self._issue_type,
            severity=self._severity,
            message_template=self._message,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def simple_column_validator(
    name: str,
    check_fn: Callable[[str, pl.LazyFrame], int],
    issue_type: str = "validation_failed",
    severity: Severity = Severity.MEDIUM,
    category: str = "custom",
    dtype_filter: set[type[pl.DataType]] | None = None,
) -> Validator:
    """Create a simple column validator in one call.

    Args:
        name: Validator name
        check_fn: Function (column, lf) -> violation_count
        issue_type: Issue type identifier
        severity: Severity level
        category: Validator category
        dtype_filter: Optional data type filter

    Returns:
        Configured Validator instance

    Example:
        validator = simple_column_validator(
            name="no_nulls",
            check_fn=lambda col, lf: lf.filter(
                pl.col(col).is_null()
            ).select(pl.len()).collect().item(),
            issue_type="null_value",
            severity=Severity.HIGH,
        )
    """
    builder = ValidatorBuilder(name).category(category)

    if dtype_filter:
        builder.for_columns(dtype_filter)

    return (
        builder.check_column(check_fn)
        .with_issue_type(issue_type)
        .with_severity(severity)
        .build()
    )


def simple_expression_validator(
    name: str,
    violation_expr: pl.Expr,
    issue_type: str = "validation_failed",
    severity: Severity = Severity.MEDIUM,
    category: str = "custom",
    columns: list[str] | None = None,
) -> Validator:
    """Create a validator from a Polars expression.

    The expression should evaluate to True for rows that are violations.

    Args:
        name: Validator name
        violation_expr: Expression that is True for violations
        issue_type: Issue type identifier
        severity: Severity level
        category: Validator category
        columns: Specific columns to check (None = all)

    Returns:
        Configured Validator instance

    Example:
        validator = simple_expression_validator(
            name="positive_values",
            violation_expr=pl.col("amount") <= 0,
            issue_type="non_positive",
            severity=Severity.HIGH,
            columns=["amount", "quantity"],
        )
    """

    def check_fn(col: str, lf: pl.LazyFrame) -> int:
        return lf.filter(violation_expr).select(pl.len()).collect().item()

    config = ValidatorConfig(columns=tuple(columns)) if columns else None

    return (
        ValidatorBuilder(name)
        .category(category)
        .with_config(config)  # type: ignore
        .check_column(check_fn)
        .with_issue_type(issue_type)
        .with_severity(severity)
        .build()
    )
