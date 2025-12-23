"""Multi-column consistency validators.

Validators for checking logical consistency across multiple columns.
"""

from typing import Any, Callable

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.multi_column.base import MultiColumnValidator
from truthound.validators.registry import register_validator


@register_validator
class ColumnConsistencyValidator(Validator):
    """Validates logical consistency between columns using rules.

    Example:
        # If status is 'shipped', ship_date must not be null
        validator = ColumnConsistencyValidator(
            rules=[
                {
                    "when": pl.col("status") == "shipped",
                    "then": pl.col("ship_date").is_not_null(),
                    "description": "Shipped orders need ship date",
                },
                {
                    "when": pl.col("status") == "cancelled",
                    "then": pl.col("cancel_reason").is_not_null(),
                    "description": "Cancelled orders need reason",
                },
            ]
        )
    """

    name = "column_consistency"
    category = "multi_column"

    def __init__(
        self,
        rules: list[dict[str, Any]],
        fail_on_first: bool = False,
        **kwargs: Any,
    ):
        """Initialize consistency validator.

        Args:
            rules: List of rule dicts with 'when', 'then', and optional 'description'
            fail_on_first: Stop after first failing rule
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.rules = rules
        self.fail_on_first = fail_on_first

        # Validate rules structure
        for i, rule in enumerate(rules):
            if "when" not in rule or "then" not in rule:
                raise ValueError(f"Rule {i} must have 'when' and 'then' keys")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for i, rule in enumerate(self.rules):
            when_expr = rule["when"]
            then_expr = rule["then"]
            description = rule.get("description", f"Rule {i}")

            # Count rows where condition is true but then is false
            result = lf.select([
                pl.len().alias("_total"),
                when_expr.sum().alias("_when_true"),
                (when_expr & ~then_expr).sum().alias("_failures"),
            ]).collect()

            when_true = result["_when_true"][0]
            failures = result["_failures"][0]

            if failures > 0:
                if self._passes_mostly(failures, when_true):
                    continue

                ratio = failures / when_true if when_true > 0 else 0

                issues.append(
                    ValidationIssue(
                        column="_consistency",
                        issue_type="consistency_rule_failed",
                        count=failures,
                        severity=self._calculate_severity(ratio),
                        details=f"Rule '{description}' failed: {failures}/{when_true} rows",
                        expected=description,
                    )
                )

                if self.fail_on_first:
                    break

        return issues


@register_validator
class ColumnMutualExclusivityValidator(MultiColumnValidator):
    """Validates that columns are mutually exclusive (only one can have value).

    Example:
        # Only one payment method should be set
        validator = ColumnMutualExclusivityValidator(
            columns=["credit_card", "bank_transfer", "paypal"],
            allow_none=True,  # Allow none to be set
        )
    """

    name = "column_mutual_exclusivity"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        allow_none: bool = False,
        allow_multiple: int = 1,
        **kwargs: Any,
    ):
        """Initialize mutual exclusivity validator.

        Args:
            columns: Columns to check
            allow_none: Allow all columns to be null/empty
            allow_multiple: Number of columns that can have values (default 1)
            **kwargs: Additional config
        """
        super().__init__(columns=columns, **kwargs)
        self.allow_none = allow_none
        self.allow_multiple = allow_multiple

    def get_validation_expression(self) -> pl.Expr:
        """Return mutual exclusivity expression."""
        # Count non-null columns
        non_null_count = pl.lit(0)
        for col in self.columns:
            non_null_count = non_null_count + pl.col(col).is_not_null().cast(pl.Int32)

        if self.allow_none:
            return non_null_count <= self.allow_multiple
        else:
            return (non_null_count >= 1) & (non_null_count <= self.allow_multiple)

    def _get_issue_type(self) -> str:
        return "mutual_exclusivity_violated"

    def _get_expected(self) -> str:
        if self.allow_multiple == 1:
            if self.allow_none:
                return f"At most one of {self.columns} should have value"
            return f"Exactly one of {self.columns} should have value"
        return f"At most {self.allow_multiple} of {self.columns} should have values"


@register_validator
class ColumnCoexistenceValidator(MultiColumnValidator):
    """Validates that columns either all have values or all are null.

    Example:
        # Address fields should all be filled or all empty
        validator = ColumnCoexistenceValidator(
            columns=["street", "city", "zip_code", "country"],
        )
    """

    name = "column_coexistence"
    category = "multi_column"

    def get_validation_expression(self) -> pl.Expr:
        """Return coexistence expression."""
        # All null or all non-null
        null_exprs = [pl.col(c).is_null() for c in self.columns]
        non_null_exprs = [pl.col(c).is_not_null() for c in self.columns]

        all_null = null_exprs[0]
        all_non_null = non_null_exprs[0]

        for i in range(1, len(self.columns)):
            all_null = all_null & null_exprs[i]
            all_non_null = all_non_null & non_null_exprs[i]

        return all_null | all_non_null

    def _get_issue_type(self) -> str:
        return "coexistence_violated"

    def _get_expected(self) -> str:
        return f"Columns {self.columns} should all have values or all be null"


@register_validator
class ColumnDependencyValidator(Validator):
    """Validates that certain columns are filled when a condition is met.

    Example:
        # When type is 'subscription', billing fields are required
        validator = ColumnDependencyValidator(
            condition_column="type",
            condition_value="subscription",
            required_columns=["billing_cycle", "next_billing_date"],
        )
    """

    name = "column_dependency"
    category = "multi_column"

    def __init__(
        self,
        condition_column: str,
        condition_value: Any | list[Any],
        required_columns: list[str],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.condition_column = condition_column
        self.condition_values = (
            condition_value if isinstance(condition_value, list) else [condition_value]
        )
        self.required_columns = required_columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Build condition expression
        condition = pl.col(self.condition_column).is_in(self.condition_values)

        # Check each required column
        for req_col in self.required_columns:
            result = lf.select([
                condition.sum().alias("_condition_true"),
                (condition & pl.col(req_col).is_null()).sum().alias("_missing"),
            ]).collect()

            condition_true = result["_condition_true"][0]
            missing = result["_missing"][0]

            if missing > 0:
                if self._passes_mostly(missing, condition_true):
                    continue

                ratio = missing / condition_true if condition_true > 0 else 0

                issues.append(
                    ValidationIssue(
                        column=req_col,
                        issue_type="column_dependency_violated",
                        count=missing,
                        severity=self._calculate_severity(ratio),
                        details=f"Column '{req_col}' is null when {self.condition_column} in {self.condition_values}",
                        expected=f"'{req_col}' required when {self.condition_column} in {self.condition_values}",
                    )
                )

        return issues


@register_validator
class ColumnImplicationValidator(MultiColumnValidator):
    """Validates implication: if column A has value X, column B must have value Y.

    Example:
        # If country is 'US', currency must be 'USD'
        validator = ColumnImplicationValidator(
            antecedent_column="country",
            antecedent_value="US",
            consequent_column="currency",
            consequent_value="USD",
        )
    """

    name = "column_implication"
    category = "multi_column"

    def __init__(
        self,
        antecedent_column: str,
        antecedent_value: Any | list[Any],
        consequent_column: str,
        consequent_value: Any | list[Any],
        **kwargs: Any,
    ):
        super().__init__(columns=[antecedent_column, consequent_column], **kwargs)
        self.antecedent_column = antecedent_column
        self.antecedent_values = (
            antecedent_value if isinstance(antecedent_value, list) else [antecedent_value]
        )
        self.consequent_column = consequent_column
        self.consequent_values = (
            consequent_value if isinstance(consequent_value, list) else [consequent_value]
        )

    def get_validation_expression(self) -> pl.Expr:
        """Return implication expression (A -> B is equivalent to ~A | B)."""
        antecedent = pl.col(self.antecedent_column).is_in(self.antecedent_values)
        consequent = pl.col(self.consequent_column).is_in(self.consequent_values)

        return ~antecedent | consequent

    def _get_issue_type(self) -> str:
        return "column_implication_violated"

    def _get_expected(self) -> str:
        return (
            f"If {self.antecedent_column} in {self.antecedent_values}, "
            f"then {self.consequent_column} must be in {self.consequent_values}"
        )
