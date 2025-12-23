"""Expression-based validators.

Validators using Polars expressions for flexible, high-performance validation.
"""

from typing import Any, Callable

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.query.base import ExpressionValidator
from truthound.validators.registry import register_validator


@register_validator
class CustomExpressionValidator(ExpressionValidator):
    """Validates rows using a custom Polars expression.

    The most flexible validator - accepts any Polars expression.

    Example:
        # Price must be positive when quantity > 0
        validator = CustomExpressionValidator(
            filter_expr=(pl.col("quantity") <= 0) | (pl.col("price") > 0),
            description="Price must be positive for non-zero quantities",
        )

        # Email domain check
        validator = CustomExpressionValidator(
            filter_expr=pl.col("email").str.contains("@company.com"),
            description="Email must be company domain",
        )
    """

    name = "custom_expression"
    category = "query"


@register_validator
class ConditionalExpressionValidator(Validator):
    """Validates using conditional logic (IF condition THEN check).

    Example:
        # If status is 'shipped', tracking_number must exist
        validator = ConditionalExpressionValidator(
            condition=pl.col("status") == "shipped",
            then_expr=pl.col("tracking_number").is_not_null(),
            description="Shipped orders must have tracking number",
        )
    """

    name = "conditional_expression"
    category = "query"

    def __init__(
        self,
        condition: pl.Expr,
        then_expr: pl.Expr,
        description: str = "Conditional validation",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.condition = condition
        self.then_expr = then_expr
        self.description = description

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Count rows where condition is true but then_expr is false
        result = lf.select([
            pl.len().alias("_total"),
            self.condition.sum().alias("_condition_true"),
            (self.condition & ~self.then_expr).sum().alias("_failures"),
        ]).collect()

        total = result["_total"][0]
        condition_true = result["_condition_true"][0]
        failures = result["_failures"][0]

        if failures > 0:
            if self._passes_mostly(failures, condition_true):
                return issues

            ratio = failures / condition_true if condition_true > 0 else 0

            issues.append(
                ValidationIssue(
                    column="_conditional",
                    issue_type="conditional_validation_failed",
                    count=failures,
                    severity=self._calculate_severity(ratio),
                    details=f"{failures}/{condition_true} rows failed: {self.description}",
                    expected=self.description,
                )
            )

        return issues


@register_validator
class MultiConditionValidator(Validator):
    """Validates multiple conditions with AND/OR logic.

    Example:
        # All conditions must pass
        validator = MultiConditionValidator(
            conditions=[
                (pl.col("age") >= 18, "Age must be 18+"),
                (pl.col("status").is_in(["active", "pending"]), "Valid status"),
                (pl.col("email").is_not_null(), "Email required"),
            ],
            logic="and",
        )

        # At least one condition must pass
        validator = MultiConditionValidator(
            conditions=[
                (pl.col("phone").is_not_null(), "Phone provided"),
                (pl.col("email").is_not_null(), "Email provided"),
            ],
            logic="or",
        )
    """

    name = "multi_condition"
    category = "query"

    def __init__(
        self,
        conditions: list[tuple[pl.Expr, str]],
        logic: str = "and",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.conditions = conditions
        self.logic = logic.lower()

        if self.logic not in ["and", "or"]:
            raise ValueError("logic must be 'and' or 'or'")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Build combined expression
        if self.logic == "and":
            combined = pl.lit(True)
            for expr, _ in self.conditions:
                combined = combined & expr
        else:  # or
            combined = pl.lit(False)
            for expr, _ in self.conditions:
                combined = combined | expr

        # Count failures
        result = lf.select([
            pl.len().alias("_total"),
            (~combined).sum().alias("_failures"),
        ]).collect()

        total = result["_total"][0]
        failures = result["_failures"][0]

        if failures > 0:
            if self._passes_mostly(failures, total):
                return issues

            ratio = failures / total if total > 0 else 0
            condition_names = [desc for _, desc in self.conditions]

            issues.append(
                ValidationIssue(
                    column="_multi_condition",
                    issue_type="multi_condition_failed",
                    count=failures,
                    severity=self._calculate_severity(ratio),
                    details=f"{failures}/{total} rows failed ({self.logic.upper()}): {', '.join(condition_names)}",
                    expected=f"All conditions ({self.logic})",
                )
            )

        return issues


@register_validator
class RowLevelValidator(Validator):
    """Validates each row using a custom Python function.

    For complex validation logic that can't be expressed as Polars expressions.
    Note: This is slower than expression-based validators for large datasets.

    Example:
        # Complex business logic
        def validate_order(row):
            if row["type"] == "subscription":
                return row["billing_cycle"] in ["monthly", "yearly"]
            return True

        validator = RowLevelValidator(
            row_validator=validate_order,
            description="Subscription orders need billing cycle",
        )
    """

    name = "row_level"
    category = "query"

    def __init__(
        self,
        row_validator: Callable[[dict[str, Any]], bool],
        description: str = "Row-level validation",
        columns: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.row_validator = row_validator
        self.description = description
        self.columns = columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        total = len(df)

        if total == 0:
            return issues

        # Select only needed columns if specified
        if self.columns:
            df = df.select(self.columns)

        # Validate each row
        failures = 0
        sample_indices = []

        for i in range(total):
            row = {col: df[col][i] for col in df.columns}
            try:
                if not self.row_validator(row):
                    failures += 1
                    if len(sample_indices) < self.config.sample_size:
                        sample_indices.append(i)
            except Exception:
                failures += 1
                if len(sample_indices) < self.config.sample_size:
                    sample_indices.append(i)

        if failures > 0:
            if self._passes_mostly(failures, total):
                return issues

            ratio = failures / total

            # Get sample values
            sample_values = []
            for idx in sample_indices:
                row_str = ", ".join(f"{col}={df[col][idx]}" for col in df.columns[:3])
                sample_values.append(f"Row {idx}: {row_str}")

            issues.append(
                ValidationIssue(
                    column="_row_level",
                    issue_type="row_validation_failed",
                    count=failures,
                    severity=self._calculate_severity(ratio),
                    details=f"{failures}/{total} rows failed: {self.description}",
                    expected=self.description,
                    sample_values=sample_values,
                )
            )

        return issues
