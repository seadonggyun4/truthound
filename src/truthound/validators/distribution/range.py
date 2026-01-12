"""Range and boundary validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    ValidationExpressionSpec,
    Validator,
    NumericValidatorMixin,
    ExpressionValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class BetweenValidator(Validator, NumericValidatorMixin, ExpressionValidatorMixin):
    """Validates that numeric values are within a specified range.

    Uses expression-based architecture for optimal performance when
    combined with other validators via ExpressionBatchExecutor.

    Example:
        # Standalone usage
        validator = BetweenValidator(min_value=0, max_value=100)
        issues = validator.validate(lf)

        # Batched usage
        from truthound.validators.base import ExpressionBatchExecutor
        executor = ExpressionBatchExecutor()
        executor.add_validator(BetweenValidator(min_value=0, max_value=100))
        executor.add_validator(NullValidator())
        all_issues = executor.execute(lf)  # Single collect()!
    """

    name = "between"
    category = "distribution"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def _build_out_of_range_expr(self, col: str) -> pl.Expr:
        """Build expression for out-of-range values."""
        if self.inclusive:
            below = (pl.col(col) < self.min_value) if self.min_value is not None else pl.lit(False)
            above = (pl.col(col) > self.max_value) if self.max_value is not None else pl.lit(False)
        else:
            below = (pl.col(col) <= self.min_value) if self.min_value is not None else pl.lit(False)
            above = (pl.col(col) >= self.max_value) if self.max_value is not None else pl.lit(False)

        return (below | above) & pl.col(col).is_not_null()

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for range checking."""
        range_str = f"[{self.min_value}, {self.max_value}]"
        specs = []
        for col in columns:
            out_of_range_expr = self._build_out_of_range_expr(col)
            specs.append(
                ValidationExpressionSpec(
                    column=col,
                    validator_name=self.name,
                    issue_type="out_of_range",
                    count_expr=out_of_range_expr.sum(),
                    non_null_expr=pl.len(),
                    severity_ratio_thresholds=(0.1, 0.05, 0.01),
                    details_template=f"{{count}} values outside {range_str}",
                    expected=range_str,
                )
            )
        return specs

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach."""
        columns = self._get_numeric_columns(lf)
        return self._validate_with_expressions(lf, columns=columns)


@register_validator
class RangeValidator(Validator, NumericValidatorMixin, ExpressionValidatorMixin):
    """Auto-detects expected ranges based on column names.

    Uses expression-based architecture for optimal performance.
    """

    name = "range"
    category = "distribution"

    KNOWN_RANGES: dict[str, tuple[float | None, float | None]] = {
        "age": (0, 150),
        "price": (0, None),
        "quantity": (0, None),
        "count": (0, None),
        "amount": (0, None),
        "percentage": (0, 100),
        "percent": (0, 100),
        "pct": (0, 100),
        "rate": (0, 100),
        "score": (0, 100),
        "rating": (0, 5),
        "year": (1900, 2100),
        "month": (1, 12),
        "day": (1, 31),
        "hour": (0, 23),
        "minute": (0, 59),
        "second": (0, 59),
    }

    def _get_column_range(self, col: str) -> tuple[float | None, float | None] | None:
        """Get known range for a column based on its name."""
        col_lower = col.lower()
        for pattern, range_tuple in self.KNOWN_RANGES.items():
            if pattern in col_lower:
                return range_tuple
        return None

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for auto-detected range checking."""
        specs = []
        for col in columns:
            range_tuple = self._get_column_range(col)
            if range_tuple is None:
                continue

            min_val, max_val = range_tuple

            # Build out-of-range expression
            below_min = (pl.col(col) < min_val) if min_val is not None else pl.lit(False)
            above_max = (pl.col(col) > max_val) if max_val is not None else pl.lit(False)
            oob_expr = (below_min | above_max) & pl.col(col).is_not_null()

            # Build range description
            if min_val is not None and max_val is not None:
                range_desc = f"[{min_val}, {max_val}]"
            elif min_val is not None:
                range_desc = f">= {min_val}"
            else:
                range_desc = f"<= {max_val}"

            specs.append(
                ValidationExpressionSpec(
                    column=col,
                    validator_name=self.name,
                    issue_type="out_of_range",
                    count_expr=oob_expr.sum(),
                    non_null_expr=pl.len(),
                    severity_ratio_thresholds=(0.1, 0.05, 0.01),
                    details_template=f"Expected {range_desc}",
                    expected=range_desc,
                )
            )
        return specs

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach."""
        columns = self._get_numeric_columns(lf)
        return self._validate_with_expressions(lf, columns=columns)


@register_validator
class PositiveValidator(Validator, NumericValidatorMixin, ExpressionValidatorMixin):
    """Validates that numeric values are positive (> 0).

    Uses expression-based architecture for optimal performance.
    """

    name = "positive"
    category = "distribution"

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for positive value checking."""
        specs = []
        for col in columns:
            non_positive_expr = (pl.col(col) <= 0) & pl.col(col).is_not_null()
            specs.append(
                ValidationExpressionSpec(
                    column=col,
                    validator_name=self.name,
                    issue_type="not_positive",
                    count_expr=non_positive_expr.sum(),
                    non_null_expr=pl.len(),
                    details_template="{count} non-positive values",
                    expected="> 0",
                )
            )
        return specs

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach."""
        columns = self._get_numeric_columns(lf)
        return self._validate_with_expressions(lf, columns=columns)


@register_validator
class NonNegativeValidator(Validator, NumericValidatorMixin, ExpressionValidatorMixin):
    """Validates that numeric values are non-negative (>= 0).

    Uses expression-based architecture for optimal performance.
    """

    name = "non_negative"
    category = "distribution"

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for non-negative value checking."""
        specs = []
        for col in columns:
            negative_expr = (pl.col(col) < 0) & pl.col(col).is_not_null()
            specs.append(
                ValidationExpressionSpec(
                    column=col,
                    validator_name=self.name,
                    issue_type="negative",
                    count_expr=negative_expr.sum(),
                    non_null_expr=pl.len(),
                    details_template="{count} negative values",
                    expected=">= 0",
                )
            )
        return specs

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach."""
        columns = self._get_numeric_columns(lf)
        return self._validate_with_expressions(lf, columns=columns)
