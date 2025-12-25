"""Example custom validator plugin.

This plugin demonstrates how to create custom validators and register
them with Truthound through the plugin system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from truthound.plugins import (
    ValidatorPlugin,
    PluginInfo,
    PluginType,
    PluginConfig,
)
from truthound.validators.base import Validator, ValidatorConfig, ValidationIssue
from truthound.types import Severity


# =============================================================================
# Custom Validators
# =============================================================================


@dataclass
class BusinessRuleConfig(ValidatorConfig):
    """Configuration for business rule validator."""

    min_order_value: float = 0.0
    max_order_value: float = 1_000_000.0
    allowed_statuses: list[str] | None = None


class PositiveValueValidator(Validator):
    """Validates that numeric columns contain only positive values.

    This is useful for columns like prices, quantities, or amounts
    that should never be negative.

    Example:
        >>> validator = PositiveValueValidator()
        >>> issues = validator.validate(lf)
    """

    name = "positive_value"
    category = "business"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for non-positive values in numeric columns."""
        issues: list[ValidationIssue] = []
        df = lf.collect()

        # Get numeric columns
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            )
        ]

        for col in self._get_target_columns(lf):
            if col not in numeric_cols:
                continue

            # Count non-positive values
            non_positive = df.filter(pl.col(col) <= 0)
            count = len(non_positive)

            if count > 0:
                # Get sample values
                samples = (
                    non_positive[col]
                    .head(self.config.sample_size if self.config else 5)
                    .to_list()
                )

                issues.append(ValidationIssue(
                    column=col,
                    issue_type=self.name,
                    count=count,
                    severity=self._calculate_severity(
                        count / len(df),
                        (0.01, 0.05, 0.10),
                    ),
                    details=f"Found {count} non-positive values",
                    expected="> 0",
                    actual=f"{count} values <= 0",
                    sample_values=samples,
                ))

        return issues


class ConsistencyValidator(Validator):
    """Validates consistency between related columns.

    Checks that related columns have consistent values. For example,
    if status is 'shipped', then ship_date should not be null.

    Configuration:
        rules: List of consistency rules as dicts with:
            - when: Column and condition (e.g., {"column": "status", "equals": "shipped"})
            - then: Column and condition to check (e.g., {"column": "ship_date", "not_null": True})

    Example:
        >>> config = ValidatorConfig(settings={
        ...     "rules": [
        ...         {
        ...             "when": {"column": "status", "equals": "shipped"},
        ...             "then": {"column": "ship_date", "not_null": True}
        ...         }
        ...     ]
        ... })
        >>> validator = ConsistencyValidator(config)
    """

    name = "consistency"
    category = "business"

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._rules = kwargs.get("rules") or (
            self.config.settings.get("rules", []) if self.config else []
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check consistency rules."""
        issues: list[ValidationIssue] = []
        df = lf.collect()

        for rule in self._rules:
            when = rule.get("when", {})
            then = rule.get("then", {})

            when_col = when.get("column")
            then_col = then.get("column")

            if not when_col or not then_col:
                continue

            if when_col not in df.columns or then_col not in df.columns:
                continue

            # Build when condition
            when_mask = self._build_condition_mask(df, when)

            # Check then condition on filtered rows
            filtered = df.filter(when_mask)
            if len(filtered) == 0:
                continue

            violations = self._check_then_condition(filtered, then)
            count = len(violations)

            if count > 0:
                issues.append(ValidationIssue(
                    column=then_col,
                    issue_type=self.name,
                    count=count,
                    severity=Severity.MEDIUM,
                    details=(
                        f"When {when_col}={when.get('equals', '?')}, "
                        f"{then_col} should {'not be null' if then.get('not_null') else 'satisfy condition'}"
                    ),
                    expected=str(then),
                    actual=f"{count} violations",
                ))

        return issues

    def _build_condition_mask(self, df: pl.DataFrame, condition: dict) -> pl.Expr:
        """Build a Polars expression from condition dict."""
        col = condition.get("column")

        if "equals" in condition:
            return pl.col(col) == condition["equals"]
        elif "not_equals" in condition:
            return pl.col(col) != condition["not_equals"]
        elif "in" in condition:
            return pl.col(col).is_in(condition["in"])
        elif "not_null" in condition:
            return pl.col(col).is_not_null()

        return pl.lit(True)

    def _check_then_condition(self, df: pl.DataFrame, condition: dict) -> pl.DataFrame:
        """Check then condition and return violations."""
        col = condition.get("column")

        if condition.get("not_null"):
            return df.filter(pl.col(col).is_null())
        elif "equals" in condition:
            return df.filter(pl.col(col) != condition["equals"])
        elif "in" in condition:
            return df.filter(~pl.col(col).is_in(condition["in"]))

        return pl.DataFrame()


class BusinessHoursValidator(Validator):
    """Validates that datetime values fall within business hours.

    Useful for validating transaction times, order timestamps, etc.

    Configuration:
        start_hour: Start of business hours (default: 9)
        end_hour: End of business hours (default: 17)
        include_weekends: Whether weekends are valid (default: False)
    """

    name = "business_hours"
    category = "business"

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        settings = self.config.settings if self.config else {}
        self._start_hour = kwargs.get("start_hour", settings.get("start_hour", 9))
        self._end_hour = kwargs.get("end_hour", settings.get("end_hour", 17))
        self._include_weekends = kwargs.get(
            "include_weekends",
            settings.get("include_weekends", False),
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check datetime values are within business hours."""
        issues: list[ValidationIssue] = []
        df = lf.collect()

        # Get datetime columns
        datetime_cols = [
            col for col in df.columns
            if df[col].dtype in (pl.Datetime, pl.Date)
        ]

        for col in self._get_target_columns(lf):
            if col not in datetime_cols:
                continue

            # Extract hour and weekday
            try:
                hours = df[col].dt.hour()
                weekdays = df[col].dt.weekday()

                # Check business hours
                outside_hours = (hours < self._start_hour) | (hours >= self._end_hour)

                # Check weekends if not included
                if not self._include_weekends:
                    on_weekend = weekdays >= 5
                    violations = outside_hours | on_weekend
                else:
                    violations = outside_hours

                count = violations.sum()

                if count and count > 0:
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type=self.name,
                        count=count,
                        severity=Severity.LOW,
                        details=(
                            f"Found {count} values outside business hours "
                            f"({self._start_hour}:00-{self._end_hour}:00)"
                        ),
                        expected=f"Business hours: {self._start_hour}:00-{self._end_hour}:00",
                        actual=f"{count} outside business hours",
                    ))

            except Exception:
                # Skip if datetime operations fail
                pass

        return issues


# =============================================================================
# Plugin Class
# =============================================================================


class CustomValidatorPlugin(ValidatorPlugin):
    """Plugin that provides custom business validation rules.

    This plugin adds three custom validators:
    - positive_value: Ensures numeric values are positive
    - consistency: Validates relationships between columns
    - business_hours: Validates timestamps are within business hours

    Example:
        >>> from truthound.plugins import PluginManager
        >>> manager = PluginManager()
        >>> manager.load_from_class(CustomValidatorPlugin)
    """

    def _get_plugin_name(self) -> str:
        return "custom-validators"

    def _get_plugin_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return "Custom business validation rules for Truthound"

    def get_validators(self) -> list[type]:
        """Return list of validator classes."""
        return [
            PositiveValueValidator,
            ConsistencyValidator,
            BusinessHoursValidator,
        ]
