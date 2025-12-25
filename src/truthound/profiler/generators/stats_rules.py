"""Statistics-based rule generator.

Generates validation rules based on:
- Distribution statistics (mean, std, min, max)
- Uniqueness metrics
- Value frequency patterns
"""

from __future__ import annotations

from typing import Any

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    Strictness,
    TableProfile,
)
from truthound.profiler.generators.base import (
    DEFAULT_THRESHOLDS,
    GeneratedRule,
    RuleBuilder,
    RuleCategory,
    RuleConfidence,
    RuleGenerator,
    register_generator,
)


@register_generator("stats")
class StatsRuleGenerator(RuleGenerator):
    """Generates statistical validation rules.

    This generator creates rules that validate:
    - Numeric ranges (min/max)
    - Distribution properties (mean, std)
    - Uniqueness constraints
    - Value set membership
    """

    name = "stats"
    description = "Generates statistical validation rules"
    categories = {RuleCategory.DISTRIBUTION, RuleCategory.UNIQUENESS}
    priority = 80

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Table-level rules
        if profile.duplicate_row_ratio > 0:
            rules.extend(self._duplicate_row_rules(profile, strictness))

        # Column-level rules
        for column in profile.columns:
            rules.extend(self.generate_for_column(column, strictness))

        return rules

    def generate_for_column(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Uniqueness rules
        rules.extend(self._uniqueness_rules(column, strictness))

        # Distribution rules (for numeric columns)
        if column.distribution:
            rules.extend(self._distribution_rules(column, strictness))

        # Categorical rules (for low-cardinality columns)
        if self._is_categorical(column):
            rules.extend(self._categorical_rules(column, strictness))

        return rules

    def _is_categorical(self, column: ColumnProfile) -> bool:
        """Check if column appears to be categorical."""
        # Low distinct count relative to row count
        if column.row_count == 0:
            return False

        distinct_ratio = column.distinct_count / column.row_count
        return (
            distinct_ratio < 0.05 and  # Less than 5% unique values
            column.distinct_count <= 100 and  # Not too many categories
            column.inferred_type in {DataType.STRING, DataType.CATEGORICAL}
        )

    def _duplicate_row_rules(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate rules for duplicate row handling."""
        rules: list[GeneratedRule] = []

        if profile.duplicate_row_ratio > 0:
            max_duplicate_ratio = {
                Strictness.LOOSE: 0.10,
                Strictness.MEDIUM: 0.05,
                Strictness.STRICT: 0.01,
            }[strictness]

            if profile.duplicate_row_ratio <= max_duplicate_ratio:
                # Current duplicate ratio is acceptable
                rules.append(
                    RuleBuilder("max_duplicate_rows")
                    .validator("DuplicateValidator")
                    .category(RuleCategory.UNIQUENESS)
                    .params(max_duplicate_ratio=max_duplicate_ratio)
                    .confidence(RuleConfidence.MEDIUM)
                    .description(
                        f"Duplicate row ratio should not exceed {max_duplicate_ratio*100}%"
                    )
                    .rationale(
                        f"Observed {profile.duplicate_row_ratio*100:.2f}% duplicate rows"
                    )
                    .build()
                )

        return rules

    def _uniqueness_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate uniqueness-related rules."""
        rules: list[GeneratedRule] = []

        if column.is_unique:
            # Column is completely unique - likely a primary key
            rules.append(
                RuleBuilder(f"unique_{column.name}")
                .validator("UniqueValidator")
                .category(RuleCategory.UNIQUENESS)
                .column(column.name)
                .confidence(RuleConfidence.HIGH)
                .description(f"Column '{column.name}' must have unique values")
                .rationale("100% unique values observed")
                .build()
            )
        elif column.unique_ratio >= DEFAULT_THRESHOLDS.get_unique_threshold(strictness):
            # High uniqueness - enforce minimum ratio
            min_ratio = column.unique_ratio * 0.95  # Allow 5% degradation

            rules.append(
                RuleBuilder(f"unique_ratio_{column.name}")
                .validator("UniqueRatioValidator")
                .category(RuleCategory.UNIQUENESS)
                .column(column.name)
                .params(min_ratio=round(min_ratio, 3))
                .confidence(RuleConfidence.MEDIUM)
                .description(
                    f"Column '{column.name}' should have at least {min_ratio*100:.1f}% unique values"
                )
                .rationale(f"Observed {column.unique_ratio*100:.1f}% unique")
                .build()
            )

        # Constant value detection
        if column.is_constant and strictness == Strictness.STRICT:
            if column.top_values and len(column.top_values) > 0:
                constant_value = column.top_values[0].value
                rules.append(
                    RuleBuilder(f"constant_{column.name}")
                    .validator("InSetValidator")
                    .category(RuleCategory.DISTRIBUTION)
                    .column(column.name)
                    .params(allowed_values=[constant_value])
                    .confidence(RuleConfidence.HIGH)
                    .description(f"Column '{column.name}' should only contain: {constant_value}")
                    .rationale("Column has constant value")
                    .build()
                )

        return rules

    def _distribution_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate distribution-based rules for numeric columns."""
        rules: list[GeneratedRule] = []
        dist = column.distribution

        if not dist:
            return rules

        # Range validation
        if dist.min is not None and dist.max is not None:
            tolerance = DEFAULT_THRESHOLDS.get_range_tolerance(strictness)
            range_width = dist.max - dist.min

            # Expand range by tolerance
            min_val = dist.min - (range_width * tolerance)
            max_val = dist.max + (range_width * tolerance)

            # Round for cleaner rules
            if isinstance(dist.min, float):
                min_val = round(min_val, 2)
                max_val = round(max_val, 2)

            rules.append(
                RuleBuilder(f"range_{column.name}")
                .validator("RangeValidator")
                .category(RuleCategory.DISTRIBUTION)
                .column(column.name)
                .params(min_value=min_val, max_value=max_val)
                .confidence(RuleConfidence.MEDIUM)
                .description(f"Column '{column.name}' values should be between {min_val} and {max_val}")
                .rationale(f"Observed range: [{dist.min}, {dist.max}]")
                .build()
            )

        # Positive value check
        if dist.min is not None and dist.min >= 0:
            rules.append(
                RuleBuilder(f"non_negative_{column.name}")
                .validator("NonNegativeValidator")
                .category(RuleCategory.DISTRIBUTION)
                .column(column.name)
                .confidence(RuleConfidence.HIGH)
                .description(f"Column '{column.name}' values should be non-negative")
                .rationale(f"Minimum observed value: {dist.min}")
                .build()
            )

            if dist.min > 0:
                rules.append(
                    RuleBuilder(f"positive_{column.name}")
                    .validator("PositiveValidator")
                    .category(RuleCategory.DISTRIBUTION)
                    .column(column.name)
                    .confidence(RuleConfidence.HIGH)
                    .description(f"Column '{column.name}' values should be positive")
                    .rationale(f"Minimum observed value: {dist.min}")
                    .build()
                )

        # Mean validation (for strict mode)
        if strictness == Strictness.STRICT and dist.mean is not None and dist.std is not None:
            # Allow mean to vary within 2 standard deviations
            mean_min = dist.mean - 2 * dist.std
            mean_max = dist.mean + 2 * dist.std

            rules.append(
                RuleBuilder(f"mean_range_{column.name}")
                .validator("MeanBetweenValidator")
                .category(RuleCategory.DISTRIBUTION)
                .column(column.name)
                .params(
                    min_value=round(mean_min, 2),
                    max_value=round(mean_max, 2),
                )
                .confidence(RuleConfidence.MEDIUM)
                .description(
                    f"Column '{column.name}' mean should be between {mean_min:.2f} and {mean_max:.2f}"
                )
                .rationale(f"Observed mean: {dist.mean:.2f}, std: {dist.std:.2f}")
                .build()
            )

        return rules

    def _categorical_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate rules for categorical columns."""
        rules: list[GeneratedRule] = []

        if not column.top_values:
            return rules

        # Get all observed values (from top values)
        # For truly categorical columns, top_values should cover most/all values
        observed_values = [v.value for v in column.top_values]

        # Only create InSet rule if we have reasonable coverage
        total_covered = sum(v.ratio for v in column.top_values)

        if total_covered >= 0.95:  # 95% coverage
            rules.append(
                RuleBuilder(f"in_set_{column.name}")
                .validator("InSetValidator")
                .category(RuleCategory.DISTRIBUTION)
                .column(column.name)
                .params(allowed_values=observed_values)
                .confidence(RuleConfidence.MEDIUM)
                .description(f"Column '{column.name}' values should be in: {observed_values}")
                .rationale(f"Top values cover {total_covered*100:.1f}% of data")
                .build()
            )

        # Distinct count validation
        rules.append(
            RuleBuilder(f"distinct_count_{column.name}")
            .validator("DistinctCountBetweenValidator")
            .category(RuleCategory.UNIQUENESS)
            .column(column.name)
            .params(
                min_count=1,
                max_count=column.distinct_count + 5,  # Allow some growth
            )
            .confidence(RuleConfidence.MEDIUM)
            .description(
                f"Column '{column.name}' should have between 1 and {column.distinct_count + 5} distinct values"
            )
            .rationale(f"Observed {column.distinct_count} distinct values")
            .build()
        )

        return rules
