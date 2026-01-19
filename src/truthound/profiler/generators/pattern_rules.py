"""Pattern-based rule generator.

Generates validation rules based on:
- Detected patterns (email, phone, UUID, etc.)
- String length constraints
- Regex patterns
- Semantic types
"""

from __future__ import annotations

from typing import Any

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    PatternMatch,
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


# Mapping of inferred types to validator classes
TYPE_TO_VALIDATOR: dict[DataType, str] = {
    DataType.EMAIL: "EmailValidator",
    DataType.URL: "UrlValidator",
    DataType.UUID: "UuidValidator",
    DataType.IP_ADDRESS: "IpAddressValidator",
    DataType.PHONE: "PhoneValidator",
    DataType.JSON: "JsonParseableValidator",
    DataType.KOREAN_RRN: "KoreanRRNValidator",
    DataType.KOREAN_PHONE: "KoreanPhoneValidator",
    DataType.KOREAN_BUSINESS_NUMBER: "KoreanBusinessNumberValidator",
}


@register_generator("pattern")
class PatternRuleGenerator(RuleGenerator):
    """Generates pattern-based validation rules.

    This generator creates rules that validate:
    - Format patterns (email, phone, UUID, etc.)
    - String length constraints
    - Custom regex patterns
    - Semantic type validation
    """

    name = "pattern"
    description = "Generates pattern-based validation rules"
    categories = {RuleCategory.FORMAT, RuleCategory.PATTERN}
    priority = 70

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        for column in profile.columns:
            rules.extend(self.generate_for_column(column, strictness))

        return rules

    def generate_for_column(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Semantic type-based rules
        rules.extend(self._semantic_type_rules(column, strictness))

        # Pattern-based rules
        rules.extend(self._pattern_match_rules(column, strictness))

        # String length rules
        rules.extend(self._string_length_rules(column, strictness))

        return rules

    def _semantic_type_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate rules based on inferred semantic type."""
        rules: list[GeneratedRule] = []

        if column.inferred_type not in TYPE_TO_VALIDATOR:
            return rules

        validator_class = TYPE_TO_VALIDATOR[column.inferred_type]
        threshold = DEFAULT_THRESHOLDS.get_pattern_threshold(strictness)

        # Determine mostly parameter based on strictness
        mostly = {
            Strictness.LOOSE: 0.90,
            Strictness.MEDIUM: 0.95,
            Strictness.STRICT: 0.99,
        }[strictness]

        rules.append(
            RuleBuilder(f"{column.inferred_type.value}_{column.name}")
            .validator(validator_class)
            .category(RuleCategory.FORMAT)
            .column(column.name)
            .mostly(mostly)
            .confidence(RuleConfidence.HIGH)
            .description(
                f"Column '{column.name}' should contain valid {column.inferred_type.value} values"
            )
            .rationale(f"Inferred type: {column.inferred_type.value}")
            .build()
        )

        return rules

    def _pattern_match_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate rules based on detected patterns."""
        rules: list[GeneratedRule] = []

        if not column.detected_patterns:
            return rules

        threshold = DEFAULT_THRESHOLDS.get_pattern_threshold(strictness)

        for pattern in column.detected_patterns:
            # Only create rules for patterns above threshold
            if pattern.match_ratio < threshold:
                continue

            # Skip patterns with empty or missing regex
            if not pattern.regex:
                continue

            # Skip if already covered by semantic type
            pattern_to_type = {
                "email": DataType.EMAIL,
                "url": DataType.URL,
                "uuid": DataType.UUID,
                "ip_address": DataType.IP_ADDRESS,
                "phone": DataType.PHONE,
                "json": DataType.JSON,
                "korean_rrn": DataType.KOREAN_RRN,
                "korean_phone": DataType.KOREAN_PHONE,
                "korean_business_number": DataType.KOREAN_BUSINESS_NUMBER,
            }

            if pattern.pattern in pattern_to_type:
                if column.inferred_type == pattern_to_type[pattern.pattern]:
                    continue  # Already covered by semantic type rule

            # Create regex validator for the pattern
            mostly = pattern.match_ratio * 0.95  # Allow small degradation

            rules.append(
                RuleBuilder(f"pattern_{pattern.pattern}_{column.name}")
                .validator("RegexValidator")
                .category(RuleCategory.PATTERN)
                .column(column.name)
                .params(pattern=pattern.regex)
                .mostly(round(mostly, 2))
                .confidence(
                    RuleConfidence.HIGH if pattern.match_ratio >= 0.99
                    else RuleConfidence.MEDIUM
                )
                .description(
                    f"Column '{column.name}' should match pattern '{pattern.pattern}'"
                )
                .rationale(
                    f"Pattern matched {pattern.match_ratio*100:.1f}% of values"
                )
                .build()
            )

        return rules

    def _string_length_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate string length validation rules."""
        rules: list[GeneratedRule] = []

        # Only for string columns with length info
        if column.min_length is None or column.max_length is None:
            return rules

        # Skip if inferred type suggests a specific format
        format_types = {
            DataType.EMAIL, DataType.URL, DataType.UUID,
            DataType.IP_ADDRESS, DataType.PHONE, DataType.JSON,
            DataType.KOREAN_RRN, DataType.KOREAN_PHONE,
            DataType.KOREAN_BUSINESS_NUMBER,
        }
        if column.inferred_type in format_types:
            return rules

        # Calculate bounds with tolerance
        tolerance = DEFAULT_THRESHOLDS.get_range_tolerance(strictness)
        length_range = column.max_length - column.min_length

        min_len = max(0, column.min_length - int(length_range * tolerance))
        max_len = column.max_length + int(length_range * tolerance) + 1

        # Special cases
        if column.min_length == column.max_length:
            # Fixed length - be more strict
            rules.append(
                RuleBuilder(f"fixed_length_{column.name}")
                .validator("LengthValidator")
                .category(RuleCategory.FORMAT)
                .column(column.name)
                .params(min_length=column.min_length, max_length=column.max_length)
                .confidence(RuleConfidence.HIGH)
                .description(
                    f"Column '{column.name}' should have exactly {column.min_length} characters"
                )
                .rationale("All observed values have same length")
                .build()
            )
        else:
            # Variable length
            rules.append(
                RuleBuilder(f"length_{column.name}")
                .validator("LengthValidator")
                .category(RuleCategory.FORMAT)
                .column(column.name)
                .params(min_length=min_len, max_length=max_len)
                .confidence(RuleConfidence.MEDIUM)
                .description(
                    f"Column '{column.name}' length should be between {min_len} and {max_len}"
                )
                .rationale(
                    f"Observed length range: [{column.min_length}, {column.max_length}]"
                )
                .build()
            )

        # Empty string check
        if column.empty_string_count == 0 and column.min_length > 0:
            rules.append(
                RuleBuilder(f"no_empty_string_{column.name}")
                .validator("EmptyStringValidator")
                .category(RuleCategory.FORMAT)
                .column(column.name)
                .params(allow_empty=False)
                .confidence(RuleConfidence.HIGH)
                .description(f"Column '{column.name}' should not contain empty strings")
                .rationale("No empty strings observed")
                .build()
            )

        return rules


@register_generator("temporal")
class TemporalRuleGenerator(RuleGenerator):
    """Generates temporal/datetime validation rules.

    This generator creates rules that validate:
    - Date ranges
    - Date ordering
    - Temporal constraints
    """

    name = "temporal"
    description = "Generates temporal validation rules"
    categories = {RuleCategory.TEMPORAL}
    priority = 60

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        for column in profile.columns:
            if column.min_date is not None or column.max_date is not None:
                rules.extend(self.generate_for_column(column, strictness))

        return rules

    def generate_for_column(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        if column.min_date is None and column.max_date is None:
            return rules

        # Date range validation
        rules.extend(self._date_range_rules(column, strictness))

        return rules

    def _date_range_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate date range validation rules."""
        rules: list[GeneratedRule] = []
        from datetime import timedelta

        if column.min_date is None or column.max_date is None:
            return rules

        # Calculate tolerance based on strictness
        date_range = column.max_date - column.min_date
        tolerance_days = {
            Strictness.LOOSE: 30,
            Strictness.MEDIUM: 7,
            Strictness.STRICT: 1,
        }[strictness]

        min_date = column.min_date - timedelta(days=tolerance_days)
        max_date = column.max_date + timedelta(days=tolerance_days)

        rules.append(
            RuleBuilder(f"date_range_{column.name}")
            .validator("DateBetweenValidator")
            .category(RuleCategory.TEMPORAL)
            .column(column.name)
            .params(
                min_date=min_date.isoformat(),
                max_date=max_date.isoformat(),
            )
            .confidence(RuleConfidence.MEDIUM)
            .description(
                f"Column '{column.name}' dates should be between "
                f"{min_date.date()} and {max_date.date()}"
            )
            .rationale(
                f"Observed date range: [{column.min_date.date()}, {column.max_date.date()}]"
            )
            .build()
        )

        # Check for past/future dates
        from datetime import datetime
        now = datetime.now()

        if column.max_date < now:
            # All dates are in the past
            rules.append(
                RuleBuilder(f"past_date_{column.name}")
                .validator("PastDateValidator")
                .category(RuleCategory.TEMPORAL)
                .column(column.name)
                .confidence(RuleConfidence.MEDIUM)
                .description(f"Column '{column.name}' should contain past dates")
                .rationale(f"All observed dates are in the past (max: {column.max_date.date()})")
                .build()
            )
        elif column.min_date > now:
            # All dates are in the future
            rules.append(
                RuleBuilder(f"future_date_{column.name}")
                .validator("FutureDateValidator")
                .category(RuleCategory.TEMPORAL)
                .column(column.name)
                .confidence(RuleConfidence.MEDIUM)
                .description(f"Column '{column.name}' should contain future dates")
                .rationale(f"All observed dates are in the future (min: {column.min_date.date()})")
                .build()
            )

        return rules
