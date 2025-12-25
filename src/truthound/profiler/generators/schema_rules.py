"""Schema-based rule generator.

Generates validation rules for:
- Column existence
- Column types
- Column order
- Table structure
"""

from __future__ import annotations

from typing import Any

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
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


@register_generator("schema")
class SchemaRuleGenerator(RuleGenerator):
    """Generates schema validation rules.

    This generator creates rules that validate:
    - Required columns exist
    - Column types are correct
    - Table has expected structure
    """

    name = "schema"
    description = "Generates schema validation rules"
    categories = {RuleCategory.SCHEMA, RuleCategory.COMPLETENESS}
    priority = 100  # High priority - schema rules are fundamental

    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        rules: list[GeneratedRule] = []

        # Table-level rules
        rules.extend(self._generate_table_rules(profile, strictness))

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

        # Column existence rule
        rules.append(self._column_exists_rule(column))

        # Column type rule
        rules.append(self._column_type_rule(column))

        # Completeness rules
        rules.extend(self._completeness_rules(column, strictness))

        return rules

    def _generate_table_rules(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate table-level schema rules."""
        rules: list[GeneratedRule] = []

        # Required columns rule
        column_names = [col.name for col in profile.columns]
        rules.append(
            RuleBuilder(f"table_required_columns")
            .validator("TableRequiredColumnsValidator")
            .category(RuleCategory.SCHEMA)
            .params(required_columns=column_names)
            .confidence(RuleConfidence.HIGH)
            .description(f"Table must have columns: {', '.join(column_names)}")
            .rationale("All columns observed in profile are required")
            .build()
        )

        # Column count rule (for strict mode)
        if strictness == Strictness.STRICT:
            rules.append(
                RuleBuilder("table_column_count")
                .validator("TableColumnCountValidator")
                .category(RuleCategory.SCHEMA)
                .params(expected_count=profile.column_count)
                .confidence(RuleConfidence.HIGH)
                .description(f"Table must have exactly {profile.column_count} columns")
                .rationale("Strict mode: enforce exact column count")
                .build()
            )

        # Table not empty rule
        if profile.row_count > 0:
            rules.append(
                RuleBuilder("table_not_empty")
                .validator("TableNotEmptyValidator")
                .category(RuleCategory.SCHEMA)
                .confidence(RuleConfidence.HIGH)
                .description("Table must not be empty")
                .rationale("Profile showed non-empty table")
                .build()
            )

        # Row count range (loose bounds based on observed data)
        if strictness in {Strictness.MEDIUM, Strictness.STRICT}:
            # Allow some variation based on strictness
            tolerance = {
                Strictness.MEDIUM: 0.5,  # 50% variation allowed
                Strictness.STRICT: 0.2,  # 20% variation allowed
            }[strictness]

            min_rows = int(profile.row_count * (1 - tolerance))
            max_rows = int(profile.row_count * (1 + tolerance))

            rules.append(
                RuleBuilder("table_row_count_range")
                .validator("TableRowCountRangeValidator")
                .category(RuleCategory.SCHEMA)
                .params(min_rows=min_rows, max_rows=max_rows)
                .confidence(RuleConfidence.MEDIUM)
                .description(f"Table should have between {min_rows} and {max_rows} rows")
                .rationale(f"Based on observed {profile.row_count} rows with {tolerance*100}% tolerance")
                .build()
            )

        return rules

    def _column_exists_rule(self, column: ColumnProfile) -> GeneratedRule:
        """Generate column existence rule."""
        return (
            RuleBuilder(f"column_exists_{column.name}")
            .validator("ColumnExistsValidator")
            .category(RuleCategory.SCHEMA)
            .column(column.name)
            .confidence(RuleConfidence.HIGH)
            .description(f"Column '{column.name}' must exist")
            .rationale("Column observed in profile")
            .build()
        )

    def _column_type_rule(self, column: ColumnProfile) -> GeneratedRule:
        """Generate column type rule."""
        # Map physical type to validator expected type
        type_mapping = {
            "Int8": "int8",
            "Int16": "int16",
            "Int32": "int32",
            "Int64": "int64",
            "UInt8": "uint8",
            "UInt16": "uint16",
            "UInt32": "uint32",
            "UInt64": "uint64",
            "Float32": "float32",
            "Float64": "float64",
            "Boolean": "bool",
            "String": "string",
            "Utf8": "string",
            "Date": "date",
            "Datetime": "datetime",
            "Time": "time",
            "Duration": "duration",
        }

        # Extract base type from physical type string
        physical_type = column.physical_type
        for key, value in type_mapping.items():
            if key in physical_type:
                expected_type = value
                break
        else:
            expected_type = "string"  # Default fallback

        return (
            RuleBuilder(f"column_type_{column.name}")
            .validator("ColumnTypeValidator")
            .category(RuleCategory.SCHEMA)
            .column(column.name)
            .params(expected_type=expected_type)
            .confidence(RuleConfidence.HIGH)
            .description(f"Column '{column.name}' must be of type '{expected_type}'")
            .rationale(f"Observed type: {column.physical_type}")
            .build()
        )

    def _completeness_rules(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate completeness rules based on null ratios."""
        rules: list[GeneratedRule] = []

        # Determine threshold based on strictness
        threshold = DEFAULT_THRESHOLDS.get_null_threshold(strictness)

        if column.null_ratio == 0:
            # Column is completely non-null - enforce it
            rules.append(
                RuleBuilder(f"not_null_{column.name}")
                .validator("NotNullValidator")
                .category(RuleCategory.COMPLETENESS)
                .column(column.name)
                .confidence(RuleConfidence.HIGH)
                .description(f"Column '{column.name}' must not contain nulls")
                .rationale("No nulls observed in profile")
                .build()
            )
        elif column.null_ratio <= threshold:
            # Some nulls but within threshold - use completeness ratio
            min_ratio = 1 - column.null_ratio
            # Add some buffer based on strictness
            buffer = {
                Strictness.LOOSE: 0.05,
                Strictness.MEDIUM: 0.02,
                Strictness.STRICT: 0.01,
            }[strictness]
            min_ratio = max(0, min_ratio - buffer)

            rules.append(
                RuleBuilder(f"completeness_{column.name}")
                .validator("CompletenessRatioValidator")
                .category(RuleCategory.COMPLETENESS)
                .column(column.name)
                .params(min_ratio=round(min_ratio, 3))
                .confidence(RuleConfidence.MEDIUM)
                .description(f"Column '{column.name}' must be at least {min_ratio*100:.1f}% complete")
                .rationale(f"Observed {(1-column.null_ratio)*100:.1f}% completeness")
                .build()
            )
        else:
            # High null ratio - just add a warning-level rule
            if strictness != Strictness.LOOSE:
                rules.append(
                    RuleBuilder(f"completeness_warning_{column.name}")
                    .validator("CompletenessRatioValidator")
                    .category(RuleCategory.COMPLETENESS)
                    .column(column.name)
                    .params(min_ratio=0.5)  # At least 50% complete
                    .confidence(RuleConfidence.LOW)
                    .description(f"Column '{column.name}' should be at least 50% complete")
                    .rationale(f"High null ratio observed: {column.null_ratio*100:.1f}%")
                    .build()
                )

        return rules
