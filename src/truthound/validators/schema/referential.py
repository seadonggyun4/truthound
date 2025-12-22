"""Referential integrity validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ReferentialIntegrityValidator(Validator):
    """Validates referential integrity between datasets.

    Example:
        # All order.customer_id values should exist in customers.id
        validator = ReferentialIntegrityValidator(
            column="customer_id",
            reference_data=customers_df,
            reference_column="id",
        )
    """

    name = "referential_integrity"
    category = "schema"

    def __init__(
        self,
        column: str,
        reference_data: pl.DataFrame | pl.LazyFrame,
        reference_column: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.reference_column = reference_column or column

        # Extract reference values
        if isinstance(reference_data, pl.LazyFrame):
            ref_df = reference_data.collect()
        else:
            ref_df = reference_data

        self.reference_values = set(
            ref_df.get_column(self.reference_column).drop_nulls().to_list()
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        column_values = df.get_column(self.column).drop_nulls()

        if len(column_values) == 0:
            return issues

        # Find orphan values
        orphan_count = 0
        sample_orphans = []

        for val in column_values.to_list():
            if val not in self.reference_values:
                orphan_count += 1
                if len(sample_orphans) < self.config.sample_size:
                    sample_orphans.append(val)

        if orphan_count > 0:
            ratio = orphan_count / len(column_values)
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="referential_integrity_violation",
                    count=orphan_count,
                    severity=Severity.CRITICAL if ratio > 0.1 else Severity.HIGH,
                    details=f"{orphan_count} values not found in reference",
                    sample_values=sample_orphans,
                )
            )

        return issues
