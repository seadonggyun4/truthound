"""Multi-column validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class MultiColumnUniqueValidator(Validator):
    """Validates that combination of columns is unique."""

    name = "multi_column_unique"
    category = "schema"

    def __init__(
        self,
        columns: list[str],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.key_columns = columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        result = lf.select([
            pl.len().alias("_total"),
            pl.struct(self.key_columns).n_unique().alias("_unique"),
        ]).collect()

        total = result["_total"][0]
        unique = result["_unique"][0]
        duplicates = total - unique

        if duplicates > 0:
            col_desc = ", ".join(self.key_columns)
            ratio = duplicates / total if total > 0 else 0

            issues.append(
                ValidationIssue(
                    column=f"[{col_desc}]",
                    issue_type="multi_column_duplicate",
                    count=duplicates,
                    severity=self._calculate_severity(ratio, (0.1, 0.05, 0.01)),
                    details=f"{duplicates} duplicate combinations",
                )
            )

        return issues
