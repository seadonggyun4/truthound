"""Type validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class TypeValidator(Validator, StringValidatorMixin):
    """Detects mixed types in string columns (numbers stored as strings)."""

    name = "type"
    category = "aggregate"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            numeric_count = 0
            string_count = 0

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                val_stripped = val.strip()
                if not val_stripped:
                    continue

                try:
                    float(val_stripped)
                    numeric_count += 1
                except ValueError:
                    string_count += 1

            total_non_null = numeric_count + string_count
            if total_non_null == 0:
                continue

            # Detect mixed types (significant portion of both)
            numeric_ratio = numeric_count / total_non_null
            string_ratio = string_count / total_non_null

            if 0.1 < numeric_ratio < 0.9:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="mixed_type",
                        count=min(numeric_count, string_count),
                        severity=Severity.MEDIUM,
                        details=f"Mixed types: {numeric_ratio:.1%} numeric, {string_ratio:.1%} string",
                    )
                )

        return issues
