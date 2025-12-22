"""Default value detection validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class DefaultValueValidator(Validator):
    """Detects excessive use of default/placeholder values.

    Example:
        validator = DefaultValueValidator(
            default_values=["N/A", "TBD", "unknown", -1, 0],
            max_ratio=0.1,
        )
    """

    name = "default_value"
    category = "completeness"

    # Common default/placeholder values
    COMMON_DEFAULTS = [
        "N/A", "n/a", "NA", "na",
        "TBD", "tbd",
        "unknown", "Unknown", "UNKNOWN",
        "none", "None", "NONE",
        "-", "--", "---",
        "null", "NULL",
        "undefined",
        0, -1, -999, 9999, 99999,
    ]

    def __init__(
        self,
        default_values: list[Any] | None = None,
        max_ratio: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.default_values = default_values or self.COMMON_DEFAULTS
        self.max_ratio = max_ratio

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        # Separate default values by type to avoid is_in type mismatch
        string_defaults = [v for v in self.default_values if isinstance(v, str)]
        numeric_defaults = [v for v in self.default_values if isinstance(v, (int, float)) and not isinstance(v, bool)]

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col)
            dtype = col_data.dtype

            # Choose appropriate defaults based on column type
            if dtype in (pl.Utf8, pl.String):
                defaults_to_check = string_defaults
            elif dtype.is_numeric():
                defaults_to_check = numeric_defaults
            else:
                # For other types, skip or check string representation
                continue

            if not defaults_to_check:
                continue

            default_count = col_data.is_in(defaults_to_check).sum()

            if default_count > 0:
                ratio = default_count / total_rows
                if ratio > self.max_ratio:
                    issues.append(
                        ValidationIssue(
                            column=col,
                            issue_type="default_value",
                            count=default_count,
                            severity=self._calculate_severity(ratio),
                            details=f"{ratio:.1%} values are defaults (max {self.max_ratio:.1%})",
                            expected=f"<= {self.max_ratio:.1%}",
                            actual=f"{ratio:.1%}",
                        )
                    )

        return issues
