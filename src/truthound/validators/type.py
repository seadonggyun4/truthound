"""Type consistency validator."""

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class TypeValidator(Validator):
    """Detects type inconsistencies in columns (mixed types in string columns)."""

    name = "type"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for type inconsistencies in string columns.

        This validator looks for string columns that contain values that could
        be parsed as numbers, suggesting mixed data types.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for columns with type inconsistencies.
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in schema.names():
            dtype = schema[col]

            # Only check string columns for mixed types
            if dtype == pl.String or dtype == pl.Utf8:
                col_data = df.get_column(col).drop_nulls()

                if len(col_data) == 0:
                    continue

                # Check how many values look like numbers
                numeric_count = 0
                for val in col_data.to_list():
                    if isinstance(val, str):
                        try:
                            float(val)
                            numeric_count += 1
                        except ValueError:
                            pass

                non_null_count = len(col_data)
                # If we have a mix of numeric-looking and non-numeric strings
                if 0 < numeric_count < non_null_count:
                    mix_ratio = numeric_count / non_null_count

                    # Only flag if it's a significant mix (not just a few outliers)
                    if 0.1 < mix_ratio < 0.9:
                        severity = Severity.MEDIUM

                        issues.append(
                            ValidationIssue(
                                column=col,
                                issue_type="mixed_type",
                                count=numeric_count,
                                severity=severity,
                                details=f"{mix_ratio:.1%} of values appear numeric in string column",
                            )
                        )

        return issues
