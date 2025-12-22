"""Within-record uniqueness validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class UniqueWithinRecordValidator(Validator):
    """Validates that specified columns have unique values within each row.

    Useful for checking that related fields don't have duplicate values.

    Example:
        # Primary and secondary contacts should be different
        validator = UniqueWithinRecordValidator(
            columns=["primary_contact", "secondary_contact"],
        )

        # All three choice fields should be unique
        validator = UniqueWithinRecordValidator(
            columns=["choice_1", "choice_2", "choice_3"],
        )
    """

    name = "unique_within_record"
    category = "uniqueness"

    def __init__(
        self,
        columns: list[str],
        ignore_nulls: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.check_columns = columns
        self.ignore_nulls = ignore_nulls

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        # Check each row for duplicates
        duplicate_rows = 0
        sample_indices = []

        for idx in range(total_rows):
            row_values = []
            for col in self.check_columns:
                val = df[col][idx]
                if self.ignore_nulls and val is None:
                    continue
                row_values.append(val)

            # Check for duplicates in this row
            if len(row_values) != len(set(row_values)):
                duplicate_rows += 1
                if len(sample_indices) < self.config.sample_size:
                    sample_indices.append(idx)

        if duplicate_rows > 0:
            if self._passes_mostly(duplicate_rows, total_rows):
                return issues

            ratio = duplicate_rows / total_rows
            col_desc = ", ".join(self.check_columns)

            # Get sample values
            samples = []
            for idx in sample_indices:
                row_vals = [str(df[col][idx]) for col in self.check_columns]
                samples.append(f"row {idx}: [{', '.join(row_vals)}]")

            issues.append(
                ValidationIssue(
                    column=f"[{col_desc}]",
                    issue_type="duplicate_within_record",
                    count=duplicate_rows,
                    severity=self._calculate_severity(ratio),
                    details=f"{duplicate_rows} rows have duplicate values across columns",
                    expected="Unique values within each row",
                    sample_values=samples,
                )
            )

        return issues


@register_validator
class AllColumnsUniqueWithinRecordValidator(Validator):
    """Validates that all non-null values in each row are unique.

    Example:
        # Each row's values should all be different
        validator = AllColumnsUniqueWithinRecordValidator()
    """

    name = "all_columns_unique_within_record"
    category = "uniqueness"

    def __init__(
        self,
        ignore_nulls: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.ignore_nulls = ignore_nulls

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        columns = self._get_target_columns(lf)
        if len(columns) < 2:
            return issues

        duplicate_rows = 0
        sample_indices = []

        for idx in range(total_rows):
            row_values = []
            for col in columns:
                val = df[col][idx]
                if self.ignore_nulls and val is None:
                    continue
                row_values.append(val)

            if len(row_values) != len(set(row_values)):
                duplicate_rows += 1
                if len(sample_indices) < self.config.sample_size:
                    sample_indices.append(idx)

        if duplicate_rows > 0:
            if self._passes_mostly(duplicate_rows, total_rows):
                return issues

            ratio = duplicate_rows / total_rows

            issues.append(
                ValidationIssue(
                    column="_all_columns",
                    issue_type="duplicate_values_in_record",
                    count=duplicate_rows,
                    severity=self._calculate_severity(ratio),
                    details=f"{duplicate_rows} rows have duplicate values",
                    expected="All column values unique within each row",
                )
            )

        return issues
