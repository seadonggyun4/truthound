"""Schema-based validator for data validation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator

if TYPE_CHECKING:
    from truthound.schema import ColumnSchema, Schema


# Type aliases for normalization - maps equivalent type names to canonical form
_TYPE_ALIASES: dict[str, str] = {
    # String types
    "utf8": "String",
    "Utf8": "String",
    "string": "String",
    "str": "String",
    # Integer types
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "i8": "Int8",
    "i16": "Int16",
    "i32": "Int32",
    "i64": "Int64",
    # Unsigned integer types
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "u8": "UInt8",
    "u16": "UInt16",
    "u32": "UInt32",
    "u64": "UInt64",
    # Float types
    "float32": "Float32",
    "float64": "Float64",
    "f32": "Float32",
    "f64": "Float64",
    "float": "Float64",
    "double": "Float64",
    # Boolean
    "bool": "Boolean",
    "boolean": "Boolean",
    # Date/Time types
    "date": "Date",
    "datetime": "Datetime",
    "time": "Time",
    "duration": "Duration",
    # Other types
    "null": "Null",
    "object": "Object",
    "categorical": "Categorical",
    "binary": "Binary",
    "list": "List",
    "struct": "Struct",
}


def _normalize_dtype(dtype_str: str) -> str:
    """Normalize dtype string for comparison.

    Handles case differences and aliases between Polars versions.

    Args:
        dtype_str: Type string to normalize (e.g., "Int64", "int64", "Utf8", "String")

    Returns:
        Normalized type string for consistent comparison.
    """
    # Remove any whitespace
    dtype_str = dtype_str.strip()

    # Check for exact alias match first
    if dtype_str in _TYPE_ALIASES:
        return _TYPE_ALIASES[dtype_str]

    # Check case-insensitive match
    dtype_lower = dtype_str.lower()
    if dtype_lower in _TYPE_ALIASES:
        return _TYPE_ALIASES[dtype_lower]

    # For complex types like Datetime(time_unit='us', time_zone=None),
    # extract just the base type name
    if "(" in dtype_str:
        base_type = dtype_str.split("(")[0].strip()
        if base_type in _TYPE_ALIASES:
            return _TYPE_ALIASES[base_type]
        if base_type.lower() in _TYPE_ALIASES:
            return _TYPE_ALIASES[base_type.lower()]
        return base_type

    # Return as-is if no normalization needed
    return dtype_str


class SchemaValidator(Validator):
    """Validates data against a predefined schema."""

    name = "schema"

    def __init__(self, schema: Schema):
        """Initialize with a schema.

        Args:
            schema: Schema to validate against.
        """
        self.schema = schema

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate data against the schema.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues found.
        """
        issues: list[ValidationIssue] = []
        data_schema = lf.collect_schema()
        aggregate_stats = self._collect_aggregate_stats(lf, data_schema)
        row_count = int(aggregate_stats["_row_count"])

        if row_count == 0:
            return issues

        issues.extend(self._validate_structure(data_schema))

        for col_name in data_schema:
            if col_name not in self.schema:
                continue

            issues.extend(
                self._validate_column(
                    lf=lf,
                    data_schema=data_schema,
                    aggregate_stats=aggregate_stats,
                    row_count=row_count,
                    col_name=col_name,
                )
            )

        return issues

    def _validate_structure(self, data_schema: dict[str, pl.DataType]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for col_name in self.schema.columns:
            if col_name not in data_schema:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="missing_column",
                        count=1,
                        severity=Severity.CRITICAL,
                        details="Column defined in schema but missing in data",
                    )
                )

        for col_name in data_schema:
            if col_name not in self.schema:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="extra_column",
                        count=1,
                        severity=Severity.LOW,
                        details="Column in data but not defined in schema",
                    )
                )

        return issues

    def _validate_column(
        self,
        *,
        lf: pl.LazyFrame,
        data_schema: dict[str, pl.DataType],
        aggregate_stats: dict[str, Any],
        row_count: int,
        col_name: str,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        col_schema = self.schema[col_name]

        actual_dtype = str(data_schema[col_name])
        expected_dtype = col_schema.dtype
        actual_normalized = _normalize_dtype(actual_dtype)
        expected_normalized = _normalize_dtype(expected_dtype)

        if actual_normalized != expected_normalized:
            issues.append(
                ValidationIssue(
                    column=col_name,
                    issue_type="type_mismatch",
                    count=row_count,
                    severity=Severity.HIGH,
                    details=f"Expected {expected_dtype}, got {actual_dtype}",
                )
            )

        issues.extend(
            self._validate_aggregate_constraints(
                col_name=col_name,
                col_schema=col_schema,
                data_type=data_schema[col_name],
                aggregate_stats=aggregate_stats,
                row_count=row_count,
            )
        )
        issues.extend(
            self._validate_targeted_value_checks(
                lf=lf,
                col_name=col_name,
                col_schema=col_schema,
            )
        )
        return issues

    def _collect_aggregate_stats(
        self,
        lf: pl.LazyFrame,
        data_schema: dict[str, pl.DataType],
    ) -> dict[str, Any]:
        expressions: list[pl.Expr] = [pl.len().alias("_row_count")]

        for col_name, data_type in data_schema.items():
            if col_name not in self.schema:
                continue

            col_schema = self.schema[col_name]
            expressions.append(pl.col(col_name).null_count().alias(self._stat_key(col_name, "null_count")))

            if col_schema.unique:
                expressions.append(pl.col(col_name).n_unique().alias(self._stat_key(col_name, "n_unique")))

            if col_schema.min_value is not None:
                expressions.append(
                    pl.col(col_name)
                    .drop_nulls()
                    .lt(col_schema.min_value)
                    .cast(pl.Int64)
                    .sum()
                    .alias(self._stat_key(col_name, "below_minimum"))
                )

            if col_schema.max_value is not None:
                expressions.append(
                    pl.col(col_name)
                    .drop_nulls()
                    .gt(col_schema.max_value)
                    .cast(pl.Int64)
                    .sum()
                    .alias(self._stat_key(col_name, "above_maximum"))
                )

            if data_type in (pl.String, pl.Utf8):
                lengths = pl.col(col_name).drop_nulls().str.len_chars()
                if col_schema.min_length is not None:
                    expressions.append(
                        lengths.lt(col_schema.min_length)
                        .cast(pl.Int64)
                        .sum()
                        .alias(self._stat_key(col_name, "string_too_short"))
                    )
                if col_schema.max_length is not None:
                    expressions.append(
                        lengths.gt(col_schema.max_length)
                        .cast(pl.Int64)
                        .sum()
                        .alias(self._stat_key(col_name, "string_too_long"))
                    )

        stats_frame = lf.select(expressions).collect(engine="streaming")
        return stats_frame.to_dicts()[0]

    def _validate_aggregate_constraints(
        self,
        *,
        col_name: str,
        col_schema: ColumnSchema,
        data_type: pl.DataType,
        aggregate_stats: dict[str, Any],
        row_count: int,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        null_count = self._int_stat(aggregate_stats, col_name, "null_count")
        non_null_count = row_count - null_count

        if not col_schema.nullable and null_count > 0:
            issues.append(
                ValidationIssue(
                    column=col_name,
                    issue_type="unexpected_nulls",
                    count=null_count,
                    severity=Severity.HIGH,
                    details="Column should not contain null values",
                )
            )

        if col_schema.unique:
            unique_count = self._int_stat(aggregate_stats, col_name, "n_unique")
            if unique_count < non_null_count:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="duplicate_values",
                        count=non_null_count - unique_count,
                        severity=Severity.HIGH,
                        details="Column should contain unique values",
                    )
                )

        if col_schema.min_value is not None:
            below_minimum = self._int_stat(aggregate_stats, col_name, "below_minimum")
            if below_minimum > 0:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="below_minimum",
                        count=below_minimum,
                        severity=Severity.MEDIUM,
                        details=f"Values below minimum {col_schema.min_value}",
                    )
                )

        if col_schema.max_value is not None:
            above_maximum = self._int_stat(aggregate_stats, col_name, "above_maximum")
            if above_maximum > 0:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="above_maximum",
                        count=above_maximum,
                        severity=Severity.MEDIUM,
                        details=f"Values above maximum {col_schema.max_value}",
                    )
                )

        if data_type in (pl.String, pl.Utf8):
            if col_schema.min_length is not None:
                too_short = self._int_stat(aggregate_stats, col_name, "string_too_short")
                if too_short > 0:
                    issues.append(
                        ValidationIssue(
                            column=col_name,
                            issue_type="string_too_short",
                            count=too_short,
                            severity=Severity.LOW,
                            details=f"Strings shorter than {col_schema.min_length}",
                        )
                    )

            if col_schema.max_length is not None:
                too_long = self._int_stat(aggregate_stats, col_name, "string_too_long")
                if too_long > 0:
                    issues.append(
                        ValidationIssue(
                            column=col_name,
                            issue_type="string_too_long",
                            count=too_long,
                            severity=Severity.LOW,
                            details=f"Strings longer than {col_schema.max_length}",
                        )
                    )

        return issues

    def _validate_targeted_value_checks(
        self,
        *,
        lf: pl.LazyFrame,
        col_name: str,
        col_schema: ColumnSchema,
    ) -> list[ValidationIssue]:
        if col_schema.allowed_values is None and col_schema.pattern is None:
            return []

        non_null = self._collect_non_null_column(lf, col_name)
        if len(non_null) == 0:
            return []

        issues: list[ValidationIssue] = []

        if col_schema.allowed_values is not None:
            allowed_set = set(col_schema.allowed_values)
            invalid_count = sum(1 for value in non_null.to_list() if value not in allowed_set)
            if invalid_count > 0:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="invalid_value",
                        count=invalid_count,
                        severity=Severity.MEDIUM,
                        details=f"Values not in allowed set: {col_schema.allowed_values[:5]}...",
                    )
                )

        if col_schema.pattern is not None:
            pattern = re.compile(col_schema.pattern)
            invalid_count = sum(
                1
                for value in non_null.to_list()
                if isinstance(value, str) and not pattern.match(value)
            )
            if invalid_count > 0:
                issues.append(
                    ValidationIssue(
                        column=col_name,
                        issue_type="pattern_mismatch",
                        count=invalid_count,
                        severity=Severity.MEDIUM,
                        details=f"Values don't match pattern: {col_schema.pattern}",
                    )
                )

        return issues

    def _collect_non_null_column(self, lf: pl.LazyFrame, col_name: str) -> pl.Series:
        collected = (
            lf.select(pl.col(col_name).drop_nulls().alias(col_name))
            .collect(engine="streaming")
            .get_column(col_name)
        )
        return collected

    @staticmethod
    def _stat_key(col_name: str, stat_name: str) -> str:
        return f"{col_name}__{stat_name}"

    @classmethod
    def _int_stat(cls, aggregate_stats: dict[str, Any], col_name: str, stat_name: str) -> int:
        value = aggregate_stats.get(cls._stat_key(col_name, stat_name), 0)
        if value is None:
            return 0
        return int(value)
