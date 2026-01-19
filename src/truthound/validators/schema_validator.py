"""Schema-based validator for data validation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator

if TYPE_CHECKING:
    from truthound.schema import Schema


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
        df = lf.collect()
        data_schema = df.schema
        row_count = len(df)

        if row_count == 0:
            return issues

        # Check for missing columns
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

        # Check for extra columns (not in schema)
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

        # Validate each column that exists in both
        for col_name in data_schema:
            if col_name not in self.schema:
                continue

            col_schema = self.schema[col_name]
            col_data = df.get_column(col_name)

            # Type check - normalize both types for comparison
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

            # Nullable check
            if not col_schema.nullable:
                null_count = col_data.null_count()
                if null_count > 0:
                    issues.append(
                        ValidationIssue(
                            column=col_name,
                            issue_type="unexpected_nulls",
                            count=null_count,
                            severity=Severity.HIGH,
                            details="Column should not contain null values",
                        )
                    )

            # Unique check
            if col_schema.unique:
                unique_count = col_data.n_unique()
                non_null_count = row_count - col_data.null_count()
                if unique_count < non_null_count:
                    dup_count = non_null_count - unique_count
                    issues.append(
                        ValidationIssue(
                            column=col_name,
                            issue_type="duplicate_values",
                            count=dup_count,
                            severity=Severity.HIGH,
                            details="Column should contain unique values",
                        )
                    )

            # Range check (min/max)
            non_null = col_data.drop_nulls()
            if len(non_null) > 0:
                if col_schema.min_value is not None:
                    below_min = non_null.filter(non_null < col_schema.min_value)
                    if len(below_min) > 0:
                        issues.append(
                            ValidationIssue(
                                column=col_name,
                                issue_type="below_minimum",
                                count=len(below_min),
                                severity=Severity.MEDIUM,
                                details=f"Values below minimum {col_schema.min_value}",
                            )
                        )

                if col_schema.max_value is not None:
                    above_max = non_null.filter(non_null > col_schema.max_value)
                    if len(above_max) > 0:
                        issues.append(
                            ValidationIssue(
                                column=col_name,
                                issue_type="above_maximum",
                                count=len(above_max),
                                severity=Severity.MEDIUM,
                                details=f"Values above maximum {col_schema.max_value}",
                            )
                        )

            # Allowed values check
            if col_schema.allowed_values is not None and len(non_null) > 0:
                allowed_set = set(col_schema.allowed_values)
                invalid_count = 0
                for val in non_null.to_list():
                    if val not in allowed_set:
                        invalid_count += 1
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

            # Pattern check (regex)
            if col_schema.pattern is not None and len(non_null) > 0:
                pattern = re.compile(col_schema.pattern)
                invalid_count = 0
                for val in non_null.to_list():
                    if isinstance(val, str) and not pattern.match(val):
                        invalid_count += 1
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

            # String length check
            if data_schema[col_name] in (pl.String, pl.Utf8) and len(non_null) > 0:
                lengths = non_null.str.len_chars()

                if col_schema.min_length is not None:
                    too_short = lengths.filter(lengths < col_schema.min_length)
                    if len(too_short) > 0:
                        issues.append(
                            ValidationIssue(
                                column=col_name,
                                issue_type="string_too_short",
                                count=len(too_short),
                                severity=Severity.LOW,
                                details=f"Strings shorter than {col_schema.min_length}",
                            )
                        )

                if col_schema.max_length is not None:
                    too_long = lengths.filter(lengths > col_schema.max_length)
                    if len(too_long) > 0:
                        issues.append(
                            ValidationIssue(
                                column=col_name,
                                issue_type="string_too_long",
                                count=len(too_long),
                                severity=Severity.LOW,
                                details=f"Strings longer than {col_schema.max_length}",
                            )
                        )

        return issues
