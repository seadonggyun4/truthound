"""JSON Schema validation."""

import json
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, StringValidatorMixin
from truthound.validators.registry import register_validator


@register_validator
class JsonSchemaValidator(Validator, StringValidatorMixin):
    """Validates that JSON values conform to a JSON Schema.

    Example:
        validator = JsonSchemaValidator(
            column="config",
            schema={
                "type": "object",
                "required": ["name", "version"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                    "enabled": {"type": "boolean"},
                }
            }
        )
    """

    name = "json_schema"
    category = "string"

    def __init__(
        self,
        schema: dict[str, Any],
        column: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.schema = schema
        self.target_column = column
        self._validator = None

        # Try to use jsonschema library if available
        try:
            import jsonschema
            self._validator = jsonschema.Draft7Validator(schema)
        except ImportError:
            # Fall back to simple validation
            pass

    def _simple_validate(self, data: Any, schema: dict[str, Any]) -> list[str]:
        """Simple JSON schema validation without jsonschema library."""
        errors = []

        schema_type = schema.get("type")
        if schema_type:
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }

            expected_type = type_map.get(schema_type)
            if expected_type and not isinstance(data, expected_type):
                errors.append(f"Expected {schema_type}, got {type(data).__name__}")
                return errors

        # Check required fields for objects
        if isinstance(data, dict):
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

            # Check properties
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    errors.extend(self._simple_validate(data[prop], prop_schema))

        # Check items for arrays
        if isinstance(data, list) and "items" in schema:
            items_schema = schema["items"]
            for i, item in enumerate(data):
                item_errors = self._simple_validate(item, items_schema)
                errors.extend([f"[{i}]: {e}" for e in item_errors])

        # Check enum
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"Value {data} not in enum {schema['enum']}")

        # Check minimum/maximum for numbers
        if isinstance(data, (int, float)):
            if "minimum" in schema and data < schema["minimum"]:
                errors.append(f"Value {data} < minimum {schema['minimum']}")
            if "maximum" in schema and data > schema["maximum"]:
                errors.append(f"Value {data} > maximum {schema['maximum']}")

        # Check minLength/maxLength for strings
        if isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                errors.append(f"String length {len(data)} < minLength {schema['minLength']}")
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errors.append(f"String length {len(data)} > maxLength {schema['maxLength']}")

        return errors

    def _validate_value(self, value: str) -> list[str]:
        """Validate a single JSON value against the schema."""
        try:
            data = json.loads(value)
        except json.JSONDecodeError as e:
            return [f"Invalid JSON: {e}"]

        if self._validator:
            # Use jsonschema library
            errors = list(self._validator.iter_errors(data))
            return [e.message for e in errors]
        else:
            # Use simple validation
            return self._simple_validate(data, self.schema)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.target_column:
            columns = [self.target_column]
        else:
            columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Use streaming for large datasets
        df = lf.collect(engine="streaming")
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            invalid_count = 0
            samples = []
            error_samples = []

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                errors = self._validate_value(val)
                if errors:
                    invalid_count += 1
                    if len(samples) < self.config.sample_size:
                        samples.append(val[:50] + "..." if len(val) > 50 else val)
                        error_samples.append(errors[0])

            if invalid_count > 0:
                if self._passes_mostly(invalid_count, len(col_data)):
                    continue

                ratio = invalid_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="json_schema_violation",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Schema validation failed: {error_samples[0] if error_samples else 'unknown'}",
                        sample_values=samples,
                    )
                )

        return issues
