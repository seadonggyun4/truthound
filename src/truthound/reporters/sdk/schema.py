"""Reporter output validation and schema support.

This module provides schema validation for reporter outputs to ensure
consistent, well-formed reports across different reporter implementations.

Example:
    >>> from truthound.reporters.sdk.schema import (
    ...     ReportSchema,
    ...     JSONSchema,
    ...     validate_output,
    ...     register_schema,
    ... )
    >>>
    >>> # Define a schema for your reporter
    >>> schema = JSONSchema({
    ...     "type": "object",
    ...     "properties": {
    ...         "summary": {"type": "object"},
    ...         "results": {"type": "array"},
    ...     },
    ...     "required": ["summary", "results"],
    ... })
    >>>
    >>> # Register for automatic validation
    >>> register_schema("my_reporter", schema)
    >>>
    >>> # Validate output
    >>> result = validate_output(output, schema)
    >>> if not result.valid:
    ...     print(f"Validation errors: {result.errors}")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from xml.etree import ElementTree

__all__ = [
    # Core classes
    "ReportSchema",
    "JSONSchema",
    "XMLSchema",
    "CSVSchema",
    "TextSchema",
    # Validation
    "ValidationResult",
    "ValidationError",
    "SchemaError",
    # Functions
    "validate_output",
    "register_schema",
    "get_schema",
    "unregister_schema",
    # Decorators
    "validate_reporter_output",
    # Utilities
    "infer_schema",
    "merge_schemas",
]


class SchemaError(Exception):
    """Exception raised for schema-related errors."""

    pass


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
    ) -> None:
        self.message = message
        self.path = path
        self.value = value
        self.expected = expected
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.path:
            parts.append(f"at path '{self.path}'")
        if self.expected:
            parts.append(f"(expected: {self.expected})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of schema validation."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_name: Optional[str] = None
    checked_at: Optional[str] = None

    def raise_if_invalid(self) -> None:
        """Raise SchemaError if validation failed."""
        if not self.valid:
            error_messages = [str(e) for e in self.errors]
            raise SchemaError(
                f"Validation failed with {len(self.errors)} error(s): "
                + "; ".join(error_messages)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "errors": [
                {
                    "message": e.message,
                    "path": e.path,
                    "value": repr(e.value) if e.value is not None else None,
                    "expected": e.expected,
                }
                for e in self.errors
            ],
            "warnings": self.warnings,
            "schema_name": self.schema_name,
            "checked_at": self.checked_at,
        }


class ReportSchema(ABC):
    """Base class for report schemas.

    Subclass this to create custom schema validators for your
    reporter output format.

    Example:
        >>> class MyCustomSchema(ReportSchema):
        ...     def validate(self, output: Any) -> ValidationResult:
        ...         errors = []
        ...         if not isinstance(output, dict):
        ...             errors.append(ValidationError("Expected dictionary"))
        ...         return ValidationResult(valid=len(errors) == 0, errors=errors)
        ...
        ...     def to_dict(self) -> Dict[str, Any]:
        ...         return {"type": "custom"}
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def validate(self, output: Any) -> ValidationResult:
        """Validate output against this schema.

        Args:
            output: The output to validate.

        Returns:
            ValidationResult with validation status and any errors.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class JSONSchema(ReportSchema):
    """JSON Schema validator for reporter output.

    Supports a subset of JSON Schema draft-07 for validating
    JSON/dictionary output from reporters.

    Example:
        >>> schema = JSONSchema({
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "count": {"type": "integer", "minimum": 0},
        ...     },
        ...     "required": ["name"],
        ... })
        >>> result = schema.validate({"name": "test", "count": 5})
        >>> print(result.valid)  # True
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        name: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        """Initialize JSON Schema validator.

        Args:
            schema: JSON Schema definition.
            name: Optional schema name.
            strict: If True, disallow additional properties by default.
        """
        super().__init__(name)
        self.schema = schema
        self.strict = strict
        self._type_validators: Dict[str, Callable] = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }

    def validate(self, output: Any) -> ValidationResult:
        """Validate output against JSON Schema."""
        from datetime import datetime

        errors: List[ValidationError] = []
        warnings: List[str] = []

        self._validate_value(output, self.schema, "", errors, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            schema_name=self.name,
            checked_at=datetime.now().isoformat(),
        )

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
        warnings: List[str],
    ) -> None:
        """Recursively validate a value against schema."""
        # Handle type validation
        if "type" in schema:
            expected_types = schema["type"]
            if isinstance(expected_types, str):
                expected_types = [expected_types]

            type_valid = any(
                self._type_validators.get(t, lambda v: False)(value)
                for t in expected_types
            )

            if not type_valid:
                errors.append(
                    ValidationError(
                        f"Invalid type: got {type(value).__name__}",
                        path=path or "$",
                        value=value,
                        expected=", ".join(expected_types),
                    )
                )
                return

        # Handle enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(
                    ValidationError(
                        f"Value not in enum",
                        path=path or "$",
                        value=value,
                        expected=str(schema["enum"]),
                    )
                )

        # Handle const validation
        if "const" in schema:
            if value != schema["const"]:
                errors.append(
                    ValidationError(
                        f"Value does not match const",
                        path=path or "$",
                        value=value,
                        expected=repr(schema["const"]),
                    )
                )

        # Handle string-specific validations
        if isinstance(value, str):
            self._validate_string(value, schema, path, errors)

        # Handle number-specific validations
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._validate_number(value, schema, path, errors)

        # Handle array validations
        if isinstance(value, list):
            self._validate_array(value, schema, path, errors, warnings)

        # Handle object validations
        if isinstance(value, dict):
            self._validate_object(value, schema, path, errors, warnings)

    def _validate_string(
        self,
        value: str,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate string-specific constraints."""
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(
                ValidationError(
                    f"String too short (length: {len(value)})",
                    path=path,
                    value=value,
                    expected=f"minLength: {schema['minLength']}",
                )
            )

        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(
                ValidationError(
                    f"String too long (length: {len(value)})",
                    path=path,
                    value=value,
                    expected=f"maxLength: {schema['maxLength']}",
                )
            )

        if "pattern" in schema:
            if not re.match(schema["pattern"], value):
                errors.append(
                    ValidationError(
                        f"String does not match pattern",
                        path=path,
                        value=value,
                        expected=f"pattern: {schema['pattern']}",
                    )
                )

        if "format" in schema:
            self._validate_format(value, schema["format"], path, errors)

    def _validate_format(
        self,
        value: str,
        format_type: str,
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate string format."""
        format_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "uri": r"^https?://[^\s]+$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "time": r"^\d{2}:\d{2}:\d{2}$",
            "date-time": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
            "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
            "ipv4": r"^(\d{1,3}\.){3}\d{1,3}$",
        }

        if format_type in format_patterns:
            if not re.match(format_patterns[format_type], value):
                errors.append(
                    ValidationError(
                        f"String does not match format",
                        path=path,
                        value=value,
                        expected=f"format: {format_type}",
                    )
                )

    def _validate_number(
        self,
        value: Union[int, float],
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate number-specific constraints."""
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(
                ValidationError(
                    f"Number below minimum",
                    path=path,
                    value=value,
                    expected=f"minimum: {schema['minimum']}",
                )
            )

        if "maximum" in schema and value > schema["maximum"]:
            errors.append(
                ValidationError(
                    f"Number above maximum",
                    path=path,
                    value=value,
                    expected=f"maximum: {schema['maximum']}",
                )
            )

        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(
                ValidationError(
                    f"Number not greater than exclusive minimum",
                    path=path,
                    value=value,
                    expected=f"exclusiveMinimum: {schema['exclusiveMinimum']}",
                )
            )

        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(
                ValidationError(
                    f"Number not less than exclusive maximum",
                    path=path,
                    value=value,
                    expected=f"exclusiveMaximum: {schema['exclusiveMaximum']}",
                )
            )

        if "multipleOf" in schema and value % schema["multipleOf"] != 0:
            errors.append(
                ValidationError(
                    f"Number is not a multiple",
                    path=path,
                    value=value,
                    expected=f"multipleOf: {schema['multipleOf']}",
                )
            )

    def _validate_array(
        self,
        value: List[Any],
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
        warnings: List[str],
    ) -> None:
        """Validate array-specific constraints."""
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(
                ValidationError(
                    f"Array too short (length: {len(value)})",
                    path=path,
                    value=f"[{len(value)} items]",
                    expected=f"minItems: {schema['minItems']}",
                )
            )

        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(
                ValidationError(
                    f"Array too long (length: {len(value)})",
                    path=path,
                    value=f"[{len(value)} items]",
                    expected=f"maxItems: {schema['maxItems']}",
                )
            )

        if "uniqueItems" in schema and schema["uniqueItems"]:
            try:
                # Check for duplicates using JSON serialization for hashability
                seen = set()
                for item in value:
                    key = json.dumps(item, sort_keys=True, default=str)
                    if key in seen:
                        errors.append(
                            ValidationError(
                                f"Array contains duplicate items",
                                path=path,
                                value=item,
                            )
                        )
                        break
                    seen.add(key)
            except Exception:
                warnings.append(f"Could not check uniqueItems at {path}")

        # Validate items
        if "items" in schema:
            item_schema = schema["items"]
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                self._validate_value(item, item_schema, item_path, errors, warnings)

    def _validate_object(
        self,
        value: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
        warnings: List[str],
    ) -> None:
        """Validate object-specific constraints."""
        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in value:
                errors.append(
                    ValidationError(
                        f"Missing required property: {prop}",
                        path=path or "$",
                        expected=f"property '{prop}'",
                    )
                )

        # Validate properties
        properties = schema.get("properties", {})
        additional_properties = schema.get(
            "additionalProperties", not self.strict
        )

        for key, prop_value in value.items():
            prop_path = f"{path}.{key}" if path else key

            if key in properties:
                self._validate_value(
                    prop_value, properties[key], prop_path, errors, warnings
                )
            elif not additional_properties:
                errors.append(
                    ValidationError(
                        f"Additional property not allowed: {key}",
                        path=prop_path,
                        value=prop_value,
                    )
                )
            elif isinstance(additional_properties, dict):
                # additionalProperties can be a schema
                self._validate_value(
                    prop_value, additional_properties, prop_path, errors, warnings
                )

        # Check property count
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(
                ValidationError(
                    f"Object has too few properties",
                    path=path or "$",
                    value=f"{len(value)} properties",
                    expected=f"minProperties: {schema['minProperties']}",
                )
            )

        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errors.append(
                ValidationError(
                    f"Object has too many properties",
                    path=path or "$",
                    value=f"{len(value)} properties",
                    expected=f"maxProperties: {schema['maxProperties']}",
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "json_schema",
            "name": self.name,
            "strict": self.strict,
            "schema": self.schema,
        }


class XMLSchema(ReportSchema):
    """XML structure validator for reporter output.

    Validates XML output structure including elements, attributes,
    and text content.

    Example:
        >>> schema = XMLSchema(
        ...     root_element="report",
        ...     required_elements=["summary", "results"],
        ...     required_attributes={"report": ["version"]},
        ... )
        >>> result = schema.validate(xml_string)
    """

    def __init__(
        self,
        root_element: str,
        required_elements: Optional[List[str]] = None,
        required_attributes: Optional[Dict[str, List[str]]] = None,
        element_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize XML Schema validator.

        Args:
            root_element: Expected root element name.
            required_elements: List of required element names.
            required_attributes: Dict mapping element names to required attributes.
            element_schemas: Dict mapping element names to validation rules.
            name: Optional schema name.
        """
        super().__init__(name)
        self.root_element = root_element
        self.required_elements = required_elements or []
        self.required_attributes = required_attributes or {}
        self.element_schemas = element_schemas or {}

    def validate(self, output: Union[str, bytes, ElementTree.Element]) -> ValidationResult:
        """Validate XML output."""
        from datetime import datetime

        errors: List[ValidationError] = []
        warnings: List[str] = []

        # Parse XML if string/bytes
        try:
            if isinstance(output, (str, bytes)):
                root = ElementTree.fromstring(output)
            elif isinstance(output, ElementTree.Element):
                root = output
            else:
                errors.append(
                    ValidationError(
                        f"Expected XML string, bytes, or Element, got {type(output).__name__}",
                        path="$",
                    )
                )
                return ValidationResult(
                    valid=False,
                    errors=errors,
                    schema_name=self.name,
                    checked_at=datetime.now().isoformat(),
                )
        except ElementTree.ParseError as e:
            errors.append(
                ValidationError(
                    f"XML parse error: {e}",
                    path="$",
                )
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                schema_name=self.name,
                checked_at=datetime.now().isoformat(),
            )

        # Validate root element
        if root.tag != self.root_element:
            errors.append(
                ValidationError(
                    f"Invalid root element",
                    path="$",
                    value=root.tag,
                    expected=self.root_element,
                )
            )

        # Validate required elements
        found_elements = {elem.tag for elem in root.iter()}
        for required in self.required_elements:
            if required not in found_elements:
                errors.append(
                    ValidationError(
                        f"Missing required element: {required}",
                        path="$",
                        expected=required,
                    )
                )

        # Validate required attributes
        for elem_name, attrs in self.required_attributes.items():
            for elem in root.iter(elem_name):
                for attr in attrs:
                    if attr not in elem.attrib:
                        errors.append(
                            ValidationError(
                                f"Missing required attribute: {attr}",
                                path=elem_name,
                                expected=attr,
                            )
                        )

        # Validate element schemas
        for elem_name, schema in self.element_schemas.items():
            for elem in root.iter(elem_name):
                self._validate_element(elem, schema, elem_name, errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            schema_name=self.name,
            checked_at=datetime.now().isoformat(),
        )

    def _validate_element(
        self,
        element: ElementTree.Element,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate element against schema."""
        # Validate text content
        if "text" in schema:
            text_schema = schema["text"]
            text = element.text or ""

            if "pattern" in text_schema:
                if not re.match(text_schema["pattern"], text):
                    errors.append(
                        ValidationError(
                            f"Element text does not match pattern",
                            path=path,
                            value=text,
                            expected=f"pattern: {text_schema['pattern']}",
                        )
                    )

            if "minLength" in text_schema and len(text) < text_schema["minLength"]:
                errors.append(
                    ValidationError(
                        f"Element text too short",
                        path=path,
                        value=text,
                        expected=f"minLength: {text_schema['minLength']}",
                    )
                )

        # Validate child count
        if "minChildren" in schema:
            if len(element) < schema["minChildren"]:
                errors.append(
                    ValidationError(
                        f"Element has too few children",
                        path=path,
                        value=f"{len(element)} children",
                        expected=f"minChildren: {schema['minChildren']}",
                    )
                )

        if "maxChildren" in schema:
            if len(element) > schema["maxChildren"]:
                errors.append(
                    ValidationError(
                        f"Element has too many children",
                        path=path,
                        value=f"{len(element)} children",
                        expected=f"maxChildren: {schema['maxChildren']}",
                    )
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "xml_schema",
            "name": self.name,
            "root_element": self.root_element,
            "required_elements": self.required_elements,
            "required_attributes": self.required_attributes,
            "element_schemas": self.element_schemas,
        }


class CSVSchema(ReportSchema):
    """CSV structure validator for reporter output.

    Validates CSV output including headers, column count,
    and column value constraints.

    Example:
        >>> schema = CSVSchema(
        ...     required_columns=["id", "name", "status"],
        ...     column_types={"id": "integer", "status": "enum:pass,fail"},
        ...     delimiter=",",
        ... )
        >>> result = schema.validate(csv_string)
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        delimiter: str = ",",
        has_header: bool = True,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize CSV Schema validator.

        Args:
            required_columns: List of required column names.
            column_types: Dict mapping column names to type constraints.
            delimiter: CSV delimiter character.
            has_header: Whether CSV has a header row.
            min_rows: Minimum number of data rows.
            max_rows: Maximum number of data rows.
            name: Optional schema name.
        """
        super().__init__(name)
        self.required_columns = required_columns or []
        self.column_types = column_types or {}
        self.delimiter = delimiter
        self.has_header = has_header
        self.min_rows = min_rows
        self.max_rows = max_rows

    def validate(self, output: str) -> ValidationResult:
        """Validate CSV output."""
        import csv
        from datetime import datetime
        from io import StringIO

        errors: List[ValidationError] = []
        warnings: List[str] = []

        if not isinstance(output, str):
            errors.append(
                ValidationError(
                    f"Expected string, got {type(output).__name__}",
                    path="$",
                )
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                schema_name=self.name,
                checked_at=datetime.now().isoformat(),
            )

        try:
            reader = csv.reader(StringIO(output), delimiter=self.delimiter)
            rows = list(reader)
        except Exception as e:
            errors.append(
                ValidationError(
                    f"CSV parse error: {e}",
                    path="$",
                )
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                schema_name=self.name,
                checked_at=datetime.now().isoformat(),
            )

        if not rows:
            if self.required_columns or self.min_rows:
                errors.append(
                    ValidationError(
                        "CSV is empty",
                        path="$",
                    )
                )
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                schema_name=self.name,
                checked_at=datetime.now().isoformat(),
            )

        # Extract header and data
        if self.has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = []
            data_rows = rows

        # Validate required columns
        if self.has_header:
            for col in self.required_columns:
                if col not in headers:
                    errors.append(
                        ValidationError(
                            f"Missing required column: {col}",
                            path="header",
                            expected=col,
                        )
                    )

        # Validate row count
        if self.min_rows is not None and len(data_rows) < self.min_rows:
            errors.append(
                ValidationError(
                    f"Too few data rows",
                    path="$",
                    value=f"{len(data_rows)} rows",
                    expected=f"minRows: {self.min_rows}",
                )
            )

        if self.max_rows is not None and len(data_rows) > self.max_rows:
            errors.append(
                ValidationError(
                    f"Too many data rows",
                    path="$",
                    value=f"{len(data_rows)} rows",
                    expected=f"maxRows: {self.max_rows}",
                )
            )

        # Validate column types
        if self.has_header and self.column_types:
            for i, row in enumerate(data_rows):
                for col_name, type_constraint in self.column_types.items():
                    if col_name in headers:
                        col_idx = headers.index(col_name)
                        if col_idx < len(row):
                            value = row[col_idx]
                            self._validate_column_type(
                                value, type_constraint, f"row[{i}].{col_name}", errors
                            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            schema_name=self.name,
            checked_at=datetime.now().isoformat(),
        )

    def _validate_column_type(
        self,
        value: str,
        type_constraint: str,
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate column value against type constraint."""
        if type_constraint == "integer":
            try:
                int(value)
            except ValueError:
                errors.append(
                    ValidationError(
                        f"Value is not an integer",
                        path=path,
                        value=value,
                        expected="integer",
                    )
                )

        elif type_constraint == "number":
            try:
                float(value)
            except ValueError:
                errors.append(
                    ValidationError(
                        f"Value is not a number",
                        path=path,
                        value=value,
                        expected="number",
                    )
                )

        elif type_constraint == "boolean":
            if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                errors.append(
                    ValidationError(
                        f"Value is not a boolean",
                        path=path,
                        value=value,
                        expected="boolean",
                    )
                )

        elif type_constraint.startswith("enum:"):
            allowed = type_constraint[5:].split(",")
            if value not in allowed:
                errors.append(
                    ValidationError(
                        f"Value not in enum",
                        path=path,
                        value=value,
                        expected=f"one of: {allowed}",
                    )
                )

        elif type_constraint.startswith("pattern:"):
            pattern = type_constraint[8:]
            if not re.match(pattern, value):
                errors.append(
                    ValidationError(
                        f"Value does not match pattern",
                        path=path,
                        value=value,
                        expected=f"pattern: {pattern}",
                    )
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "csv_schema",
            "name": self.name,
            "required_columns": self.required_columns,
            "column_types": self.column_types,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
        }


class TextSchema(ReportSchema):
    """Text/plain format validator for reporter output.

    Validates text output using patterns, line constraints,
    and content requirements.

    Example:
        >>> schema = TextSchema(
        ...     required_patterns=[r"Summary:", r"Total: \d+"],
        ...     forbidden_patterns=[r"ERROR:", r"FATAL:"],
        ...     min_lines=5,
        ... )
        >>> result = schema.validate(text_output)
    """

    def __init__(
        self,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
        min_lines: Optional[int] = None,
        max_lines: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        encoding: str = "utf-8",
        name: Optional[str] = None,
    ) -> None:
        """Initialize Text Schema validator.

        Args:
            required_patterns: Regex patterns that must appear in output.
            forbidden_patterns: Regex patterns that must not appear.
            min_lines: Minimum number of lines.
            max_lines: Maximum number of lines.
            min_length: Minimum total character length.
            max_length: Maximum total character length.
            encoding: Expected text encoding.
            name: Optional schema name.
        """
        super().__init__(name)
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_length = min_length
        self.max_length = max_length
        self.encoding = encoding

    def validate(self, output: Union[str, bytes]) -> ValidationResult:
        """Validate text output."""
        from datetime import datetime

        errors: List[ValidationError] = []
        warnings: List[str] = []

        # Convert bytes to string
        if isinstance(output, bytes):
            try:
                output = output.decode(self.encoding)
            except UnicodeDecodeError as e:
                errors.append(
                    ValidationError(
                        f"Encoding error: {e}",
                        path="$",
                        expected=f"encoding: {self.encoding}",
                    )
                )
                return ValidationResult(
                    valid=False,
                    errors=errors,
                    schema_name=self.name,
                    checked_at=datetime.now().isoformat(),
                )

        if not isinstance(output, str):
            errors.append(
                ValidationError(
                    f"Expected string or bytes, got {type(output).__name__}",
                    path="$",
                )
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                schema_name=self.name,
                checked_at=datetime.now().isoformat(),
            )

        # Validate length
        if self.min_length is not None and len(output) < self.min_length:
            errors.append(
                ValidationError(
                    f"Text too short",
                    path="$",
                    value=f"{len(output)} chars",
                    expected=f"minLength: {self.min_length}",
                )
            )

        if self.max_length is not None and len(output) > self.max_length:
            errors.append(
                ValidationError(
                    f"Text too long",
                    path="$",
                    value=f"{len(output)} chars",
                    expected=f"maxLength: {self.max_length}",
                )
            )

        # Validate line count
        lines = output.splitlines()
        if self.min_lines is not None and len(lines) < self.min_lines:
            errors.append(
                ValidationError(
                    f"Too few lines",
                    path="$",
                    value=f"{len(lines)} lines",
                    expected=f"minLines: {self.min_lines}",
                )
            )

        if self.max_lines is not None and len(lines) > self.max_lines:
            errors.append(
                ValidationError(
                    f"Too many lines",
                    path="$",
                    value=f"{len(lines)} lines",
                    expected=f"maxLines: {self.max_lines}",
                )
            )

        # Validate required patterns
        for pattern in self.required_patterns:
            if not re.search(pattern, output):
                errors.append(
                    ValidationError(
                        f"Required pattern not found",
                        path="$",
                        expected=f"pattern: {pattern}",
                    )
                )

        # Validate forbidden patterns
        for pattern in self.forbidden_patterns:
            match = re.search(pattern, output)
            if match:
                errors.append(
                    ValidationError(
                        f"Forbidden pattern found",
                        path="$",
                        value=match.group(),
                        expected=f"not: {pattern}",
                    )
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            schema_name=self.name,
            checked_at=datetime.now().isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "text_schema",
            "name": self.name,
            "required_patterns": self.required_patterns,
            "forbidden_patterns": self.forbidden_patterns,
            "min_lines": self.min_lines,
            "max_lines": self.max_lines,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "encoding": self.encoding,
        }


# Schema registry
_schema_registry: Dict[str, ReportSchema] = {}


def register_schema(name: str, schema: ReportSchema) -> None:
    """Register a schema for a reporter.

    Args:
        name: Schema name (typically reporter name).
        schema: The schema to register.

    Example:
        >>> schema = JSONSchema({...})
        >>> register_schema("my_reporter", schema)
    """
    _schema_registry[name] = schema


def get_schema(name: str) -> Optional[ReportSchema]:
    """Get a registered schema by name.

    Args:
        name: Schema name to look up.

    Returns:
        The registered schema, or None if not found.
    """
    return _schema_registry.get(name)


def unregister_schema(name: str) -> bool:
    """Unregister a schema.

    Args:
        name: Schema name to unregister.

    Returns:
        True if schema was removed, False if not found.
    """
    if name in _schema_registry:
        del _schema_registry[name]
        return True
    return False


def validate_output(
    output: Any,
    schema: Optional[ReportSchema] = None,
    schema_name: Optional[str] = None,
) -> ValidationResult:
    """Validate output against a schema.

    Args:
        output: The output to validate.
        schema: Schema to validate against.
        schema_name: Name of a registered schema to use.

    Returns:
        ValidationResult with validation status.

    Raises:
        ValueError: If neither schema nor schema_name provided.
    """
    if schema is None and schema_name is not None:
        schema = get_schema(schema_name)
        if schema is None:
            raise ValueError(f"No schema registered with name: {schema_name}")

    if schema is None:
        raise ValueError("Must provide either schema or schema_name")

    return schema.validate(output)


def validate_reporter_output(
    schema: Optional[ReportSchema] = None,
    schema_name: Optional[str] = None,
    raise_on_error: bool = False,
) -> Callable:
    """Decorator to validate reporter output.

    Args:
        schema: Schema to validate against.
        schema_name: Name of registered schema to use.
        raise_on_error: If True, raise SchemaError on validation failure.

    Returns:
        Decorator function.

    Example:
        >>> @validate_reporter_output(schema=my_schema, raise_on_error=True)
        ... def render(self, results):
        ...     return {"summary": ..., "results": ...}
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)

            result = validate_output(output, schema=schema, schema_name=schema_name)

            if raise_on_error:
                result.raise_if_invalid()

            return output

        return wrapper

    return decorator


def infer_schema(output: Any) -> ReportSchema:
    """Infer a schema from sample output.

    This function attempts to generate a schema based on
    the structure of the provided output.

    Args:
        output: Sample output to infer schema from.

    Returns:
        Inferred ReportSchema.

    Example:
        >>> sample = {"name": "test", "count": 5, "items": [1, 2, 3]}
        >>> schema = infer_schema(sample)
        >>> result = schema.validate(new_output)
    """
    if isinstance(output, dict):
        return _infer_json_schema(output)
    elif isinstance(output, str):
        if output.strip().startswith("<"):
            return _infer_xml_schema(output)
        elif "," in output.split("\n")[0]:
            return _infer_csv_schema(output)
        else:
            return _infer_text_schema(output)
    else:
        # Default to JSON schema for other types
        return JSONSchema({"type": _python_to_json_type(type(output))})


def _python_to_json_type(python_type: type) -> str:
    """Convert Python type to JSON Schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    return type_map.get(python_type, "string")


def _infer_json_schema(obj: Dict[str, Any]) -> JSONSchema:
    """Infer JSON Schema from dictionary."""

    def infer_property_schema(value: Any) -> Dict[str, Any]:
        if value is None:
            return {"type": "null"}
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, list):
            if value:
                item_schema = infer_property_schema(value[0])
                return {"type": "array", "items": item_schema}
            return {"type": "array"}
        elif isinstance(value, dict):
            properties = {}
            for k, v in value.items():
                properties[k] = infer_property_schema(v)
            return {"type": "object", "properties": properties}
        return {}

    properties = {}
    required = []

    for key, value in obj.items():
        properties[key] = infer_property_schema(value)
        if value is not None:
            required.append(key)

    return JSONSchema(
        {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        name="inferred",
    )


def _infer_xml_schema(xml_str: str) -> XMLSchema:
    """Infer XML Schema from XML string."""
    try:
        root = ElementTree.fromstring(xml_str)
        elements = {elem.tag for elem in root.iter()}
        elements.discard(root.tag)

        return XMLSchema(
            root_element=root.tag,
            required_elements=list(elements),
            name="inferred",
        )
    except Exception:
        return XMLSchema(root_element="root", name="inferred")


def _infer_csv_schema(csv_str: str) -> CSVSchema:
    """Infer CSV Schema from CSV string."""
    import csv
    from io import StringIO

    try:
        reader = csv.reader(StringIO(csv_str))
        rows = list(reader)

        if rows:
            headers = rows[0]
            return CSVSchema(
                required_columns=headers,
                has_header=True,
                min_rows=max(0, len(rows) - 1),
                name="inferred",
            )
    except Exception:
        pass

    return CSVSchema(name="inferred")


def _infer_text_schema(text: str) -> TextSchema:
    """Infer Text Schema from text string."""
    lines = text.splitlines()

    return TextSchema(
        min_lines=len(lines),
        min_length=len(text) // 2,  # Allow some variation
        max_length=len(text) * 2,
        name="inferred",
    )


def merge_schemas(
    schemas: List[ReportSchema],
    name: Optional[str] = None,
) -> ReportSchema:
    """Merge multiple schemas into a composite schema.

    The merged schema validates against all provided schemas.

    Args:
        schemas: List of schemas to merge.
        name: Optional name for the merged schema.

    Returns:
        A new schema that combines all validations.

    Example:
        >>> schema1 = JSONSchema({"type": "object"})
        >>> schema2 = TextSchema(min_length=10)
        >>> merged = merge_schemas([schema1, schema2])
    """

    class CompositeSchema(ReportSchema):
        def __init__(self, schemas: List[ReportSchema], name: Optional[str]) -> None:
            super().__init__(name or "composite")
            self.schemas = schemas

        def validate(self, output: Any) -> ValidationResult:
            from datetime import datetime

            all_errors: List[ValidationError] = []
            all_warnings: List[str] = []

            for schema in self.schemas:
                result = schema.validate(output)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)

            return ValidationResult(
                valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                schema_name=self.name,
                checked_at=datetime.now().isoformat(),
            )

        def to_dict(self) -> Dict[str, Any]:
            return {
                "type": "composite",
                "name": self.name,
                "schemas": [s.to_dict() for s in self.schemas],
            }

    return CompositeSchema(schemas, name)
