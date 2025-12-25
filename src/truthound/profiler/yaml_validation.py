"""YAML validation and error handling for custom patterns.

This module provides comprehensive YAML validation with detailed,
user-friendly error messages for pattern configuration files.

Key Features:
- JSON Schema-based validation
- Detailed error messages with line numbers
- Suggestions for common mistakes
- Path-based error reporting
- YAML syntax error handling

Example:
    from truthound.profiler.yaml_validation import (
        YAMLValidator,
        ValidationError,
        validate_pattern_yaml,
    )

    # Validate YAML content
    try:
        result = validate_pattern_yaml(yaml_content)
    except ValidationError as e:
        print(e.format_error())

    # With file path for better error messages
    result = validate_pattern_yaml(yaml_content, source_path="patterns.yaml")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Error Types
# =============================================================================


class ErrorSeverity(str, Enum):
    """Severity levels for validation errors."""

    ERROR = "error"  # Must be fixed
    WARNING = "warning"  # Should be fixed
    INFO = "info"  # Informational


class ErrorCode(str, Enum):
    """Error codes for categorization."""

    # YAML Syntax
    YAML_SYNTAX = "YAML001"
    YAML_ENCODING = "YAML002"
    YAML_STRUCTURE = "YAML003"

    # Schema
    MISSING_REQUIRED = "SCHEMA001"
    INVALID_TYPE = "SCHEMA002"
    INVALID_VALUE = "SCHEMA003"
    UNKNOWN_FIELD = "SCHEMA004"
    CONSTRAINT_VIOLATION = "SCHEMA005"

    # Pattern Specific
    INVALID_REGEX = "PATTERN001"
    REGEX_COMPLEXITY = "PATTERN002"
    EXAMPLE_MISMATCH = "PATTERN003"
    DUPLICATE_ID = "PATTERN004"
    CIRCULAR_EXTENDS = "PATTERN005"

    # General
    FILE_NOT_FOUND = "FILE001"
    PERMISSION_DENIED = "FILE002"
    ENCODING_ERROR = "FILE003"


@dataclass
class SourceLocation:
    """Location in source file.

    Attributes:
        line: Line number (1-based)
        column: Column number (1-based)
        path: JSON path to the error location
    """

    line: int = 0
    column: int = 0
    path: str = ""

    def __str__(self) -> str:
        if self.line and self.column:
            return f"line {self.line}, column {self.column}"
        elif self.line:
            return f"line {self.line}"
        elif self.path:
            return f"at {self.path}"
        return "unknown location"


@dataclass
class ValidationError:
    """Detailed validation error.

    Attributes:
        code: Error code for categorization
        message: Human-readable error message
        severity: Error severity
        location: Source location
        context: Contextual information
        suggestion: Suggested fix
        source_snippet: Relevant source code snippet
    """

    code: ErrorCode
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    location: SourceLocation = field(default_factory=SourceLocation)
    context: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    source_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "location": {
                "line": self.location.line,
                "column": self.location.column,
                "path": self.location.path,
            },
            "context": self.context,
            "suggestion": self.suggestion,
        }

    def format_error(self, source_path: str = "") -> str:
        """Format error for display.

        Args:
            source_path: Path to source file for context

        Returns:
            Formatted error message
        """
        lines = []

        # Header with severity and code
        severity_symbol = {
            ErrorSeverity.ERROR: "✗",
            ErrorSeverity.WARNING: "⚠",
            ErrorSeverity.INFO: "ℹ",
        }[self.severity]

        header = f"{severity_symbol} [{self.code.value}] {self.message}"
        lines.append(header)

        # Location
        if source_path:
            loc_str = str(self.location) if self.location.line else ""
            if loc_str:
                lines.append(f"   --> {source_path}:{loc_str}")
            else:
                lines.append(f"   --> {source_path}")
        elif self.location.path:
            lines.append(f"   --> at: {self.location.path}")

        # Source snippet
        if self.source_snippet:
            lines.append("")
            for i, snippet_line in enumerate(self.source_snippet.split("\n")):
                if self.location.line:
                    line_num = self.location.line + i
                    lines.append(f"    {line_num} | {snippet_line}")
                else:
                    lines.append(f"    | {snippet_line}")

            # Pointer to error column
            if self.location.column:
                pointer = " " * (self.location.column + 5) + "^"
                lines.append(pointer)

        # Suggestion
        if self.suggestion:
            lines.append("")
            lines.append(f"   💡 Suggestion: {self.suggestion}")

        return "\n".join(lines)


class YAMLValidationException(Exception):
    """Exception raised for YAML validation errors.

    Attributes:
        errors: List of validation errors
        source_path: Path to source file
    """

    def __init__(
        self,
        errors: list[ValidationError],
        source_path: str = "",
    ):
        self.errors = errors
        self.source_path = source_path

        # Create summary message
        error_count = sum(1 for e in errors if e.severity == ErrorSeverity.ERROR)
        warning_count = sum(1 for e in errors if e.severity == ErrorSeverity.WARNING)

        message = f"{error_count} error(s)"
        if warning_count:
            message += f", {warning_count} warning(s)"

        super().__init__(message)

    def format_errors(self) -> str:
        """Format all errors for display."""
        lines = []

        for error in self.errors:
            lines.append(error.format_error(self.source_path))
            lines.append("")

        # Summary
        error_count = sum(1 for e in self.errors if e.severity == ErrorSeverity.ERROR)
        warning_count = sum(1 for e in self.errors if e.severity == ErrorSeverity.WARNING)

        lines.append(f"Found {error_count} error(s) and {warning_count} warning(s)")

        return "\n".join(lines)


# =============================================================================
# Schema Definition
# =============================================================================


PATTERN_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+$",
            "description": "Schema version (e.g., '1.0')",
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "description": "Configuration name",
        },
        "description": {
            "type": "string",
            "maxLength": 1000,
            "description": "Configuration description",
        },
        "extends": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parent configurations to inherit from",
        },
        "patterns": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["regex"],
                "properties": {
                    "name": {"type": "string"},
                    "regex": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 100},
                    "data_type": {"type": "string"},
                    "min_match_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "description": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "required": ["value"],
                                    "properties": {
                                        "value": {"type": "string"},
                                        "should_match": {"type": "boolean"},
                                        "description": {"type": "string"},
                                    },
                                },
                            ],
                        },
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "enabled": {"type": "boolean"},
                    "case_sensitive": {"type": "boolean"},
                    "multiline": {"type": "boolean"},
                },
            },
            "description": "Pattern definitions",
        },
        "groups": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "priority_boost": {"type": "integer"},
                    "patterns": {"type": "object"},
                },
            },
            "description": "Pattern groups",
        },
        "metadata": {
            "type": "object",
            "description": "Additional metadata",
        },
    },
}


# =============================================================================
# YAML Parser with Error Tracking
# =============================================================================


class YAMLParseResult:
    """Result of YAML parsing with error tracking.

    Attributes:
        data: Parsed data (if successful)
        errors: List of parse errors
        line_mapping: Mapping of JSON paths to line numbers
    """

    def __init__(self):
        self.data: dict[str, Any] | None = None
        self.errors: list[ValidationError] = []
        self.line_mapping: dict[str, int] = {}

    @property
    def success(self) -> bool:
        """Check if parsing was successful."""
        return self.data is not None and not any(
            e.severity == ErrorSeverity.ERROR for e in self.errors
        )


def parse_yaml_with_locations(content: str) -> YAMLParseResult:
    """Parse YAML content and track locations.

    Args:
        content: YAML content string

    Returns:
        YAMLParseResult with data and location mapping
    """
    result = YAMLParseResult()

    try:
        import yaml

        # Try to use ruamel.yaml for better error messages
        try:
            from ruamel.yaml import YAML
            from ruamel.yaml.error import YAMLError

            yaml_parser = YAML()
            yaml_parser.preserve_quotes = True

            try:
                data = yaml_parser.load(content)
                result.data = dict(data) if data else {}
                _extract_line_mapping(data, result.line_mapping)
            except YAMLError as e:
                result.errors.append(_convert_ruamel_error(e))

        except ImportError:
            # Fall back to PyYAML
            try:
                result.data = yaml.safe_load(content) or {}
                _estimate_line_mapping(content, result.data, result.line_mapping)
            except yaml.YAMLError as e:
                result.errors.append(_convert_pyyaml_error(e))

    except ImportError:
        result.errors.append(ValidationError(
            code=ErrorCode.YAML_SYNTAX,
            message="No YAML parser available. Install PyYAML: pip install pyyaml",
            severity=ErrorSeverity.ERROR,
            suggestion="Run: pip install pyyaml",
        ))

    return result


def _extract_line_mapping(
    data: Any,
    mapping: dict[str, int],
    path: str = "",
) -> None:
    """Extract line mapping from ruamel.yaml data."""
    if hasattr(data, "lc"):
        line = getattr(data.lc, "line", 0)
        if line:
            mapping[path] = line + 1  # Convert to 1-based

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            _extract_line_mapping(value, mapping, new_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            _extract_line_mapping(item, mapping, new_path)


def _estimate_line_mapping(
    content: str,
    data: Any,
    mapping: dict[str, int],
    path: str = "",
) -> None:
    """Estimate line mapping by searching for keys in content."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key

            # Search for key in content
            pattern = rf'^\s*{re.escape(key)}\s*:'
            for i, line in enumerate(content.split('\n'), 1):
                if re.match(pattern, line):
                    mapping[new_path] = i
                    break

            _estimate_line_mapping(content, value, mapping, new_path)


def _convert_pyyaml_error(error: Any) -> ValidationError:
    """Convert PyYAML error to ValidationError."""
    import yaml

    line = 0
    column = 0

    if hasattr(error, 'problem_mark') and error.problem_mark:
        line = error.problem_mark.line + 1
        column = error.problem_mark.column + 1

    message = str(error.problem) if hasattr(error, 'problem') else str(error)

    # Create helpful suggestions based on common errors
    suggestion = ""
    if "could not find expected ':'" in message:
        suggestion = "Check for missing colons after keys"
    elif "found character" in message and "cannot start" in message:
        suggestion = "Special characters may need to be quoted"
    elif "expected" in message.lower() and "block end" in message.lower():
        suggestion = "Check indentation - YAML uses spaces, not tabs"

    return ValidationError(
        code=ErrorCode.YAML_SYNTAX,
        message=message,
        severity=ErrorSeverity.ERROR,
        location=SourceLocation(line=line, column=column),
        suggestion=suggestion,
    )


def _convert_ruamel_error(error: Any) -> ValidationError:
    """Convert ruamel.yaml error to ValidationError."""
    line = 0
    column = 0

    if hasattr(error, 'problem_mark') and error.problem_mark:
        line = error.problem_mark.line + 1
        column = error.problem_mark.column + 1

    return ValidationError(
        code=ErrorCode.YAML_SYNTAX,
        message=str(error),
        severity=ErrorSeverity.ERROR,
        location=SourceLocation(line=line, column=column),
    )


# =============================================================================
# Schema Validator
# =============================================================================


class SchemaValidator:
    """Validates data against JSON Schema with detailed errors.

    Provides better error messages than standard JSON Schema validators.
    """

    def __init__(self, schema: dict[str, Any]):
        """Initialize validator.

        Args:
            schema: JSON Schema dictionary
        """
        self.schema = schema

    def validate(
        self,
        data: dict[str, Any],
        line_mapping: dict[str, int] | None = None,
    ) -> list[ValidationError]:
        """Validate data against schema.

        Args:
            data: Data to validate
            line_mapping: Optional mapping of paths to line numbers

        Returns:
            List of validation errors
        """
        errors = []
        line_mapping = line_mapping or {}

        self._validate_object(
            data,
            self.schema,
            path="",
            errors=errors,
            line_mapping=line_mapping,
        )

        return errors

    def _validate_object(
        self,
        data: Any,
        schema: dict[str, Any],
        path: str,
        errors: list[ValidationError],
        line_mapping: dict[str, int],
    ) -> None:
        """Validate an object against schema."""
        schema_type = schema.get("type", "any")

        # Type check
        if not self._check_type(data, schema_type):
            expected = schema_type
            actual = type(data).__name__
            errors.append(ValidationError(
                code=ErrorCode.INVALID_TYPE,
                message=f"Expected type '{expected}', got '{actual}'",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(path, 0),
                    path=path,
                ),
                context={"expected": expected, "actual": actual},
                suggestion=self._get_type_suggestion(schema_type, data),
            ))
            return

        if schema_type == "object" and isinstance(data, dict):
            self._validate_object_properties(
                data, schema, path, errors, line_mapping
            )

        elif schema_type == "array" and isinstance(data, list):
            self._validate_array_items(
                data, schema, path, errors, line_mapping
            )

        elif schema_type == "string" and isinstance(data, str):
            self._validate_string(data, schema, path, errors, line_mapping)

        elif schema_type == "number" or schema_type == "integer":
            self._validate_number(data, schema, path, errors, line_mapping)

    def _validate_object_properties(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        path: str,
        errors: list[ValidationError],
        line_mapping: dict[str, int],
    ) -> None:
        """Validate object properties."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_props = schema.get("additionalProperties", True)

        # Check required properties
        for req in required:
            if req not in data:
                errors.append(ValidationError(
                    code=ErrorCode.MISSING_REQUIRED,
                    message=f"Missing required field: '{req}'",
                    severity=ErrorSeverity.ERROR,
                    location=SourceLocation(
                        line=line_mapping.get(path, 0),
                        path=path,
                    ),
                    context={"field": req},
                    suggestion=f"Add the required field '{req}'",
                ))

        # Validate each property
        for key, value in data.items():
            prop_path = f"{path}.{key}" if path else key

            if key in properties:
                self._validate_object(
                    value,
                    properties[key],
                    prop_path,
                    errors,
                    line_mapping,
                )
            elif isinstance(additional_props, dict):
                self._validate_object(
                    value,
                    additional_props,
                    prop_path,
                    errors,
                    line_mapping,
                )
            elif not additional_props:
                errors.append(ValidationError(
                    code=ErrorCode.UNKNOWN_FIELD,
                    message=f"Unknown field: '{key}'",
                    severity=ErrorSeverity.WARNING,
                    location=SourceLocation(
                        line=line_mapping.get(prop_path, 0),
                        path=prop_path,
                    ),
                    suggestion=f"Remove unknown field or check spelling",
                ))

    def _validate_array_items(
        self,
        data: list[Any],
        schema: dict[str, Any],
        path: str,
        errors: list[ValidationError],
        line_mapping: dict[str, int],
    ) -> None:
        """Validate array items."""
        items_schema = schema.get("items")

        if items_schema:
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"

                # Handle oneOf
                if "oneOf" in items_schema:
                    valid = False
                    for option in items_schema["oneOf"]:
                        test_errors: list[ValidationError] = []
                        self._validate_object(
                            item, option, item_path, test_errors, line_mapping
                        )
                        if not test_errors:
                            valid = True
                            break

                    if not valid:
                        errors.append(ValidationError(
                            code=ErrorCode.INVALID_VALUE,
                            message="Value doesn't match any allowed format",
                            severity=ErrorSeverity.ERROR,
                            location=SourceLocation(
                                line=line_mapping.get(item_path, 0),
                                path=item_path,
                            ),
                        ))
                else:
                    self._validate_object(
                        item, items_schema, item_path, errors, line_mapping
                    )

    def _validate_string(
        self,
        data: str,
        schema: dict[str, Any],
        path: str,
        errors: list[ValidationError],
        line_mapping: dict[str, int],
    ) -> None:
        """Validate string value."""
        min_length = schema.get("minLength", 0)
        max_length = schema.get("maxLength", float("inf"))
        pattern = schema.get("pattern")

        if len(data) < min_length:
            errors.append(ValidationError(
                code=ErrorCode.CONSTRAINT_VIOLATION,
                message=f"String too short (min: {min_length})",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(path, 0),
                    path=path,
                ),
            ))

        if len(data) > max_length:
            errors.append(ValidationError(
                code=ErrorCode.CONSTRAINT_VIOLATION,
                message=f"String too long (max: {max_length})",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(path, 0),
                    path=path,
                ),
            ))

        if pattern and not re.match(pattern, data):
            errors.append(ValidationError(
                code=ErrorCode.CONSTRAINT_VIOLATION,
                message=f"String doesn't match pattern: {pattern}",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(path, 0),
                    path=path,
                ),
            ))

    def _validate_number(
        self,
        data: int | float,
        schema: dict[str, Any],
        path: str,
        errors: list[ValidationError],
        line_mapping: dict[str, int],
    ) -> None:
        """Validate number value."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        if minimum is not None and data < minimum:
            errors.append(ValidationError(
                code=ErrorCode.CONSTRAINT_VIOLATION,
                message=f"Value {data} is below minimum {minimum}",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(path, 0),
                    path=path,
                ),
            ))

        if maximum is not None and data > maximum:
            errors.append(ValidationError(
                code=ErrorCode.CONSTRAINT_VIOLATION,
                message=f"Value {data} exceeds maximum {maximum}",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(path, 0),
                    path=path,
                ),
            ))

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if expected_type == "any":
            return True

        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected = type_mapping.get(expected_type)
        if expected is None:
            return True

        # Special case: integers are valid numbers
        if expected_type == "number" and isinstance(value, bool):
            return False

        return isinstance(value, expected)

    def _get_type_suggestion(self, expected: str, actual: Any) -> str:
        """Get suggestion for type mismatch."""
        suggestions = {
            "string": "Wrap value in quotes",
            "integer": "Remove quotes or decimal point",
            "number": "Use a numeric value",
            "boolean": "Use 'true' or 'false' (lowercase)",
            "array": "Use YAML list syntax (- item)",
            "object": "Use YAML object syntax (key: value)",
        }
        return suggestions.get(expected, "")


# =============================================================================
# Pattern-Specific Validation
# =============================================================================


class PatternValidator:
    """Validates pattern-specific rules."""

    def validate(
        self,
        data: dict[str, Any],
        line_mapping: dict[str, int] | None = None,
    ) -> list[ValidationError]:
        """Validate pattern configuration.

        Args:
            data: Parsed pattern configuration
            line_mapping: Line number mapping

        Returns:
            List of validation errors
        """
        errors = []
        line_mapping = line_mapping or {}
        seen_ids: set[str] = set()

        patterns = data.get("patterns", {})
        groups = data.get("groups", {})

        # Validate patterns
        for pattern_id, pattern in patterns.items():
            path = f"patterns.{pattern_id}"
            errors.extend(self._validate_pattern(
                pattern_id, pattern, path, line_mapping
            ))

            # Check for duplicates
            if pattern_id in seen_ids:
                errors.append(ValidationError(
                    code=ErrorCode.DUPLICATE_ID,
                    message=f"Duplicate pattern ID: '{pattern_id}'",
                    severity=ErrorSeverity.ERROR,
                    location=SourceLocation(
                        line=line_mapping.get(path, 0),
                        path=path,
                    ),
                ))
            seen_ids.add(pattern_id)

        # Validate groups
        for group_id, group in groups.items():
            group_path = f"groups.{group_id}"

            if "patterns" in group:
                for pattern_id, pattern in group["patterns"].items():
                    pattern_path = f"{group_path}.patterns.{pattern_id}"
                    errors.extend(self._validate_pattern(
                        pattern_id, pattern, pattern_path, line_mapping
                    ))

                    if pattern_id in seen_ids:
                        errors.append(ValidationError(
                            code=ErrorCode.DUPLICATE_ID,
                            message=f"Duplicate pattern ID: '{pattern_id}'",
                            severity=ErrorSeverity.WARNING,
                            location=SourceLocation(
                                line=line_mapping.get(pattern_path, 0),
                                path=pattern_path,
                            ),
                            suggestion="Pattern IDs should be unique across all groups",
                        ))
                    seen_ids.add(pattern_id)

        # Check for circular extends
        extends = data.get("extends", [])
        if extends:
            errors.extend(self._check_circular_extends(extends, line_mapping))

        return errors

    def _validate_pattern(
        self,
        pattern_id: str,
        pattern: dict[str, Any],
        path: str,
        line_mapping: dict[str, int],
    ) -> list[ValidationError]:
        """Validate a single pattern."""
        errors = []

        # Validate regex
        regex = pattern.get("regex", "")
        if regex:
            regex_errors = self._validate_regex(regex, path, line_mapping)
            errors.extend(regex_errors)

            # Validate examples if regex is valid
            if not regex_errors:
                errors.extend(self._validate_examples(
                    regex, pattern.get("examples", []), path, line_mapping
                ))

        return errors

    def _validate_regex(
        self,
        regex: str,
        path: str,
        line_mapping: dict[str, int],
    ) -> list[ValidationError]:
        """Validate regex pattern."""
        errors = []

        try:
            compiled = re.compile(regex)

            # Check for potentially problematic patterns
            if regex.startswith(".*") and regex.endswith(".*"):
                errors.append(ValidationError(
                    code=ErrorCode.REGEX_COMPLEXITY,
                    message="Pattern starts and ends with '.*' - may match unintended strings",
                    severity=ErrorSeverity.WARNING,
                    location=SourceLocation(
                        line=line_mapping.get(f"{path}.regex", 0),
                        path=f"{path}.regex",
                    ),
                    suggestion="Consider using anchors (^ and $) for more precise matching",
                ))

            # Check for catastrophic backtracking potential
            if re.search(r"\(.*\+.*\)\+|\(.*\*.*\)\*", regex):
                errors.append(ValidationError(
                    code=ErrorCode.REGEX_COMPLEXITY,
                    message="Pattern may cause catastrophic backtracking",
                    severity=ErrorSeverity.WARNING,
                    location=SourceLocation(
                        line=line_mapping.get(f"{path}.regex", 0),
                        path=f"{path}.regex",
                    ),
                    suggestion="Avoid nested quantifiers like (a+)+",
                ))

        except re.error as e:
            errors.append(ValidationError(
                code=ErrorCode.INVALID_REGEX,
                message=f"Invalid regular expression: {e}",
                severity=ErrorSeverity.ERROR,
                location=SourceLocation(
                    line=line_mapping.get(f"{path}.regex", 0),
                    path=f"{path}.regex",
                ),
                context={"regex": regex, "error": str(e)},
                suggestion=self._get_regex_suggestion(str(e)),
            ))

        return errors

    def _validate_examples(
        self,
        regex: str,
        examples: list[Any],
        path: str,
        line_mapping: dict[str, int],
    ) -> list[ValidationError]:
        """Validate examples against regex."""
        errors = []

        try:
            compiled = re.compile(regex)
        except re.error:
            return errors  # Regex error already reported

        for i, example in enumerate(examples):
            example_path = f"{path}.examples[{i}]"

            if isinstance(example, str):
                value = example
                should_match = True
            elif isinstance(example, dict):
                value = example.get("value", "")
                should_match = example.get("should_match", True)
            else:
                continue

            actual_match = bool(compiled.match(value))

            if actual_match != should_match:
                expected = "match" if should_match else "not match"
                actual = "matches" if actual_match else "doesn't match"

                errors.append(ValidationError(
                    code=ErrorCode.EXAMPLE_MISMATCH,
                    message=f"Example '{value}' should {expected} but {actual}",
                    severity=ErrorSeverity.ERROR,
                    location=SourceLocation(
                        line=line_mapping.get(example_path, 0),
                        path=example_path,
                    ),
                    context={
                        "value": value,
                        "should_match": should_match,
                        "actual_match": actual_match,
                    },
                ))

        return errors

    def _check_circular_extends(
        self,
        extends: list[str],
        line_mapping: dict[str, int],
    ) -> list[ValidationError]:
        """Check for circular extends references."""
        # Note: Full circular detection would require loading referenced files
        # This is a placeholder for the detection logic
        return []

    def _get_regex_suggestion(self, error_message: str) -> str:
        """Get suggestion for regex error."""
        suggestions = {
            "unterminated": "Check for missing closing brackets, parentheses, or quotes",
            "unbalanced": "Count opening and closing brackets/parentheses",
            "nothing to repeat": "Quantifiers (+, *, ?) need something to repeat",
            "bad escape": "Use double backslash (\\\\) or raw string (r'...')",
            "unknown group": "Check group reference syntax: (?P<name>...) or (?:...)",
        }

        for pattern, suggestion in suggestions.items():
            if pattern in error_message.lower():
                return suggestion

        return "Check regex syntax at https://regex101.com"


# =============================================================================
# Main Validation Function
# =============================================================================


def validate_pattern_yaml(
    content: str,
    source_path: str = "",
    strict: bool = False,
) -> dict[str, Any]:
    """Validate pattern YAML content.

    Args:
        content: YAML content string
        source_path: Source file path for error messages
        strict: Treat warnings as errors

    Returns:
        Parsed and validated data

    Raises:
        YAMLValidationException: If validation fails
    """
    # Parse YAML
    parse_result = parse_yaml_with_locations(content)

    if not parse_result.success:
        raise YAMLValidationException(parse_result.errors, source_path)

    # Schema validation
    schema_validator = SchemaValidator(PATTERN_SCHEMA)
    schema_errors = schema_validator.validate(
        parse_result.data,
        parse_result.line_mapping,
    )

    # Pattern-specific validation
    pattern_validator = PatternValidator()
    pattern_errors = pattern_validator.validate(
        parse_result.data,
        parse_result.line_mapping,
    )

    all_errors = parse_result.errors + schema_errors + pattern_errors

    # Filter by severity
    if strict:
        errors = all_errors
    else:
        errors = [e for e in all_errors if e.severity == ErrorSeverity.ERROR]

    if errors:
        raise YAMLValidationException(all_errors, source_path)

    return parse_result.data


def validate_pattern_file(
    file_path: str | Path,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate a pattern YAML file.

    Args:
        file_path: Path to YAML file
        strict: Treat warnings as errors

    Returns:
        Parsed and validated data

    Raises:
        YAMLValidationException: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise YAMLValidationException(
            [ValidationError(
                code=ErrorCode.FILE_NOT_FOUND,
                message=f"File not found: {file_path}",
                severity=ErrorSeverity.ERROR,
            )],
            str(file_path),
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except PermissionError:
        raise YAMLValidationException(
            [ValidationError(
                code=ErrorCode.PERMISSION_DENIED,
                message=f"Permission denied: {file_path}",
                severity=ErrorSeverity.ERROR,
            )],
            str(file_path),
        )
    except UnicodeDecodeError as e:
        raise YAMLValidationException(
            [ValidationError(
                code=ErrorCode.ENCODING_ERROR,
                message=f"Encoding error: {e}",
                severity=ErrorSeverity.ERROR,
                suggestion="Ensure the file is saved as UTF-8",
            )],
            str(file_path),
        )

    return validate_pattern_yaml(content, str(file_path), strict)
