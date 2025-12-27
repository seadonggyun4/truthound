"""Tests for the Reporter SDK schema validation module."""

import json
import pytest
from xml.etree import ElementTree

from truthound.reporters.sdk.schema import (
    # Core classes
    ReportSchema,
    JSONSchema,
    XMLSchema,
    CSVSchema,
    TextSchema,
    # Validation
    ValidationResult,
    ValidationError,
    SchemaError,
    # Functions
    validate_output,
    register_schema,
    get_schema,
    unregister_schema,
    validate_reporter_output,
    infer_schema,
    merge_schemas,
)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ValidationError("Test error")
        assert error.message == "Test error"
        assert error.path is None
        assert str(error) == "Test error"

    def test_error_with_path(self):
        """Test error with path."""
        error = ValidationError("Invalid type", path="data.name")
        assert "data.name" in str(error)

    def test_error_with_expected(self):
        """Test error with expected value."""
        error = ValidationError("Invalid type", path="$", expected="string")
        assert "expected: string" in str(error)

    def test_error_with_value(self):
        """Test error with value."""
        error = ValidationError("Invalid", value=123)
        assert error.value == 123


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(valid=True)
        assert result.valid
        assert len(result.errors) == 0

    def test_invalid_result(self):
        """Test invalid result."""
        errors = [ValidationError("Error 1"), ValidationError("Error 2")]
        result = ValidationResult(valid=False, errors=errors)
        assert not result.valid
        assert len(result.errors) == 2

    def test_raise_if_invalid(self):
        """Test raise_if_invalid method."""
        result = ValidationResult(valid=True)
        result.raise_if_invalid()  # Should not raise

        errors = [ValidationError("Error")]
        result = ValidationResult(valid=False, errors=errors)
        with pytest.raises(SchemaError):
            result.raise_if_invalid()

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = ValidationResult(
            valid=True,
            warnings=["Warning 1"],
            schema_name="test",
        )
        d = result.to_dict()
        assert d["valid"] is True
        assert d["schema_name"] == "test"
        assert "Warning 1" in d["warnings"]


class TestJSONSchema:
    """Tests for JSONSchema class."""

    def test_basic_types(self):
        """Test basic type validation."""
        schema = JSONSchema({"type": "string"})
        assert schema.validate("hello").valid
        assert not schema.validate(123).valid

        schema = JSONSchema({"type": "integer"})
        assert schema.validate(42).valid
        assert not schema.validate("42").valid
        assert not schema.validate(True).valid  # bool should not match integer

        schema = JSONSchema({"type": "boolean"})
        assert schema.validate(True).valid
        assert schema.validate(False).valid
        assert not schema.validate(1).valid

    def test_multiple_types(self):
        """Test multiple allowed types."""
        schema = JSONSchema({"type": ["string", "null"]})
        assert schema.validate("hello").valid
        assert schema.validate(None).valid
        assert not schema.validate(123).valid

    def test_object_validation(self):
        """Test object validation with properties."""
        schema = JSONSchema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        })

        assert schema.validate({"name": "Alice", "age": 30}).valid
        assert schema.validate({"name": "Bob"}).valid
        assert not schema.validate({"age": 30}).valid  # missing name
        assert not schema.validate({"name": 123}).valid  # wrong type

    def test_array_validation(self):
        """Test array validation."""
        schema = JSONSchema({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 5,
        })

        assert schema.validate([1, 2, 3]).valid
        assert not schema.validate([]).valid  # too few
        assert not schema.validate([1, 2, 3, 4, 5, 6]).valid  # too many
        assert not schema.validate([1, "two", 3]).valid  # wrong item type

    def test_string_constraints(self):
        """Test string validation constraints."""
        schema = JSONSchema({
            "type": "string",
            "minLength": 2,
            "maxLength": 10,
            "pattern": r"^[a-z]+$",
        })

        assert schema.validate("hello").valid
        assert not schema.validate("a").valid  # too short
        assert not schema.validate("verylongstring").valid  # too long
        assert not schema.validate("Hello").valid  # pattern mismatch

    def test_number_constraints(self):
        """Test number validation constraints."""
        schema = JSONSchema({
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "multipleOf": 5,
        })

        assert schema.validate(50).valid
        assert schema.validate(0).valid
        assert schema.validate(100).valid
        assert not schema.validate(-10).valid  # below minimum
        assert not schema.validate(110).valid  # above maximum
        assert not schema.validate(42).valid  # not multiple of 5

    def test_exclusive_constraints(self):
        """Test exclusive minimum/maximum."""
        schema = JSONSchema({
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 100,
        })

        assert schema.validate(50).valid
        assert not schema.validate(0).valid  # equals exclusive minimum
        assert not schema.validate(100).valid  # equals exclusive maximum

    def test_enum_validation(self):
        """Test enum validation."""
        schema = JSONSchema({
            "type": "string",
            "enum": ["red", "green", "blue"],
        })

        assert schema.validate("red").valid
        assert not schema.validate("yellow").valid

    def test_const_validation(self):
        """Test const validation."""
        schema = JSONSchema({
            "type": "string",
            "const": "fixed_value",
        })

        assert schema.validate("fixed_value").valid
        assert not schema.validate("other_value").valid

    def test_format_validation(self):
        """Test format validation."""
        # Email
        schema = JSONSchema({"type": "string", "format": "email"})
        assert schema.validate("user@example.com").valid
        assert not schema.validate("not-an-email").valid

        # Date
        schema = JSONSchema({"type": "string", "format": "date"})
        assert schema.validate("2024-01-15").valid
        assert not schema.validate("01-15-2024").valid

        # UUID
        schema = JSONSchema({"type": "string", "format": "uuid"})
        assert schema.validate("123e4567-e89b-12d3-a456-426614174000").valid
        assert not schema.validate("not-a-uuid").valid

    def test_nested_object(self):
        """Test nested object validation."""
        schema = JSONSchema({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["name"],
                },
            },
        })

        valid_data = {"user": {"name": "Alice", "email": "alice@example.com"}}
        assert schema.validate(valid_data).valid

        invalid_data = {"user": {"email": "alice@example.com"}}  # missing name
        result = schema.validate(invalid_data)
        assert not result.valid

    def test_unique_items(self):
        """Test uniqueItems validation."""
        schema = JSONSchema({
            "type": "array",
            "uniqueItems": True,
        })

        assert schema.validate([1, 2, 3]).valid
        assert not schema.validate([1, 2, 2]).valid

    def test_additional_properties_false(self):
        """Test strict mode with additionalProperties."""
        schema = JSONSchema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        )

        assert schema.validate({"name": "test"}).valid
        assert not schema.validate({"name": "test", "extra": "value"}).valid

    def test_strict_mode(self):
        """Test strict mode option."""
        schema = JSONSchema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
            strict=True,
        )

        assert schema.validate({"name": "test"}).valid
        assert not schema.validate({"name": "test", "extra": "value"}).valid

    def test_to_dict(self):
        """Test to_dict method."""
        schema = JSONSchema(
            {"type": "object"},
            name="test_schema",
            strict=True,
        )
        d = schema.to_dict()
        assert d["type"] == "json_schema"
        assert d["name"] == "test_schema"
        assert d["strict"] is True


class TestXMLSchema:
    """Tests for XMLSchema class."""

    def test_basic_xml_validation(self):
        """Test basic XML structure validation."""
        schema = XMLSchema(
            root_element="report",
            required_elements=["summary"],
        )

        valid_xml = "<report><summary>Test</summary></report>"
        assert schema.validate(valid_xml).valid

        invalid_xml = "<report><other>Test</other></report>"
        assert not schema.validate(invalid_xml).valid

    def test_wrong_root_element(self):
        """Test wrong root element detection."""
        schema = XMLSchema(root_element="report")
        result = schema.validate("<data><item/></data>")
        assert not result.valid
        assert any("root element" in str(e).lower() for e in result.errors)

    def test_required_attributes(self):
        """Test required attribute validation."""
        schema = XMLSchema(
            root_element="report",
            required_attributes={"report": ["version"]},
        )

        valid_xml = '<report version="1.0"><data/></report>'
        assert schema.validate(valid_xml).valid

        invalid_xml = "<report><data/></report>"
        assert not schema.validate(invalid_xml).valid

    def test_element_object(self):
        """Test validation with Element object."""
        schema = XMLSchema(root_element="root")
        elem = ElementTree.fromstring("<root><child/></root>")
        assert schema.validate(elem).valid

    def test_invalid_xml(self):
        """Test invalid XML handling."""
        schema = XMLSchema(root_element="root")
        result = schema.validate("<invalid>no closing tag")
        assert not result.valid
        assert any("parse" in str(e).lower() for e in result.errors)

    def test_to_dict(self):
        """Test to_dict method."""
        schema = XMLSchema(
            root_element="report",
            required_elements=["summary"],
            name="xml_test",
        )
        d = schema.to_dict()
        assert d["type"] == "xml_schema"
        assert d["root_element"] == "report"


class TestCSVSchema:
    """Tests for CSVSchema class."""

    def test_basic_csv_validation(self):
        """Test basic CSV validation."""
        schema = CSVSchema(required_columns=["id", "name"])
        csv_data = "id,name,age\n1,Alice,30\n2,Bob,25"
        assert schema.validate(csv_data).valid

    def test_missing_column(self):
        """Test missing required column detection."""
        schema = CSVSchema(required_columns=["id", "name", "email"])
        csv_data = "id,name\n1,Alice\n2,Bob"
        result = schema.validate(csv_data)
        assert not result.valid
        assert any("email" in str(e) for e in result.errors)

    def test_row_count_validation(self):
        """Test min/max row validation."""
        schema = CSVSchema(min_rows=2, max_rows=5)

        valid_csv = "a,b\n1,2\n3,4\n5,6"
        assert schema.validate(valid_csv).valid

        too_few = "a,b\n1,2"  # Only 1 data row
        assert not schema.validate(too_few).valid

        too_many = "a,b\n" + "\n".join(f"{i},{i}" for i in range(10))
        assert not schema.validate(too_many).valid

    def test_column_type_validation(self):
        """Test column type validation."""
        schema = CSVSchema(
            required_columns=["id", "status"],
            column_types={
                "id": "integer",
                "status": "enum:active,inactive",
            },
        )

        valid_csv = "id,status\n1,active\n2,inactive"
        assert schema.validate(valid_csv).valid

        invalid_csv = "id,status\n1,unknown\n2,inactive"
        assert not schema.validate(invalid_csv).valid

    def test_number_type(self):
        """Test number column type."""
        schema = CSVSchema(column_types={"value": "number"})

        valid_csv = "value\n1.5\n2.7\n3.14"
        assert schema.validate(valid_csv).valid

        invalid_csv = "value\n1.5\nabc\n3.14"
        assert not schema.validate(invalid_csv).valid

    def test_pattern_type(self):
        """Test pattern column type."""
        schema = CSVSchema(column_types={"code": r"pattern:[A-Z]{3}\d{3}"})

        valid_csv = "code\nABC123\nDEF456"
        assert schema.validate(valid_csv).valid

        invalid_csv = "code\nabc123\nDEF456"
        assert not schema.validate(invalid_csv).valid

    def test_custom_delimiter(self):
        """Test custom delimiter."""
        schema = CSVSchema(required_columns=["a", "b"], delimiter=";")
        csv_data = "a;b\n1;2\n3;4"
        assert schema.validate(csv_data).valid

    def test_to_dict(self):
        """Test to_dict method."""
        schema = CSVSchema(
            required_columns=["a", "b"],
            delimiter=";",
            name="csv_test",
        )
        d = schema.to_dict()
        assert d["type"] == "csv_schema"
        assert d["delimiter"] == ";"


class TestTextSchema:
    """Tests for TextSchema class."""

    def test_basic_text_validation(self):
        """Test basic text validation."""
        schema = TextSchema(min_length=10, max_length=100)
        assert schema.validate("This is a valid text.").valid
        assert not schema.validate("Short").valid
        assert not schema.validate("x" * 200).valid

    def test_line_count_validation(self):
        """Test line count validation."""
        schema = TextSchema(min_lines=3, max_lines=5)

        valid_text = "Line 1\nLine 2\nLine 3\nLine 4"
        assert schema.validate(valid_text).valid

        too_few = "Line 1\nLine 2"
        assert not schema.validate(too_few).valid

        too_many = "\n".join(f"Line {i}" for i in range(10))
        assert not schema.validate(too_many).valid

    def test_required_patterns(self):
        """Test required pattern validation."""
        schema = TextSchema(required_patterns=[r"Total: \d+", r"Status: (OK|FAILED)"])

        valid_text = "Report\nTotal: 100\nStatus: OK"
        assert schema.validate(valid_text).valid

        invalid_text = "Report\nTotal: 100"  # Missing status
        assert not schema.validate(invalid_text).valid

    def test_forbidden_patterns(self):
        """Test forbidden pattern validation."""
        schema = TextSchema(forbidden_patterns=[r"ERROR", r"FATAL"])

        valid_text = "All operations completed successfully."
        assert schema.validate(valid_text).valid

        invalid_text = "ERROR: Something went wrong."
        result = schema.validate(invalid_text)
        assert not result.valid

    def test_bytes_input(self):
        """Test bytes input with encoding."""
        schema = TextSchema(min_length=5)

        valid_bytes = b"Hello World"
        assert schema.validate(valid_bytes).valid

    def test_encoding_error(self):
        """Test encoding error handling."""
        schema = TextSchema(encoding="ascii")
        invalid_bytes = "Hello 世界".encode("utf-8")
        result = schema.validate(invalid_bytes)
        assert not result.valid

    def test_to_dict(self):
        """Test to_dict method."""
        schema = TextSchema(
            required_patterns=[r"\d+"],
            min_lines=1,
            name="text_test",
        )
        d = schema.to_dict()
        assert d["type"] == "text_schema"
        assert d["min_lines"] == 1


class TestSchemaRegistry:
    """Tests for schema registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        # Clear any existing schemas
        from truthound.reporters.sdk import schema
        schema._schema_registry.clear()

    def test_register_and_get(self):
        """Test registering and getting schemas."""
        test_schema = JSONSchema({"type": "object"}, name="test")
        register_schema("my_reporter", test_schema)

        retrieved = get_schema("my_reporter")
        assert retrieved is test_schema

    def test_get_nonexistent(self):
        """Test getting nonexistent schema."""
        assert get_schema("nonexistent") is None

    def test_unregister(self):
        """Test unregistering schema."""
        test_schema = JSONSchema({"type": "object"})
        register_schema("temp", test_schema)

        assert unregister_schema("temp")
        assert get_schema("temp") is None
        assert not unregister_schema("temp")  # Already removed

    def test_validate_output_with_name(self):
        """Test validate_output with schema name."""
        test_schema = JSONSchema({"type": "string"})
        register_schema("string_schema", test_schema)

        result = validate_output("hello", schema_name="string_schema")
        assert result.valid

    def test_validate_output_with_schema(self):
        """Test validate_output with schema object."""
        test_schema = JSONSchema({"type": "integer"})
        result = validate_output(42, schema=test_schema)
        assert result.valid

    def test_validate_output_errors(self):
        """Test validate_output error cases."""
        with pytest.raises(ValueError):
            validate_output("test")  # No schema provided

        with pytest.raises(ValueError):
            validate_output("test", schema_name="nonexistent")


class TestValidateReporterOutputDecorator:
    """Tests for the validate_reporter_output decorator."""

    def test_decorator_passes(self):
        """Test decorator with valid output."""
        schema = JSONSchema({"type": "object"})

        @validate_reporter_output(schema=schema)
        def render():
            return {}

        result = render()
        assert result == {}

    def test_decorator_raises(self):
        """Test decorator raises on invalid output."""
        schema = JSONSchema({"type": "object"})

        @validate_reporter_output(schema=schema, raise_on_error=True)
        def render():
            return "not an object"

        with pytest.raises(SchemaError):
            render()


class TestInferSchema:
    """Tests for schema inference."""

    def test_infer_from_dict(self):
        """Test schema inference from dictionary."""
        data = {"name": "test", "count": 5, "items": [1, 2, 3]}
        schema = infer_schema(data)

        assert isinstance(schema, JSONSchema)
        # Should validate similar structure
        assert schema.validate({"name": "other", "count": 10, "items": [4, 5]}).valid

    def test_infer_from_xml(self):
        """Test schema inference from XML."""
        xml_str = "<root><child>value</child></root>"
        schema = infer_schema(xml_str)

        assert isinstance(schema, XMLSchema)

    def test_infer_from_csv(self):
        """Test schema inference from CSV."""
        csv_str = "a,b,c\n1,2,3\n4,5,6"
        schema = infer_schema(csv_str)

        assert isinstance(schema, CSVSchema)

    def test_infer_from_text(self):
        """Test schema inference from plain text."""
        text = "This is just some text\nwith multiple lines."
        schema = infer_schema(text)

        assert isinstance(schema, TextSchema)


class TestMergeSchemas:
    """Tests for schema merging."""

    def test_merge_two_schemas(self):
        """Test merging two schemas."""
        schema1 = JSONSchema({"type": "object"})
        schema2 = TextSchema(min_length=10)

        merged = merge_schemas([schema1, schema2], name="merged")

        # Should validate against both
        result = merged.validate({"key": "value"})
        # JSON passes but TextSchema should fail (dict is not string)
        assert not result.valid  # TextSchema expects string

    def test_merge_compatible_schemas(self):
        """Test merging compatible schemas."""
        schema1 = TextSchema(min_length=5)
        schema2 = TextSchema(required_patterns=[r"\d+"])

        merged = merge_schemas([schema1, schema2])

        # Valid for both
        assert merged.validate("Total: 100").valid

        # Invalid for schema1 (too short)
        assert not merged.validate("1").valid

        # Invalid for schema2 (no digits)
        assert not merged.validate("hello world").valid

    def test_merge_to_dict(self):
        """Test merged schema to_dict."""
        schema1 = JSONSchema({"type": "object"})
        schema2 = TextSchema()

        merged = merge_schemas([schema1, schema2], name="test_merged")
        d = merged.to_dict()

        assert d["type"] == "composite"
        assert d["name"] == "test_merged"
        assert len(d["schemas"]) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_json_schema(self):
        """Test empty JSON schema (accepts anything)."""
        schema = JSONSchema({})
        assert schema.validate("anything").valid
        assert schema.validate(123).valid
        assert schema.validate([1, 2, 3]).valid

    def test_deeply_nested_validation(self):
        """Test deeply nested structure validation."""
        schema = JSONSchema({
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "string",
                                },
                            },
                        },
                    },
                },
            },
        })

        valid_data = {"level1": {"level2": {"level3": "value"}}}
        assert schema.validate(valid_data).valid

        invalid_data = {"level1": {"level2": {"level3": 123}}}
        result = schema.validate(invalid_data)
        assert not result.valid
        assert "level3" in result.errors[0].path

    def test_empty_csv(self):
        """Test empty CSV validation."""
        schema = CSVSchema()
        assert schema.validate("").valid

        schema_with_requirements = CSVSchema(required_columns=["id"])
        assert not schema_with_requirements.validate("").valid

    def test_non_string_input_to_text_schema(self):
        """Test non-string input to TextSchema."""
        schema = TextSchema()
        result = schema.validate(123)
        assert not result.valid
