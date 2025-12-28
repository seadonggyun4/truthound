"""Tests for validator internationalization (i18n) system.

This test suite covers:
- Message codes and templates
- Locale management
- Message formatting
- Catalog operations
- Builder pattern
"""

import pytest
from pathlib import Path
import tempfile
import json

from truthound.validators.i18n import (
    ValidatorMessageCode,
    ValidatorI18n,
    get_validator_message,
    set_validator_locale,
    get_validator_locale,
    format_issue_message,
    ValidatorMessageCatalog,
    get_default_messages,
    get_korean_messages,
    get_japanese_messages,
    get_chinese_messages,
    get_german_messages,
    get_french_messages,
    get_spanish_messages,
)
from truthound.validators.i18n.catalogs import (
    CatalogBuilder,
    get_all_catalogs,
    get_supported_locales,
    create_custom_catalog,
)


class TestValidatorMessageCode:
    """Test ValidatorMessageCode enum."""

    def test_null_codes_exist(self):
        """Test null-related codes exist."""
        assert ValidatorMessageCode.NULL_VALUES_FOUND.value == "null.values_found"
        assert ValidatorMessageCode.NULL_COLUMN_EMPTY.value == "null.column_empty"
        assert ValidatorMessageCode.NULL_ABOVE_THRESHOLD.value == "null.above_threshold"

    def test_unique_codes_exist(self):
        """Test uniqueness-related codes exist."""
        assert ValidatorMessageCode.UNIQUE_DUPLICATES_FOUND.value == "unique.duplicates_found"
        assert ValidatorMessageCode.UNIQUE_KEY_VIOLATION.value == "unique.key_violation"

    def test_type_codes_exist(self):
        """Test type-related codes exist."""
        assert ValidatorMessageCode.TYPE_MISMATCH.value == "type.mismatch"
        assert ValidatorMessageCode.TYPE_COERCION_FAILED.value == "type.coercion_failed"

    def test_format_codes_exist(self):
        """Test format-related codes exist."""
        assert ValidatorMessageCode.FORMAT_INVALID_EMAIL.value == "format.invalid_email"
        assert ValidatorMessageCode.FORMAT_PATTERN_MISMATCH.value == "format.pattern_mismatch"

    def test_range_codes_exist(self):
        """Test range-related codes exist."""
        assert ValidatorMessageCode.RANGE_OUT_OF_BOUNDS.value == "range.out_of_bounds"
        assert ValidatorMessageCode.RANGE_OUTLIER_DETECTED.value == "range.outlier_detected"

    def test_general_codes_exist(self):
        """Test general validation codes exist."""
        assert ValidatorMessageCode.VALIDATION_FAILED.value == "validation.failed"
        assert ValidatorMessageCode.VALIDATION_ERROR.value == "validation.error"


class TestValidatorI18n:
    """Test ValidatorI18n singleton class."""

    def setup_method(self):
        """Reset to default locale before each test."""
        set_validator_locale("en")

    def test_singleton_instance(self):
        """Test singleton pattern."""
        i18n1 = ValidatorI18n.get_instance()
        i18n2 = ValidatorI18n.get_instance()
        assert i18n1 is i18n2

    def test_default_locale(self):
        """Test default locale is English."""
        assert get_validator_locale() == "en"

    def test_set_locale(self):
        """Test setting locale."""
        set_validator_locale("ko")
        assert get_validator_locale() == "ko"

        set_validator_locale("ja")
        assert get_validator_locale() == "ja"

    def test_set_locale_with_region(self):
        """Test setting locale with region code."""
        set_validator_locale("ko_KR")
        assert get_validator_locale() == "ko"

        set_validator_locale("en_US")
        assert get_validator_locale() == "en"

    def test_fallback_for_unknown_locale(self):
        """Test fallback to English for unknown locale."""
        set_validator_locale("xx")  # Unknown locale
        assert get_validator_locale() == "en"

    def test_get_english_message(self):
        """Test getting English message."""
        set_validator_locale("en")
        msg = get_validator_message(
            ValidatorMessageCode.NULL_VALUES_FOUND,
            column="email",
            count=10,
        )
        assert msg == "Found 10 null values in column 'email'"

    def test_get_korean_message(self):
        """Test getting Korean message."""
        set_validator_locale("ko")
        msg = get_validator_message(
            ValidatorMessageCode.NULL_VALUES_FOUND,
            column="email",
            count=10,
        )
        assert msg == "'email' 컬럼에서 10개의 null 값이 발견되었습니다"

    def test_get_japanese_message(self):
        """Test getting Japanese message."""
        set_validator_locale("ja")
        msg = get_validator_message(
            ValidatorMessageCode.NULL_VALUES_FOUND,
            column="email",
            count=10,
        )
        assert msg == "列'email'で10個のnull値が見つかりました"

    def test_get_message_with_string_code(self):
        """Test getting message using string code."""
        msg = get_validator_message(
            "null.values_found",
            column="test",
            count=5,
        )
        assert "5" in msg
        assert "test" in msg

    def test_message_fallback_to_english(self):
        """Test fallback to English for missing translations."""
        set_validator_locale("de")
        # German catalog is incomplete, should fallback
        msg = get_validator_message(
            ValidatorMessageCode.REF_FOREIGN_KEY_VIOLATION,
            column="user_id",
            count=3,
        )
        # Should get English message
        assert "3" in msg
        assert "user_id" in msg

    def test_missing_placeholder_handling(self):
        """Test handling of missing placeholders."""
        msg = get_validator_message(
            ValidatorMessageCode.NULL_VALUES_FOUND,
            column="test",
            # Missing 'count'
        )
        assert "missing" in msg.lower() or "count" in msg

    def test_add_custom_catalog(self):
        """Test adding custom catalog."""
        i18n = ValidatorI18n.get_instance()
        i18n.add_catalog("test", {
            "null.values_found": "Custom: {count} nulls in {column}",
        })

        set_validator_locale("test")
        msg = get_validator_message(
            ValidatorMessageCode.NULL_VALUES_FOUND,
            column="email",
            count=5,
        )
        assert msg == "Custom: 5 nulls in email"


class TestFormatIssueMessage:
    """Test format_issue_message convenience function."""

    def setup_method(self):
        """Reset to default locale before each test."""
        set_validator_locale("en")

    def test_format_null_issue(self):
        """Test formatting null issue."""
        msg = format_issue_message("null", "email", 10)
        assert "10" in msg
        assert "email" in msg

    def test_format_duplicate_issue(self):
        """Test formatting duplicate issue."""
        msg = format_issue_message("duplicate", "id", 5)
        assert "5" in msg
        assert "id" in msg

    def test_format_type_mismatch_issue(self):
        """Test formatting type mismatch issue."""
        msg = format_issue_message(
            "type_mismatch",
            "age",
            3,
            expected="int",
            actual="str",
        )
        assert "age" in msg

    def test_format_pattern_mismatch_issue(self):
        """Test formatting pattern mismatch issue."""
        msg = format_issue_message(
            "pattern_mismatch",
            "phone",
            7,
            pattern=r"\d{3}-\d{4}",
        )
        assert "phone" in msg

    def test_format_unknown_issue_type(self):
        """Test formatting unknown issue type."""
        msg = format_issue_message(
            "unknown_type",
            "col",
            1,
            reason="test error",
        )
        # Should fallback to validation.failed
        assert "col" in msg


class TestValidatorMessageCatalog:
    """Test ValidatorMessageCatalog class."""

    def test_create_from_dict(self):
        """Test creating catalog from dictionary."""
        catalog = ValidatorMessageCatalog.from_dict("test", {
            "null.values_found": "Test: {count} nulls",
            "unique.duplicates_found": "Test: {count} dups",
        })

        assert catalog.locale == "test"
        assert len(catalog) == 2
        assert "null.values_found" in catalog
        assert catalog.get("null.values_found") == "Test: {count} nulls"

    def test_catalog_get_methods(self):
        """Test catalog getter methods."""
        catalog = ValidatorMessageCatalog.from_dict("en", {
            "key1": "value1",
            "key2": "value2",
        })

        assert catalog["key1"] == "value1"
        assert catalog.get("key1") == "value1"
        assert catalog.get("nonexistent", "default") == "default"

    def test_catalog_iteration(self):
        """Test catalog iteration."""
        messages = {"a": "A", "b": "B", "c": "C"}
        catalog = ValidatorMessageCatalog.from_dict("en", messages)

        assert list(catalog) == list(messages.keys())
        assert catalog.keys() == list(messages.keys())
        assert catalog.values() == list(messages.values())
        assert catalog.items() == list(messages.items())

    def test_catalog_merge(self):
        """Test merging catalogs."""
        catalog1 = ValidatorMessageCatalog.from_dict("en", {
            "key1": "original1",
            "key2": "original2",
        })
        catalog2 = ValidatorMessageCatalog.from_dict("en", {
            "key2": "updated2",
            "key3": "new3",
        })

        merged = catalog1.merge(catalog2)

        assert merged["key1"] == "original1"  # From catalog1
        assert merged["key2"] == "updated2"   # Overwritten by catalog2
        assert merged["key3"] == "new3"       # From catalog2

    def test_catalog_extend(self):
        """Test extending catalog."""
        catalog = ValidatorMessageCatalog.from_dict("en", {
            "key1": "value1",
        })

        extended = catalog.extend({
            "key2": "value2",
        })

        assert "key1" in extended
        assert "key2" in extended
        assert len(extended) == 2

    def test_catalog_to_dict(self):
        """Test converting catalog to dictionary."""
        catalog = ValidatorMessageCatalog.from_dict(
            "en",
            {"key": "value"},
            metadata={"name": "Test"},
        )

        data = catalog.to_dict()

        assert data["locale"] == "en"
        assert data["messages"] == {"key": "value"}
        assert data["metadata"] == {"name": "Test"}

    def test_catalog_json_roundtrip(self):
        """Test JSON serialization/deserialization."""
        catalog = ValidatorMessageCatalog.from_dict(
            "en",
            {
                "null.values_found": "Test message",
                "unique.duplicates_found": "Another message",
            },
            metadata={"version": "1.0"},
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            path = Path(f.name)

        try:
            catalog.to_json(path)

            # Verify JSON content
            with open(path) as f:
                data = json.load(f)
            assert data["locale"] == "en"
            assert "null.values_found" in data["messages"]

            # Load back
            loaded = ValidatorMessageCatalog.from_json("en", path)
            assert loaded.get("null.values_found") == catalog.get("null.values_found")
        finally:
            path.unlink()

    def test_catalog_from_flat_json(self):
        """Test loading from flat JSON (no messages wrapper)."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump({"key1": "value1", "key2": "value2"}, f)
            path = Path(f.name)

        try:
            catalog = ValidatorMessageCatalog.from_json("en", path)
            assert catalog.get("key1") == "value1"
            assert catalog.get("key2") == "value2"
        finally:
            path.unlink()


class TestCatalogBuilder:
    """Test CatalogBuilder class."""

    def test_basic_builder(self):
        """Test basic builder usage."""
        catalog = (
            CatalogBuilder("test")
            .add("key1", "value1")
            .add("key2", "value2")
            .build()
        )

        assert catalog.locale == "test"
        assert catalog.get("key1") == "value1"
        assert catalog.get("key2") == "value2"

    def test_builder_add_null(self):
        """Test adding null messages."""
        catalog = (
            CatalogBuilder("test")
            .add_null(
                values_found="Null: {count}",
                column_empty="Empty: {column}",
            )
            .build()
        )

        assert "null.values_found" in catalog
        assert "null.column_empty" in catalog

    def test_builder_add_unique(self):
        """Test adding unique messages."""
        catalog = (
            CatalogBuilder("test")
            .add_unique(
                duplicates_found="Dups: {count}",
                key_violation="Key violation",
            )
            .build()
        )

        assert "unique.duplicates_found" in catalog
        assert "unique.key_violation" in catalog

    def test_builder_add_type(self):
        """Test adding type messages."""
        catalog = (
            CatalogBuilder("test")
            .add_type(
                mismatch="Type mismatch",
                coercion_failed="Coercion failed",
            )
            .build()
        )

        assert "type.mismatch" in catalog
        assert "type.coercion_failed" in catalog

    def test_builder_add_format(self):
        """Test adding format messages."""
        catalog = (
            CatalogBuilder("test")
            .add_format(
                invalid_email="Bad email",
                invalid_phone="Bad phone",
            )
            .build()
        )

        assert "format.invalid_email" in catalog
        assert "format.invalid_phone" in catalog

    def test_builder_add_range(self):
        """Test adding range messages."""
        catalog = (
            CatalogBuilder("test")
            .add_range(
                out_of_bounds="OOB",
                outlier_detected="Outlier",
            )
            .build()
        )

        assert "range.out_of_bounds" in catalog
        assert "range.outlier_detected" in catalog

    def test_builder_add_validation(self):
        """Test adding validation messages."""
        catalog = (
            CatalogBuilder("test")
            .add_validation(
                failed="Failed",
                error="Error",
            )
            .build()
        )

        assert "validation.failed" in catalog
        assert "validation.error" in catalog

    def test_builder_with_metadata(self):
        """Test adding metadata."""
        catalog = (
            CatalogBuilder("test")
            .add("key", "value")
            .with_metadata(name="Test Catalog", version="1.0")
            .build()
        )

        assert catalog.metadata["name"] == "Test Catalog"
        assert catalog.metadata["version"] == "1.0"

    def test_builder_via_catalog_class(self):
        """Test builder accessed via catalog class."""
        catalog = (
            ValidatorMessageCatalog.builder("test")
            .add("key", "value")
            .build()
        )

        assert catalog.locale == "test"
        assert "key" in catalog


class TestPrebuiltCatalogs:
    """Test pre-built message catalogs."""

    def test_get_default_messages(self):
        """Test getting default English messages."""
        catalog = get_default_messages()

        assert catalog.locale == "en"
        assert "null.values_found" in catalog
        assert "unique.duplicates_found" in catalog
        assert "validation.error" in catalog

    def test_get_korean_messages(self):
        """Test getting Korean messages."""
        catalog = get_korean_messages()

        assert catalog.locale == "ko"
        assert "null.values_found" in catalog
        # Verify it's actually Korean
        assert "컬럼" in catalog.get("null.values_found")

    def test_get_japanese_messages(self):
        """Test getting Japanese messages."""
        catalog = get_japanese_messages()

        assert catalog.locale == "ja"
        assert "null.values_found" in catalog
        # Verify it's actually Japanese
        assert "列" in catalog.get("null.values_found")

    def test_get_chinese_messages(self):
        """Test getting Chinese messages."""
        catalog = get_chinese_messages()

        assert catalog.locale == "zh"
        assert "null.values_found" in catalog
        # Verify it's actually Chinese
        assert "列" in catalog.get("null.values_found")

    def test_get_german_messages(self):
        """Test getting German messages."""
        catalog = get_german_messages()

        assert catalog.locale == "de"
        # German is partial, but should have some messages
        assert len(catalog) > 0

    def test_get_french_messages(self):
        """Test getting French messages."""
        catalog = get_french_messages()

        assert catalog.locale == "fr"
        assert len(catalog) > 0

    def test_get_spanish_messages(self):
        """Test getting Spanish messages."""
        catalog = get_spanish_messages()

        assert catalog.locale == "es"
        assert len(catalog) > 0

    def test_get_all_catalogs(self):
        """Test getting all catalogs."""
        catalogs = get_all_catalogs()

        assert "en" in catalogs
        assert "ko" in catalogs
        assert "ja" in catalogs
        assert "zh" in catalogs
        assert "de" in catalogs
        assert "fr" in catalogs
        assert "es" in catalogs

    def test_get_supported_locales(self):
        """Test getting supported locales."""
        locales = get_supported_locales()

        assert "en" in locales
        assert "ko" in locales
        assert "ja" in locales
        assert len(locales) == 7


class TestCreateCustomCatalog:
    """Test create_custom_catalog function."""

    def test_create_from_english(self):
        """Test creating custom catalog based on English."""
        catalog = (
            create_custom_catalog("en_CA", "Canadian English", "en")
            .add("format.invalid_phone", "Invalid Canadian phone: {count}")
            .build()
        )

        assert catalog.locale == "en_CA"
        # Should have base English messages
        assert "null.values_found" in catalog
        # Should have custom message
        assert catalog.get("format.invalid_phone") == "Invalid Canadian phone: {count}"
        # Should have metadata
        assert catalog.metadata["name"] == "Canadian English"
        assert catalog.metadata["base"] == "en"

    def test_create_from_korean(self):
        """Test creating custom catalog based on Korean."""
        catalog = (
            create_custom_catalog("ko_formal", "한국어 (존칭)", "ko")
            .add("validation.failed", "'{column}' 컬럼 검증에 실패하였습니다: {reason}")
            .build()
        )

        assert catalog.locale == "ko_formal"
        # Should have base Korean messages
        assert "컬럼" in catalog.get("null.values_found")
        # Should have customization
        assert "실패하였습니다" in catalog.get("validation.failed")

    def test_create_from_nonexistent_base(self):
        """Test creating catalog with non-existent base falls back to English."""
        catalog = (
            create_custom_catalog("xx", "Custom", "nonexistent")
            .add("custom.key", "custom value")
            .build()
        )

        # Should fallback to English base
        assert "null.values_found" in catalog
        assert "Found" in catalog.get("null.values_found")


class TestMessageFormatting:
    """Test message formatting with various parameters."""

    def setup_method(self):
        """Reset to default locale before each test."""
        set_validator_locale("en")

    def test_format_with_percentage(self):
        """Test formatting with percentage."""
        msg = get_validator_message(
            ValidatorMessageCode.NULL_ABOVE_THRESHOLD,
            column="status",
            ratio=0.15,
            threshold=0.10,
        )
        assert "15.0%" in msg
        assert "10.0%" in msg

    def test_format_with_float(self):
        """Test formatting with float values."""
        msg = get_validator_message(
            ValidatorMessageCode.STAT_MEAN_OUT_OF_RANGE,
            column="temperature",
            mean=23.456,
            min=20,
            max=22,
        )
        assert "23.46" in msg  # Formatted to 2 decimal places

    def test_format_with_list(self):
        """Test formatting with list values."""
        msg = get_validator_message(
            ValidatorMessageCode.UNIQUE_COMPOSITE_DUPLICATES,
            columns=["id", "date"],
            count=5,
        )
        assert "['id', 'date']" in msg or "id" in msg

    def test_format_range_message(self):
        """Test formatting range message."""
        msg = get_validator_message(
            ValidatorMessageCode.RANGE_OUT_OF_BOUNDS,
            column="age",
            count=10,
            min=0,
            max=120,
        )
        assert "10" in msg
        assert "age" in msg
        assert "0" in msg
        assert "120" in msg
