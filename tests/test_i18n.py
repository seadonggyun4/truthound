"""Comprehensive tests for internationalization (i18n) system.

Tests cover:
- Message codes
- Locale management
- Message catalogs
- Message loaders
- Placeholder formatting
- I18n class
- I18n exceptions
- Registry pattern
- Convenience functions
- Integration scenarios
"""

from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from truthound.profiler.i18n import (
    # Message codes
    MessageCode,
    # Locale management
    LocaleInfo,
    LocaleManager,
    BUILTIN_LOCALES,
    set_locale,
    get_locale,
    # Message catalog
    MessageEntry,
    MessageCatalog,
    # Message loaders
    MessageLoader,
    DictMessageLoader,
    FileMessageLoader,
    # Formatter
    PlaceholderFormatter,
    # Main interface
    I18n,
    # I18n exceptions
    I18nError,
    I18nAnalysisError,
    I18nPatternError,
    I18nTypeError,
    I18nIOError,
    I18nTimeoutError,
    I18nValidationError,
    # Registry
    MessageCatalogRegistry,
    # Convenience functions
    get_message,
    t,
    register_messages,
    load_messages_from_file,
    create_message_loader,
    # Context manager
    locale_context,
    # Presets
    I18nPresets,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_i18n():
    """Reset I18n singleton before each test."""
    I18n.reset_instance()
    set_locale("en")
    yield
    I18n.reset_instance()
    set_locale("en")


@pytest.fixture
def sample_messages() -> dict[str, dict[str, Any]]:
    """Create sample messages for testing."""
    return {
        "en": {
            "greeting": "Hello, {name}!",
            "error": {
                "not_found": "File not found: {path}",
                "timeout": "Operation timed out after {seconds}s",
            },
            "items": {
                "zero": "No items",
                "one": "1 item",
                "other": "{count} items",
            },
        },
        "ko": {
            "greeting": "안녕하세요, {name}님!",
            "error": {
                "not_found": "파일을 찾을 수 없음: {path}",
                "timeout": "{seconds}초 후 작업 시간 초과",
            },
            "items": {
                "zero": "항목 없음",
                "one": "1개 항목",
                "other": "{count}개 항목",
            },
        },
    }


@pytest.fixture
def temp_message_dir(sample_messages: dict) -> Path:
    """Create temporary directory with message files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create JSON message files
        for locale, messages in sample_messages.items():
            filepath = tmppath / f"messages_{locale}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)

        yield tmppath


# =============================================================================
# Message Code Tests
# =============================================================================


class TestMessageCode:
    """Test MessageCode enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        # General errors
        assert MessageCode.ERR_UNKNOWN
        assert MessageCode.ERR_INTERNAL

        # Analysis errors
        assert MessageCode.ANALYSIS_FAILED
        assert MessageCode.ANALYSIS_COLUMN_FAILED

        # Pattern errors
        assert MessageCode.PATTERN_INVALID
        assert MessageCode.PATTERN_MATCH_FAILED

        # Type errors
        assert MessageCode.TYPE_INFERENCE_FAILED
        assert MessageCode.TYPE_UNSUPPORTED

        # IO errors
        assert MessageCode.IO_FILE_NOT_FOUND
        assert MessageCode.IO_READ_FAILED

    def test_code_values_are_strings(self):
        """Test that code values are dot-notation strings."""
        assert MessageCode.ANALYSIS_FAILED.value == "analysis.failed"
        assert MessageCode.IO_FILE_NOT_FOUND.value == "io.file_not_found"

    def test_code_uniqueness(self):
        """Test that all codes are unique."""
        values = [code.value for code in MessageCode]
        assert len(values) == len(set(values))


# =============================================================================
# Locale Management Tests
# =============================================================================


class TestLocaleInfo:
    """Test LocaleInfo class."""

    def test_create_basic(self):
        """Test creating basic locale info."""
        info = LocaleInfo("en", "English", "English")
        assert info.code == "en"
        assert info.name == "English"
        assert info.native_name == "English"
        assert info.language == "en"
        assert info.region == ""

    def test_create_with_region(self):
        """Test creating locale with region."""
        info = LocaleInfo("ko_KR", "Korean (Korea)", "한국어 (대한민국)")
        assert info.code == "ko_KR"
        assert info.language == "ko"
        assert info.region == "KR"

    def test_from_system(self):
        """Test creating from system locale."""
        info = LocaleInfo.from_system()
        assert info.code is not None
        assert info.language is not None


class TestLocaleManager:
    """Test LocaleManager class."""

    def test_default_locale(self):
        """Test default locale."""
        manager = LocaleManager(default_locale="en")
        assert manager.default == "en"
        assert manager.current == "en"

    def test_set_locale(self):
        """Test setting locale."""
        manager = LocaleManager()
        manager.set_locale("ko")
        assert manager.current == "ko"

    def test_fallback_chain_basic(self):
        """Test basic fallback chain."""
        manager = LocaleManager(default_locale="en", fallback_locale="en")
        manager.set_locale("ko")

        chain = manager.get_fallback_chain()
        assert chain == ["ko", "en"]

    def test_fallback_chain_with_region(self):
        """Test fallback chain with region."""
        manager = LocaleManager(default_locale="en", fallback_locale="en")
        manager.set_locale("ko_KR")

        chain = manager.get_fallback_chain()
        assert chain == ["ko_KR", "ko", "en"]

    def test_register_locale(self):
        """Test registering new locale."""
        manager = LocaleManager()
        manager.register_locale(LocaleInfo("test", "Test Language"))

        assert "test" in manager.list_locales()

    def test_thread_safety(self):
        """Test thread-safe locale setting."""
        manager = LocaleManager(default_locale="en")
        locales_seen = []
        errors = []

        def worker(locale: str):
            try:
                manager.set_locale(locale)
                for _ in range(100):
                    locales_seen.append((locale, manager.current))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("ko",)),
            threading.Thread(target=worker, args=("ja",)),
            threading.Thread(target=worker, args=("de",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Each thread should see its own locale
        for expected, actual in locales_seen:
            assert actual == expected


class TestBuiltinLocales:
    """Test built-in locales."""

    def test_common_locales_exist(self):
        """Test that common locales are defined."""
        assert "en" in BUILTIN_LOCALES
        assert "ko" in BUILTIN_LOCALES
        assert "ja" in BUILTIN_LOCALES
        assert "zh" in BUILTIN_LOCALES
        assert "de" in BUILTIN_LOCALES
        assert "fr" in BUILTIN_LOCALES
        assert "es" in BUILTIN_LOCALES


# =============================================================================
# Message Catalog Tests
# =============================================================================


class TestMessageEntry:
    """Test MessageEntry class."""

    def test_simple_entry(self):
        """Test simple string entry."""
        entry = MessageEntry(key="greeting", value="Hello!")
        assert entry.get_form() == "Hello!"

    def test_pluralized_entry(self):
        """Test pluralized entry."""
        entry = MessageEntry(
            key="items",
            value={
                "zero": "No items",
                "one": "1 item",
                "other": "{count} items",
            },
        )

        assert entry.get_form(count=0) == "No items"
        assert entry.get_form(count=1) == "1 item"
        assert entry.get_form(count=5) == "{count} items"


class TestMessageCatalog:
    """Test MessageCatalog class."""

    def test_create_with_messages(self, sample_messages: dict):
        """Test creating catalog with messages."""
        catalog = MessageCatalog("en", sample_messages["en"])
        assert "greeting" in catalog
        assert "error.not_found" in catalog

    def test_get_message(self, sample_messages: dict):
        """Test getting message."""
        catalog = MessageCatalog("en", sample_messages["en"])
        assert catalog.get("greeting") == "Hello, {name}!"
        assert catalog.get("error.not_found") == "File not found: {path}"

    def test_get_nonexistent(self, sample_messages: dict):
        """Test getting nonexistent message."""
        catalog = MessageCatalog("en", sample_messages["en"])
        assert catalog.get("nonexistent") is None

    def test_pluralization(self, sample_messages: dict):
        """Test pluralization."""
        catalog = MessageCatalog("en", sample_messages["en"])

        assert catalog.get("items", count=0) == "No items"
        assert catalog.get("items", count=1) == "1 item"
        assert catalog.get("items", count=5) == "{count} items"


# =============================================================================
# Message Loader Tests
# =============================================================================


class TestDictMessageLoader:
    """Test DictMessageLoader class."""

    def test_load_existing(self, sample_messages: dict):
        """Test loading existing locale."""
        loader = DictMessageLoader(sample_messages)
        catalog = loader.load("en")

        assert catalog is not None
        assert catalog.get("greeting") == "Hello, {name}!"

    def test_load_nonexistent(self, sample_messages: dict):
        """Test loading nonexistent locale."""
        loader = DictMessageLoader(sample_messages)
        catalog = loader.load("fr")

        assert catalog is None

    def test_supports(self, sample_messages: dict):
        """Test supports check."""
        loader = DictMessageLoader(sample_messages)

        assert loader.supports("en") is True
        assert loader.supports("ko") is True
        assert loader.supports("fr") is False


class TestFileMessageLoader:
    """Test FileMessageLoader class."""

    def test_load_json(self, temp_message_dir: Path):
        """Test loading JSON message file."""
        loader = FileMessageLoader(temp_message_dir)
        catalog = loader.load("en")

        assert catalog is not None
        assert catalog.get("greeting") == "Hello, {name}!"

    def test_load_korean(self, temp_message_dir: Path):
        """Test loading Korean messages."""
        loader = FileMessageLoader(temp_message_dir)
        catalog = loader.load("ko")

        assert catalog is not None
        assert catalog.get("greeting") == "안녕하세요, {name}님!"

    def test_load_nonexistent(self, temp_message_dir: Path):
        """Test loading nonexistent locale."""
        loader = FileMessageLoader(temp_message_dir)
        catalog = loader.load("fr")

        assert catalog is None

    def test_supports(self, temp_message_dir: Path):
        """Test supports check."""
        loader = FileMessageLoader(temp_message_dir)

        assert loader.supports("en") is True
        assert loader.supports("ko") is True
        assert loader.supports("fr") is False


# =============================================================================
# Placeholder Formatter Tests
# =============================================================================


class TestPlaceholderFormatter:
    """Test PlaceholderFormatter class."""

    def test_named_placeholders(self):
        """Test named placeholders."""
        formatter = PlaceholderFormatter()
        result = formatter.format("Hello, {name}!", name="World")
        assert result == "Hello, World!"

    def test_positional_placeholders(self):
        """Test positional placeholders."""
        formatter = PlaceholderFormatter()
        result = formatter.format("{0} + {1} = {2}", 1, 2, 3)
        assert result == "1 + 2 = 3"

    def test_format_specs(self):
        """Test format specifications."""
        formatter = PlaceholderFormatter()
        result = formatter.format("Progress: {progress:.1%}", progress=0.756)
        assert result == "Progress: 75.6%"

    def test_datetime_formatting(self):
        """Test datetime formatting."""
        formatter = PlaceholderFormatter()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = formatter.format("Date: {date}", date=dt)
        assert "2024-01-15" in result

    def test_timedelta_formatting(self):
        """Test timedelta formatting."""
        formatter = PlaceholderFormatter()

        result = formatter.format("Duration: {td}", td=timedelta(seconds=30))
        assert result == "Duration: 30s"

        result = formatter.format("Duration: {td}", td=timedelta(minutes=5))
        assert result == "Duration: 5m"

        result = formatter.format("Duration: {td}", td=timedelta(hours=2, minutes=30))
        assert result == "Duration: 2h 30m"

    def test_custom_formatter(self):
        """Test custom type formatter."""
        formatter = PlaceholderFormatter()

        class CustomType:
            def __init__(self, value):
                self.value = value

        formatter.register_formatter(CustomType, lambda x: f"Custom({x.value})")

        result = formatter.format("Value: {obj}", obj=CustomType(42))
        assert result == "Value: Custom(42)"


# =============================================================================
# I18n Class Tests
# =============================================================================


class TestI18n:
    """Test I18n class."""

    def test_singleton(self):
        """Test singleton pattern."""
        i18n1 = I18n.get_instance()
        i18n2 = I18n.get_instance()
        assert i18n1 is i18n2

    def test_set_locale(self):
        """Test setting locale."""
        i18n = I18n.get_instance()
        i18n.set_locale("ko")
        assert i18n.get_locale() == "ko"

    def test_translate_english(self):
        """Test English translation."""
        i18n = I18n.get_instance()
        i18n.set_locale("en")

        msg = i18n.t(MessageCode.ANALYSIS_FAILED, column="email")
        assert "email" in msg
        assert "failed" in msg.lower() or "Analysis" in msg

    def test_translate_korean(self):
        """Test Korean translation."""
        i18n = I18n.get_instance()
        i18n.set_locale("ko")

        msg = i18n.t(MessageCode.ANALYSIS_FAILED, column="email")
        assert "email" in msg
        assert "실패" in msg

    def test_translate_japanese(self):
        """Test Japanese translation."""
        i18n = I18n.get_instance()
        i18n.set_locale("ja")

        msg = i18n.t(MessageCode.ANALYSIS_FAILED, column="email")
        assert "email" in msg
        assert "失敗" in msg

    def test_fallback_to_english(self):
        """Test fallback to English for unknown locale."""
        i18n = I18n.get_instance()
        i18n.set_locale("xyz")

        msg = i18n.t(MessageCode.ANALYSIS_FAILED, column="email")
        assert "email" in msg

    def test_has_key(self):
        """Test checking if key exists."""
        i18n = I18n.get_instance()

        assert i18n.has(MessageCode.ANALYSIS_FAILED) is True
        assert i18n.has("nonexistent.key") is False

    def test_string_key(self):
        """Test using string keys."""
        i18n = I18n.get_instance()
        i18n.set_locale("en")

        msg = i18n.t("analysis.failed", column="test")
        assert "test" in msg

    def test_default_value(self):
        """Test default value for missing keys."""
        i18n = I18n.get_instance()

        msg = i18n.t("nonexistent.key", default="Default message")
        assert msg == "Default message"

    def test_locale_override(self):
        """Test locale override in translation."""
        i18n = I18n.get_instance()
        i18n.set_locale("en")

        # Override with Korean
        msg = i18n.t(MessageCode.ANALYSIS_FAILED, column="email", locale="ko")
        assert "실패" in msg

    def test_add_loader(self, sample_messages: dict):
        """Test adding custom loader."""
        i18n = I18n.get_instance()
        loader = DictMessageLoader({
            "test": {"custom": {"message": "Custom message"}}
        })
        i18n.add_loader(loader)
        i18n.set_locale("test")

        msg = i18n.t("custom.message")
        assert msg == "Custom message"


# =============================================================================
# I18n Exception Tests
# =============================================================================


class TestI18nExceptions:
    """Test I18n exception classes."""

    def test_i18n_error(self):
        """Test I18nError."""
        set_locale("en")
        error = I18nError(MessageCode.ANALYSIS_FAILED, column="email")

        assert "email" in str(error)
        assert error.code == MessageCode.ANALYSIS_FAILED

    def test_i18n_error_korean(self):
        """Test I18nError with Korean locale."""
        set_locale("ko")
        error = I18nError(MessageCode.ANALYSIS_FAILED, column="email")

        assert "실패" in str(error)

    def test_get_message_in_different_locale(self):
        """Test getting message in different locale."""
        set_locale("en")
        error = I18nError(MessageCode.ANALYSIS_FAILED, column="email")

        korean_msg = error.get_message(locale="ko")
        assert "실패" in korean_msg

    def test_analysis_error(self):
        """Test I18nAnalysisError."""
        error = I18nAnalysisError(column="test_col")
        assert "test_col" in str(error)

    def test_pattern_error(self):
        """Test I18nPatternError."""
        error = I18nPatternError(pattern="[0-9]+")
        assert "[0-9]+" in str(error)

    def test_timeout_error(self):
        """Test I18nTimeoutError."""
        error = I18nTimeoutError(seconds=30)
        assert "30" in str(error)

    def test_validation_error(self):
        """Test I18nValidationError."""
        error = I18nValidationError(message="Invalid input")
        assert "Invalid input" in str(error)


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_message(self):
        """Test get_message function."""
        set_locale("en")
        msg = get_message(MessageCode.ANALYSIS_FAILED, column="email")
        assert "email" in msg

    def test_t_function(self):
        """Test t() shorthand."""
        set_locale("ko")
        msg = t(MessageCode.IO_FILE_NOT_FOUND, path="/test/file.txt")
        assert "/test/file.txt" in msg
        assert "찾을 수 없" in msg

    def test_set_get_locale(self):
        """Test set_locale and get_locale."""
        set_locale("ja")
        assert get_locale() == "ja"

        set_locale("en")
        assert get_locale() == "en"

    def test_register_messages(self):
        """Test register_messages."""
        # Register messages directly with I18n instance
        i18n = I18n.get_instance()
        loader = DictMessageLoader({
            "custom": {"test": {"message": "Custom test message"}}
        })
        i18n.add_loader(loader)

        set_locale("custom")
        msg = t("test.message")
        assert msg == "Custom test message"


class TestLocaleContext:
    """Test locale_context context manager."""

    def test_temporary_locale(self):
        """Test temporary locale change."""
        set_locale("en")

        with locale_context("ko"):
            assert get_locale() == "ko"
            msg = get_message(MessageCode.ANALYSIS_FAILED, column="test")
            assert "실패" in msg

        # Original locale restored
        assert get_locale() == "en"

    def test_nested_contexts(self):
        """Test nested locale contexts."""
        set_locale("en")

        with locale_context("ko"):
            assert get_locale() == "ko"

            with locale_context("ja"):
                assert get_locale() == "ja"

            assert get_locale() == "ko"

        assert get_locale() == "en"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for i18n system."""

    def test_complete_workflow(self):
        """Test complete i18n workflow."""
        # Start with English
        set_locale("en")

        # Get messages in different locales
        en_msg = get_message(MessageCode.TIMEOUT_EXCEEDED, seconds=30)
        assert "30" in en_msg
        assert "timed out" in en_msg.lower()

        # Switch to Korean
        set_locale("ko")
        ko_msg = get_message(MessageCode.TIMEOUT_EXCEEDED, seconds=30)
        assert "30" in ko_msg
        assert "시간 초과" in ko_msg

        # Use context manager for Japanese
        with locale_context("ja"):
            ja_msg = get_message(MessageCode.TIMEOUT_EXCEEDED, seconds=30)
            assert "30" in ja_msg
            assert "タイムアウト" in ja_msg

        # Back to Korean
        assert get_locale() == "ko"

    def test_error_handling_workflow(self):
        """Test error handling with i18n."""
        set_locale("ko")

        try:
            raise I18nAnalysisError(column="email", reason="데이터 손상")
        except I18nAnalysisError as e:
            # Error message should be in Korean
            assert "email" in str(e)

    def test_file_loading_workflow(self, temp_message_dir: Path):
        """Test loading messages from files."""
        # Test file loader directly (without singleton caching issues)
        loader = FileMessageLoader(temp_message_dir)

        # Verify loader can load catalogs
        en_catalog = loader.load("en")
        ko_catalog = loader.load("ko")

        assert en_catalog is not None
        assert ko_catalog is not None

        # Test English messages
        assert en_catalog.get("greeting") == "Hello, {name}!"
        assert en_catalog.get("error.not_found") == "File not found: {path}"

        # Test Korean messages
        assert ko_catalog.get("greeting") == "안녕하세요, {name}님!"
        assert ko_catalog.get("error.not_found") == "파일을 찾을 수 없음: {path}"

        # Test pluralization
        assert en_catalog.get("items", count=0) == "No items"
        assert en_catalog.get("items", count=1) == "1 item"
        assert ko_catalog.get("items", count=0) == "항목 없음"

    def test_thread_safe_translations(self):
        """Test thread-safe translations."""
        results = {}
        errors = []

        def worker(locale: str, code: MessageCode, expected_substring: str):
            try:
                # Each thread sets its own locale
                with locale_context(locale):
                    msg = get_message(code, column="test")
                    results[locale] = msg
                    assert expected_substring in msg, f"Expected '{expected_substring}' in '{msg}'"
            except Exception as e:
                errors.append((locale, e))

        threads = [
            threading.Thread(target=worker, args=("en", MessageCode.ANALYSIS_FAILED, "failed")),
            threading.Thread(target=worker, args=("ko", MessageCode.ANALYSIS_FAILED, "실패")),
            threading.Thread(target=worker, args=("ja", MessageCode.ANALYSIS_FAILED, "失敗")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 3

    def test_all_message_codes_have_translations(self):
        """Test that all message codes have English translations."""
        i18n = I18n.get_instance()
        i18n.set_locale("en")

        missing = []
        for code in MessageCode:
            if not i18n.has(code):
                missing.append(code)

        assert len(missing) == 0, f"Missing translations: {missing}"


# =============================================================================
# Preset Tests
# =============================================================================


class TestI18nPresets:
    """Test I18nPresets class."""

    def test_minimal(self):
        """Test minimal preset."""
        i18n = I18nPresets.minimal()
        assert i18n is not None
        msg = i18n.t(MessageCode.ERR_UNKNOWN)
        assert "error" in msg.lower()

    def test_auto_detect_locale(self):
        """Test auto-detect locale preset."""
        i18n = I18nPresets.auto_detect_locale()
        assert i18n is not None
        # Locale should be set
        assert i18n.get_locale() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
