"""Comprehensive tests for enterprise i18n features.

This test suite covers:
- CLDR plural rules
- RTL language support (BiDi)
- Number and date formatting
- Regional dialect support
- TMS integration
- Dynamic catalog loading
- Context-based messages
- Extended language catalogs
"""

import pytest
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile
import json

# Import all i18n modules
from truthound.validators.i18n import (
    # Core
    ValidatorMessageCode,
    get_validator_message,
    set_validator_locale,
    get_validator_locale,

    # Catalogs
    ValidatorMessageCatalog,
    get_default_messages,
    get_russian_messages,
    get_italian_messages,
    get_portuguese_messages,
    get_arabic_messages,
    get_hebrew_messages,

    # Protocols
    LocaleInfo,
    TextDirection,
    PluralCategory,
    NumberStyle,
    DateStyle,
    TimeStyle,
    MessageContext,

    # Plural
    CLDRPluralRules,
    get_plural_category,
    pluralize,

    # BiDi
    BiDiHandler,
    BiDiConfig,
    detect_direction,
    wrap_bidi,
    is_rtl_language,

    # Formatting
    LocaleNumberFormatter,
    LocaleDateFormatter,
    format_number,
    format_currency,
    format_date,
    format_relative_time,

    # Dialects
    DialectRegistry,
    DialectResolver,
    get_fallback_chain,

    # TMS
    TMSManager,
    CrowdinProvider,
    LokaliseProvider,
    TMSConfig,
    TMSProvider,

    # Loading
    CatalogManager,
    ContextResolver,
    FileSystemStorage,
    MemoryStorage,
    LRUCache,
)


# ==============================================================================
# LocaleInfo Tests
# ==============================================================================

class TestLocaleInfo:
    """Tests for LocaleInfo parsing and matching."""

    def test_parse_simple_locale(self):
        """Test parsing simple locale code."""
        locale = LocaleInfo.parse("en")
        assert locale.language == "en"
        assert locale.region is None
        assert locale.script is None
        assert locale.tag == "en"

    def test_parse_locale_with_region(self):
        """Test parsing locale with region."""
        locale = LocaleInfo.parse("en-US")
        assert locale.language == "en"
        assert locale.region == "US"
        assert locale.tag == "en-US"

    def test_parse_locale_with_underscore(self):
        """Test parsing locale with underscore separator."""
        locale = LocaleInfo.parse("ko_KR")
        assert locale.language == "ko"
        assert locale.region == "KR"

    def test_parse_locale_with_script(self):
        """Test parsing locale with script."""
        locale = LocaleInfo.parse("zh-Hans")
        assert locale.language == "zh"
        assert locale.script == "Hans"

    def test_parse_full_locale(self):
        """Test parsing full locale tag."""
        locale = LocaleInfo.parse("zh-Hans-CN")
        assert locale.language == "zh"
        assert locale.script == "Hans"
        assert locale.region == "CN"

    def test_rtl_direction(self):
        """Test RTL language detection."""
        assert LocaleInfo.parse("ar").direction == TextDirection.RTL
        assert LocaleInfo.parse("he").direction == TextDirection.RTL
        assert LocaleInfo.parse("fa").direction == TextDirection.RTL
        assert LocaleInfo.parse("en").direction == TextDirection.LTR
        assert LocaleInfo.parse("ko").direction == TextDirection.LTR

    def test_locale_matching(self):
        """Test locale matching."""
        en_us = LocaleInfo.parse("en-US")
        en = LocaleInfo.parse("en")

        assert en_us.matches(en, strict=False)
        assert not en_us.matches(en, strict=True)
        assert en.matches(en, strict=True)


# ==============================================================================
# Plural Rules Tests
# ==============================================================================

class TestPluralRules:
    """Tests for CLDR plural rules."""

    def test_english_plural(self):
        """Test English plural rules."""
        assert get_plural_category(1, "en") == PluralCategory.ONE
        assert get_plural_category(2, "en") == PluralCategory.OTHER
        assert get_plural_category(0, "en") == PluralCategory.OTHER
        assert get_plural_category(100, "en") == PluralCategory.OTHER

    def test_russian_plural(self):
        """Test Russian plural rules (one, few, many, other)."""
        # One: 1, 21, 31...
        assert get_plural_category(1, "ru") == PluralCategory.ONE
        assert get_plural_category(21, "ru") == PluralCategory.ONE
        assert get_plural_category(31, "ru") == PluralCategory.ONE

        # Few: 2-4, 22-24, 32-34...
        assert get_plural_category(2, "ru") == PluralCategory.FEW
        assert get_plural_category(3, "ru") == PluralCategory.FEW
        assert get_plural_category(4, "ru") == PluralCategory.FEW
        assert get_plural_category(22, "ru") == PluralCategory.FEW

        # Many: 0, 5-20, 25-30...
        assert get_plural_category(0, "ru") == PluralCategory.MANY
        assert get_plural_category(5, "ru") == PluralCategory.MANY
        assert get_plural_category(11, "ru") == PluralCategory.MANY
        assert get_plural_category(12, "ru") == PluralCategory.MANY
        assert get_plural_category(14, "ru") == PluralCategory.MANY

    def test_arabic_plural(self):
        """Test Arabic plural rules (zero, one, two, few, many, other)."""
        assert get_plural_category(0, "ar") == PluralCategory.ZERO
        assert get_plural_category(1, "ar") == PluralCategory.ONE
        assert get_plural_category(2, "ar") == PluralCategory.TWO
        assert get_plural_category(5, "ar") == PluralCategory.FEW
        assert get_plural_category(11, "ar") == PluralCategory.MANY

    def test_japanese_plural(self):
        """Test Japanese plural rules (no distinction)."""
        assert get_plural_category(0, "ja") == PluralCategory.OTHER
        assert get_plural_category(1, "ja") == PluralCategory.OTHER
        assert get_plural_category(5, "ja") == PluralCategory.OTHER

    def test_korean_plural(self):
        """Test Korean plural rules (no distinction)."""
        assert get_plural_category(0, "ko") == PluralCategory.OTHER
        assert get_plural_category(1, "ko") == PluralCategory.OTHER
        assert get_plural_category(100, "ko") == PluralCategory.OTHER

    def test_pluralize_english(self):
        """Test pluralize function for English."""
        result = pluralize(
            count=1,
            forms={
                PluralCategory.ONE: "{count} file",
                PluralCategory.OTHER: "{count} files",
            },
            locale="en",
        )
        assert result == "1 file"

        result = pluralize(
            count=5,
            forms={
                PluralCategory.ONE: "{count} file",
                PluralCategory.OTHER: "{count} files",
            },
            locale="en",
        )
        assert result == "5 files"

    def test_pluralize_russian(self):
        """Test pluralize function for Russian."""
        forms = {
            PluralCategory.ONE: "{count} файл",
            PluralCategory.FEW: "{count} файла",
            PluralCategory.MANY: "{count} файлов",
            PluralCategory.OTHER: "{count} файла",
        }

        assert pluralize(1, forms, "ru") == "1 файл"
        assert pluralize(2, forms, "ru") == "2 файла"
        assert pluralize(5, forms, "ru") == "5 файлов"
        assert pluralize(21, forms, "ru") == "21 файл"
        assert pluralize(22, forms, "ru") == "22 файла"

    def test_pluralize_with_string_keys(self):
        """Test pluralize with string keys."""
        result = pluralize(
            count=1,
            forms={"one": "{count} item", "other": "{count} items"},
            locale="en",
        )
        assert result == "1 item"


# ==============================================================================
# BiDi Tests
# ==============================================================================

class TestBiDi:
    """Tests for bidirectional text handling."""

    def test_detect_ltr_text(self):
        """Test detecting LTR text."""
        assert detect_direction("Hello World") == TextDirection.LTR
        assert detect_direction("안녕하세요") == TextDirection.LTR
        assert detect_direction("こんにちは") == TextDirection.LTR

    def test_detect_rtl_text(self):
        """Test detecting RTL text."""
        assert detect_direction("مرحبا") == TextDirection.RTL
        assert detect_direction("שלום") == TextDirection.RTL

    def test_detect_mixed_text(self):
        """Test detecting mixed direction (first strong wins)."""
        # English first
        assert detect_direction("Hello مرحبا") == TextDirection.LTR
        # Arabic first
        assert detect_direction("مرحبا Hello") == TextDirection.RTL

    def test_is_rtl_language(self):
        """Test RTL language detection."""
        assert is_rtl_language("ar") is True
        assert is_rtl_language("he") is True
        assert is_rtl_language("fa") is True
        assert is_rtl_language("ur") is True
        assert is_rtl_language("en") is False
        assert is_rtl_language("ko") is False

    def test_wrap_bidi_rtl(self):
        """Test wrapping text with RTL controls."""
        result = wrap_bidi("Hello", TextDirection.RTL)
        assert "\u2067" in result  # RLI
        assert "\u2069" in result  # PDI

    def test_wrap_bidi_ltr(self):
        """Test wrapping text with LTR controls."""
        result = wrap_bidi("مرحبا", TextDirection.LTR)
        assert "\u2066" in result  # LRI
        assert "\u2069" in result  # PDI

    def test_bidi_handler_analyze(self):
        """Test BiDi stats analysis."""
        handler = BiDiHandler()
        stats = handler.analyze("Hello مرحبا 123")

        assert stats.ltr_chars > 0
        assert stats.rtl_chars > 0
        assert stats.neutral_chars > 0
        assert stats.is_mixed is True

    def test_bidi_handler_format_mixed(self):
        """Test formatting mixed content."""
        handler = BiDiHandler()
        result = handler.format_mixed(
            "Name: {name}",
            {"name": "أحمد"},
            base_direction=TextDirection.LTR,
        )
        assert "أحمد" in result


# ==============================================================================
# Number Formatting Tests
# ==============================================================================

class TestNumberFormatting:
    """Tests for locale-aware number formatting."""

    def test_english_decimal(self):
        """Test English decimal formatting."""
        result = format_number(1234567.89, "en")
        assert result == "1,234,567.89"

    def test_german_decimal(self):
        """Test German decimal formatting."""
        result = format_number(1234567.89, "de")
        assert result == "1.234.567,89"

    def test_french_decimal(self):
        """Test French decimal formatting."""
        result = format_number(1234567.89, "fr")
        assert result == "1 234 567,89"

    def test_korean_decimal(self):
        """Test Korean decimal formatting."""
        result = format_number(1234567.89, "ko")
        assert result == "1,234,567.89"

    def test_currency_usd(self):
        """Test USD currency formatting."""
        result = format_currency(1234.56, "USD", "en")
        assert "$" in result
        assert "1,234.56" in result

    def test_currency_eur_german(self):
        """Test EUR currency formatting in German."""
        result = format_currency(1234.56, "EUR", "de")
        assert "€" in result
        assert "1.234,56" in result

    def test_percent_formatting(self):
        """Test percent formatting."""
        formatter = LocaleNumberFormatter()
        result = formatter.format(0.1234, LocaleInfo.parse("en"), NumberStyle.PERCENT)
        assert "12.34%" in result.formatted

    def test_compact_formatting(self):
        """Test compact number formatting."""
        formatter = LocaleNumberFormatter()

        # English
        result = formatter.format(1500000, LocaleInfo.parse("en"), NumberStyle.COMPACT)
        assert "M" in result.formatted

        # Korean
        result = formatter.format(15000, LocaleInfo.parse("ko"), NumberStyle.COMPACT)
        assert "만" in result.formatted

    def test_ordinal_formatting(self):
        """Test ordinal number formatting."""
        formatter = LocaleNumberFormatter()

        # English ordinals
        result = formatter.format(1, LocaleInfo.parse("en"), NumberStyle.ORDINAL)
        assert result.formatted == "1st"

        result = formatter.format(2, LocaleInfo.parse("en"), NumberStyle.ORDINAL)
        assert result.formatted == "2nd"

        result = formatter.format(3, LocaleInfo.parse("en"), NumberStyle.ORDINAL)
        assert result.formatted == "3rd"

        result = formatter.format(4, LocaleInfo.parse("en"), NumberStyle.ORDINAL)
        assert result.formatted == "4th"

        result = formatter.format(11, LocaleInfo.parse("en"), NumberStyle.ORDINAL)
        assert result.formatted == "11th"


# ==============================================================================
# Date Formatting Tests
# ==============================================================================

class TestDateFormatting:
    """Tests for locale-aware date formatting."""

    def test_date_medium_english(self):
        """Test medium date format in English."""
        test_date = date(2024, 12, 28)
        result = format_date(test_date, "en", DateStyle.MEDIUM)
        assert "Dec" in result
        assert "28" in result
        assert "2024" in result

    def test_date_long_korean(self):
        """Test long date format in Korean."""
        test_date = date(2024, 12, 28)
        result = format_date(test_date, "ko", DateStyle.LONG)
        assert "2024년" in result
        assert "12월" in result
        assert "28일" in result

    def test_date_iso(self):
        """Test ISO date format."""
        test_date = date(2024, 12, 28)
        result = format_date(test_date, "en", DateStyle.ISO)
        assert result == "2024-12-28"

    def test_relative_time_past(self):
        """Test relative time for past dates."""
        reference = datetime(2024, 12, 28, 12, 0, 0)
        past = reference - timedelta(days=2)
        result = format_relative_time(past, reference=reference, locale="en")
        assert "2 days ago" in result

    def test_relative_time_future(self):
        """Test relative time for future dates."""
        reference = datetime(2024, 12, 28, 12, 0, 0)
        future = reference + timedelta(hours=3)
        result = format_relative_time(future, reference=reference, locale="en")
        assert "in 3 hour" in result

    def test_relative_time_korean(self):
        """Test relative time in Korean."""
        reference = datetime(2024, 12, 28, 12, 0, 0)
        past = reference - timedelta(days=1)
        result = format_relative_time(past, reference=reference, locale="ko")
        assert "어제" in result

    def test_relative_time_japanese(self):
        """Test relative time in Japanese."""
        reference = datetime(2024, 12, 28, 12, 0, 0)
        past = reference - timedelta(days=1)
        result = format_relative_time(past, reference=reference, locale="ja")
        assert "昨日" in result


# ==============================================================================
# Dialect Tests
# ==============================================================================

class TestDialects:
    """Tests for regional dialect support."""

    def test_dialect_registry_default_dialects(self):
        """Test default dialects are registered."""
        registry = DialectRegistry()

        assert registry.get_dialect(LocaleInfo.parse("en-GB")) is not None
        assert registry.get_dialect(LocaleInfo.parse("en-US")) is None  # Not registered
        assert registry.get_dialect(LocaleInfo.parse("pt-BR")) is not None
        assert registry.get_dialect(LocaleInfo.parse("zh-Hant-TW")) is not None

    def test_fallback_chain(self):
        """Test fallback chain generation."""
        chain = get_fallback_chain("en-GB")
        assert "en-GB" in chain
        assert "en" in chain
        assert chain.index("en-GB") < chain.index("en")

    def test_dialect_spelling_rules(self):
        """Test spelling transformation rules."""
        registry = DialectRegistry()
        dialect = registry.get_dialect(LocaleInfo.parse("en-GB"))

        assert dialect is not None
        result = dialect.apply_spelling_rules("The color is nice")
        assert "colour" in result

    def test_dialect_resolver(self):
        """Test dialect-aware message resolution."""
        registry = DialectRegistry()
        resolver = DialectResolver(registry)

        catalogs = {
            "en": {"greeting": "Hello", "farewell": "Goodbye"},
            "en-GB": {"greeting": "Hello"},  # farewell falls back to en
        }

        result = resolver.resolve(
            "greeting",
            LocaleInfo.parse("en-GB"),
            catalogs,
        )
        assert result == "Hello"

        result = resolver.resolve(
            "farewell",
            LocaleInfo.parse("en-GB"),
            catalogs,
        )
        assert result == "Goodbye"  # Fallback to en


# ==============================================================================
# Dynamic Loading Tests
# ==============================================================================

class TestDynamicLoading:
    """Tests for dynamic catalog loading."""

    def test_lru_cache_basic(self):
        """Test LRU cache basic operations."""
        cache = LRUCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert len(cache) == 3

    def test_lru_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = LRUCache(max_size=2)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_memory_storage(self):
        """Test in-memory catalog storage."""
        storage = MemoryStorage()

        catalog = {"key1": "value1", "key2": "value2"}
        locale = LocaleInfo.parse("en")

        storage.save(locale, catalog)
        assert storage.exists(locale)

        loaded = storage.load(locale)
        assert loaded == catalog

    def test_filesystem_storage(self):
        """Test filesystem catalog storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir)
            locale = LocaleInfo.parse("en")
            catalog = {"test.key": "Test message"}

            storage.save(locale, catalog)
            assert storage.exists(locale)

            loaded = storage.load(locale)
            assert loaded == catalog

    def test_catalog_manager_lazy_loading(self):
        """Test catalog manager with lazy loading."""
        storage = MemoryStorage()
        storage.save(
            LocaleInfo.parse("en"),
            {"greeting": "Hello"},
        )
        storage.save(
            LocaleInfo.parse("ko"),
            {"greeting": "안녕하세요"},
        )

        manager = CatalogManager(storage=storage, lazy=True)

        # Catalog not loaded yet
        assert not manager.is_loaded(LocaleInfo.parse("ko"))

        # Load on demand
        catalog = manager.get_catalog(LocaleInfo.parse("ko"))
        assert catalog["greeting"] == "안녕하세요"

    def test_context_resolver(self):
        """Test context-aware message resolution."""
        storage = MemoryStorage()
        storage.save(
            LocaleInfo.parse("ko"),
            {
                "greeting": "안녕",
                "greeting@formal": "안녕하십니까",
                "greeting@informal": "안녕",
            },
        )

        manager = CatalogManager(storage=storage)
        resolver = ContextResolver(manager)

        # Default
        result = resolver.resolve("greeting", LocaleInfo.parse("ko"))
        assert result.message == "안녕"

        # Formal
        result = resolver.resolve(
            "greeting",
            LocaleInfo.parse("ko"),
            context=MessageContext.FORMAL,
        )
        assert result.message == "안녕하십니까"

    def test_context_resolver_plural(self):
        """Test pluralized message resolution."""
        storage = MemoryStorage()
        storage.save(
            LocaleInfo.parse("en"),
            {
                "files#one": "{count} file",
                "files#other": "{count} files",
            },
        )

        manager = CatalogManager(storage=storage)
        resolver = ContextResolver(manager)

        result = resolver.resolve_plural("files", 1, LocaleInfo.parse("en"))
        assert result.message == "1 file"

        result = resolver.resolve_plural("files", 5, LocaleInfo.parse("en"))
        assert result.message == "5 files"


# ==============================================================================
# TMS Integration Tests
# ==============================================================================

class TestTMSIntegration:
    """Tests for TMS integration."""

    def test_tms_manager_providers(self):
        """Test TMS manager provider management."""
        manager = TMSManager()

        # Initially no providers
        assert manager.get_provider("test") is None

    def test_crowdin_provider_init(self):
        """Test Crowdin provider initialization."""
        provider = CrowdinProvider(
            api_key="test-key",
            project_id="test-project",
        )

        assert provider.config.api_key == "test-key"
        assert provider.config.project_id == "test-project"
        assert provider.config.provider == TMSProvider.CROWDIN

    def test_lokalise_provider_init(self):
        """Test Lokalise provider initialization."""
        provider = LokaliseProvider(
            api_key="test-key",
            project_id="test-project",
        )

        assert provider.config.provider == TMSProvider.LOKALISE


# ==============================================================================
# Extended Catalog Tests
# ==============================================================================

class TestExtendedCatalogs:
    """Tests for extended language catalogs."""

    def test_russian_catalog_complete(self):
        """Test Russian catalog completeness."""
        catalog = get_russian_messages()

        assert catalog.locale == "ru"
        assert "null.values_found" in catalog
        assert "unique.duplicates_found" in catalog
        assert "validation.error" in catalog
        assert catalog.metadata.get("complete") is True

    def test_italian_catalog_complete(self):
        """Test Italian catalog completeness."""
        catalog = get_italian_messages()

        assert catalog.locale == "it"
        assert "null.values_found" in catalog
        assert catalog.metadata.get("complete") is True

    def test_portuguese_catalog_complete(self):
        """Test Portuguese catalog completeness."""
        catalog = get_portuguese_messages()

        assert catalog.locale == "pt"
        assert "null.values_found" in catalog
        assert catalog.metadata.get("complete") is True

    def test_arabic_catalog_complete(self):
        """Test Arabic catalog completeness."""
        catalog = get_arabic_messages()

        assert catalog.locale == "ar"
        assert "null.values_found" in catalog
        assert catalog.metadata.get("direction") == "rtl"

    def test_hebrew_catalog_complete(self):
        """Test Hebrew catalog completeness."""
        catalog = get_hebrew_messages()

        assert catalog.locale == "he"
        assert "null.values_found" in catalog
        assert catalog.metadata.get("direction") == "rtl"

    def test_russian_message_formatting(self):
        """Test Russian message formatting."""
        catalog = get_russian_messages()
        template = catalog.get("null.values_found")

        assert template is not None
        message = template.format(count=10, column="email")
        assert "10" in message
        assert "email" in message
        assert "пустых" in message  # "empty" in Russian


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestI18nIntegration:
    """Integration tests for the full i18n system."""

    def setup_method(self):
        """Reset to default locale before each test."""
        set_validator_locale("en")

    def test_full_localization_flow(self):
        """Test complete localization workflow."""
        # Set Korean locale
        set_validator_locale("ko")
        assert get_validator_locale() == "ko"

        # Get localized message
        msg = get_validator_message(
            ValidatorMessageCode.NULL_VALUES_FOUND,
            column="email",
            count=10,
        )
        assert "email" in msg
        assert "10" in msg
        assert "null" in msg or "컬럼" in msg  # Korean or English fallback

    def test_message_catalog_extension(self):
        """Test extending message catalogs."""
        catalog = get_default_messages()

        # Extend with custom messages
        extended = catalog.extend({
            "custom.message": "Custom validation message",
        })

        assert "null.values_found" in extended  # Original
        assert "custom.message" in extended  # Added

    def test_catalog_merge(self):
        """Test merging catalogs."""
        base = ValidatorMessageCatalog.from_dict("en", {
            "key1": "Value 1",
            "key2": "Value 2",
        })
        override = ValidatorMessageCatalog.from_dict("en", {
            "key2": "Updated Value 2",
            "key3": "Value 3",
        })

        merged = base.merge(override)

        assert merged["key1"] == "Value 1"
        assert merged["key2"] == "Updated Value 2"
        assert merged["key3"] == "Value 3"
