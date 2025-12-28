"""Tests for i18n catalog module."""

import pytest
from truthound.datadocs.i18n.catalog import (
    ReportCatalog,
    CatalogBuilder,
    CatalogRegistry,
    get_catalog,
    get_supported_locales,
)


class TestReportCatalog:
    """Tests for ReportCatalog class."""

    def test_create_empty_catalog(self):
        """Test creating empty catalog."""
        catalog = ReportCatalog(locale="en")
        assert catalog.locale == "en"
        assert len(catalog) == 0

    def test_create_catalog_with_messages(self):
        """Test creating catalog with messages."""
        catalog = ReportCatalog(
            locale="ko",
            messages={"report.title": "데이터 품질 보고서"},
        )
        assert catalog["report.title"] == "데이터 품질 보고서"

    def test_get_message(self):
        """Test getting a message."""
        catalog = ReportCatalog(
            locale="en",
            messages={"key": "value"},
        )
        assert catalog.get("key") == "value"
        assert catalog.get("nonexistent") == "nonexistent"
        assert catalog.get("nonexistent", "default") == "default"

    def test_get_with_params(self):
        """Test getting message with parameters."""
        catalog = ReportCatalog(
            locale="en",
            messages={"alert.count": "{count} alerts found"},
        )
        result = catalog.get("alert.count", count=5)
        assert result == "5 alerts found"

    def test_contains(self):
        """Test __contains__."""
        catalog = ReportCatalog(
            locale="en",
            messages={"key": "value"},
        )
        assert "key" in catalog
        assert "nonexistent" not in catalog

    def test_merge(self):
        """Test merging catalogs."""
        catalog1 = ReportCatalog(
            locale="en",
            messages={"a": "1", "b": "2"},
        )
        catalog2 = ReportCatalog(
            locale="en",
            messages={"b": "3", "c": "4"},
        )
        merged = catalog1.merge(catalog2)
        assert merged["a"] == "1"
        assert merged["b"] == "3"  # Overwritten
        assert merged["c"] == "4"

    def test_extend(self):
        """Test extending catalog."""
        catalog = ReportCatalog(
            locale="en",
            messages={"a": "1"},
        )
        extended = catalog.extend({"b": "2"})
        assert extended["a"] == "1"
        assert extended["b"] == "2"
        assert "b" not in catalog  # Original unchanged

    def test_to_dict(self):
        """Test converting to dict."""
        catalog = ReportCatalog(
            locale="en",
            messages={"key": "value"},
            metadata={"name": "English"},
        )
        d = catalog.to_dict()
        assert d["locale"] == "en"
        assert d["messages"]["key"] == "value"
        assert d["metadata"]["name"] == "English"

    def test_from_dict(self):
        """Test creating from dict."""
        catalog = ReportCatalog.from_dict(
            "ja",
            {"report.title": "レポート"},
            metadata={"name": "Japanese"},
        )
        assert catalog.locale == "ja"
        assert catalog["report.title"] == "レポート"


class TestCatalogBuilder:
    """Tests for CatalogBuilder class."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        catalog = (
            CatalogBuilder("en")
            .add("key1", "value1")
            .add("key2", "value2")
            .build()
        )
        assert catalog.locale == "en"
        assert catalog["key1"] == "value1"
        assert catalog["key2"] == "value2"

    def test_builder_report_section(self):
        """Test adding report section."""
        catalog = (
            CatalogBuilder("ko")
            .add_report_section(
                title="보고서",
                subtitle="부제",
                summary="요약",
                details="상세",
            )
            .build()
        )
        assert catalog["report.title"] == "보고서"
        assert catalog["report.subtitle"] == "부제"

    def test_builder_quality_labels(self):
        """Test adding quality labels."""
        catalog = (
            CatalogBuilder("en")
            .add_quality_labels(
                excellent="Excellent",
                good="Good",
                fair="Fair",
                poor="Poor",
                critical="Critical",
            )
            .build()
        )
        assert catalog["quality.excellent"] == "Excellent"
        assert catalog["quality.critical"] == "Critical"

    def test_builder_with_metadata(self):
        """Test adding metadata."""
        catalog = (
            CatalogBuilder("en")
            .with_metadata(name="English", complete=True)
            .build()
        )
        assert catalog.metadata["name"] == "English"
        assert catalog.metadata["complete"] is True


class TestCatalogRegistry:
    """Tests for CatalogRegistry class."""

    def test_get_default_locale(self):
        """Test getting default locale."""
        registry = CatalogRegistry()
        catalog = registry.get("en")
        assert catalog.locale == "en"
        assert "report.title" in catalog

    def test_get_korean(self):
        """Test getting Korean catalog."""
        registry = CatalogRegistry()
        catalog = registry.get("ko")
        assert catalog.locale == "ko"
        assert catalog["report.title"] == "데이터 품질 보고서"

    def test_get_japanese(self):
        """Test getting Japanese catalog."""
        registry = CatalogRegistry()
        catalog = registry.get("ja")
        assert catalog.locale == "ja"
        assert "レポート" in catalog["report.title"]

    def test_get_fallback(self):
        """Test fallback to default locale."""
        registry = CatalogRegistry()
        catalog = registry.get("nonexistent")
        assert catalog.locale == "en"  # Fallback to English

    def test_get_with_region(self):
        """Test getting with region code."""
        registry = CatalogRegistry()
        catalog = registry.get("en_US")
        assert catalog.locale == "en"  # Falls back to base language

    def test_register_custom(self):
        """Test registering custom catalog."""
        registry = CatalogRegistry()
        custom = ReportCatalog(
            locale="custom",
            messages={"test": "Custom message"},
        )
        registry.register(custom)
        result = registry.get("custom")
        assert result["test"] == "Custom message"

    def test_list_locales(self):
        """Test listing locales."""
        registry = CatalogRegistry()
        locales = registry.list_locales()
        assert "en" in locales
        assert "ko" in locales
        assert "ja" in locales


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_catalog(self):
        """Test get_catalog function."""
        catalog = get_catalog("en")
        assert catalog.locale == "en"

    def test_get_supported_locales(self):
        """Test get_supported_locales function."""
        locales = get_supported_locales()
        assert isinstance(locales, list)
        assert len(locales) > 0
        assert "en" in locales
