"""Tests for engine context module."""

import pytest
from truthound.datadocs.engine.context import (
    ReportContext,
    ReportData,
    TranslatableString,
)


class TestReportData:
    """Tests for ReportData class."""

    def test_create_empty_data(self):
        """Test creating empty report data."""
        data = ReportData()
        assert data.sections == {}
        assert data.metadata == {}
        assert data.alerts == []
        assert data.recommendations == []
        assert data.charts == []
        assert data.tables == []
        assert data.raw == {}

    def test_create_data_with_sections(self):
        """Test creating report data with sections."""
        data = ReportData(
            sections={"overview": {"row_count": 1000}},
            metadata={"title": "Test Report"},
        )
        assert data.sections["overview"]["row_count"] == 1000
        assert data.metadata["title"] == "Test Report"

    def test_with_section(self):
        """Test adding a section."""
        data = ReportData()
        new_data = data.with_section("columns", {"id": {"type": "int"}})
        assert "columns" in new_data.sections
        assert data.sections == {}  # Original unchanged

    def test_with_alert(self):
        """Test adding an alert."""
        data = ReportData()
        new_data = data.with_alert({"title": "High null ratio", "severity": "warning"})
        assert len(new_data.alerts) == 1
        assert new_data.alerts[0]["title"] == "High null ratio"
        assert data.alerts == []  # Original unchanged

    def test_with_metadata(self):
        """Test adding metadata."""
        data = ReportData(metadata={"version": "1.0"})
        new_data = data.with_metadata(title="Report", author="Test")
        assert new_data.metadata["title"] == "Report"
        assert new_data.metadata["version"] == "1.0"


class TestReportContext:
    """Tests for ReportContext class."""

    def test_create_default_context(self):
        """Test creating context with defaults."""
        ctx = ReportContext(data=ReportData())
        assert ctx.locale == "en"
        assert ctx.theme == "default"
        assert ctx.template == "default"
        assert ctx.title == "Data Quality Report"
        assert ctx.subtitle == ""

    def test_create_custom_context(self):
        """Test creating context with custom values."""
        data = ReportData(metadata={"title": "Test Report"})
        ctx = ReportContext(
            data=data,
            locale="ko",
            theme="dark",
        )
        assert ctx.locale == "ko"
        assert ctx.theme == "dark"
        assert ctx.title == "Test Report"

    def test_with_locale(self):
        """Test changing locale."""
        ctx = ReportContext(data=ReportData(), locale="en")
        new_ctx = ctx.with_locale("ja")
        assert new_ctx.locale == "ja"
        assert ctx.locale == "en"  # Original unchanged

    def test_with_theme(self):
        """Test changing theme."""
        ctx = ReportContext(data=ReportData(), theme="default")
        new_ctx = ctx.with_theme("corporate")
        assert new_ctx.theme == "corporate"
        assert ctx.theme == "default"

    def test_with_option(self):
        """Test adding options."""
        ctx = ReportContext(data=ReportData())
        new_ctx = ctx.with_option("show_charts", True)
        assert new_ctx.options["show_charts"] is True
        assert "show_charts" not in ctx.options

    def test_context_is_frozen(self):
        """Test that context is immutable."""
        ctx = ReportContext(data=ReportData())
        with pytest.raises((TypeError, AttributeError)):
            ctx.locale = "ko"  # type: ignore


class TestTranslatableString:
    """Tests for TranslatableString class."""

    def test_create_translatable(self):
        """Test creating translatable string."""
        t = TranslatableString("report.title")
        assert t.key == "report.title"
        assert t.default is None
        assert t.params == {}

    def test_create_with_default(self):
        """Test creating with default value."""
        t = TranslatableString("report.title", default="Report")
        assert t.default == "Report"

    def test_create_with_params(self):
        """Test creating with parameters."""
        t = TranslatableString("alert.count", params={"count": 5})
        assert t.params["count"] == 5

    def test_str_representation(self):
        """Test string representation uses default via format method."""
        t = TranslatableString("key", default="Default")
        # TranslatableString doesn't implement __str__, use format or default
        assert t.default == "Default"
        assert t.format("{value}") == "Default"  # Falls back to default on KeyError

    def test_str_without_default(self):
        """Test format method without default returns key."""
        t = TranslatableString("report.title")
        # format() returns key when template has missing params and no default
        assert t.format("{missing}") == "report.title"
