"""Tests for Quality Reporter Factory."""

import pytest

from truthound.reporters.quality.factory import (
    get_quality_reporter,
    register_quality_reporter,
    unregister_quality_reporter,
    list_quality_formats,
    is_quality_format_available,
    create_console_reporter,
    create_json_reporter,
    create_html_reporter,
    create_markdown_reporter,
)
from truthound.reporters.quality.reporters import (
    ConsoleQualityReporter,
    JsonQualityReporter,
    MarkdownQualityReporter,
    HtmlQualityReporter,
    JUnitQualityReporter,
)
from truthound.reporters.quality.base import (
    BaseQualityReporter,
    QualityReporterError,
)
from truthound.reporters.quality.config import QualityReporterConfig


# =============================================================================
# Test get_quality_reporter
# =============================================================================


class TestGetQualityReporter:
    """Tests for get_quality_reporter function."""

    def test_get_console_reporter(self):
        """Test getting console reporter."""
        reporter = get_quality_reporter("console")
        assert isinstance(reporter, ConsoleQualityReporter)

    def test_get_json_reporter(self):
        """Test getting JSON reporter."""
        reporter = get_quality_reporter("json")
        assert isinstance(reporter, JsonQualityReporter)

    def test_get_html_reporter(self):
        """Test getting HTML reporter."""
        reporter = get_quality_reporter("html")
        assert isinstance(reporter, HtmlQualityReporter)

    def test_get_markdown_reporter(self):
        """Test getting Markdown reporter."""
        reporter = get_quality_reporter("markdown")
        assert isinstance(reporter, MarkdownQualityReporter)

    def test_get_junit_reporter(self):
        """Test getting JUnit reporter."""
        reporter = get_quality_reporter("junit")
        assert isinstance(reporter, JUnitQualityReporter)

    def test_format_aliases(self):
        """Test format aliases work."""
        # md -> markdown
        reporter = get_quality_reporter("md")
        assert isinstance(reporter, MarkdownQualityReporter)

        # terminal -> console
        reporter = get_quality_reporter("terminal")
        assert isinstance(reporter, ConsoleQualityReporter)

        # rich -> console
        reporter = get_quality_reporter("rich")
        assert isinstance(reporter, ConsoleQualityReporter)

        # xml -> junit
        reporter = get_quality_reporter("xml")
        assert isinstance(reporter, JUnitQualityReporter)

    def test_case_insensitive(self):
        """Test format is case insensitive."""
        reporter = get_quality_reporter("JSON")
        assert isinstance(reporter, JsonQualityReporter)

        reporter = get_quality_reporter("Console")
        assert isinstance(reporter, ConsoleQualityReporter)

    def test_strips_whitespace(self):
        """Test format strips whitespace."""
        reporter = get_quality_reporter("  json  ")
        assert isinstance(reporter, JsonQualityReporter)

    def test_unknown_format_raises_error(self):
        """Test unknown format raises error."""
        with pytest.raises(QualityReporterError, match="Unknown"):
            get_quality_reporter("unknown_format")

    def test_with_config(self):
        """Test passing config to reporter."""
        config = QualityReporterConfig(title="Test Report")
        reporter = get_quality_reporter("json", config=config)

        assert reporter.config.title == "Test Report"

    def test_with_kwargs(self):
        """Test passing kwargs to reporter."""
        reporter = get_quality_reporter("json", indent=4)
        assert reporter._indent == 4


# =============================================================================
# Test register/unregister
# =============================================================================


class TestRegisterUnregister:
    """Tests for register and unregister functions."""

    def test_register_custom_reporter(self):
        """Test registering a custom reporter."""

        @register_quality_reporter("custom_test")
        class CustomReporter(BaseQualityReporter):
            name = "custom_test"

            def render(self, data):
                return "custom output"

        reporter = get_quality_reporter("custom_test")
        assert reporter.name == "custom_test"

        # Clean up
        unregister_quality_reporter("custom_test")

    def test_unregister_reporter(self):
        """Test unregistering a reporter."""

        @register_quality_reporter("to_unregister")
        class TempReporter(BaseQualityReporter):
            name = "to_unregister"

            def render(self, data):
                return ""

        # Should work initially
        reporter = get_quality_reporter("to_unregister")
        assert reporter is not None

        # Unregister
        result = unregister_quality_reporter("to_unregister")
        assert result is True

        # Should fail now
        with pytest.raises(QualityReporterError):
            get_quality_reporter("to_unregister")

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent reporter."""
        result = unregister_quality_reporter("nonexistent")
        assert result is False


# =============================================================================
# Test list_quality_formats
# =============================================================================


class TestListQualityFormats:
    """Tests for list_quality_formats function."""

    def test_returns_list(self):
        """Test returns a list."""
        formats = list_quality_formats()
        assert isinstance(formats, list)

    def test_includes_builtin_formats(self):
        """Test includes built-in formats."""
        formats = list_quality_formats()

        assert "console" in formats
        assert "json" in formats
        assert "html" in formats
        assert "markdown" in formats
        assert "junit" in formats

    def test_includes_aliases(self):
        """Test includes aliases."""
        formats = list_quality_formats()

        assert "md" in formats
        assert "terminal" in formats
        assert "rich" in formats
        assert "xml" in formats

    def test_sorted(self):
        """Test list is sorted."""
        formats = list_quality_formats()
        assert formats == sorted(formats)


# =============================================================================
# Test is_quality_format_available
# =============================================================================


class TestIsQualityFormatAvailable:
    """Tests for is_quality_format_available function."""

    def test_builtin_formats_available(self):
        """Test built-in formats are available."""
        assert is_quality_format_available("console")
        assert is_quality_format_available("json")
        assert is_quality_format_available("html")
        assert is_quality_format_available("markdown")
        assert is_quality_format_available("junit")

    def test_aliases_available(self):
        """Test aliases are available."""
        assert is_quality_format_available("md")
        assert is_quality_format_available("terminal")
        assert is_quality_format_available("rich")
        assert is_quality_format_available("xml")

    def test_unknown_not_available(self):
        """Test unknown format not available."""
        assert not is_quality_format_available("unknown_format")

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_quality_format_available("JSON")
        assert is_quality_format_available("Console")


# =============================================================================
# Test convenience functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_console_reporter(self):
        """Test create_console_reporter."""
        reporter = create_console_reporter()
        assert isinstance(reporter, ConsoleQualityReporter)

    def test_create_json_reporter(self):
        """Test create_json_reporter."""
        reporter = create_json_reporter()
        assert isinstance(reporter, JsonQualityReporter)

    def test_create_html_reporter(self):
        """Test create_html_reporter."""
        reporter = create_html_reporter()
        assert isinstance(reporter, HtmlQualityReporter)

    def test_create_markdown_reporter(self):
        """Test create_markdown_reporter."""
        reporter = create_markdown_reporter()
        assert isinstance(reporter, MarkdownQualityReporter)

    def test_create_with_kwargs(self):
        """Test create functions accept kwargs."""
        reporter = create_json_reporter(indent=4)
        assert reporter._indent == 4

        config = QualityReporterConfig(title="Test")
        reporter = create_console_reporter(config=config)
        assert reporter.config.title == "Test"
