"""Tests for i18n formatting module."""

import pytest
from datetime import datetime
from truthound.datadocs.i18n.formatting import (
    NumberFormatter,
    DateFormatter,
    format_number,
    format_percentage,
    format_date,
    format_datetime,
    format_duration,
    get_locale_formats,
)


class TestNumberFormatter:
    """Tests for NumberFormatter class."""

    def test_english_format(self):
        """Test English number formatting."""
        formatter = NumberFormatter("en")
        assert formatter.format(1234567) == "1,234,567"
        assert formatter.format(1234567.89) == "1,234,567.89"

    def test_german_format(self):
        """Test German number formatting."""
        formatter = NumberFormatter("de")
        assert formatter.format(1234567) == "1.234.567"
        # German uses comma for decimal
        assert "," in formatter.format(1234567.89)

    def test_french_format(self):
        """Test French number formatting."""
        formatter = NumberFormatter("fr")
        # French uses space for thousands
        result = formatter.format(1234567)
        assert " " in result or result == "1 234 567"

    def test_with_decimals(self):
        """Test specifying decimal places."""
        formatter = NumberFormatter("en")
        assert formatter.format(3.14159, decimals=2) == "3.14"
        assert formatter.format(3.1, decimals=3) == "3.100"

    def test_without_grouping(self):
        """Test without thousands grouping."""
        formatter = NumberFormatter("en")
        assert formatter.format(1234567, group_digits=False) == "1234567"

    def test_percentage(self):
        """Test percentage formatting."""
        formatter = NumberFormatter("en")
        assert formatter.format_percentage(0.5) == "50.0%"
        assert formatter.format_percentage(0.123, decimals=1) == "12.3%"

    def test_currency(self):
        """Test currency formatting."""
        formatter = NumberFormatter("en")
        result = formatter.format_currency(1234.56)
        assert "$" in result
        assert "1,234.56" in result

    def test_currency_german(self):
        """Test German currency formatting."""
        formatter = NumberFormatter("de")
        result = formatter.format_currency(1234.56)
        assert "€" in result

    def test_compact_notation(self):
        """Test compact notation."""
        formatter = NumberFormatter("en")
        assert formatter.format_compact(1500) == "1.5K"
        assert formatter.format_compact(1500000) == "1.5M"
        assert formatter.format_compact(1500000000) == "1.5B"
        assert formatter.format_compact(500) == "500.0"

    def test_negative_compact(self):
        """Test negative numbers in compact notation."""
        formatter = NumberFormatter("en")
        assert formatter.format_compact(-1500) == "-1.5K"


class TestDateFormatter:
    """Tests for DateFormatter class."""

    def test_english_date(self):
        """Test English date formatting."""
        formatter = DateFormatter("en")
        dt = datetime(2025, 12, 28)
        result = formatter.format(dt)
        assert "2025" in result
        assert "12" in result
        assert "28" in result

    def test_korean_date(self):
        """Test Korean date formatting."""
        formatter = DateFormatter("ko")
        dt = datetime(2025, 12, 28)
        result = formatter.format(dt)
        assert "년" in result
        assert "월" in result
        assert "일" in result

    def test_japanese_date(self):
        """Test Japanese date formatting."""
        formatter = DateFormatter("ja")
        dt = datetime(2025, 12, 28)
        result = formatter.format(dt)
        assert "年" in result
        assert "月" in result
        assert "日" in result

    def test_datetime_format(self):
        """Test datetime formatting."""
        formatter = DateFormatter("en")
        dt = datetime(2025, 12, 28, 14, 30, 45)
        result = formatter.format_datetime(dt)
        assert "2025" in result
        assert "14" in result or "2" in result  # 24h or 12h

    def test_time_format(self):
        """Test time formatting."""
        formatter = DateFormatter("en")
        dt = datetime(2025, 12, 28, 14, 30, 45)
        result = formatter.format_time(dt)
        assert "14:30:45" in result

    def test_relative_time_seconds(self):
        """Test relative time for seconds."""
        formatter = DateFormatter("en")
        now = datetime(2025, 12, 28, 12, 0, 0)
        dt = datetime(2025, 12, 28, 11, 59, 30)
        result = formatter.format_relative(dt, now)
        assert "just now" in result

    def test_relative_time_minutes(self):
        """Test relative time for minutes."""
        formatter = DateFormatter("en")
        now = datetime(2025, 12, 28, 12, 0, 0)
        dt = datetime(2025, 12, 28, 11, 55, 0)
        result = formatter.format_relative(dt, now)
        assert "minute" in result

    def test_relative_time_hours(self):
        """Test relative time for hours."""
        formatter = DateFormatter("en")
        now = datetime(2025, 12, 28, 12, 0, 0)
        dt = datetime(2025, 12, 28, 9, 0, 0)
        result = formatter.format_relative(dt, now)
        assert "hour" in result

    def test_relative_time_days(self):
        """Test relative time for days."""
        formatter = DateFormatter("en")
        now = datetime(2025, 12, 28, 12, 0, 0)
        dt = datetime(2025, 12, 25, 12, 0, 0)
        result = formatter.format_relative(dt, now)
        assert "day" in result


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_format_number(self):
        """Test format_number function."""
        assert format_number(1234, "en") == "1,234"

    def test_format_percentage(self):
        """Test format_percentage function."""
        assert format_percentage(0.5, "en") == "50.0%"

    def test_format_date(self):
        """Test format_date function."""
        dt = datetime(2025, 12, 28)
        result = format_date(dt, "en")
        assert "2025" in result

    def test_format_datetime(self):
        """Test format_datetime function."""
        dt = datetime(2025, 12, 28, 14, 30)
        result = format_datetime(dt, "en")
        assert "2025" in result

    def test_format_duration(self):
        """Test format_duration function."""
        assert format_duration(30) == "30.0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3700) == "1h 1m"


class TestLocaleFormats:
    """Tests for locale format settings."""

    def test_english_formats(self):
        """Test English format settings."""
        formats = get_locale_formats("en")
        assert formats.decimal_separator == "."
        assert formats.thousands_separator == ","

    def test_german_formats(self):
        """Test German format settings."""
        formats = get_locale_formats("de")
        assert formats.decimal_separator == ","
        assert formats.thousands_separator == "."
        assert formats.currency_symbol == "€"

    def test_fallback_formats(self):
        """Test fallback for unknown locale."""
        formats = get_locale_formats("unknown")
        assert formats.decimal_separator == "."  # Falls back to English

    def test_rtl_locale(self):
        """Test RTL locale metadata."""
        formats = get_locale_formats("ar")
        # Arabic should have specific formatting
        assert formats is not None
