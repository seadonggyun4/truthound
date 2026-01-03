"""Locale-aware formatting for reports.

This module provides number, date, and time formatting
with locale-specific conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class LocaleFormats:
    """Locale-specific format settings.

    Attributes:
        decimal_separator: Decimal point character.
        thousands_separator: Thousands grouping character.
        date_format: Date format string.
        datetime_format: Datetime format string.
        time_format: Time format string.
        currency_symbol: Currency symbol.
        currency_format: Currency format pattern.
        percent_format: Percentage format pattern.
    """

    decimal_separator: str = "."
    thousands_separator: str = ","
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    time_format: str = "%H:%M:%S"
    currency_symbol: str = "$"
    currency_format: str = "{symbol}{value}"
    percent_format: str = "{value}%"


# Locale format registry
_LOCALE_FORMATS: dict[str, LocaleFormats] = {
    "en": LocaleFormats(),
    "en_US": LocaleFormats(date_format="%m/%d/%Y"),
    "en_GB": LocaleFormats(date_format="%d/%m/%Y", currency_symbol="£"),
    "ko": LocaleFormats(
        date_format="%Y년 %m월 %d일",
        datetime_format="%Y년 %m월 %d일 %H:%M:%S",
        currency_symbol="₩",
        currency_format="{symbol}{value}",
    ),
    "ja": LocaleFormats(
        date_format="%Y年%m月%d日",
        datetime_format="%Y年%m月%d日 %H:%M:%S",
        currency_symbol="¥",
    ),
    "zh": LocaleFormats(
        date_format="%Y年%m月%d日",
        datetime_format="%Y年%m月%d日 %H:%M:%S",
        currency_symbol="¥",
    ),
    "de": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d.%m.%Y",
        currency_symbol="€",
        currency_format="{value} {symbol}",
    ),
    "fr": LocaleFormats(
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d/%m/%Y",
        currency_symbol="€",
        currency_format="{value} {symbol}",
    ),
    "es": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        currency_symbol="€",
        currency_format="{value} {symbol}",
    ),
    "pt": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        currency_symbol="R$",
    ),
    "it": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        currency_symbol="€",
        currency_format="{value} {symbol}",
    ),
    "ru": LocaleFormats(
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d.%m.%Y",
        currency_symbol="₽",
        currency_format="{value} {symbol}",
    ),
    "ar": LocaleFormats(
        date_format="%Y/%m/%d",
        currency_symbol="ر.س",
        currency_format="{value} {symbol}",
    ),
    "th": LocaleFormats(
        date_format="%d/%m/%Y",
        currency_symbol="฿",
    ),
    "vi": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        currency_symbol="₫",
        currency_format="{value} {symbol}",
    ),
    "id": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        currency_symbol="Rp",
    ),
    "tr": LocaleFormats(
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d.%m.%Y",
        currency_symbol="₺",
        currency_format="{value} {symbol}",
    ),
}


def get_locale_formats(locale: str) -> LocaleFormats:
    """Get format settings for a locale.

    Args:
        locale: Locale code.

    Returns:
        LocaleFormats instance.
    """
    if locale in _LOCALE_FORMATS:
        return _LOCALE_FORMATS[locale]

    lang = locale.split("_")[0].split("-")[0]
    if lang in _LOCALE_FORMATS:
        return _LOCALE_FORMATS[lang]

    return _LOCALE_FORMATS["en"]


class NumberFormatter:
    """Locale-aware number formatter.

    Example:
        formatter = NumberFormatter("de")
        formatter.format(1234567.89)  # "1.234.567,89"
    """

    def __init__(self, locale: str = "en") -> None:
        self._locale = locale
        self._formats = get_locale_formats(locale)

    def format(
        self,
        value: float | int,
        decimals: int | None = None,
        group_digits: bool = True,
    ) -> str:
        """Format a number.

        Args:
            value: Number to format.
            decimals: Decimal places (None for auto).
            group_digits: Use thousands separator.

        Returns:
            Formatted string.
        """
        if decimals is not None:
            value = round(value, decimals)

        # Handle negative values
        is_negative = value < 0
        value = abs(value)

        # Split integer and decimal parts
        if decimals == 0:
            int_part = str(int(value))
            dec_part = ""
        elif isinstance(value, int) and decimals is None:
            # Integer with auto decimals: no decimal part
            int_part = str(int(value))
            dec_part = ""
        else:
            # Use Decimal for precise string representation to avoid
            # floating-point precision issues (e.g., 1234567.89 -> 1234567.8899999999)
            from decimal import Decimal, ROUND_HALF_UP

            if decimals is not None:
                # Round to specified decimals
                dec_value = Decimal(str(value)).quantize(
                    Decimal(10) ** -decimals, rounding=ROUND_HALF_UP
                )
                str_val = str(dec_value)
            else:
                # Auto: remove trailing zeros
                dec_value = Decimal(str(value))
                str_val = str(dec_value.normalize())
                # Handle scientific notation for very small numbers
                if "E" in str_val:
                    str_val = f"{float(dec_value):.10f}".rstrip("0").rstrip(".")

            if "." in str_val:
                int_part, dec_part = str_val.split(".")
                if decimals is not None:
                    dec_part = dec_part[:decimals].ljust(decimals, "0")
            else:
                int_part = str_val
                dec_part = ""

        # Group integer digits
        if group_digits and len(int_part) > 3:
            parts = []
            while len(int_part) > 3:
                parts.append(int_part[-3:])
                int_part = int_part[:-3]
            parts.append(int_part)
            int_part = self._formats.thousands_separator.join(reversed(parts))

        # Combine
        result = int_part
        if dec_part:
            result = f"{int_part}{self._formats.decimal_separator}{dec_part}"

        if is_negative:
            result = f"-{result}"

        return result

    def format_percentage(
        self,
        value: float,
        decimals: int = 1,
    ) -> str:
        """Format as percentage.

        Args:
            value: Value (0.5 = 50%).
            decimals: Decimal places.

        Returns:
            Formatted percentage.
        """
        pct_value = value * 100
        formatted = self.format(pct_value, decimals)
        return self._formats.percent_format.format(value=formatted)

    def format_currency(
        self,
        value: float,
        decimals: int = 2,
        symbol: str | None = None,
    ) -> str:
        """Format as currency.

        Args:
            value: Value.
            decimals: Decimal places.
            symbol: Currency symbol (None for locale default).

        Returns:
            Formatted currency.
        """
        formatted = self.format(value, decimals)
        sym = symbol or self._formats.currency_symbol
        return self._formats.currency_format.format(
            symbol=sym,
            value=formatted,
        )

    def format_compact(
        self,
        value: float | int,
        decimals: int = 1,
    ) -> str:
        """Format with compact notation (K, M, B).

        Args:
            value: Value.
            decimals: Decimal places.

        Returns:
            Compact formatted string.
        """
        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if abs_value >= 1_000_000_000:
            return f"{sign}{self.format(abs_value / 1_000_000_000, decimals)}B"
        elif abs_value >= 1_000_000:
            return f"{sign}{self.format(abs_value / 1_000_000, decimals)}M"
        elif abs_value >= 1_000:
            return f"{sign}{self.format(abs_value / 1_000, decimals)}K"
        else:
            return self.format(value, decimals)


class DateFormatter:
    """Locale-aware date formatter.

    Example:
        formatter = DateFormatter("ko")
        formatter.format(datetime.now())  # "2025년 12월 28일"
    """

    def __init__(self, locale: str = "en") -> None:
        self._locale = locale
        self._formats = get_locale_formats(locale)

    def format(self, dt: datetime) -> str:
        """Format a date.

        Args:
            dt: Datetime object.

        Returns:
            Formatted date string.
        """
        return dt.strftime(self._formats.date_format)

    def format_datetime(self, dt: datetime) -> str:
        """Format datetime.

        Args:
            dt: Datetime object.

        Returns:
            Formatted datetime string.
        """
        return dt.strftime(self._formats.datetime_format)

    def format_time(self, dt: datetime) -> str:
        """Format time.

        Args:
            dt: Datetime object.

        Returns:
            Formatted time string.
        """
        return dt.strftime(self._formats.time_format)

    def format_relative(self, dt: datetime, now: datetime | None = None) -> str:
        """Format as relative time.

        Args:
            dt: Target datetime.
            now: Reference time (None for current).

        Returns:
            Relative time string (e.g., "2 hours ago").
        """
        if now is None:
            now = datetime.now()

        diff = now - dt
        seconds = diff.total_seconds()

        if seconds < 0:
            return self._format_future(-seconds)
        return self._format_past(seconds)

    def _format_past(self, seconds: float) -> str:
        """Format past time."""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds < 2592000:
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif seconds < 31536000:
            months = int(seconds / 2592000)
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = int(seconds / 31536000)
            return f"{years} year{'s' if years != 1 else ''} ago"

    def _format_future(self, seconds: float) -> str:
        """Format future time."""
        if seconds < 60:
            return "in a moment"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"in {mins} minute{'s' if mins != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"in {hours} hour{'s' if hours != 1 else ''}"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"in {days} day{'s' if days != 1 else ''}"
        else:
            weeks = int(seconds / 604800)
            return f"in {weeks} week{'s' if weeks != 1 else ''}"


def format_number(
    value: float | int,
    locale: str = "en",
    decimals: int | None = None,
) -> str:
    """Format a number for a locale.

    Args:
        value: Number to format.
        locale: Locale code.
        decimals: Decimal places.

    Returns:
        Formatted string.
    """
    return NumberFormatter(locale).format(value, decimals)


def format_percentage(
    value: float,
    locale: str = "en",
    decimals: int = 1,
) -> str:
    """Format as percentage.

    Args:
        value: Value (0.5 = 50%).
        locale: Locale code.
        decimals: Decimal places.

    Returns:
        Formatted percentage.
    """
    return NumberFormatter(locale).format_percentage(value, decimals)


def format_date(
    dt: datetime,
    locale: str = "en",
) -> str:
    """Format a date.

    Args:
        dt: Datetime object.
        locale: Locale code.

    Returns:
        Formatted date.
    """
    return DateFormatter(locale).format(dt)


def format_datetime(
    dt: datetime,
    locale: str = "en",
) -> str:
    """Format datetime.

    Args:
        dt: Datetime object.
        locale: Locale code.

    Returns:
        Formatted datetime.
    """
    return DateFormatter(locale).format_datetime(dt)


def format_duration(
    seconds: float | int,
    locale: str = "en",
) -> str:
    """Format duration.

    Args:
        seconds: Duration in seconds.
        locale: Locale code.

    Returns:
        Formatted duration.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"
