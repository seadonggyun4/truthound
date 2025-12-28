"""Locale-Aware Number and Date Formatting.

This module provides comprehensive locale-aware formatting for:
- Numbers (decimal, currency, percent, scientific, compact, ordinal)
- Dates (short, medium, long, full, ISO, relative)
- Times (short, medium, long, full)
- Durations (human-readable time spans)

Features:
- CLDR-based formatting patterns
- Locale-specific number symbols (decimal, grouping, etc.)
- Multiple calendar support
- Relative time formatting ("2 days ago")
- Currency formatting with symbol positioning

Usage:
    from truthound.validators.i18n.formatting import (
        LocaleNumberFormatter,
        LocaleDateFormatter,
        format_number,
        format_date,
        format_currency,
    )

    # Format numbers
    format_number(1234567.89, "de")  # "1.234.567,89"
    format_number(1234567.89, "en")  # "1,234,567.89"

    # Format currency
    format_currency(1234.56, "USD", "en")  # "$1,234.56"
    format_currency(1234.56, "EUR", "de")  # "1.234,56 €"

    # Format dates
    format_date(datetime.now(), "ko", DateStyle.LONG)  # "2024년 12월 28일"
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from truthound.validators.i18n.protocols import (
    BaseDateFormatter,
    BaseNumberFormatter,
    DateStyle,
    FormattedDate,
    FormattedNumber,
    LocaleInfo,
    NumberStyle,
    TextDirection,
    TimeStyle,
)


# ==============================================================================
# Locale Data: Number Symbols
# ==============================================================================

@dataclass
class NumberSymbols:
    """Locale-specific number symbols.

    Based on CLDR number symbols data.
    """
    decimal: str = "."
    group: str = ","
    minus: str = "-"
    plus: str = "+"
    percent: str = "%"
    per_mille: str = "‰"
    exponential: str = "E"
    infinity: str = "∞"
    nan: str = "NaN"
    currency_decimal: str | None = None
    currency_group: str | None = None

    def get_currency_decimal(self) -> str:
        """Get decimal separator for currency (falls back to decimal)."""
        return self.currency_decimal or self.decimal

    def get_currency_group(self) -> str:
        """Get grouping separator for currency (falls back to group)."""
        return self.currency_group or self.group


# Number symbols by locale
_NUMBER_SYMBOLS: dict[str, NumberSymbols] = {
    # Default (English)
    "en": NumberSymbols(),

    # German, Austrian German
    "de": NumberSymbols(decimal=",", group="."),
    "de_AT": NumberSymbols(decimal=",", group=" "),

    # French
    "fr": NumberSymbols(decimal=",", group=" ", currency_group=" "),
    "fr_CH": NumberSymbols(decimal=",", group=" "),

    # Spanish
    "es": NumberSymbols(decimal=",", group="."),
    "es_MX": NumberSymbols(decimal=".", group=","),

    # Italian
    "it": NumberSymbols(decimal=",", group="."),

    # Portuguese
    "pt": NumberSymbols(decimal=",", group="."),
    "pt_BR": NumberSymbols(decimal=",", group="."),

    # Russian
    "ru": NumberSymbols(decimal=",", group=" "),

    # Polish
    "pl": NumberSymbols(decimal=",", group=" "),

    # Dutch
    "nl": NumberSymbols(decimal=",", group="."),

    # Czech
    "cs": NumberSymbols(decimal=",", group=" "),

    # Swedish, Norwegian, Danish, Finnish
    "sv": NumberSymbols(decimal=",", group=" "),
    "no": NumberSymbols(decimal=",", group=" "),
    "nb": NumberSymbols(decimal=",", group=" "),
    "da": NumberSymbols(decimal=",", group="."),
    "fi": NumberSymbols(decimal=",", group=" "),

    # Korean, Japanese, Chinese
    "ko": NumberSymbols(decimal=".", group=","),
    "ja": NumberSymbols(decimal=".", group=","),
    "zh": NumberSymbols(decimal=".", group=","),
    "zh_TW": NumberSymbols(decimal=".", group=","),

    # Arabic
    "ar": NumberSymbols(decimal="٫", group="٬", percent="٪", minus="-"),

    # Hebrew
    "he": NumberSymbols(decimal=".", group=","),

    # Hindi
    "hi": NumberSymbols(decimal=".", group=","),

    # Thai
    "th": NumberSymbols(decimal=".", group=","),

    # Vietnamese
    "vi": NumberSymbols(decimal=",", group="."),

    # Indonesian
    "id": NumberSymbols(decimal=",", group="."),

    # Turkish
    "tr": NumberSymbols(decimal=",", group="."),

    # Greek
    "el": NumberSymbols(decimal=",", group="."),

    # Hungarian
    "hu": NumberSymbols(decimal=",", group=" "),

    # Romanian
    "ro": NumberSymbols(decimal=",", group="."),

    # Ukrainian
    "uk": NumberSymbols(decimal=",", group=" "),

    # Swiss locales
    "de_CH": NumberSymbols(decimal=".", group="'"),
    "it_CH": NumberSymbols(decimal=".", group="'"),

    # Indian English
    "en_IN": NumberSymbols(decimal=".", group=","),
}


def get_number_symbols(locale: LocaleInfo) -> NumberSymbols:
    """Get number symbols for a locale.

    Args:
        locale: Target locale

    Returns:
        NumberSymbols for the locale
    """
    # Try full locale tag first
    key = f"{locale.language}_{locale.region}" if locale.region else locale.language
    if key in _NUMBER_SYMBOLS:
        return _NUMBER_SYMBOLS[key]

    # Fall back to language only
    if locale.language in _NUMBER_SYMBOLS:
        return _NUMBER_SYMBOLS[locale.language]

    # Default to English
    return _NUMBER_SYMBOLS["en"]


# ==============================================================================
# Locale Data: Currency Information
# ==============================================================================

@dataclass
class CurrencyInfo:
    """Currency formatting information."""
    code: str
    symbol: str
    narrow_symbol: str | None = None
    name: str = ""
    decimal_digits: int = 2
    rounding: int = 0


# Common currencies
_CURRENCIES: dict[str, CurrencyInfo] = {
    "USD": CurrencyInfo("USD", "$", "$", "US Dollar", 2),
    "EUR": CurrencyInfo("EUR", "€", "€", "Euro", 2),
    "GBP": CurrencyInfo("GBP", "£", "£", "British Pound", 2),
    "JPY": CurrencyInfo("JPY", "¥", "¥", "Japanese Yen", 0),
    "CNY": CurrencyInfo("CNY", "¥", "¥", "Chinese Yuan", 2),
    "KRW": CurrencyInfo("KRW", "₩", "₩", "Korean Won", 0),
    "INR": CurrencyInfo("INR", "₹", "₹", "Indian Rupee", 2),
    "BRL": CurrencyInfo("BRL", "R$", "R$", "Brazilian Real", 2),
    "RUB": CurrencyInfo("RUB", "₽", "₽", "Russian Ruble", 2),
    "AUD": CurrencyInfo("AUD", "A$", "$", "Australian Dollar", 2),
    "CAD": CurrencyInfo("CAD", "CA$", "$", "Canadian Dollar", 2),
    "CHF": CurrencyInfo("CHF", "CHF", "CHF", "Swiss Franc", 2),
    "HKD": CurrencyInfo("HKD", "HK$", "$", "Hong Kong Dollar", 2),
    "SGD": CurrencyInfo("SGD", "S$", "$", "Singapore Dollar", 2),
    "SEK": CurrencyInfo("SEK", "kr", "kr", "Swedish Krona", 2),
    "NOK": CurrencyInfo("NOK", "kr", "kr", "Norwegian Krone", 2),
    "DKK": CurrencyInfo("DKK", "kr", "kr", "Danish Krone", 2),
    "NZD": CurrencyInfo("NZD", "NZ$", "$", "New Zealand Dollar", 2),
    "MXN": CurrencyInfo("MXN", "MX$", "$", "Mexican Peso", 2),
    "ZAR": CurrencyInfo("ZAR", "R", "R", "South African Rand", 2),
    "TRY": CurrencyInfo("TRY", "₺", "₺", "Turkish Lira", 2),
    "PLN": CurrencyInfo("PLN", "zł", "zł", "Polish Zloty", 2),
    "THB": CurrencyInfo("THB", "฿", "฿", "Thai Baht", 2),
    "IDR": CurrencyInfo("IDR", "Rp", "Rp", "Indonesian Rupiah", 0),
    "MYR": CurrencyInfo("MYR", "RM", "RM", "Malaysian Ringgit", 2),
    "PHP": CurrencyInfo("PHP", "₱", "₱", "Philippine Peso", 2),
    "VND": CurrencyInfo("VND", "₫", "₫", "Vietnamese Dong", 0),
    "AED": CurrencyInfo("AED", "د.إ", "د.إ", "UAE Dirham", 2),
    "SAR": CurrencyInfo("SAR", "ر.س", "ر.س", "Saudi Riyal", 2),
    "ILS": CurrencyInfo("ILS", "₪", "₪", "Israeli Shekel", 2),
}


def get_currency_info(code: str) -> CurrencyInfo:
    """Get currency information.

    Args:
        code: ISO 4217 currency code

    Returns:
        CurrencyInfo for the currency
    """
    return _CURRENCIES.get(code.upper(), CurrencyInfo(code, code, code, code))


# ==============================================================================
# Locale Data: Date/Time Patterns
# ==============================================================================

@dataclass
class DateTimePatterns:
    """Locale-specific date/time patterns."""
    # Date patterns by style
    date_short: str = "M/d/yy"
    date_medium: str = "MMM d, y"
    date_long: str = "MMMM d, y"
    date_full: str = "EEEE, MMMM d, y"

    # Time patterns by style
    time_short: str = "h:mm a"
    time_medium: str = "h:mm:ss a"
    time_long: str = "h:mm:ss a z"
    time_full: str = "h:mm:ss a zzzz"

    # Combined datetime pattern
    datetime_pattern: str = "{date} {time}"

    # Month names
    months_wide: list[str] = field(default_factory=lambda: [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    months_abbreviated: list[str] = field(default_factory=lambda: [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])

    # Day names
    days_wide: list[str] = field(default_factory=lambda: [
        "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday"
    ])
    days_abbreviated: list[str] = field(default_factory=lambda: [
        "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"
    ])

    # AM/PM
    am: str = "AM"
    pm: str = "PM"

    # Era names
    era_abbreviated: list[str] = field(default_factory=lambda: ["BC", "AD"])
    era_wide: list[str] = field(default_factory=lambda: ["Before Christ", "Anno Domini"])


# Date patterns by locale
_DATE_PATTERNS: dict[str, DateTimePatterns] = {
    "en": DateTimePatterns(),

    "en_GB": DateTimePatterns(
        date_short="dd/MM/y",
        date_medium="d MMM y",
        date_long="d MMMM y",
        date_full="EEEE, d MMMM y",
    ),

    "de": DateTimePatterns(
        date_short="dd.MM.yy",
        date_medium="dd.MM.y",
        date_long="d. MMMM y",
        date_full="EEEE, d. MMMM y",
        time_short="HH:mm",
        time_medium="HH:mm:ss",
        months_wide=[
            "Januar", "Februar", "März", "April", "Mai", "Juni",
            "Juli", "August", "September", "Oktober", "November", "Dezember"
        ],
        months_abbreviated=["Jan.", "Feb.", "März", "Apr.", "Mai", "Juni",
                           "Juli", "Aug.", "Sep.", "Okt.", "Nov.", "Dez."],
        days_wide=["Sonntag", "Montag", "Dienstag", "Mittwoch",
                   "Donnerstag", "Freitag", "Samstag"],
        days_abbreviated=["So.", "Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa."],
    ),

    "fr": DateTimePatterns(
        date_short="dd/MM/y",
        date_medium="d MMM y",
        date_long="d MMMM y",
        date_full="EEEE d MMMM y",
        time_short="HH:mm",
        time_medium="HH:mm:ss",
        months_wide=[
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ],
        months_abbreviated=["janv.", "févr.", "mars", "avr.", "mai", "juin",
                           "juil.", "août", "sept.", "oct.", "nov.", "déc."],
        days_wide=["dimanche", "lundi", "mardi", "mercredi",
                   "jeudi", "vendredi", "samedi"],
        days_abbreviated=["dim.", "lun.", "mar.", "mer.", "jeu.", "ven.", "sam."],
    ),

    "es": DateTimePatterns(
        date_short="d/M/yy",
        date_medium="d MMM y",
        date_long="d 'de' MMMM 'de' y",
        date_full="EEEE, d 'de' MMMM 'de' y",
        time_short="H:mm",
        time_medium="H:mm:ss",
        months_wide=[
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
        ],
        months_abbreviated=["ene.", "feb.", "mar.", "abr.", "may.", "jun.",
                           "jul.", "ago.", "sept.", "oct.", "nov.", "dic."],
        days_wide=["domingo", "lunes", "martes", "miércoles",
                   "jueves", "viernes", "sábado"],
        days_abbreviated=["dom.", "lun.", "mar.", "mié.", "jue.", "vie.", "sáb."],
    ),

    "it": DateTimePatterns(
        date_short="dd/MM/yy",
        date_medium="d MMM y",
        date_long="d MMMM y",
        date_full="EEEE d MMMM y",
        time_short="HH:mm",
        time_medium="HH:mm:ss",
        months_wide=[
            "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"
        ],
        months_abbreviated=["gen", "feb", "mar", "apr", "mag", "giu",
                           "lug", "ago", "set", "ott", "nov", "dic"],
        days_wide=["domenica", "lunedì", "martedì", "mercoledì",
                   "giovedì", "venerdì", "sabato"],
        days_abbreviated=["dom", "lun", "mar", "mer", "gio", "ven", "sab"],
    ),

    "pt": DateTimePatterns(
        date_short="dd/MM/y",
        date_medium="d 'de' MMM 'de' y",
        date_long="d 'de' MMMM 'de' y",
        date_full="EEEE, d 'de' MMMM 'de' y",
        time_short="HH:mm",
        time_medium="HH:mm:ss",
        months_wide=[
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
        ],
        months_abbreviated=["jan.", "fev.", "mar.", "abr.", "mai.", "jun.",
                           "jul.", "ago.", "set.", "out.", "nov.", "dez."],
        days_wide=["domingo", "segunda-feira", "terça-feira", "quarta-feira",
                   "quinta-feira", "sexta-feira", "sábado"],
        days_abbreviated=["dom.", "seg.", "ter.", "qua.", "qui.", "sex.", "sáb."],
    ),

    "ru": DateTimePatterns(
        date_short="dd.MM.y",
        date_medium="d MMM y 'г.'",
        date_long="d MMMM y 'г.'",
        date_full="EEEE, d MMMM y 'г.'",
        time_short="HH:mm",
        time_medium="HH:mm:ss",
        months_wide=[
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ],
        months_abbreviated=["янв.", "февр.", "мар.", "апр.", "мая", "июн.",
                           "июл.", "авг.", "сент.", "окт.", "нояб.", "дек."],
        days_wide=["воскресенье", "понедельник", "вторник", "среда",
                   "четверг", "пятница", "суббота"],
        days_abbreviated=["вс", "пн", "вт", "ср", "чт", "пт", "сб"],
    ),

    "ja": DateTimePatterns(
        date_short="y/MM/dd",
        date_medium="y年M月d日",
        date_long="y年M月d日",
        date_full="y年M月d日EEEE",
        time_short="H:mm",
        time_medium="H:mm:ss",
        time_long="H時mm分ss秒 z",
        months_wide=["1月", "2月", "3月", "4月", "5月", "6月",
                     "7月", "8月", "9月", "10月", "11月", "12月"],
        months_abbreviated=["1月", "2月", "3月", "4月", "5月", "6月",
                           "7月", "8月", "9月", "10月", "11月", "12月"],
        days_wide=["日曜日", "月曜日", "火曜日", "水曜日",
                   "木曜日", "金曜日", "土曜日"],
        days_abbreviated=["日", "月", "火", "水", "木", "金", "土"],
        am="午前",
        pm="午後",
    ),

    "ko": DateTimePatterns(
        date_short="yy. M. d.",
        date_medium="y년 M월 d일",
        date_long="y년 M월 d일",
        date_full="y년 M월 d일 EEEE",
        time_short="a h:mm",
        time_medium="a h:mm:ss",
        months_wide=["1월", "2월", "3월", "4월", "5월", "6월",
                     "7월", "8월", "9월", "10월", "11월", "12월"],
        months_abbreviated=["1월", "2월", "3월", "4월", "5월", "6월",
                           "7월", "8월", "9월", "10월", "11월", "12월"],
        days_wide=["일요일", "월요일", "화요일", "수요일",
                   "목요일", "금요일", "토요일"],
        days_abbreviated=["일", "월", "화", "수", "목", "금", "토"],
        am="오전",
        pm="오후",
    ),

    "zh": DateTimePatterns(
        date_short="y/M/d",
        date_medium="y年M月d日",
        date_long="y年M月d日",
        date_full="y年M月d日EEEE",
        time_short="HH:mm",
        time_medium="HH:mm:ss",
        months_wide=["一月", "二月", "三月", "四月", "五月", "六月",
                     "七月", "八月", "九月", "十月", "十一月", "十二月"],
        months_abbreviated=["1月", "2月", "3月", "4月", "5月", "6月",
                           "7月", "8月", "9月", "10月", "11月", "12月"],
        days_wide=["星期日", "星期一", "星期二", "星期三",
                   "星期四", "星期五", "星期六"],
        days_abbreviated=["周日", "周一", "周二", "周三", "周四", "周五", "周六"],
        am="上午",
        pm="下午",
    ),

    "ar": DateTimePatterns(
        date_short="d/M/y",
        date_medium="dd/MM/y",
        date_long="d MMMM y",
        date_full="EEEE، d MMMM y",
        time_short="h:mm a",
        time_medium="h:mm:ss a",
        months_wide=[
            "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
            "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
        ],
        months_abbreviated=[
            "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
            "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
        ],
        days_wide=["الأحد", "الاثنين", "الثلاثاء", "الأربعاء",
                   "الخميس", "الجمعة", "السبت"],
        days_abbreviated=["أحد", "إثنين", "ثلاثاء", "أربعاء", "خميس", "جمعة", "سبت"],
        am="ص",
        pm="م",
    ),

    "he": DateTimePatterns(
        date_short="d.M.y",
        date_medium="d בMMM y",
        date_long="d בMMMM y",
        date_full="EEEE, d בMMMM y",
        time_short="H:mm",
        time_medium="H:mm:ss",
        months_wide=[
            "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
            "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"
        ],
        months_abbreviated=[
            "ינו׳", "פבר׳", "מרץ", "אפר׳", "מאי", "יוני",
            "יולי", "אוג׳", "ספט׳", "אוק׳", "נוב׳", "דצמ׳"
        ],
        days_wide=["יום ראשון", "יום שני", "יום שלישי", "יום רביעי",
                   "יום חמישי", "יום שישי", "יום שבת"],
        days_abbreviated=["יום א׳", "יום ב׳", "יום ג׳", "יום ד׳", "יום ה׳", "יום ו׳", "שבת"],
    ),
}


def get_date_patterns(locale: LocaleInfo) -> DateTimePatterns:
    """Get date/time patterns for a locale.

    Args:
        locale: Target locale

    Returns:
        DateTimePatterns for the locale
    """
    key = f"{locale.language}_{locale.region}" if locale.region else locale.language
    if key in _DATE_PATTERNS:
        return _DATE_PATTERNS[key]

    if locale.language in _DATE_PATTERNS:
        return _DATE_PATTERNS[locale.language]

    return _DATE_PATTERNS["en"]


# ==============================================================================
# Relative Time Data
# ==============================================================================

@dataclass
class RelativeTimeUnit:
    """Relative time unit patterns."""
    past_one: str
    past_other: str
    future_one: str
    future_other: str


@dataclass
class RelativeTimeData:
    """Locale-specific relative time patterns."""
    now: str = "now"
    second: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} second ago", "{0} seconds ago", "in {0} second", "in {0} seconds"
    ))
    minute: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} minute ago", "{0} minutes ago", "in {0} minute", "in {0} minutes"
    ))
    hour: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} hour ago", "{0} hours ago", "in {0} hour", "in {0} hours"
    ))
    day: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} day ago", "{0} days ago", "in {0} day", "in {0} days"
    ))
    week: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} week ago", "{0} weeks ago", "in {0} week", "in {0} weeks"
    ))
    month: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} month ago", "{0} months ago", "in {0} month", "in {0} months"
    ))
    year: RelativeTimeUnit = field(default_factory=lambda: RelativeTimeUnit(
        "{0} year ago", "{0} years ago", "in {0} year", "in {0} years"
    ))
    yesterday: str = "yesterday"
    today: str = "today"
    tomorrow: str = "tomorrow"


_RELATIVE_TIME: dict[str, RelativeTimeData] = {
    "en": RelativeTimeData(),

    "ko": RelativeTimeData(
        now="지금",
        second=RelativeTimeUnit("{0}초 전", "{0}초 전", "{0}초 후", "{0}초 후"),
        minute=RelativeTimeUnit("{0}분 전", "{0}분 전", "{0}분 후", "{0}분 후"),
        hour=RelativeTimeUnit("{0}시간 전", "{0}시간 전", "{0}시간 후", "{0}시간 후"),
        day=RelativeTimeUnit("{0}일 전", "{0}일 전", "{0}일 후", "{0}일 후"),
        week=RelativeTimeUnit("{0}주 전", "{0}주 전", "{0}주 후", "{0}주 후"),
        month=RelativeTimeUnit("{0}개월 전", "{0}개월 전", "{0}개월 후", "{0}개월 후"),
        year=RelativeTimeUnit("{0}년 전", "{0}년 전", "{0}년 후", "{0}년 후"),
        yesterday="어제",
        today="오늘",
        tomorrow="내일",
    ),

    "ja": RelativeTimeData(
        now="今",
        second=RelativeTimeUnit("{0}秒前", "{0}秒前", "{0}秒後", "{0}秒後"),
        minute=RelativeTimeUnit("{0}分前", "{0}分前", "{0}分後", "{0}分後"),
        hour=RelativeTimeUnit("{0}時間前", "{0}時間前", "{0}時間後", "{0}時間後"),
        day=RelativeTimeUnit("{0}日前", "{0}日前", "{0}日後", "{0}日後"),
        week=RelativeTimeUnit("{0}週間前", "{0}週間前", "{0}週間後", "{0}週間後"),
        month=RelativeTimeUnit("{0}か月前", "{0}か月前", "{0}か月後", "{0}か月後"),
        year=RelativeTimeUnit("{0}年前", "{0}年前", "{0}年後", "{0}年後"),
        yesterday="昨日",
        today="今日",
        tomorrow="明日",
    ),

    "zh": RelativeTimeData(
        now="现在",
        second=RelativeTimeUnit("{0}秒钟前", "{0}秒钟前", "{0}秒钟后", "{0}秒钟后"),
        minute=RelativeTimeUnit("{0}分钟前", "{0}分钟前", "{0}分钟后", "{0}分钟后"),
        hour=RelativeTimeUnit("{0}小时前", "{0}小时前", "{0}小时后", "{0}小时后"),
        day=RelativeTimeUnit("{0}天前", "{0}天前", "{0}天后", "{0}天后"),
        week=RelativeTimeUnit("{0}周前", "{0}周前", "{0}周后", "{0}周后"),
        month=RelativeTimeUnit("{0}个月前", "{0}个月前", "{0}个月后", "{0}个月后"),
        year=RelativeTimeUnit("{0}年前", "{0}年前", "{0}年后", "{0}年后"),
        yesterday="昨天",
        today="今天",
        tomorrow="明天",
    ),

    "de": RelativeTimeData(
        now="jetzt",
        second=RelativeTimeUnit("vor {0} Sekunde", "vor {0} Sekunden", "in {0} Sekunde", "in {0} Sekunden"),
        minute=RelativeTimeUnit("vor {0} Minute", "vor {0} Minuten", "in {0} Minute", "in {0} Minuten"),
        hour=RelativeTimeUnit("vor {0} Stunde", "vor {0} Stunden", "in {0} Stunde", "in {0} Stunden"),
        day=RelativeTimeUnit("vor {0} Tag", "vor {0} Tagen", "in {0} Tag", "in {0} Tagen"),
        week=RelativeTimeUnit("vor {0} Woche", "vor {0} Wochen", "in {0} Woche", "in {0} Wochen"),
        month=RelativeTimeUnit("vor {0} Monat", "vor {0} Monaten", "in {0} Monat", "in {0} Monaten"),
        year=RelativeTimeUnit("vor {0} Jahr", "vor {0} Jahren", "in {0} Jahr", "in {0} Jahren"),
        yesterday="gestern",
        today="heute",
        tomorrow="morgen",
    ),

    "fr": RelativeTimeData(
        now="maintenant",
        second=RelativeTimeUnit("il y a {0} seconde", "il y a {0} secondes", "dans {0} seconde", "dans {0} secondes"),
        minute=RelativeTimeUnit("il y a {0} minute", "il y a {0} minutes", "dans {0} minute", "dans {0} minutes"),
        hour=RelativeTimeUnit("il y a {0} heure", "il y a {0} heures", "dans {0} heure", "dans {0} heures"),
        day=RelativeTimeUnit("il y a {0} jour", "il y a {0} jours", "dans {0} jour", "dans {0} jours"),
        week=RelativeTimeUnit("il y a {0} semaine", "il y a {0} semaines", "dans {0} semaine", "dans {0} semaines"),
        month=RelativeTimeUnit("il y a {0} mois", "il y a {0} mois", "dans {0} mois", "dans {0} mois"),
        year=RelativeTimeUnit("il y a {0} an", "il y a {0} ans", "dans {0} an", "dans {0} ans"),
        yesterday="hier",
        today="aujourd'hui",
        tomorrow="demain",
    ),

    "es": RelativeTimeData(
        now="ahora",
        second=RelativeTimeUnit("hace {0} segundo", "hace {0} segundos", "dentro de {0} segundo", "dentro de {0} segundos"),
        minute=RelativeTimeUnit("hace {0} minuto", "hace {0} minutos", "dentro de {0} minuto", "dentro de {0} minutos"),
        hour=RelativeTimeUnit("hace {0} hora", "hace {0} horas", "dentro de {0} hora", "dentro de {0} horas"),
        day=RelativeTimeUnit("hace {0} día", "hace {0} días", "dentro de {0} día", "dentro de {0} días"),
        week=RelativeTimeUnit("hace {0} semana", "hace {0} semanas", "dentro de {0} semana", "dentro de {0} semanas"),
        month=RelativeTimeUnit("hace {0} mes", "hace {0} meses", "dentro de {0} mes", "dentro de {0} meses"),
        year=RelativeTimeUnit("hace {0} año", "hace {0} años", "dentro de {0} año", "dentro de {0} años"),
        yesterday="ayer",
        today="hoy",
        tomorrow="mañana",
    ),

    "ru": RelativeTimeData(
        now="сейчас",
        second=RelativeTimeUnit("{0} секунду назад", "{0} секунд назад", "через {0} секунду", "через {0} секунд"),
        minute=RelativeTimeUnit("{0} минуту назад", "{0} минут назад", "через {0} минуту", "через {0} минут"),
        hour=RelativeTimeUnit("{0} час назад", "{0} часов назад", "через {0} час", "через {0} часов"),
        day=RelativeTimeUnit("{0} день назад", "{0} дней назад", "через {0} день", "через {0} дней"),
        week=RelativeTimeUnit("{0} неделю назад", "{0} недель назад", "через {0} неделю", "через {0} недель"),
        month=RelativeTimeUnit("{0} месяц назад", "{0} месяцев назад", "через {0} месяц", "через {0} месяцев"),
        year=RelativeTimeUnit("{0} год назад", "{0} лет назад", "через {0} год", "через {0} лет"),
        yesterday="вчера",
        today="сегодня",
        tomorrow="завтра",
    ),

    "ar": RelativeTimeData(
        now="الآن",
        second=RelativeTimeUnit("قبل {0} ثانية", "قبل {0} ثوانٍ", "خلال {0} ثانية", "خلال {0} ثوانٍ"),
        minute=RelativeTimeUnit("قبل {0} دقيقة", "قبل {0} دقائق", "خلال {0} دقيقة", "خلال {0} دقائق"),
        hour=RelativeTimeUnit("قبل {0} ساعة", "قبل {0} ساعات", "خلال {0} ساعة", "خلال {0} ساعات"),
        day=RelativeTimeUnit("قبل {0} يوم", "قبل {0} أيام", "خلال {0} يوم", "خلال {0} أيام"),
        week=RelativeTimeUnit("قبل {0} أسبوع", "قبل {0} أسابيع", "خلال {0} أسبوع", "خلال {0} أسابيع"),
        month=RelativeTimeUnit("قبل {0} شهر", "قبل {0} أشهر", "خلال {0} شهر", "خلال {0} أشهر"),
        year=RelativeTimeUnit("قبل {0} سنة", "قبل {0} سنوات", "خلال {0} سنة", "خلال {0} سنوات"),
        yesterday="أمس",
        today="اليوم",
        tomorrow="غدًا",
    ),
}


def get_relative_time_data(locale: LocaleInfo) -> RelativeTimeData:
    """Get relative time patterns for a locale."""
    key = f"{locale.language}_{locale.region}" if locale.region else locale.language
    if key in _RELATIVE_TIME:
        return _RELATIVE_TIME[key]

    if locale.language in _RELATIVE_TIME:
        return _RELATIVE_TIME[locale.language]

    return _RELATIVE_TIME["en"]


# ==============================================================================
# Number Formatter Implementation
# ==============================================================================

class LocaleNumberFormatter(BaseNumberFormatter):
    """Locale-aware number formatter.

    Supports multiple formatting styles and locale-specific symbols.

    Example:
        formatter = LocaleNumberFormatter()

        # Decimal formatting
        formatter.format(1234567.89, LocaleInfo.parse("de"))
        # -> FormattedNumber(formatted="1.234.567,89")

        # Currency formatting
        formatter.format(1234.56, LocaleInfo.parse("en"), NumberStyle.CURRENCY, currency="USD")
        # -> FormattedNumber(formatted="$1,234.56")

        # Percent formatting
        formatter.format(0.1234, LocaleInfo.parse("en"), NumberStyle.PERCENT)
        # -> FormattedNumber(formatted="12.34%")
    """

    def __init__(self, default_precision: int = 2) -> None:
        """Initialize formatter.

        Args:
            default_precision: Default decimal precision
        """
        self.default_precision = default_precision

    def format(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        style: NumberStyle = NumberStyle.DECIMAL,
        **options: Any,
    ) -> FormattedNumber:
        """Format a number according to locale rules.

        Args:
            value: Number to format
            locale: Target locale
            style: Formatting style
            **options: Additional options:
                - precision: Decimal places
                - currency: Currency code (for CURRENCY style)
                - use_grouping: Whether to use grouping separators
                - min_fraction_digits: Minimum fraction digits
                - max_fraction_digits: Maximum fraction digits
                - compact_display: "short" or "long" (for COMPACT style)

        Returns:
            Formatted number result
        """
        symbols = get_number_symbols(locale)
        direction = locale.direction

        precision = options.get("precision", self.default_precision)
        use_grouping = options.get("use_grouping", True)

        if style == NumberStyle.CURRENCY:
            return self._format_currency(value, locale, symbols, direction, **options)
        elif style == NumberStyle.PERCENT:
            return self._format_percent(value, locale, symbols, direction, precision)
        elif style == NumberStyle.SCIENTIFIC:
            return self._format_scientific(value, locale, symbols, direction, precision)
        elif style == NumberStyle.COMPACT:
            return self._format_compact(value, locale, symbols, direction, **options)
        elif style == NumberStyle.ORDINAL:
            return self._format_ordinal(value, locale)
        else:
            return self._format_decimal(value, locale, symbols, direction, precision, use_grouping)

    def _format_decimal(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        symbols: NumberSymbols,
        direction: TextDirection,
        precision: int,
        use_grouping: bool,
    ) -> FormattedNumber:
        """Format as decimal number."""
        # Handle special values
        if math.isnan(float(value)):
            return FormattedNumber(value=value, formatted=symbols.nan, direction=direction)
        if math.isinf(float(value)):
            sign = "" if value > 0 else symbols.minus
            return FormattedNumber(value=value, formatted=f"{sign}{symbols.infinity}", direction=direction)

        # Round and format
        if isinstance(value, int):
            int_part = str(abs(value))
            frac_part = ""
        else:
            # Round to precision
            rounded = round(float(value), precision)
            parts = f"{abs(rounded):.{precision}f}".split(".")
            int_part = parts[0]
            frac_part = parts[1] if len(parts) > 1 else ""

        # Apply grouping
        if use_grouping and len(int_part) > 3:
            int_part = self._apply_grouping(int_part, symbols.group, locale)

        # Build formatted string
        if frac_part:
            formatted = f"{int_part}{symbols.decimal}{frac_part}"
        else:
            formatted = int_part

        # Add sign
        if value < 0:
            formatted = f"{symbols.minus}{formatted}"

        return FormattedNumber(
            value=value,
            formatted=formatted,
            direction=direction,
            parts={"integer": int_part, "decimal": frac_part},
        )

    def _apply_grouping(self, int_part: str, group_sep: str, locale: LocaleInfo) -> str:
        """Apply grouping separators to integer part."""
        # Indian numbering system uses 2,2,3 grouping
        if locale.language == "hi" or (locale.language == "en" and locale.region == "IN"):
            # First group of 3, then groups of 2
            if len(int_part) <= 3:
                return int_part
            result = int_part[-3:]
            remaining = int_part[:-3]
            while remaining:
                result = remaining[-2:] + group_sep + result
                remaining = remaining[:-2]
            return result

        # Standard 3-digit grouping
        groups = []
        while len(int_part) > 3:
            groups.insert(0, int_part[-3:])
            int_part = int_part[:-3]
        groups.insert(0, int_part)
        return group_sep.join(groups)

    def _format_currency(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        symbols: NumberSymbols,
        direction: TextDirection,
        **options: Any,
    ) -> FormattedNumber:
        """Format as currency."""
        currency_code = options.get("currency", "USD")
        currency_info = get_currency_info(currency_code)
        precision = currency_info.decimal_digits

        # Format the number part
        decimal_result = self._format_decimal(
            value, locale, symbols, direction, precision, True
        )

        # Get currency symbol
        use_narrow = options.get("narrow", False)
        symbol = currency_info.narrow_symbol if use_narrow else currency_info.symbol

        # Determine symbol position based on locale
        # Most European locales put symbol after, English-speaking before
        symbol_after = locale.language in ("de", "fr", "es", "it", "pt", "nl", "pl", "cs", "ru", "da", "sv", "no", "fi")

        if symbol_after:
            formatted = f"{decimal_result.formatted} {symbol}"
        else:
            formatted = f"{symbol}{decimal_result.formatted}"

        return FormattedNumber(
            value=value,
            formatted=formatted,
            direction=direction,
            parts={"symbol": symbol, "amount": decimal_result.formatted},
        )

    def _format_percent(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        symbols: NumberSymbols,
        direction: TextDirection,
        precision: int,
    ) -> FormattedNumber:
        """Format as percentage."""
        percent_value = float(value) * 100
        decimal_result = self._format_decimal(
            percent_value, locale, symbols, direction, precision, False
        )

        # Some locales put space before percent sign
        space = " " if locale.language in ("fr", "de", "ru") else ""
        formatted = f"{decimal_result.formatted}{space}{symbols.percent}"

        return FormattedNumber(
            value=value,
            formatted=formatted,
            direction=direction,
        )

    def _format_scientific(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        symbols: NumberSymbols,
        direction: TextDirection,
        precision: int,
    ) -> FormattedNumber:
        """Format in scientific notation."""
        formatted = f"{float(value):.{precision}e}"
        # Replace E with locale symbol
        formatted = formatted.replace("e", symbols.exponential)
        # Replace decimal point
        formatted = formatted.replace(".", symbols.decimal)

        return FormattedNumber(
            value=value,
            formatted=formatted,
            direction=direction,
        )

    def _format_compact(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        symbols: NumberSymbols,
        direction: TextDirection,
        **options: Any,
    ) -> FormattedNumber:
        """Format in compact notation (1K, 1M, etc.)."""
        abs_value = abs(float(value))

        # Define suffixes by locale
        suffixes_en = [
            (1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")
        ]
        suffixes_ko = [
            (1e12, "조"), (1e8, "억"), (1e4, "만")
        ]
        suffixes_ja = [
            (1e12, "兆"), (1e8, "億"), (1e4, "万")
        ]
        suffixes_zh = [
            (1e12, "万亿"), (1e8, "亿"), (1e4, "万")
        ]

        if locale.language == "ko":
            suffixes = suffixes_ko
        elif locale.language == "ja":
            suffixes = suffixes_ja
        elif locale.language == "zh":
            suffixes = suffixes_zh
        else:
            suffixes = suffixes_en

        for threshold, suffix in suffixes:
            if abs_value >= threshold:
                compact_value = float(value) / threshold
                formatted = f"{compact_value:.1f}".rstrip("0").rstrip(".")
                formatted = formatted.replace(".", symbols.decimal)
                if float(value) < 0:
                    formatted = f"{symbols.minus}{formatted}"
                return FormattedNumber(
                    value=value,
                    formatted=f"{formatted}{suffix}",
                    direction=direction,
                )

        # No compaction needed
        return self._format_decimal(value, locale, symbols, direction, 0, True)

    def _format_ordinal(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
    ) -> FormattedNumber:
        """Format as ordinal number."""
        n = int(value)

        if locale.language == "en":
            # English ordinals: 1st, 2nd, 3rd, 4th...
            if 11 <= n % 100 <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            formatted = f"{n}{suffix}"

        elif locale.language == "ko":
            formatted = f"제{n}"

        elif locale.language == "ja":
            formatted = f"第{n}"

        elif locale.language == "zh":
            formatted = f"第{n}"

        elif locale.language in ("de", "nl", "da", "sv", "no"):
            # German-style: add period
            formatted = f"{n}."

        elif locale.language in ("fr", "es", "it", "pt"):
            # Romance languages with gender
            if n == 1:
                formatted = "1ᵉʳ" if locale.language == "fr" else f"{n}º"
            else:
                formatted = f"{n}ᵉ" if locale.language == "fr" else f"{n}º"

        elif locale.language == "ru":
            formatted = f"{n}-й"

        else:
            formatted = str(n)

        return FormattedNumber(
            value=value,
            formatted=formatted,
            direction=locale.direction,
        )

    def parse(
        self,
        text: str,
        locale: LocaleInfo,
        style: NumberStyle = NumberStyle.DECIMAL,
    ) -> float | int | Decimal | None:
        """Parse a localized number string."""
        symbols = get_number_symbols(locale)

        # Clean the text
        cleaned = text.strip()

        # Remove grouping separators
        cleaned = cleaned.replace(symbols.group, "")

        # Normalize decimal separator
        cleaned = cleaned.replace(symbols.decimal, ".")

        # Handle percent
        if style == NumberStyle.PERCENT:
            cleaned = cleaned.replace(symbols.percent, "").strip()
            try:
                return float(cleaned) / 100
            except ValueError:
                return None

        # Handle currency symbols
        if style == NumberStyle.CURRENCY:
            # Remove common currency symbols
            for symbol in ["$", "€", "£", "¥", "₩", "₹"]:
                cleaned = cleaned.replace(symbol, "")
            cleaned = cleaned.strip()

        try:
            # Try integer first
            if "." not in cleaned:
                return int(cleaned)
            return float(cleaned)
        except ValueError:
            return None


# ==============================================================================
# Date Formatter Implementation
# ==============================================================================

class LocaleDateFormatter(BaseDateFormatter):
    """Locale-aware date/time formatter.

    Supports multiple date styles and relative time formatting.

    Example:
        formatter = LocaleDateFormatter()

        # Date formatting
        formatter.format_date(date.today(), LocaleInfo.parse("ko"), DateStyle.LONG)
        # -> FormattedDate(formatted="2024년 12월 28일")

        # Relative time
        formatter.format_relative(datetime.now() - timedelta(days=2), locale=LocaleInfo.parse("en"))
        # -> FormattedDate(formatted="2 days ago")
    """

    def format_date(
        self,
        value: datetime | date,
        locale: LocaleInfo,
        style: DateStyle = DateStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a date according to locale rules."""
        if style == DateStyle.ISO:
            if isinstance(value, datetime):
                formatted = value.strftime("%Y-%m-%d")
            else:
                formatted = value.strftime("%Y-%m-%d")
            return FormattedDate(value=value, formatted=formatted, direction=locale.direction)

        if style == DateStyle.RELATIVE:
            return self.format_relative(value, locale=locale)

        patterns = get_date_patterns(locale)
        direction = locale.direction

        if style == DateStyle.SHORT:
            formatted = self._apply_pattern(value, patterns.date_short, patterns)
        elif style == DateStyle.MEDIUM:
            formatted = self._apply_pattern(value, patterns.date_medium, patterns)
        elif style == DateStyle.LONG:
            formatted = self._apply_pattern(value, patterns.date_long, patterns)
        elif style == DateStyle.FULL:
            formatted = self._apply_pattern(value, patterns.date_full, patterns)
        else:
            formatted = self._apply_pattern(value, patterns.date_medium, patterns)

        return FormattedDate(value=value, formatted=formatted, direction=direction)

    def format_time(
        self,
        value: datetime | time,
        locale: LocaleInfo,
        style: TimeStyle = TimeStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a time according to locale rules."""
        patterns = get_date_patterns(locale)
        direction = locale.direction

        if isinstance(value, time):
            value = datetime.combine(date.today(), value)

        if style == TimeStyle.SHORT:
            formatted = self._apply_pattern(value, patterns.time_short, patterns)
        elif style == TimeStyle.MEDIUM:
            formatted = self._apply_pattern(value, patterns.time_medium, patterns)
        elif style == TimeStyle.LONG:
            formatted = self._apply_pattern(value, patterns.time_long, patterns)
        elif style == TimeStyle.FULL:
            formatted = self._apply_pattern(value, patterns.time_full, patterns)
        else:
            formatted = self._apply_pattern(value, patterns.time_medium, patterns)

        return FormattedDate(value=value, formatted=formatted, direction=direction)

    def _apply_pattern(
        self,
        value: datetime | date,
        pattern: str,
        patterns: DateTimePatterns,
    ) -> str:
        """Apply a date/time pattern to a value."""
        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime.combine(value, time.min)

        result = pattern

        # Year
        result = result.replace("yyyy", str(value.year))
        result = result.replace("yyy", str(value.year))
        result = result.replace("yy", str(value.year)[-2:])
        result = result.replace("y", str(value.year))

        # Month
        month_idx = value.month - 1
        result = result.replace("MMMM", patterns.months_wide[month_idx])
        result = result.replace("MMM", patterns.months_abbreviated[month_idx])
        result = result.replace("MM", f"{value.month:02d}")
        result = result.replace("M", str(value.month))

        # Day
        result = result.replace("dd", f"{value.day:02d}")
        result = result.replace("d", str(value.day))

        # Day of week
        dow_idx = value.weekday()  # 0=Monday in Python
        dow_idx = (dow_idx + 1) % 7  # Convert to 0=Sunday
        result = result.replace("EEEE", patterns.days_wide[dow_idx])
        result = result.replace("EEE", patterns.days_abbreviated[dow_idx])
        result = result.replace("E", patterns.days_abbreviated[dow_idx])

        # Hour (24-hour)
        result = result.replace("HH", f"{value.hour:02d}")
        result = result.replace("H", str(value.hour))

        # Hour (12-hour)
        hour12 = value.hour % 12 or 12
        result = result.replace("hh", f"{hour12:02d}")
        result = result.replace("h", str(hour12))

        # Minute
        result = result.replace("mm", f"{value.minute:02d}")
        # Don't replace single 'm' as it might be part of AM/PM

        # Second
        result = result.replace("ss", f"{value.second:02d}")
        result = result.replace("s", str(value.second))

        # AM/PM
        am_pm = patterns.am if value.hour < 12 else patterns.pm
        result = result.replace("a", am_pm)

        # Handle quoted literals (e.g., 'de' in Spanish patterns)
        result = re.sub(r"'([^']*)'", r"\1", result)

        return result

    def format_relative(
        self,
        value: datetime | date,
        reference: datetime | date | None = None,
        locale: LocaleInfo | None = None,
    ) -> FormattedDate:
        """Format a relative date (e.g., "2 days ago")."""
        if locale is None:
            locale = LocaleInfo.parse("en")

        if reference is None:
            reference = datetime.now()

        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime.combine(value, time.min)
        if isinstance(reference, date) and not isinstance(reference, datetime):
            reference = datetime.combine(reference, time.min)

        delta = value - reference
        rel_data = get_relative_time_data(locale)

        # Check for special cases
        days = delta.days
        if days == 0 and abs(delta.total_seconds()) < 60:
            return FormattedDate(value=value, formatted=rel_data.now, direction=locale.direction)
        if days == 0:
            pass  # Handle hours/minutes below
        elif days == 1:
            return FormattedDate(value=value, formatted=rel_data.tomorrow, direction=locale.direction)
        elif days == -1:
            return FormattedDate(value=value, formatted=rel_data.yesterday, direction=locale.direction)

        # Calculate appropriate unit
        total_seconds = abs(delta.total_seconds())
        is_future = delta.total_seconds() > 0

        if total_seconds < 60:
            n = int(total_seconds)
            unit = rel_data.second
        elif total_seconds < 3600:
            n = int(total_seconds / 60)
            unit = rel_data.minute
        elif total_seconds < 86400:
            n = int(total_seconds / 3600)
            unit = rel_data.hour
        elif total_seconds < 604800:
            n = abs(days)
            unit = rel_data.day
        elif total_seconds < 2592000:  # ~30 days
            n = abs(days) // 7
            unit = rel_data.week
        elif total_seconds < 31536000:  # ~365 days
            n = abs(days) // 30
            unit = rel_data.month
        else:
            n = abs(days) // 365
            unit = rel_data.year

        # Select pattern based on plurality and direction
        if is_future:
            pattern = unit.future_one if n == 1 else unit.future_other
        else:
            pattern = unit.past_one if n == 1 else unit.past_other

        formatted = pattern.format(n)

        return FormattedDate(value=value, formatted=formatted, direction=locale.direction)


# ==============================================================================
# Convenience Functions
# ==============================================================================

_number_formatter = LocaleNumberFormatter()
_date_formatter = LocaleDateFormatter()


def format_number(
    value: float | int | Decimal,
    locale: str | LocaleInfo,
    style: NumberStyle = NumberStyle.DECIMAL,
    **options: Any,
) -> str:
    """Format a number for a locale.

    Args:
        value: Number to format
        locale: Target locale (string or LocaleInfo)
        style: Formatting style
        **options: Additional formatting options

    Returns:
        Formatted number string

    Example:
        format_number(1234567.89, "de")  # "1.234.567,89"
        format_number(0.15, "en", NumberStyle.PERCENT)  # "15.00%"
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)
    return _number_formatter.format(value, locale, style, **options).formatted


def format_currency(
    value: float | int | Decimal,
    currency: str,
    locale: str | LocaleInfo,
    **options: Any,
) -> str:
    """Format a currency value.

    Args:
        value: Amount
        currency: ISO 4217 currency code
        locale: Target locale

    Returns:
        Formatted currency string

    Example:
        format_currency(1234.56, "USD", "en")  # "$1,234.56"
        format_currency(1234.56, "EUR", "de")  # "1.234,56 €"
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)
    return _number_formatter.format(
        value, locale, NumberStyle.CURRENCY, currency=currency, **options
    ).formatted


def format_date(
    value: datetime | date,
    locale: str | LocaleInfo,
    style: DateStyle = DateStyle.MEDIUM,
    **options: Any,
) -> str:
    """Format a date for a locale.

    Args:
        value: Date to format
        locale: Target locale
        style: Formatting style

    Returns:
        Formatted date string

    Example:
        format_date(date.today(), "ko", DateStyle.LONG)  # "2024년 12월 28일"
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)
    return _date_formatter.format_date(value, locale, style, **options).formatted


def format_time(
    value: datetime | time,
    locale: str | LocaleInfo,
    style: TimeStyle = TimeStyle.MEDIUM,
    **options: Any,
) -> str:
    """Format a time for a locale.

    Args:
        value: Time to format
        locale: Target locale
        style: Formatting style

    Returns:
        Formatted time string
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)
    return _date_formatter.format_time(value, locale, style, **options).formatted


def format_relative_time(
    value: datetime | date,
    reference: datetime | date | None = None,
    locale: str | LocaleInfo = "en",
) -> str:
    """Format a relative time.

    Args:
        value: Date/time to format
        reference: Reference date (default: now)
        locale: Target locale

    Returns:
        Relative time string (e.g., "2 days ago")

    Example:
        format_relative_time(datetime.now() - timedelta(hours=3), locale="en")
        # -> "3 hours ago"
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)
    return _date_formatter.format_relative(value, reference, locale).formatted
