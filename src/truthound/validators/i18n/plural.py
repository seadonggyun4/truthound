"""CLDR Plural Rules Implementation.

This module provides comprehensive CLDR-compliant plural rule handling,
supporting cardinal and ordinal pluralization for 100+ languages.

Features:
- Full CLDR plural category support (zero, one, two, few, many, other)
- Language-specific plural rules
- Ordinal number support (1st, 2nd, 3rd...)
- Extensible rule registration

Usage:
    from truthound.validators.i18n.plural import (
        CLDRPluralRules,
        PluralCategory,
        pluralize,
    )

    # Get plural category
    rules = CLDRPluralRules()
    category = rules.get_category(5, LocaleInfo.parse("ru"))
    # -> PluralCategory.MANY

    # Pluralize a message
    message = pluralize(
        count=3,
        forms={
            PluralCategory.ONE: "{count} file",
            PluralCategory.OTHER: "{count} files",
        },
        locale="en",
    )
    # -> "3 files"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from truthound.validators.i18n.protocols import (
    BasePluralRuleProvider,
    LocaleInfo,
    PluralCategory,
)


# Type for plural rule function
PluralRuleFunc = Callable[[float | int], PluralCategory]


@dataclass
class PluralOperands:
    """CLDR plural operands for a number.

    See: https://unicode.org/reports/tr35/tr35-numbers.html#Operands

    Attributes:
        n: Absolute value of the source number
        i: Integer digits of n
        v: Number of visible fraction digits with trailing zeros
        w: Number of visible fraction digits without trailing zeros
        f: Visible fraction digits with trailing zeros
        t: Visible fraction digits without trailing zeros
        c: Compact decimal exponent value (if using compact notation)
    """
    n: float  # absolute value
    i: int    # integer digits
    v: int    # visible fraction digits (with trailing zeros)
    w: int    # visible fraction digits (without trailing zeros)
    f: int    # fraction digits as integer (with trailing zeros)
    t: int    # fraction digits as integer (without trailing zeros)
    c: int = 0  # compact exponent

    @classmethod
    def from_number(cls, n: float | int, fraction_digits: int | None = None) -> "PluralOperands":
        """Create operands from a number.

        Args:
            n: The number
            fraction_digits: Explicit fraction digits (for formatted numbers)

        Returns:
            PluralOperands instance
        """
        abs_n = abs(n)
        i = int(abs_n)

        if isinstance(n, int):
            return cls(n=float(abs_n), i=i, v=0, w=0, f=0, t=0)

        # Get fractional part
        str_n = str(abs_n)
        if "." in str_n:
            fraction_str = str_n.split(".")[1]
            v = len(fraction_str) if fraction_digits is None else fraction_digits
            f = int(fraction_str) if fraction_str else 0
            # Remove trailing zeros for t and w
            t_str = fraction_str.rstrip("0")
            w = len(t_str)
            t = int(t_str) if t_str else 0
        else:
            v = w = f = t = 0

        return cls(n=float(abs_n), i=i, v=v, w=w, f=f, t=t)


class CLDRPluralRules(BasePluralRuleProvider):
    """CLDR-compliant plural rule provider.

    Implements plural rules for 100+ languages based on Unicode CLDR.

    Example:
        rules = CLDRPluralRules()

        # Cardinal plurals
        rules.get_category(1, LocaleInfo.parse("en"))  # ONE
        rules.get_category(2, LocaleInfo.parse("en"))  # OTHER
        rules.get_category(1, LocaleInfo.parse("ru"))  # ONE
        rules.get_category(2, LocaleInfo.parse("ru"))  # FEW
        rules.get_category(5, LocaleInfo.parse("ru"))  # MANY

        # Ordinal plurals
        rules.get_category(1, LocaleInfo.parse("en"), ordinal=True)  # ONE (1st)
        rules.get_category(2, LocaleInfo.parse("en"), ordinal=True)  # TWO (2nd)
        rules.get_category(3, LocaleInfo.parse("en"), ordinal=True)  # FEW (3rd)
        rules.get_category(4, LocaleInfo.parse("en"), ordinal=True)  # OTHER (4th)
    """

    def __init__(self) -> None:
        self._cardinal_rules: dict[str, PluralRuleFunc] = {}
        self._ordinal_rules: dict[str, PluralRuleFunc] = {}
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default CLDR plural rules."""
        # ==========================================
        # CARDINAL RULES
        # ==========================================

        # English, German, Dutch, Italian, Portuguese, Spanish
        # One: i = 1 and v = 0
        def english_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            return PluralCategory.OTHER

        for lang in ["en", "de", "nl", "it", "pt", "es", "ca", "gl", "da", "no", "nb", "nn", "sv", "fi", "et", "hu", "tr", "id", "ms", "tl"]:
            self._cardinal_rules[lang] = english_cardinal

        # French, Brazilian Portuguese
        # One: i = 0,1
        def french_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            if op.i in (0, 1):
                return PluralCategory.ONE
            return PluralCategory.OTHER

        self._cardinal_rules["fr"] = french_cardinal
        self._cardinal_rules["pt_BR"] = french_cardinal

        # Russian, Ukrainian, Serbian, Croatian, Bosnian, Polish
        # One: v = 0 and i % 10 = 1 and i % 100 != 11
        # Few: v = 0 and i % 10 = 2..4 and i % 100 != 12..14
        # Many: v = 0 and i % 10 = 0 or v = 0 and i % 10 = 5..9 or v = 0 and i % 100 = 11..14
        def slavic_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            i10 = op.i % 10
            i100 = op.i % 100

            if op.v == 0 and i10 == 1 and i100 != 11:
                return PluralCategory.ONE
            if op.v == 0 and 2 <= i10 <= 4 and not (12 <= i100 <= 14):
                return PluralCategory.FEW
            if op.v == 0 and (i10 == 0 or 5 <= i10 <= 9 or 11 <= i100 <= 14):
                return PluralCategory.MANY
            return PluralCategory.OTHER

        for lang in ["ru", "uk", "sr", "hr", "bs"]:
            self._cardinal_rules[lang] = slavic_cardinal

        # Polish (slightly different from other Slavic)
        # One: i = 1 and v = 0
        # Few: v = 0 and i % 10 = 2..4 and i % 100 != 12..14
        # Many: v = 0 and i != 1 and i % 10 = 0..1 or v = 0 and i % 10 = 5..9 or v = 0 and i % 100 = 12..14
        def polish_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            i10 = op.i % 10
            i100 = op.i % 100

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if op.v == 0 and 2 <= i10 <= 4 and not (12 <= i100 <= 14):
                return PluralCategory.FEW
            if op.v == 0 and (op.i != 1 and i10 in (0, 1) or 5 <= i10 <= 9 or 12 <= i100 <= 14):
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["pl"] = polish_cardinal

        # Arabic
        # Zero: n = 0
        # One: n = 1
        # Two: n = 2
        # Few: n % 100 = 3..10
        # Many: n % 100 = 11..99
        def arabic_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            n100 = int(op.n) % 100

            if op.n == 0:
                return PluralCategory.ZERO
            if op.n == 1:
                return PluralCategory.ONE
            if op.n == 2:
                return PluralCategory.TWO
            if 3 <= n100 <= 10:
                return PluralCategory.FEW
            if 11 <= n100 <= 99:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["ar"] = arabic_cardinal

        # Hebrew
        # One: i = 1 and v = 0
        # Two: i = 2 and v = 0
        # Many: v = 0 and not i = 0..10 and i % 10 = 0
        def hebrew_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if op.i == 2 and op.v == 0:
                return PluralCategory.TWO
            if op.v == 0 and not (0 <= op.i <= 10) and op.i % 10 == 0:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["he"] = hebrew_cardinal

        # Japanese, Korean, Chinese, Vietnamese, Thai, Indonesian, Malay
        # No plural distinction
        def no_plural(n: float | int) -> PluralCategory:
            return PluralCategory.OTHER

        for lang in ["ja", "ko", "zh", "vi", "th"]:
            self._cardinal_rules[lang] = no_plural

        # Welsh
        # Zero: n = 0
        # One: n = 1
        # Two: n = 2
        # Few: n = 3
        # Many: n = 6
        def welsh_cardinal(n: float | int) -> PluralCategory:
            if n == 0:
                return PluralCategory.ZERO
            if n == 1:
                return PluralCategory.ONE
            if n == 2:
                return PluralCategory.TWO
            if n == 3:
                return PluralCategory.FEW
            if n == 6:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["cy"] = welsh_cardinal

        # Irish
        # One: n = 1
        # Two: n = 2
        # Few: n = 3..6
        # Many: n = 7..10
        def irish_cardinal(n: float | int) -> PluralCategory:
            if n == 1:
                return PluralCategory.ONE
            if n == 2:
                return PluralCategory.TWO
            if 3 <= n <= 6:
                return PluralCategory.FEW
            if 7 <= n <= 10:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["ga"] = irish_cardinal

        # Latvian
        # Zero: n % 10 = 0 or n % 100 = 11..19 or v = 2 and f % 100 = 11..19
        # One: n % 10 = 1 and n % 100 != 11 or v = 2 and f % 10 = 1 and f % 100 != 11 or v != 2 and f % 10 = 1
        def latvian_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            n10 = int(op.n) % 10
            n100 = int(op.n) % 100
            f10 = op.f % 10
            f100 = op.f % 100

            if n10 == 0 or 11 <= n100 <= 19 or (op.v == 2 and 11 <= f100 <= 19):
                return PluralCategory.ZERO
            if (n10 == 1 and n100 != 11) or (op.v == 2 and f10 == 1 and f100 != 11) or (op.v != 2 and f10 == 1):
                return PluralCategory.ONE
            return PluralCategory.OTHER

        self._cardinal_rules["lv"] = latvian_cardinal

        # Lithuanian
        # One: n % 10 = 1 and n % 100 != 11..19
        # Few: n % 10 = 2..9 and n % 100 != 11..19
        def lithuanian_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            n10 = int(op.n) % 10
            n100 = int(op.n) % 100

            if n10 == 1 and not (11 <= n100 <= 19):
                return PluralCategory.ONE
            if 2 <= n10 <= 9 and not (11 <= n100 <= 19):
                return PluralCategory.FEW
            return PluralCategory.MANY if op.f != 0 else PluralCategory.OTHER

        self._cardinal_rules["lt"] = lithuanian_cardinal

        # Czech, Slovak
        # One: i = 1 and v = 0
        # Few: i = 2..4 and v = 0
        # Many: v != 0
        def czech_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if 2 <= op.i <= 4 and op.v == 0:
                return PluralCategory.FEW
            if op.v != 0:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["cs"] = czech_cardinal
        self._cardinal_rules["sk"] = czech_cardinal

        # Romanian, Moldavian
        # One: i = 1 and v = 0
        # Few: v != 0 or n = 0 or n % 100 = 2..19
        def romanian_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            n100 = int(op.n) % 100

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if op.v != 0 or op.n == 0 or 2 <= n100 <= 19:
                return PluralCategory.FEW
            return PluralCategory.OTHER

        self._cardinal_rules["ro"] = romanian_cardinal
        self._cardinal_rules["mo"] = romanian_cardinal

        # Slovenian
        # One: v = 0 and i % 100 = 1
        # Two: v = 0 and i % 100 = 2
        # Few: v = 0 and i % 100 = 3..4 or v != 0
        def slovenian_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            i100 = op.i % 100

            if op.v == 0 and i100 == 1:
                return PluralCategory.ONE
            if op.v == 0 and i100 == 2:
                return PluralCategory.TWO
            if op.v == 0 and 3 <= i100 <= 4 or op.v != 0:
                return PluralCategory.FEW
            return PluralCategory.OTHER

        self._cardinal_rules["sl"] = slovenian_cardinal

        # Maltese
        # One: n = 1
        # Few: n = 0 or n % 100 = 2..10
        # Many: n % 100 = 11..19
        def maltese_cardinal(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            n100 = int(op.n) % 100

            if op.n == 1:
                return PluralCategory.ONE
            if op.n == 0 or 2 <= n100 <= 10:
                return PluralCategory.FEW
            if 11 <= n100 <= 19:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["mt"] = maltese_cardinal

        # ==========================================
        # ORDINAL RULES
        # ==========================================

        # English ordinals: 1st, 2nd, 3rd, 4th...
        def english_ordinal(n: float | int) -> PluralCategory:
            n10 = int(n) % 10
            n100 = int(n) % 100

            if n10 == 1 and n100 != 11:
                return PluralCategory.ONE  # 1st, 21st, 31st...
            if n10 == 2 and n100 != 12:
                return PluralCategory.TWO  # 2nd, 22nd, 32nd...
            if n10 == 3 and n100 != 13:
                return PluralCategory.FEW  # 3rd, 23rd, 33rd...
            return PluralCategory.OTHER  # 4th, 5th, 11th, 12th...

        self._ordinal_rules["en"] = english_ordinal

        # Italian ordinals
        # Many: n = 11 or n = 8 or n = 80 or n = 800
        def italian_ordinal(n: float | int) -> PluralCategory:
            if n in (11, 8, 80, 800):
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._ordinal_rules["it"] = italian_ordinal

        # Swedish ordinals
        # One: n % 10 = 1,2 and n % 100 != 11,12
        def swedish_ordinal(n: float | int) -> PluralCategory:
            n10 = int(n) % 10
            n100 = int(n) % 100

            if n10 in (1, 2) and n100 not in (11, 12):
                return PluralCategory.ONE
            return PluralCategory.OTHER

        self._ordinal_rules["sv"] = swedish_ordinal

        # Welsh ordinals
        # Zero: n = 0,7,8,9
        # One: n = 1
        # Two: n = 2
        # Few: n = 3,4
        # Many: n = 5,6
        def welsh_ordinal(n: float | int) -> PluralCategory:
            if n in (0, 7, 8, 9):
                return PluralCategory.ZERO
            if n == 1:
                return PluralCategory.ONE
            if n == 2:
                return PluralCategory.TWO
            if n in (3, 4):
                return PluralCategory.FEW
            if n in (5, 6):
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._ordinal_rules["cy"] = welsh_ordinal

        # Catalan ordinals
        # One: n = 1,3
        # Two: n = 2
        # Few: n = 4
        def catalan_ordinal(n: float | int) -> PluralCategory:
            if n in (1, 3):
                return PluralCategory.ONE
            if n == 2:
                return PluralCategory.TWO
            if n == 4:
                return PluralCategory.FEW
            return PluralCategory.OTHER

        self._ordinal_rules["ca"] = catalan_ordinal

        # Hungarian ordinals
        # One: n = 1,5
        def hungarian_ordinal(n: float | int) -> PluralCategory:
            if n in (1, 5):
                return PluralCategory.ONE
            return PluralCategory.OTHER

        self._ordinal_rules["hu"] = hungarian_ordinal

        # Georgian ordinals
        # One: i = 1
        # Many: i = 0 or i % 100 = 2..20,40,60,80
        def georgian_ordinal(n: float | int) -> PluralCategory:
            i = int(n)
            i100 = i % 100

            if i == 1:
                return PluralCategory.ONE
            if i == 0 or 2 <= i100 <= 20 or i100 in (40, 60, 80):
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._ordinal_rules["ka"] = georgian_ordinal

        # Default ordinal (no distinction)
        def default_ordinal(n: float | int) -> PluralCategory:
            return PluralCategory.OTHER

        # Apply default ordinal to languages without specific rules
        for lang in ["ko", "ja", "zh", "vi", "th", "ar", "he", "fa"]:
            self._ordinal_rules[lang] = default_ordinal

    def register_cardinal_rule(self, language: str, rule: PluralRuleFunc) -> None:
        """Register a custom cardinal plural rule.

        Args:
            language: ISO 639-1 language code
            rule: Plural rule function
        """
        self._cardinal_rules[language] = rule

    def register_ordinal_rule(self, language: str, rule: PluralRuleFunc) -> None:
        """Register a custom ordinal plural rule.

        Args:
            language: ISO 639-1 language code
            rule: Plural rule function
        """
        self._ordinal_rules[language] = rule

    def get_category(
        self,
        count: float | int,
        locale: LocaleInfo,
        ordinal: bool = False,
    ) -> PluralCategory:
        """Get the plural category for a number.

        Args:
            count: The number to categorize
            locale: Target locale
            ordinal: If True, use ordinal rules

        Returns:
            Appropriate plural category
        """
        rules = self._ordinal_rules if ordinal else self._cardinal_rules

        # Try exact language match first
        if locale.language in rules:
            return rules[locale.language](count)

        # Try language with region
        if locale.region:
            key = f"{locale.language}_{locale.region}"
            if key in rules:
                return rules[key](count)

        # Default: use OTHER
        return PluralCategory.OTHER

    def get_supported_languages(self, ordinal: bool = False) -> list[str]:
        """Get list of supported language codes.

        Args:
            ordinal: If True, return ordinal-supported languages

        Returns:
            List of language codes
        """
        rules = self._ordinal_rules if ordinal else self._cardinal_rules
        return list(rules.keys())


# Global instance
_plural_rules = CLDRPluralRules()


def get_plural_category(
    count: float | int,
    locale: str | LocaleInfo,
    ordinal: bool = False,
) -> PluralCategory:
    """Get the plural category for a number.

    Args:
        count: The number to categorize
        locale: Target locale (string or LocaleInfo)
        ordinal: If True, use ordinal rules

    Returns:
        Appropriate plural category

    Example:
        get_plural_category(1, "en")  # ONE
        get_plural_category(2, "en")  # OTHER
        get_plural_category(5, "ru")  # MANY
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)

    return _plural_rules.get_category(count, locale, ordinal)


def pluralize(
    count: float | int,
    forms: dict[PluralCategory | str, str],
    locale: str | LocaleInfo = "en",
    ordinal: bool = False,
    **params,
) -> str:
    """Select and format a pluralized message.

    Args:
        count: Number for pluralization
        forms: Dictionary mapping plural categories to message templates
        locale: Target locale
        ordinal: If True, use ordinal rules
        **params: Additional format parameters

    Returns:
        Formatted message

    Example:
        # English example
        pluralize(
            count=3,
            forms={
                PluralCategory.ONE: "{count} file",
                PluralCategory.OTHER: "{count} files",
            },
            locale="en",
        )
        # -> "3 files"

        # Russian example
        pluralize(
            count=3,
            forms={
                PluralCategory.ONE: "{count} файл",
                PluralCategory.FEW: "{count} файла",
                PluralCategory.MANY: "{count} файлов",
                PluralCategory.OTHER: "{count} файла",
            },
            locale="ru",
        )
        # -> "3 файла"

        # Using string keys
        pluralize(
            count=1,
            forms={
                "one": "{count} item",
                "other": "{count} items",
            },
            locale="en",
        )
        # -> "1 item"
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)

    category = _plural_rules.get_category(count, locale, ordinal)

    # Normalize form keys to PluralCategory
    normalized_forms: dict[PluralCategory, str] = {}
    for key, value in forms.items():
        if isinstance(key, str):
            try:
                cat = PluralCategory(key.lower())
            except ValueError:
                continue
        else:
            cat = key
        normalized_forms[cat] = value

    # Get template
    template = _plural_rules.get_plural_form(count, normalized_forms, locale)

    # Format with count and additional params
    try:
        return template.format(count=count, **params)
    except KeyError:
        return template


def get_plural_rules() -> CLDRPluralRules:
    """Get the global plural rules instance.

    Returns:
        CLDRPluralRules instance
    """
    return _plural_rules
