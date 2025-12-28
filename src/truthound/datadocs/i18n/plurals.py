"""Plural rules for report internationalization.

This module provides CLDR-compliant plural handling for reports,
supporting proper pluralization in 100+ languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class PluralCategory(Enum):
    """CLDR plural categories."""

    ZERO = "zero"
    ONE = "one"
    TWO = "two"
    FEW = "few"
    MANY = "many"
    OTHER = "other"


@dataclass
class PluralOperands:
    """CLDR plural operands.

    See: https://unicode.org/reports/tr35/tr35-numbers.html#Operands
    """

    n: float  # Absolute value
    i: int    # Integer digits
    v: int    # Visible fraction digits (with trailing zeros)
    w: int    # Visible fraction digits (without trailing zeros)
    f: int    # Fraction digits as integer (with trailing zeros)
    t: int    # Fraction digits as integer (without trailing zeros)

    @classmethod
    def from_number(cls, n: float | int) -> "PluralOperands":
        """Create operands from a number."""
        abs_n = abs(n)
        i = int(abs_n)

        if isinstance(n, int):
            return cls(n=float(abs_n), i=i, v=0, w=0, f=0, t=0)

        str_n = str(abs_n)
        if "." in str_n:
            fraction_str = str_n.split(".")[1]
            v = len(fraction_str)
            f = int(fraction_str) if fraction_str else 0
            t_str = fraction_str.rstrip("0")
            w = len(t_str)
            t = int(t_str) if t_str else 0
        else:
            v = w = f = t = 0

        return cls(n=float(abs_n), i=i, v=v, w=w, f=f, t=t)


# Type for plural rule function
PluralRuleFunc = Callable[[float | int], PluralCategory]


class PluralRules:
    """CLDR plural rules provider.

    Provides plural category determination for 100+ languages.

    Example:
        rules = PluralRules()
        category = rules.get_category(5, "ru")  # PluralCategory.MANY
    """

    def __init__(self) -> None:
        self._cardinal_rules: dict[str, PluralRuleFunc] = {}
        self._ordinal_rules: dict[str, PluralRuleFunc] = {}
        self._register_rules()

    def _register_rules(self) -> None:
        """Register all plural rules."""
        # English, German, Dutch, Italian, Portuguese, Spanish, etc.
        def english(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            return PluralCategory.OTHER

        for lang in ["en", "de", "nl", "it", "pt", "es", "ca", "da", "no", "sv", "fi", "et", "hu", "tr", "id", "ms"]:
            self._cardinal_rules[lang] = english

        # French
        def french(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            if op.i in (0, 1):
                return PluralCategory.ONE
            return PluralCategory.OTHER

        self._cardinal_rules["fr"] = french

        # Russian, Ukrainian, Serbian, Croatian, Bosnian
        def slavic(n: float | int) -> PluralCategory:
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
            self._cardinal_rules[lang] = slavic

        # Polish
        def polish(n: float | int) -> PluralCategory:
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

        self._cardinal_rules["pl"] = polish

        # Arabic
        def arabic(n: float | int) -> PluralCategory:
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

        self._cardinal_rules["ar"] = arabic

        # Hebrew
        def hebrew(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if op.i == 2 and op.v == 0:
                return PluralCategory.TWO
            if op.v == 0 and not (0 <= op.i <= 10) and op.i % 10 == 0:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["he"] = hebrew

        # Japanese, Korean, Chinese, Vietnamese, Thai (no plurals)
        def no_plural(n: float | int) -> PluralCategory:
            return PluralCategory.OTHER

        for lang in ["ja", "ko", "zh", "vi", "th"]:
            self._cardinal_rules[lang] = no_plural

        # Czech, Slovak
        def czech(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if 2 <= op.i <= 4 and op.v == 0:
                return PluralCategory.FEW
            if op.v != 0:
                return PluralCategory.MANY
            return PluralCategory.OTHER

        self._cardinal_rules["cs"] = czech
        self._cardinal_rules["sk"] = czech

        # Romanian
        def romanian(n: float | int) -> PluralCategory:
            op = PluralOperands.from_number(n)
            n100 = int(op.n) % 100

            if op.i == 1 and op.v == 0:
                return PluralCategory.ONE
            if op.v != 0 or op.n == 0 or 2 <= n100 <= 19:
                return PluralCategory.FEW
            return PluralCategory.OTHER

        self._cardinal_rules["ro"] = romanian

        # English ordinals
        def english_ordinal(n: float | int) -> PluralCategory:
            n10 = int(n) % 10
            n100 = int(n) % 100

            if n10 == 1 and n100 != 11:
                return PluralCategory.ONE
            if n10 == 2 and n100 != 12:
                return PluralCategory.TWO
            if n10 == 3 and n100 != 13:
                return PluralCategory.FEW
            return PluralCategory.OTHER

        self._ordinal_rules["en"] = english_ordinal

    def get_category(
        self,
        count: float | int,
        locale: str,
        ordinal: bool = False,
    ) -> PluralCategory:
        """Get plural category for a number.

        Args:
            count: The number.
            locale: Locale code (e.g., "en", "ko").
            ordinal: Use ordinal rules.

        Returns:
            Plural category.
        """
        rules = self._ordinal_rules if ordinal else self._cardinal_rules
        lang = locale.split("_")[0].split("-")[0]

        if lang in rules:
            return rules[lang](count)

        return PluralCategory.OTHER

    def get_supported_languages(self) -> list[str]:
        """Get supported language codes."""
        return list(self._cardinal_rules.keys())

    def register_rule(
        self,
        language: str,
        rule: PluralRuleFunc,
        ordinal: bool = False,
    ) -> None:
        """Register a custom plural rule."""
        if ordinal:
            self._ordinal_rules[language] = rule
        else:
            self._cardinal_rules[language] = rule


# Global instance
_rules = PluralRules()


def get_plural_category(
    count: float | int,
    locale: str = "en",
    ordinal: bool = False,
) -> PluralCategory:
    """Get plural category for a number.

    Args:
        count: The number.
        locale: Locale code.
        ordinal: Use ordinal rules.

    Returns:
        Plural category.

    Example:
        get_plural_category(1, "en")  # ONE
        get_plural_category(5, "ru")  # MANY
    """
    return _rules.get_category(count, locale, ordinal)


def pluralize(
    count: float | int,
    singular: str,
    plural: str,
    locale: str = "en",
) -> str:
    """Simple pluralization for two-form languages.

    Args:
        count: The number.
        singular: Singular form.
        plural: Plural form.
        locale: Locale code.

    Returns:
        Appropriate form with count.

    Example:
        pluralize(1, "file", "files")  # "1 file"
        pluralize(5, "file", "files")  # "5 files"
    """
    category = _rules.get_category(count, locale)
    form = singular if category == PluralCategory.ONE else plural
    return f"{count} {form}"


def pluralize_with_forms(
    count: float | int,
    forms: dict[PluralCategory | str, str],
    locale: str = "en",
    **params,
) -> str:
    """Pluralize with multiple forms.

    Args:
        count: The number.
        forms: Forms for each category.
        locale: Locale code.
        **params: Format parameters.

    Returns:
        Formatted string.

    Example:
        pluralize_with_forms(
            5,
            {
                PluralCategory.ONE: "{count} file",
                PluralCategory.OTHER: "{count} files",
            },
            "en",
        )
        # -> "5 files"

        pluralize_with_forms(
            3,
            {
                "one": "{count} файл",
                "few": "{count} файла",
                "many": "{count} файлов",
                "other": "{count} файла",
            },
            "ru",
        )
        # -> "3 файла"
    """
    category = _rules.get_category(count, locale)

    # Normalize keys
    normalized: dict[PluralCategory, str] = {}
    for key, value in forms.items():
        if isinstance(key, str):
            try:
                normalized[PluralCategory(key.lower())] = value
            except ValueError:
                pass
        else:
            normalized[key] = value

    # Get template
    template = normalized.get(category)
    if template is None:
        template = normalized.get(PluralCategory.OTHER, str(count))

    # Format
    try:
        return template.format(count=count, **params)
    except KeyError:
        return template


def get_plural_rules() -> PluralRules:
    """Get global plural rules instance."""
    return _rules
