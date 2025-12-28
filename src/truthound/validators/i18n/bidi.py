"""Bidirectional (BiDi) Text Support.

This module provides comprehensive support for bidirectional text handling,
including RTL (Right-to-Left) languages like Arabic, Hebrew, Persian, and Urdu.

Features:
- Text direction detection
- Unicode BiDi control character handling
- RTL-aware message formatting
- Mirror character support for UI elements

Usage:
    from truthound.validators.i18n.bidi import (
        BiDiHandler,
        TextDirection,
        detect_direction,
        wrap_bidi,
    )

    # Detect text direction
    direction = detect_direction("مرحبا")  # RTL
    direction = detect_direction("Hello")  # LTR

    # Wrap text with BiDi controls
    wrapped = wrap_bidi("Hello", TextDirection.RTL)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable

from truthound.validators.i18n.protocols import LocaleInfo, TextDirection


# Unicode BiDi control characters
class BiDiControl:
    """Unicode Bidirectional control characters."""

    # Explicit directional embeddings
    LRE = "\u202A"  # Left-to-Right Embedding
    RLE = "\u202B"  # Right-to-Left Embedding
    PDF = "\u202C"  # Pop Directional Formatting

    # Explicit directional overrides
    LRO = "\u202D"  # Left-to-Right Override
    RLO = "\u202E"  # Right-to-Left Override

    # Explicit directional isolates (Unicode 6.3+)
    LRI = "\u2066"  # Left-to-Right Isolate
    RLI = "\u2067"  # Right-to-Left Isolate
    FSI = "\u2068"  # First Strong Isolate
    PDI = "\u2069"  # Pop Directional Isolate

    # Explicit marks
    LRM = "\u200E"  # Left-to-Right Mark
    RLM = "\u200F"  # Right-to-Left Mark
    ALM = "\u061C"  # Arabic Letter Mark

    @classmethod
    def is_control(cls, char: str) -> bool:
        """Check if character is a BiDi control character."""
        return char in (
            cls.LRE, cls.RLE, cls.PDF,
            cls.LRO, cls.RLO,
            cls.LRI, cls.RLI, cls.FSI, cls.PDI,
            cls.LRM, cls.RLM, cls.ALM,
        )


@dataclass
class BiDiStats:
    """Statistics about bidirectional content in text.

    Attributes:
        total_chars: Total number of characters
        ltr_chars: Number of LTR characters
        rtl_chars: Number of RTL characters
        neutral_chars: Number of neutral characters
        control_chars: Number of BiDi control characters
    """
    total_chars: int = 0
    ltr_chars: int = 0
    rtl_chars: int = 0
    neutral_chars: int = 0
    control_chars: int = 0

    @property
    def dominant_direction(self) -> TextDirection:
        """Get the dominant text direction."""
        if self.rtl_chars > self.ltr_chars:
            return TextDirection.RTL
        elif self.ltr_chars > self.rtl_chars:
            return TextDirection.LTR
        return TextDirection.AUTO

    @property
    def is_mixed(self) -> bool:
        """Check if text contains mixed directions."""
        return self.ltr_chars > 0 and self.rtl_chars > 0


@dataclass
class BiDiConfig:
    """Configuration for BiDi handling.

    Attributes:
        use_isolates: Use isolate characters (Unicode 6.3+) instead of embeddings
        strip_controls: Strip existing BiDi controls before processing
        auto_detect: Automatically detect direction from content
        mirror_punctuation: Mirror punctuation for RTL context
    """
    use_isolates: bool = True
    strip_controls: bool = False
    auto_detect: bool = True
    mirror_punctuation: bool = True


# Mirrored character pairs for RTL rendering
_MIRROR_PAIRS: dict[str, str] = {
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
    "«": "»",
    "»": "«",
    "‹": "›",
    "›": "‹",
    "⟨": "⟩",
    "⟩": "⟨",
    "【": "】",
    "】": "【",
    "〈": "〉",
    "〉": "〈",
    "《": "》",
    "》": "《",
    "「": "」",
    "」": "「",
    "『": "』",
    "』": "『",
}


# RTL script ranges in Unicode
_RTL_RANGES = [
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),   # Syriac
    (0x0750, 0x077F),   # Arabic Supplement
    (0x0780, 0x07BF),   # Thaana
    (0x07C0, 0x07FF),   # N'Ko
    (0x0800, 0x083F),   # Samaritan
    (0x0840, 0x085F),   # Mandaic
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB1D, 0xFB4F),   # Hebrew Presentation Forms
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    (0x10800, 0x10FFF), # Various RTL scripts
]


@lru_cache(maxsize=1024)
def _is_rtl_char(char: str) -> bool:
    """Check if a character is RTL.

    Uses Unicode bidirectional category to determine direction.
    """
    if not char:
        return False

    code = ord(char)

    # Check known RTL ranges first (faster)
    for start, end in _RTL_RANGES:
        if start <= code <= end:
            return True

    # Check Unicode bidi category
    try:
        category = unicodedata.bidirectional(char)
        return category in ("R", "AL", "RLE", "RLO", "RLI")
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=1024)
def _is_ltr_char(char: str) -> bool:
    """Check if a character is LTR."""
    if not char:
        return False

    try:
        category = unicodedata.bidirectional(char)
        return category in ("L", "LRE", "LRO", "LRI")
    except (ValueError, TypeError):
        return False


class BiDiHandler:
    """Handler for bidirectional text processing.

    Provides comprehensive BiDi support including:
    - Direction detection
    - Control character insertion
    - Mirror character handling
    - Mixed content formatting

    Example:
        handler = BiDiHandler()

        # Detect direction
        direction = handler.detect("مرحبا بالعالم")
        # -> TextDirection.RTL

        # Wrap text
        wrapped = handler.wrap("Price: $100", TextDirection.RTL)
        # -> "⁧Price: $100⁩"

        # Format mixed content
        formatted = handler.format_mixed(
            "Welcome {name}",
            {"name": "أحمد"},
            base_direction=TextDirection.LTR,
        )
    """

    def __init__(self, config: BiDiConfig | None = None) -> None:
        """Initialize BiDi handler.

        Args:
            config: BiDi configuration
        """
        self.config = config or BiDiConfig()

    def analyze(self, text: str) -> BiDiStats:
        """Analyze bidirectional content in text.

        Args:
            text: Text to analyze

        Returns:
            BiDiStats with character counts
        """
        stats = BiDiStats(total_chars=len(text))

        for char in text:
            if BiDiControl.is_control(char):
                stats.control_chars += 1
            elif _is_rtl_char(char):
                stats.rtl_chars += 1
            elif _is_ltr_char(char):
                stats.ltr_chars += 1
            else:
                stats.neutral_chars += 1

        return stats

    def detect(self, text: str) -> TextDirection:
        """Detect the dominant text direction.

        Uses the "first strong" algorithm - finds the first strongly
        directional character and returns its direction.

        Args:
            text: Text to analyze

        Returns:
            Detected text direction
        """
        for char in text:
            if _is_rtl_char(char):
                return TextDirection.RTL
            if _is_ltr_char(char):
                return TextDirection.LTR

        return TextDirection.AUTO

    def detect_locale_direction(self, locale: LocaleInfo) -> TextDirection:
        """Get the text direction for a locale.

        Args:
            locale: Locale to check

        Returns:
            Text direction for the locale
        """
        return locale.direction

    def wrap(
        self,
        text: str,
        direction: TextDirection,
        use_isolates: bool | None = None,
    ) -> str:
        """Wrap text with BiDi control characters.

        Args:
            text: Text to wrap
            direction: Target direction
            use_isolates: Use isolate characters (default from config)

        Returns:
            Text wrapped with BiDi controls
        """
        if direction == TextDirection.AUTO:
            return text

        use_isolates = use_isolates if use_isolates is not None else self.config.use_isolates

        if use_isolates:
            # Use directional isolates (Unicode 6.3+)
            if direction == TextDirection.RTL:
                return f"{BiDiControl.RLI}{text}{BiDiControl.PDI}"
            else:
                return f"{BiDiControl.LRI}{text}{BiDiControl.PDI}"
        else:
            # Use directional embeddings (older approach)
            if direction == TextDirection.RTL:
                return f"{BiDiControl.RLE}{text}{BiDiControl.PDF}"
            else:
                return f"{BiDiControl.LRE}{text}{BiDiControl.PDF}"

    def wrap_override(
        self,
        text: str,
        direction: TextDirection,
    ) -> str:
        """Wrap text with BiDi override characters.

        Override forces all characters to display in the specified direction.
        Use with caution as it can break number and punctuation rendering.

        Args:
            text: Text to wrap
            direction: Target direction

        Returns:
            Text wrapped with BiDi overrides
        """
        if direction == TextDirection.AUTO:
            return text

        if direction == TextDirection.RTL:
            return f"{BiDiControl.RLO}{text}{BiDiControl.PDF}"
        else:
            return f"{BiDiControl.LRO}{text}{BiDiControl.PDF}"

    def add_mark(self, text: str, direction: TextDirection) -> str:
        """Add directional mark to text.

        Useful for ensuring correct rendering at text boundaries.

        Args:
            text: Text to mark
            direction: Target direction

        Returns:
            Text with directional mark
        """
        if direction == TextDirection.RTL:
            return f"{BiDiControl.RLM}{text}"
        elif direction == TextDirection.LTR:
            return f"{BiDiControl.LRM}{text}"
        return text

    def strip_controls(self, text: str) -> str:
        """Remove all BiDi control characters from text.

        Args:
            text: Text to clean

        Returns:
            Text without BiDi controls
        """
        pattern = (
            f"[{BiDiControl.LRE}{BiDiControl.RLE}{BiDiControl.PDF}"
            f"{BiDiControl.LRO}{BiDiControl.RLO}"
            f"{BiDiControl.LRI}{BiDiControl.RLI}{BiDiControl.FSI}{BiDiControl.PDI}"
            f"{BiDiControl.LRM}{BiDiControl.RLM}{BiDiControl.ALM}]"
        )
        return re.sub(pattern, "", text)

    def mirror_char(self, char: str) -> str:
        """Get the mirrored version of a character.

        Args:
            char: Character to mirror

        Returns:
            Mirrored character or original if no mirror
        """
        return _MIRROR_PAIRS.get(char, char)

    def mirror_text(self, text: str) -> str:
        """Mirror all mirrorable characters in text.

        Args:
            text: Text to mirror

        Returns:
            Text with mirrored characters
        """
        return "".join(self.mirror_char(c) for c in text)

    def format_mixed(
        self,
        template: str,
        values: dict[str, str],
        base_direction: TextDirection = TextDirection.LTR,
    ) -> str:
        """Format a template with mixed-direction values.

        Automatically wraps values with appropriate BiDi controls
        based on their content direction.

        Args:
            template: Template string with {placeholders}
            values: Dictionary of placeholder values
            base_direction: Base direction of the template

        Returns:
            Formatted string with proper BiDi handling
        """
        # Wrap each value according to its direction
        wrapped_values = {}
        for key, value in values.items():
            value_direction = self.detect(value)

            # If value direction differs from base, wrap it
            if value_direction != TextDirection.AUTO and value_direction != base_direction:
                wrapped_values[key] = self.wrap(value, value_direction)
            else:
                wrapped_values[key] = value

        try:
            return template.format(**wrapped_values)
        except KeyError:
            return template

    def normalize_whitespace(
        self,
        text: str,
        direction: TextDirection = TextDirection.AUTO,
    ) -> str:
        """Normalize whitespace for BiDi text.

        Ensures proper whitespace handling at direction boundaries.

        Args:
            text: Text to normalize
            direction: Expected direction

        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Trim spaces at BiDi boundaries
        controls = (
            BiDiControl.LRE, BiDiControl.RLE, BiDiControl.PDF,
            BiDiControl.LRI, BiDiControl.RLI, BiDiControl.PDI,
        )
        for ctrl in controls:
            text = text.replace(f" {ctrl}", ctrl)
            text = text.replace(f"{ctrl} ", ctrl)

        return text.strip()


# RTL language codes
RTL_LANGUAGES = frozenset([
    "ar",  # Arabic
    "arc", # Aramaic
    "arz", # Egyptian Arabic
    "az_arab",  # Azerbaijani (Arabic script)
    "ckb", # Central Kurdish
    "dv",  # Divehi/Maldivian
    "fa",  # Persian/Farsi
    "ff_adlm", # Fulah (Adlam script)
    "he",  # Hebrew
    "khw", # Khowar
    "ks",  # Kashmiri
    "ku_arab", # Kurdish (Arabic script)
    "mzn", # Mazanderani
    "nqo", # N'Ko
    "pnb", # Western Punjabi
    "ps",  # Pashto
    "sd",  # Sindhi
    "sdh", # Southern Kurdish
    "syr", # Syriac
    "ug",  # Uyghur
    "ur",  # Urdu
    "yi",  # Yiddish
])


def is_rtl_language(language: str) -> bool:
    """Check if a language code represents an RTL language.

    Args:
        language: ISO 639 language code

    Returns:
        True if RTL language
    """
    # Normalize language code
    lang = language.lower().replace("-", "_")

    # Check exact match
    if lang in RTL_LANGUAGES:
        return True

    # Check base language (before underscore)
    base = lang.split("_")[0]
    return base in RTL_LANGUAGES


def detect_direction(text: str) -> TextDirection:
    """Detect text direction (convenience function).

    Args:
        text: Text to analyze

    Returns:
        Detected direction
    """
    handler = BiDiHandler()
    return handler.detect(text)


def wrap_bidi(
    text: str,
    direction: TextDirection,
    use_isolates: bool = True,
) -> str:
    """Wrap text with BiDi controls (convenience function).

    Args:
        text: Text to wrap
        direction: Target direction
        use_isolates: Use isolate characters

    Returns:
        Wrapped text
    """
    handler = BiDiHandler(BiDiConfig(use_isolates=use_isolates))
    return handler.wrap(text, direction)


def get_locale_direction(locale: str | LocaleInfo) -> TextDirection:
    """Get text direction for a locale.

    Args:
        locale: Locale string or LocaleInfo

    Returns:
        Text direction
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)

    return locale.direction
