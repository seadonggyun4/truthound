"""i18n Protocol Definitions and Base Abstractions.

This module defines the core protocols (interfaces) for the i18n system,
enabling extensibility and loose coupling between components.

Protocols:
- PluralRuleProvider: CLDR plural rule handling
- NumberFormatter: Locale-aware number formatting
- DateFormatter: Locale-aware date/time formatting
- TextDirectionProvider: RTL/LTR text direction
- MessageResolver: Message resolution with context
- CatalogLoader: Dynamic catalog loading
- TranslationService: External TMS integration
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable


# ==============================================================================
# Enums and Type Definitions
# ==============================================================================

class PluralCategory(str, Enum):
    """CLDR plural categories.

    Based on Unicode CLDR plural rules:
    https://cldr.unicode.org/index/cldr-spec/plural-rules
    """
    ZERO = "zero"      # 0 items
    ONE = "one"        # 1 item (singular)
    TWO = "two"        # 2 items (dual)
    FEW = "few"        # Small number (e.g., 2-4 in Slavic languages)
    MANY = "many"      # Larger number (e.g., 5+ in Slavic languages)
    OTHER = "other"    # Default/fallback


class TextDirection(str, Enum):
    """Text direction for layout."""
    LTR = "ltr"  # Left-to-right (English, Korean, etc.)
    RTL = "rtl"  # Right-to-left (Arabic, Hebrew, etc.)
    AUTO = "auto"  # Automatic detection


class NumberStyle(str, Enum):
    """Number formatting style."""
    DECIMAL = "decimal"
    CURRENCY = "currency"
    PERCENT = "percent"
    SCIENTIFIC = "scientific"
    COMPACT = "compact"
    ORDINAL = "ordinal"


class DateStyle(str, Enum):
    """Date formatting style."""
    SHORT = "short"      # 12/31/24
    MEDIUM = "medium"    # Dec 31, 2024
    LONG = "long"        # December 31, 2024
    FULL = "full"        # Tuesday, December 31, 2024
    ISO = "iso"          # 2024-12-31
    RELATIVE = "relative"  # 2 days ago


class TimeStyle(str, Enum):
    """Time formatting style."""
    SHORT = "short"      # 3:30 PM
    MEDIUM = "medium"    # 3:30:00 PM
    LONG = "long"        # 3:30:00 PM UTC
    FULL = "full"        # 3:30:00 PM Coordinated Universal Time


class MessageContext(str, Enum):
    """Context for message selection."""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MARKETING = "marketing"


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass(frozen=True)
class LocaleInfo:
    """Complete locale information.

    Attributes:
        language: ISO 639-1 language code (e.g., "en", "ko")
        region: ISO 3166-1 region code (e.g., "US", "GB")
        script: ISO 15924 script code (e.g., "Latn", "Hans")
        variant: Locale variant (e.g., "formal", "informal")
    """
    language: str
    region: str | None = None
    script: str | None = None
    variant: str | None = None

    @property
    def tag(self) -> str:
        """Get BCP 47 language tag."""
        parts = [self.language]
        if self.script:
            parts.append(self.script)
        if self.region:
            parts.append(self.region)
        if self.variant:
            parts.append(self.variant)
        return "-".join(parts)

    @property
    def direction(self) -> TextDirection:
        """Get default text direction for this locale."""
        rtl_languages = {"ar", "he", "fa", "ur", "yi", "ps", "sd"}
        if self.language in rtl_languages:
            return TextDirection.RTL
        return TextDirection.LTR

    @classmethod
    def parse(cls, tag: str) -> "LocaleInfo":
        """Parse a locale tag.

        Supports formats:
        - Simple: "en", "ko"
        - With region: "en-US", "ko-KR", "en_US", "ko_KR"
        - With script: "zh-Hans", "zh-Hant"
        - Full: "zh-Hans-CN", "sr-Latn-RS"

        Args:
            tag: Locale tag string

        Returns:
            Parsed LocaleInfo
        """
        # Normalize separator
        parts = tag.replace("_", "-").split("-")

        language = parts[0].lower()
        region = None
        script = None
        variant = None

        for part in parts[1:]:
            if len(part) == 4 and part.isalpha():
                # Script code (4 letters)
                script = part.capitalize()
            elif len(part) == 2 and part.isalpha():
                # Region code (2 letters)
                region = part.upper()
            elif len(part) == 3 and part.isdigit():
                # UN M.49 region code (3 digits)
                region = part
            else:
                # Variant or extension
                variant = part.lower()

        return cls(
            language=language,
            region=region,
            script=script,
            variant=variant,
        )

    def matches(self, other: "LocaleInfo", strict: bool = False) -> bool:
        """Check if this locale matches another.

        Args:
            other: Locale to compare with
            strict: If True, require exact match

        Returns:
            True if locales match
        """
        if strict:
            return (
                self.language == other.language
                and self.region == other.region
                and self.script == other.script
                and self.variant == other.variant
            )

        # Non-strict: language must match, others are optional
        if self.language != other.language:
            return False

        if other.region and self.region != other.region:
            return False

        if other.script and self.script != other.script:
            return False

        return True


@dataclass
class FormattedNumber:
    """Result of number formatting.

    Attributes:
        value: Original numeric value
        formatted: Formatted string representation
        direction: Text direction
        parts: Component parts (for advanced rendering)
    """
    value: float | int | Decimal
    formatted: str
    direction: TextDirection = TextDirection.LTR
    parts: dict[str, str] = field(default_factory=dict)


@dataclass
class FormattedDate:
    """Result of date/time formatting.

    Attributes:
        value: Original datetime value
        formatted: Formatted string representation
        direction: Text direction
        calendar: Calendar system used
    """
    value: datetime | date | time
    formatted: str
    direction: TextDirection = TextDirection.LTR
    calendar: str = "gregorian"


@dataclass
class PluralizedMessage:
    """Result of message pluralization.

    Attributes:
        message: Formatted message
        count: Original count value
        category: Plural category used
    """
    message: str
    count: float | int
    category: PluralCategory


@dataclass
class ResolvedMessage:
    """Result of message resolution.

    Attributes:
        key: Original message key
        message: Resolved and formatted message
        locale: Locale used
        context: Context used
        fallback: Whether fallback was used
    """
    key: str
    message: str
    locale: LocaleInfo
    context: MessageContext | None = None
    fallback: bool = False


# ==============================================================================
# Protocols (Interfaces)
# ==============================================================================

@runtime_checkable
class PluralRuleProvider(Protocol):
    """Protocol for plural rule handling.

    Implementations should follow CLDR plural rules:
    https://cldr.unicode.org/index/cldr-spec/plural-rules
    """

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
            ordinal: If True, use ordinal rules (1st, 2nd, 3rd...)

        Returns:
            Appropriate plural category
        """
        ...

    def get_plural_form(
        self,
        count: float | int,
        forms: dict[PluralCategory, str],
        locale: LocaleInfo,
    ) -> str:
        """Select appropriate plural form.

        Args:
            count: The number to pluralize
            forms: Dictionary of category -> message template
            locale: Target locale

        Returns:
            Selected message template
        """
        ...


@runtime_checkable
class NumberFormatter(Protocol):
    """Protocol for locale-aware number formatting."""

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
            **options: Additional options (currency, precision, etc.)

        Returns:
            Formatted number result
        """
        ...

    def parse(
        self,
        text: str,
        locale: LocaleInfo,
        style: NumberStyle = NumberStyle.DECIMAL,
    ) -> float | int | Decimal | None:
        """Parse a localized number string.

        Args:
            text: Localized number string
            locale: Source locale
            style: Expected format style

        Returns:
            Parsed number or None if invalid
        """
        ...


@runtime_checkable
class DateFormatter(Protocol):
    """Protocol for locale-aware date/time formatting."""

    def format_date(
        self,
        value: datetime | date,
        locale: LocaleInfo,
        style: DateStyle = DateStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a date according to locale rules.

        Args:
            value: Date to format
            locale: Target locale
            style: Formatting style
            **options: Additional options (timezone, calendar, etc.)

        Returns:
            Formatted date result
        """
        ...

    def format_time(
        self,
        value: datetime | time,
        locale: LocaleInfo,
        style: TimeStyle = TimeStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a time according to locale rules.

        Args:
            value: Time to format
            locale: Target locale
            style: Formatting style
            **options: Additional options (timezone, etc.)

        Returns:
            Formatted time result
        """
        ...

    def format_datetime(
        self,
        value: datetime,
        locale: LocaleInfo,
        date_style: DateStyle = DateStyle.MEDIUM,
        time_style: TimeStyle = TimeStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a datetime according to locale rules.

        Args:
            value: Datetime to format
            locale: Target locale
            date_style: Date formatting style
            time_style: Time formatting style
            **options: Additional options

        Returns:
            Formatted datetime result
        """
        ...

    def format_relative(
        self,
        value: datetime | date,
        reference: datetime | date | None = None,
        locale: LocaleInfo | None = None,
    ) -> FormattedDate:
        """Format a relative date (e.g., "2 days ago").

        Args:
            value: Date to format
            reference: Reference date (default: now)
            locale: Target locale

        Returns:
            Formatted relative date
        """
        ...


@runtime_checkable
class TextDirectionProvider(Protocol):
    """Protocol for text direction handling."""

    def get_direction(self, locale: LocaleInfo) -> TextDirection:
        """Get text direction for a locale.

        Args:
            locale: Target locale

        Returns:
            Text direction
        """
        ...

    def wrap_bidi(
        self,
        text: str,
        direction: TextDirection,
        embed: bool = True,
    ) -> str:
        """Wrap text with bidirectional control characters.

        Args:
            text: Text to wrap
            direction: Intended direction
            embed: If True, use embedding; otherwise use override

        Returns:
            Text with bidi controls
        """
        ...


@runtime_checkable
class MessageResolver(Protocol):
    """Protocol for context-aware message resolution."""

    def resolve(
        self,
        key: str,
        locale: LocaleInfo,
        context: MessageContext | None = None,
        **params: Any,
    ) -> ResolvedMessage:
        """Resolve a message with context.

        Args:
            key: Message key
            locale: Target locale
            context: Message context
            **params: Format parameters

        Returns:
            Resolved message
        """
        ...

    def resolve_plural(
        self,
        key: str,
        count: float | int,
        locale: LocaleInfo,
        context: MessageContext | None = None,
        **params: Any,
    ) -> PluralizedMessage:
        """Resolve a pluralized message.

        Args:
            key: Message key (base, without plural suffix)
            count: Number for pluralization
            locale: Target locale
            context: Message context
            **params: Additional format parameters

        Returns:
            Pluralized message
        """
        ...


@runtime_checkable
class CatalogLoader(Protocol):
    """Protocol for dynamic message catalog loading."""

    def load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        """Load message catalog for a locale.

        Args:
            locale: Target locale
            namespace: Optional namespace (e.g., "validators", "errors")

        Returns:
            Dictionary of message key -> template
        """
        ...

    def is_loaded(self, locale: LocaleInfo) -> bool:
        """Check if a locale catalog is loaded.

        Args:
            locale: Locale to check

        Returns:
            True if loaded
        """
        ...

    def unload(self, locale: LocaleInfo) -> None:
        """Unload a locale catalog from memory.

        Args:
            locale: Locale to unload
        """
        ...

    def get_available_locales(self) -> list[LocaleInfo]:
        """Get list of available locales.

        Returns:
            List of available LocaleInfo
        """
        ...


@runtime_checkable
class TranslationService(Protocol):
    """Protocol for external Translation Management System (TMS) integration."""

    def sync_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
    ) -> dict[str, str]:
        """Sync local catalog with TMS.

        Args:
            locale: Target locale
            catalog: Local message catalog

        Returns:
            Updated catalog with TMS translations
        """
        ...

    def push_new_keys(
        self,
        keys: list[str],
        source_locale: LocaleInfo,
        source_messages: dict[str, str],
    ) -> bool:
        """Push new message keys to TMS for translation.

        Args:
            keys: New message keys
            source_locale: Source locale (usually "en")
            source_messages: Source message templates

        Returns:
            True if successful
        """
        ...

    def get_translation_status(
        self,
        locale: LocaleInfo,
    ) -> dict[str, float]:
        """Get translation completion status.

        Args:
            locale: Target locale

        Returns:
            Dictionary with completion percentages by namespace
        """
        ...


# ==============================================================================
# Abstract Base Classes
# ==============================================================================

class BasePluralRuleProvider(ABC):
    """Abstract base class for plural rule providers."""

    @abstractmethod
    def get_category(
        self,
        count: float | int,
        locale: LocaleInfo,
        ordinal: bool = False,
    ) -> PluralCategory:
        """Get the plural category for a number."""
        pass

    def get_plural_form(
        self,
        count: float | int,
        forms: dict[PluralCategory, str],
        locale: LocaleInfo,
    ) -> str:
        """Select appropriate plural form."""
        category = self.get_category(count, locale)

        # Try exact match first
        if category in forms:
            return forms[category]

        # Fallback to OTHER
        if PluralCategory.OTHER in forms:
            return forms[PluralCategory.OTHER]

        # Last resort: return first available form
        return next(iter(forms.values()))


class BaseNumberFormatter(ABC):
    """Abstract base class for number formatters."""

    @abstractmethod
    def format(
        self,
        value: float | int | Decimal,
        locale: LocaleInfo,
        style: NumberStyle = NumberStyle.DECIMAL,
        **options: Any,
    ) -> FormattedNumber:
        """Format a number according to locale rules."""
        pass

    def parse(
        self,
        text: str,
        locale: LocaleInfo,
        style: NumberStyle = NumberStyle.DECIMAL,
    ) -> float | int | Decimal | None:
        """Parse a localized number string."""
        # Default implementation: try to parse after cleaning
        try:
            # Remove common thousands separators and normalize decimal
            cleaned = text.strip()
            # Basic implementation - subclasses should override
            return float(cleaned.replace(",", "").replace(" ", ""))
        except ValueError:
            return None


class BaseDateFormatter(ABC):
    """Abstract base class for date formatters."""

    @abstractmethod
    def format_date(
        self,
        value: datetime | date,
        locale: LocaleInfo,
        style: DateStyle = DateStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a date according to locale rules."""
        pass

    @abstractmethod
    def format_time(
        self,
        value: datetime | time,
        locale: LocaleInfo,
        style: TimeStyle = TimeStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a time according to locale rules."""
        pass

    def format_datetime(
        self,
        value: datetime,
        locale: LocaleInfo,
        date_style: DateStyle = DateStyle.MEDIUM,
        time_style: TimeStyle = TimeStyle.MEDIUM,
        **options: Any,
    ) -> FormattedDate:
        """Format a datetime according to locale rules."""
        date_result = self.format_date(value, locale, date_style, **options)
        time_result = self.format_time(value, locale, time_style, **options)

        # Combine date and time
        combined = f"{date_result.formatted} {time_result.formatted}"

        return FormattedDate(
            value=value,
            formatted=combined,
            direction=date_result.direction,
            calendar=date_result.calendar,
        )

    def format_relative(
        self,
        value: datetime | date,
        reference: datetime | date | None = None,
        locale: LocaleInfo | None = None,
    ) -> FormattedDate:
        """Format a relative date (e.g., "2 days ago")."""
        if reference is None:
            reference = datetime.now()

        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime.combine(value, time.min)
        if isinstance(reference, date) and not isinstance(reference, datetime):
            reference = datetime.combine(reference, time.min)

        delta = value - reference
        days = delta.days

        # Basic English implementation - subclasses should override
        if days == 0:
            return FormattedDate(value=value, formatted="today")
        elif days == 1:
            return FormattedDate(value=value, formatted="tomorrow")
        elif days == -1:
            return FormattedDate(value=value, formatted="yesterday")
        elif days > 0:
            return FormattedDate(value=value, formatted=f"in {days} days")
        else:
            return FormattedDate(value=value, formatted=f"{-days} days ago")


class BaseCatalogLoader(ABC):
    """Abstract base class for catalog loaders."""

    def __init__(self) -> None:
        self._loaded_catalogs: dict[str, dict[str, str]] = {}

    @abstractmethod
    def _do_load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        """Internal load implementation."""
        pass

    def load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        """Load message catalog for a locale."""
        cache_key = f"{locale.tag}:{namespace or 'default'}"

        if cache_key not in self._loaded_catalogs:
            self._loaded_catalogs[cache_key] = self._do_load(locale, namespace)

        return self._loaded_catalogs[cache_key]

    def is_loaded(self, locale: LocaleInfo) -> bool:
        """Check if a locale catalog is loaded."""
        return any(
            key.startswith(locale.tag)
            for key in self._loaded_catalogs
        )

    def unload(self, locale: LocaleInfo) -> None:
        """Unload a locale catalog from memory."""
        keys_to_remove = [
            key for key in self._loaded_catalogs
            if key.startswith(locale.tag)
        ]
        for key in keys_to_remove:
            del self._loaded_catalogs[key]

    @abstractmethod
    def get_available_locales(self) -> list[LocaleInfo]:
        """Get list of available locales."""
        pass


class BaseTranslationService(ABC):
    """Abstract base class for TMS integrations."""

    def __init__(self, api_key: str | None = None, project_id: str | None = None):
        self.api_key = api_key
        self.project_id = project_id

    @abstractmethod
    def sync_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
    ) -> dict[str, str]:
        """Sync local catalog with TMS."""
        pass

    @abstractmethod
    def push_new_keys(
        self,
        keys: list[str],
        source_locale: LocaleInfo,
        source_messages: dict[str, str],
    ) -> bool:
        """Push new message keys to TMS for translation."""
        pass

    @abstractmethod
    def get_translation_status(
        self,
        locale: LocaleInfo,
    ) -> dict[str, float]:
        """Get translation completion status."""
        pass
