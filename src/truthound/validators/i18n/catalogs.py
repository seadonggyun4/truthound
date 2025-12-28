"""Message catalogs for validator internationalization.

This module provides structured message catalogs that can be:
- Extended with custom messages
- Loaded from external files
- Merged with existing catalogs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import json


@dataclass
class ValidatorMessageCatalog:
    """A collection of validator messages for a specific locale.

    Catalogs can be created from dictionaries, JSON files, or built
    incrementally using the fluent API.

    Example:
        # Create from dictionary
        catalog = ValidatorMessageCatalog.from_dict("ko", {
            "null.values_found": "'{column}' 컬럼에서 {count}개의 null 값이 발견되었습니다",
        })

        # Load from JSON file
        catalog = ValidatorMessageCatalog.from_json("ko", Path("messages/ko.json"))

        # Build incrementally
        catalog = (
            ValidatorMessageCatalog.builder("ko")
            .add("null.values_found", "...")
            .add("unique.duplicates_found", "...")
            .build()
        )
    """

    locale: str
    messages: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get a message by key.

        Args:
            key: Message key
            default: Default value if not found

        Returns:
            Message template or default
        """
        return self.messages.get(key, default)

    def __getitem__(self, key: str) -> str:
        """Get a message by key."""
        return self.messages[key]

    def __contains__(self, key: str) -> bool:
        """Check if a message key exists."""
        return key in self.messages

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)

    def __iter__(self) -> Iterator[str]:
        """Iterate over message keys."""
        return iter(self.messages)

    def keys(self) -> list[str]:
        """Return all message keys."""
        return list(self.messages.keys())

    def values(self) -> list[str]:
        """Return all message templates."""
        return list(self.messages.values())

    def items(self) -> list[tuple[str, str]]:
        """Return all key-value pairs."""
        return list(self.messages.items())

    def merge(self, other: "ValidatorMessageCatalog") -> "ValidatorMessageCatalog":
        """Merge with another catalog (other takes precedence).

        Args:
            other: Another catalog to merge with

        Returns:
            New merged catalog
        """
        merged_messages = {**self.messages, **other.messages}
        merged_metadata = {**self.metadata, **other.metadata}
        return ValidatorMessageCatalog(
            locale=self.locale,
            messages=merged_messages,
            metadata=merged_metadata,
        )

    def extend(self, messages: dict[str, str]) -> "ValidatorMessageCatalog":
        """Extend catalog with additional messages.

        Args:
            messages: Additional messages to add

        Returns:
            New extended catalog
        """
        return ValidatorMessageCatalog(
            locale=self.locale,
            messages={**self.messages, **messages},
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "locale": self.locale,
            "messages": self.messages.copy(),
            "metadata": self.metadata.copy(),
        }

    def to_json(self, path: Path) -> None:
        """Save catalog to JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(
        cls,
        locale: str,
        messages: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> "ValidatorMessageCatalog":
        """Create catalog from dictionary.

        Args:
            locale: Locale code
            messages: Message dictionary
            metadata: Optional metadata

        Returns:
            New catalog
        """
        return cls(
            locale=locale,
            messages=messages.copy(),
            metadata=metadata or {},
        )

    @classmethod
    def from_json(cls, locale: str, path: Path) -> "ValidatorMessageCatalog":
        """Load catalog from JSON file.

        Args:
            locale: Locale code
            path: JSON file path

        Returns:
            New catalog
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both flat message dict and full catalog format
        if "messages" in data:
            messages = data["messages"]
            metadata = data.get("metadata", {})
        else:
            messages = data
            metadata = {}

        return cls(locale=locale, messages=messages, metadata=metadata)

    @classmethod
    def builder(cls, locale: str) -> "CatalogBuilder":
        """Create a catalog builder.

        Args:
            locale: Locale code

        Returns:
            Builder instance
        """
        return CatalogBuilder(locale)


class CatalogBuilder:
    """Fluent builder for ValidatorMessageCatalog."""

    def __init__(self, locale: str) -> None:
        """Initialize builder.

        Args:
            locale: Locale code
        """
        self._locale = locale
        self._messages: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}

    def add(self, key: str, template: str) -> "CatalogBuilder":
        """Add a message.

        Args:
            key: Message key
            template: Message template

        Returns:
            Self for chaining
        """
        self._messages[key] = template
        return self

    def add_null(
        self,
        values_found: str,
        column_empty: str | None = None,
        above_threshold: str | None = None,
    ) -> "CatalogBuilder":
        """Add null-related messages.

        Args:
            values_found: Message for null values found
            column_empty: Message for empty column
            above_threshold: Message for null ratio above threshold

        Returns:
            Self for chaining
        """
        self._messages["null.values_found"] = values_found
        if column_empty:
            self._messages["null.column_empty"] = column_empty
        if above_threshold:
            self._messages["null.above_threshold"] = above_threshold
        return self

    def add_unique(
        self,
        duplicates_found: str,
        composite_duplicates: str | None = None,
        key_violation: str | None = None,
    ) -> "CatalogBuilder":
        """Add uniqueness-related messages.

        Args:
            duplicates_found: Message for duplicates found
            composite_duplicates: Message for composite duplicates
            key_violation: Message for key violation

        Returns:
            Self for chaining
        """
        self._messages["unique.duplicates_found"] = duplicates_found
        if composite_duplicates:
            self._messages["unique.composite_duplicates"] = composite_duplicates
        if key_violation:
            self._messages["unique.key_violation"] = key_violation
        return self

    def add_type(
        self,
        mismatch: str,
        coercion_failed: str | None = None,
        inference_failed: str | None = None,
    ) -> "CatalogBuilder":
        """Add type-related messages.

        Args:
            mismatch: Message for type mismatch
            coercion_failed: Message for coercion failure
            inference_failed: Message for inference failure

        Returns:
            Self for chaining
        """
        self._messages["type.mismatch"] = mismatch
        if coercion_failed:
            self._messages["type.coercion_failed"] = coercion_failed
        if inference_failed:
            self._messages["type.inference_failed"] = inference_failed
        return self

    def add_format(
        self,
        invalid_email: str | None = None,
        invalid_phone: str | None = None,
        invalid_date: str | None = None,
        invalid_url: str | None = None,
        pattern_mismatch: str | None = None,
    ) -> "CatalogBuilder":
        """Add format-related messages.

        Args:
            invalid_email: Message for invalid email
            invalid_phone: Message for invalid phone
            invalid_date: Message for invalid date
            invalid_url: Message for invalid URL
            pattern_mismatch: Message for pattern mismatch

        Returns:
            Self for chaining
        """
        if invalid_email:
            self._messages["format.invalid_email"] = invalid_email
        if invalid_phone:
            self._messages["format.invalid_phone"] = invalid_phone
        if invalid_date:
            self._messages["format.invalid_date"] = invalid_date
        if invalid_url:
            self._messages["format.invalid_url"] = invalid_url
        if pattern_mismatch:
            self._messages["format.pattern_mismatch"] = pattern_mismatch
        return self

    def add_range(
        self,
        out_of_bounds: str | None = None,
        below_minimum: str | None = None,
        above_maximum: str | None = None,
        outlier_detected: str | None = None,
    ) -> "CatalogBuilder":
        """Add range-related messages.

        Args:
            out_of_bounds: Message for out of bounds
            below_minimum: Message for below minimum
            above_maximum: Message for above maximum
            outlier_detected: Message for outlier detected

        Returns:
            Self for chaining
        """
        if out_of_bounds:
            self._messages["range.out_of_bounds"] = out_of_bounds
        if below_minimum:
            self._messages["range.below_minimum"] = below_minimum
        if above_maximum:
            self._messages["range.above_maximum"] = above_maximum
        if outlier_detected:
            self._messages["range.outlier_detected"] = outlier_detected
        return self

    def add_validation(
        self,
        failed: str | None = None,
        skipped: str | None = None,
        error: str | None = None,
    ) -> "CatalogBuilder":
        """Add general validation messages.

        Args:
            failed: Message for validation failed
            skipped: Message for validation skipped
            error: Message for validation error

        Returns:
            Self for chaining
        """
        if failed:
            self._messages["validation.failed"] = failed
        if skipped:
            self._messages["validation.skipped"] = skipped
        if error:
            self._messages["validation.error"] = error
        return self

    def with_metadata(self, **metadata: Any) -> "CatalogBuilder":
        """Add metadata.

        Args:
            **metadata: Metadata key-value pairs

        Returns:
            Self for chaining
        """
        self._metadata.update(metadata)
        return self

    def build(self) -> ValidatorMessageCatalog:
        """Build the catalog.

        Returns:
            New catalog instance
        """
        return ValidatorMessageCatalog(
            locale=self._locale,
            messages=self._messages.copy(),
            metadata=self._metadata.copy(),
        )


# Pre-built catalog accessor functions

def get_default_messages() -> ValidatorMessageCatalog:
    """Get the default English message catalog.

    Returns:
        English message catalog
    """
    from truthound.validators.i18n.messages import _DEFAULT_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "en",
        _DEFAULT_MESSAGES,
        metadata={"name": "English", "complete": True},
    )


def get_korean_messages() -> ValidatorMessageCatalog:
    """Get the Korean message catalog.

    Returns:
        Korean message catalog
    """
    from truthound.validators.i18n.messages import _KOREAN_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "ko",
        _KOREAN_MESSAGES,
        metadata={"name": "한국어", "complete": True},
    )


def get_japanese_messages() -> ValidatorMessageCatalog:
    """Get the Japanese message catalog.

    Returns:
        Japanese message catalog
    """
    from truthound.validators.i18n.messages import _JAPANESE_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "ja",
        _JAPANESE_MESSAGES,
        metadata={"name": "日本語", "complete": True},
    )


def get_chinese_messages() -> ValidatorMessageCatalog:
    """Get the Chinese (Simplified) message catalog.

    Returns:
        Chinese message catalog
    """
    from truthound.validators.i18n.messages import _CHINESE_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "zh",
        _CHINESE_MESSAGES,
        metadata={"name": "中文", "complete": False},
    )


def get_german_messages() -> ValidatorMessageCatalog:
    """Get the German message catalog.

    Returns:
        German message catalog
    """
    from truthound.validators.i18n.messages import _GERMAN_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "de",
        _GERMAN_MESSAGES,
        metadata={"name": "Deutsch", "complete": False},
    )


def get_french_messages() -> ValidatorMessageCatalog:
    """Get the French message catalog.

    Returns:
        French message catalog
    """
    from truthound.validators.i18n.messages import _FRENCH_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "fr",
        _FRENCH_MESSAGES,
        metadata={"name": "Français", "complete": False},
    )


def get_spanish_messages() -> ValidatorMessageCatalog:
    """Get the Spanish message catalog.

    Returns:
        Spanish message catalog
    """
    from truthound.validators.i18n.messages import _SPANISH_MESSAGES
    return ValidatorMessageCatalog.from_dict(
        "es",
        _SPANISH_MESSAGES,
        metadata={"name": "Español", "complete": False},
    )


def get_all_catalogs() -> dict[str, ValidatorMessageCatalog]:
    """Get all available message catalogs.

    Returns:
        Dictionary of locale code to catalog
    """
    return {
        "en": get_default_messages(),
        "ko": get_korean_messages(),
        "ja": get_japanese_messages(),
        "zh": get_chinese_messages(),
        "de": get_german_messages(),
        "fr": get_french_messages(),
        "es": get_spanish_messages(),
    }


def get_supported_locales() -> list[str]:
    """Get list of supported locale codes.

    Returns:
        List of locale codes
    """
    return ["en", "ko", "ja", "zh", "de", "fr", "es"]


def create_custom_catalog(
    locale: str,
    name: str,
    base_locale: str = "en",
) -> CatalogBuilder:
    """Create a builder for a custom catalog based on an existing one.

    This is useful for creating variants or extending existing catalogs.

    Args:
        locale: New locale code
        name: Human-readable name
        base_locale: Base locale to extend from

    Returns:
        Builder pre-populated with base messages

    Example:
        # Create a Canadian English variant
        catalog = (
            create_custom_catalog("en_CA", "Canadian English", "en")
            .add("format.invalid_phone",
                 "Found {count} invalid Canadian phone numbers in column '{column}'")
            .build()
        )
    """
    catalogs = {
        "en": get_default_messages,
        "ko": get_korean_messages,
        "ja": get_japanese_messages,
        "zh": get_chinese_messages,
        "de": get_german_messages,
        "fr": get_french_messages,
        "es": get_spanish_messages,
    }

    base_catalog = catalogs.get(base_locale, get_default_messages)()
    builder = CatalogBuilder(locale)
    builder._messages = base_catalog.messages.copy()
    builder._metadata = {"name": name, "base": base_locale}

    return builder
