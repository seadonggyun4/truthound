"""Regional Dialect Support.

This module provides comprehensive support for regional language variants,
enabling locale-specific message customization while maintaining fallback
to the base language.

Features:
- Dialect hierarchy with fallback chain
- Regional vocabulary variations
- Spelling variations (US/UK English, etc.)
- Script variations (Simplified/Traditional Chinese)
- Regional formatting preferences

Usage:
    from truthound.validators.i18n.dialects import (
        DialectRegistry,
        DialectResolver,
        create_dialect,
    )

    # Register a dialect
    registry = DialectRegistry()
    registry.register_dialect(
        locale=LocaleInfo.parse("en-GB"),
        base="en",
        overrides={
            "format.invalid_email": "Found {count} invalid e-mail addresses",
        },
    )

    # Resolve with dialect
    resolver = DialectResolver(registry)
    message = resolver.resolve("format.invalid_email", LocaleInfo.parse("en-GB"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from truthound.validators.i18n.protocols import LocaleInfo


@dataclass
class DialectDefinition:
    """Definition of a regional dialect.

    Attributes:
        locale: Full locale identifier
        base_language: Base language code to fall back to
        name: Human-readable name
        native_name: Name in the native language
        overrides: Message key overrides for this dialect
        vocabulary: Regional vocabulary mappings
        spelling_rules: Spelling transformation rules
        metadata: Additional metadata
    """
    locale: LocaleInfo
    base_language: str
    name: str
    native_name: str
    overrides: dict[str, str] = field(default_factory=dict)
    vocabulary: dict[str, str] = field(default_factory=dict)
    spelling_rules: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        """Get the locale tag."""
        return self.locale.tag

    def get_fallback_chain(self) -> list[str]:
        """Get the fallback locale chain.

        Returns:
            List of locale tags from most specific to most general
        """
        chain = [self.tag]

        # Add base language if different
        if self.base_language != self.locale.language:
            chain.append(self.base_language)
        elif self.locale.region:
            chain.append(self.locale.language)

        return chain

    def apply_spelling_rules(self, text: str) -> str:
        """Apply spelling transformation rules to text.

        Args:
            text: Input text

        Returns:
            Text with spelling transformations applied
        """
        result = text
        for pattern, replacement in self.spelling_rules:
            result = result.replace(pattern, replacement)
        return result

    def apply_vocabulary(self, text: str) -> str:
        """Apply vocabulary mappings to text.

        Args:
            text: Input text

        Returns:
            Text with vocabulary replacements applied
        """
        result = text
        for term, replacement in self.vocabulary.items():
            result = result.replace(term, replacement)
        return result


class DialectRegistry:
    """Registry for managing dialect definitions.

    Provides dialect registration, lookup, and fallback chain resolution.

    Example:
        registry = DialectRegistry()

        # Register British English
        registry.register_dialect(
            locale=LocaleInfo.parse("en-GB"),
            base="en",
            name="British English",
            native_name="British English",
            spelling_rules=[
                ("color", "colour"),
                ("center", "centre"),
                ("organize", "organise"),
            ],
        )

        # Get dialect
        dialect = registry.get_dialect(LocaleInfo.parse("en-GB"))
    """

    def __init__(self) -> None:
        self._dialects: dict[str, DialectDefinition] = {}
        self._register_default_dialects()

    def _register_default_dialects(self) -> None:
        """Register commonly used dialects."""
        # ==================================================
        # English Variants
        # ==================================================

        # British English
        self.register_dialect(
            locale=LocaleInfo.parse("en-GB"),
            base="en",
            name="British English",
            native_name="British English",
            spelling_rules=[
                ("color", "colour"),
                ("center", "centre"),
                ("analyze", "analyse"),
                ("realize", "realise"),
                ("organize", "organise"),
                ("behavior", "behaviour"),
                ("favor", "favour"),
                ("honor", "honour"),
                ("labor", "labour"),
                ("neighbor", "neighbour"),
                ("catalog", "catalogue"),
                ("dialog", "dialogue"),
                ("program", "programme"),  # except for computer programs
            ],
            vocabulary={
                "e-mail": "email",
                "cell phone": "mobile phone",
                "apartment": "flat",
            },
        )

        # Australian English
        self.register_dialect(
            locale=LocaleInfo.parse("en-AU"),
            base="en-GB",  # Based on British English
            name="Australian English",
            native_name="Australian English",
            vocabulary={
                "trash can": "rubbish bin",
            },
        )

        # Canadian English
        self.register_dialect(
            locale=LocaleInfo.parse("en-CA"),
            base="en",
            name="Canadian English",
            native_name="Canadian English",
            spelling_rules=[
                ("color", "colour"),
                ("center", "centre"),
            ],
        )

        # Indian English
        self.register_dialect(
            locale=LocaleInfo.parse("en-IN"),
            base="en-GB",
            name="Indian English",
            native_name="Indian English",
            vocabulary={
                "one hundred thousand": "one lakh",
                "ten million": "one crore",
            },
        )

        # ==================================================
        # Spanish Variants
        # ==================================================

        # Latin American Spanish
        self.register_dialect(
            locale=LocaleInfo.parse("es-419"),  # Latin America and Caribbean
            base="es",
            name="Latin American Spanish",
            native_name="Español latinoamericano",
            vocabulary={
                "ordenador": "computadora",
                "móvil": "celular",
                "coche": "carro",
            },
        )

        # Mexican Spanish
        self.register_dialect(
            locale=LocaleInfo.parse("es-MX"),
            base="es-419",
            name="Mexican Spanish",
            native_name="Español mexicano",
        )

        # Argentine Spanish
        self.register_dialect(
            locale=LocaleInfo.parse("es-AR"),
            base="es-419",
            name="Argentine Spanish",
            native_name="Español argentino",
            vocabulary={
                "tú": "vos",
            },
        )

        # ==================================================
        # Portuguese Variants
        # ==================================================

        # Brazilian Portuguese
        self.register_dialect(
            locale=LocaleInfo.parse("pt-BR"),
            base="pt",
            name="Brazilian Portuguese",
            native_name="Português brasileiro",
            vocabulary={
                "autocarro": "ônibus",
                "telemóvel": "celular",
                "ficheiro": "arquivo",
                "ecrã": "tela",
            },
            overrides={
                "null.values_found": "Encontrados {count} valores nulos na coluna '{column}'",
                "unique.duplicates_found": "Encontrados {count} valores duplicados na coluna '{column}'",
            },
        )

        # European Portuguese
        self.register_dialect(
            locale=LocaleInfo.parse("pt-PT"),
            base="pt",
            name="European Portuguese",
            native_name="Português europeu",
        )

        # ==================================================
        # French Variants
        # ==================================================

        # Canadian French
        self.register_dialect(
            locale=LocaleInfo.parse("fr-CA"),
            base="fr",
            name="Canadian French",
            native_name="Français canadien",
            vocabulary={
                "courriel": "courriel",  # Different from "e-mail" used in France
                "fin de semaine": "fin de semaine",  # Instead of "week-end"
            },
        )

        # Belgian French
        self.register_dialect(
            locale=LocaleInfo.parse("fr-BE"),
            base="fr",
            name="Belgian French",
            native_name="Français de Belgique",
            vocabulary={
                "soixante-dix": "septante",
                "quatre-vingts": "quatre-vingts",
                "quatre-vingt-dix": "nonante",
            },
        )

        # Swiss French
        self.register_dialect(
            locale=LocaleInfo.parse("fr-CH"),
            base="fr",
            name="Swiss French",
            native_name="Français de Suisse",
            vocabulary={
                "soixante-dix": "septante",
                "quatre-vingts": "huitante",
                "quatre-vingt-dix": "nonante",
            },
        )

        # ==================================================
        # German Variants
        # ==================================================

        # Austrian German
        self.register_dialect(
            locale=LocaleInfo.parse("de-AT"),
            base="de",
            name="Austrian German",
            native_name="Österreichisches Deutsch",
            vocabulary={
                "Januar": "Jänner",
                "Tomaten": "Paradeiser",
                "Kartoffeln": "Erdäpfel",
            },
        )

        # Swiss German (Standard)
        self.register_dialect(
            locale=LocaleInfo.parse("de-CH"),
            base="de",
            name="Swiss German",
            native_name="Schweizerdeutsch",
            spelling_rules=[
                ("ß", "ss"),  # Swiss German doesn't use ß
            ],
        )

        # ==================================================
        # Chinese Variants
        # ==================================================

        # Simplified Chinese (Mainland China)
        self.register_dialect(
            locale=LocaleInfo.parse("zh-Hans-CN"),
            base="zh",
            name="Simplified Chinese",
            native_name="简体中文",
            metadata={"script": "Hans"},
        )

        # Traditional Chinese (Taiwan)
        self.register_dialect(
            locale=LocaleInfo.parse("zh-Hant-TW"),
            base="zh",
            name="Traditional Chinese (Taiwan)",
            native_name="繁體中文（台灣）",
            metadata={"script": "Hant"},
            vocabulary={
                "软件": "軟體",
                "网络": "網路",
                "服务器": "伺服器",
                "数据": "資料",
                "信息": "訊息",
            },
            overrides={
                "null.values_found": "在欄位'{column}'中發現{count}個空值",
                "unique.duplicates_found": "在欄位'{column}'中發現{count}個重複值",
            },
        )

        # Traditional Chinese (Hong Kong)
        self.register_dialect(
            locale=LocaleInfo.parse("zh-Hant-HK"),
            base="zh-Hant-TW",
            name="Traditional Chinese (Hong Kong)",
            native_name="繁體中文（香港）",
            metadata={"script": "Hant"},
        )

        # ==================================================
        # Arabic Variants
        # ==================================================

        # Modern Standard Arabic
        self.register_dialect(
            locale=LocaleInfo.parse("ar"),
            base="ar",
            name="Modern Standard Arabic",
            native_name="العربية الفصحى",
        )

        # Egyptian Arabic
        self.register_dialect(
            locale=LocaleInfo.parse("ar-EG"),
            base="ar",
            name="Egyptian Arabic",
            native_name="العربية المصرية",
        )

        # Gulf Arabic
        self.register_dialect(
            locale=LocaleInfo.parse("ar-AE"),
            base="ar",
            name="Gulf Arabic",
            native_name="العربية الخليجية",
        )

        # ==================================================
        # Korean Variants
        # ==================================================

        # South Korean
        self.register_dialect(
            locale=LocaleInfo.parse("ko-KR"),
            base="ko",
            name="South Korean",
            native_name="한국어 (대한민국)",
        )

        # North Korean
        self.register_dialect(
            locale=LocaleInfo.parse("ko-KP"),
            base="ko",
            name="North Korean",
            native_name="조선말",
            vocabulary={
                "컴퓨터": "콤퓨터",
                "아이스크림": "얼음과자",
            },
        )

    def register_dialect(
        self,
        locale: LocaleInfo | str,
        base: str,
        name: str,
        native_name: str,
        overrides: dict[str, str] | None = None,
        vocabulary: dict[str, str] | None = None,
        spelling_rules: list[tuple[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DialectDefinition:
        """Register a new dialect.

        Args:
            locale: Locale identifier
            base: Base language code
            name: Human-readable name
            native_name: Name in native language
            overrides: Message key overrides
            vocabulary: Vocabulary mappings
            spelling_rules: Spelling transformation rules
            metadata: Additional metadata

        Returns:
            Created DialectDefinition
        """
        if isinstance(locale, str):
            locale = LocaleInfo.parse(locale)

        dialect = DialectDefinition(
            locale=locale,
            base_language=base,
            name=name,
            native_name=native_name,
            overrides=overrides or {},
            vocabulary=vocabulary or {},
            spelling_rules=spelling_rules or [],
            metadata=metadata or {},
        )

        self._dialects[locale.tag] = dialect
        return dialect

    def get_dialect(self, locale: LocaleInfo | str) -> DialectDefinition | None:
        """Get a dialect by locale.

        Args:
            locale: Locale to look up

        Returns:
            DialectDefinition if found, None otherwise
        """
        if isinstance(locale, str):
            locale = LocaleInfo.parse(locale)

        return self._dialects.get(locale.tag)

    def get_fallback_chain(self, locale: LocaleInfo | str) -> list[str]:
        """Get the fallback chain for a locale.

        Args:
            locale: Starting locale

        Returns:
            List of locale tags from most specific to most general
        """
        if isinstance(locale, str):
            locale = LocaleInfo.parse(locale)

        chain = []
        seen = set()

        current_tag = locale.tag
        while current_tag and current_tag not in seen:
            chain.append(current_tag)
            seen.add(current_tag)

            dialect = self._dialects.get(current_tag)
            if dialect:
                current_tag = dialect.base_language
            else:
                # Try removing specificity
                current_locale = LocaleInfo.parse(current_tag)
                if current_locale.region:
                    current_tag = current_locale.language
                elif current_locale.script:
                    current_tag = current_locale.language
                else:
                    break

        # Ensure base language is always in chain
        base_lang = locale.language
        if base_lang not in chain:
            chain.append(base_lang)

        return chain

    def list_dialects(self, base_language: str | None = None) -> list[DialectDefinition]:
        """List all registered dialects.

        Args:
            base_language: Optional filter by base language

        Returns:
            List of DialectDefinition
        """
        dialects = list(self._dialects.values())
        if base_language:
            dialects = [d for d in dialects if d.base_language == base_language or d.locale.language == base_language]
        return dialects

    def __iter__(self) -> Iterator[DialectDefinition]:
        """Iterate over all dialects."""
        return iter(self._dialects.values())

    def __len__(self) -> int:
        """Return number of registered dialects."""
        return len(self._dialects)


class DialectResolver:
    """Resolver for dialect-aware message lookup.

    Resolves messages by walking the dialect fallback chain.

    Example:
        resolver = DialectResolver(registry)

        # Resolve with fallback
        message = resolver.resolve(
            key="null.values_found",
            locale=LocaleInfo.parse("en-GB"),
            catalogs={"en": en_messages, "en-GB": gb_messages},
        )
    """

    def __init__(
        self,
        registry: DialectRegistry | None = None,
    ) -> None:
        """Initialize resolver.

        Args:
            registry: Dialect registry (creates default if not provided)
        """
        self.registry = registry or DialectRegistry()

    def resolve(
        self,
        key: str,
        locale: LocaleInfo | str,
        catalogs: dict[str, dict[str, str]],
        apply_transformations: bool = True,
    ) -> str | None:
        """Resolve a message key with dialect fallback.

        Args:
            key: Message key
            locale: Target locale
            catalogs: Available message catalogs
            apply_transformations: Apply spelling/vocabulary transformations

        Returns:
            Resolved message or None if not found
        """
        if isinstance(locale, str):
            locale = LocaleInfo.parse(locale)

        # Get fallback chain
        chain = self.registry.get_fallback_chain(locale)

        # Walk the chain looking for the key
        for tag in chain:
            # Check dialect overrides first
            dialect = self.registry.get_dialect(tag)
            if dialect and key in dialect.overrides:
                message = dialect.overrides[key]
                if apply_transformations:
                    message = dialect.apply_spelling_rules(message)
                    message = dialect.apply_vocabulary(message)
                return message

            # Check catalog
            if tag in catalogs and key in catalogs[tag]:
                message = catalogs[tag][key]
                if apply_transformations and dialect:
                    message = dialect.apply_spelling_rules(message)
                    message = dialect.apply_vocabulary(message)
                return message

        return None

    def resolve_all(
        self,
        locale: LocaleInfo | str,
        catalogs: dict[str, dict[str, str]],
    ) -> dict[str, str]:
        """Resolve all messages for a locale with dialect fallback.

        Args:
            locale: Target locale
            catalogs: Available message catalogs

        Returns:
            Complete message dictionary
        """
        if isinstance(locale, str):
            locale = LocaleInfo.parse(locale)

        result: dict[str, str] = {}
        chain = self.registry.get_fallback_chain(locale)

        # Start from most general and work up to most specific
        for tag in reversed(chain):
            # Add catalog messages
            if tag in catalogs:
                result.update(catalogs[tag])

            # Add dialect overrides
            dialect = self.registry.get_dialect(tag)
            if dialect:
                result.update(dialect.overrides)

        return result

    def get_supported_locales(self) -> list[LocaleInfo]:
        """Get all locales with dialect definitions.

        Returns:
            List of supported LocaleInfo
        """
        return [d.locale for d in self.registry]


# Global registry instance
_dialect_registry = DialectRegistry()


def get_dialect_registry() -> DialectRegistry:
    """Get the global dialect registry.

    Returns:
        DialectRegistry instance
    """
    return _dialect_registry


def register_dialect(
    locale: LocaleInfo | str,
    base: str,
    name: str,
    native_name: str,
    overrides: dict[str, str] | None = None,
    vocabulary: dict[str, str] | None = None,
    spelling_rules: list[tuple[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DialectDefinition:
    """Register a dialect in the global registry.

    Args:
        locale: Locale identifier
        base: Base language code
        name: Human-readable name
        native_name: Name in native language
        overrides: Message key overrides
        vocabulary: Vocabulary mappings
        spelling_rules: Spelling transformation rules
        metadata: Additional metadata

    Returns:
        Created DialectDefinition
    """
    return _dialect_registry.register_dialect(
        locale=locale,
        base=base,
        name=name,
        native_name=native_name,
        overrides=overrides,
        vocabulary=vocabulary,
        spelling_rules=spelling_rules,
        metadata=metadata,
    )


def get_fallback_chain(locale: LocaleInfo | str) -> list[str]:
    """Get the fallback chain for a locale.

    Args:
        locale: Target locale

    Returns:
        List of locale tags in fallback order
    """
    return _dialect_registry.get_fallback_chain(locale)


def create_dialect(
    locale: str,
    base: str,
    name: str,
    native_name: str | None = None,
) -> DialectDefinition:
    """Create a new dialect definition (convenience function).

    Args:
        locale: Locale tag string
        base: Base language
        name: Display name
        native_name: Native display name

    Returns:
        DialectDefinition
    """
    return DialectDefinition(
        locale=LocaleInfo.parse(locale),
        base_language=base,
        name=name,
        native_name=native_name or name,
    )
