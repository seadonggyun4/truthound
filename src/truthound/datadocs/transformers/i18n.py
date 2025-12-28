"""Internationalization transformer for report generation.

This module provides transformers for translating report content
into different languages using message catalogs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from truthound.datadocs.engine.context import (
    ReportContext,
    ReportData,
    TranslatableString,
)
from truthound.datadocs.transformers.base import BaseTransformer


@runtime_checkable
class MessageCatalog(Protocol):
    """Protocol for message catalogs.

    A message catalog provides translations for a specific locale.
    """

    @property
    def locale(self) -> str:
        """Get the locale code for this catalog."""
        ...

    def get(
        self,
        key: str,
        default: str | None = None,
        **params: Any,
    ) -> str:
        """Get a translated message.

        Args:
            key: Message key.
            default: Default value if not found.
            **params: Parameters for formatting.

        Returns:
            Translated and formatted message.
        """
        ...

    def has(self, key: str) -> bool:
        """Check if a message key exists.

        Args:
            key: Message key.

        Returns:
            True if the key exists.
        """
        ...


@dataclass
class SimpleMessageCatalog:
    """Simple in-memory message catalog.

    This is a basic implementation of the MessageCatalog protocol.
    For production use, consider using the full i18n system.

    Example:
        catalog = SimpleMessageCatalog(
            locale="ko",
            messages={
                "report.title": "데이터 품질 리포트",
                "report.quality_score": "품질 점수: {score}점",
            }
        )
    """
    locale: str
    messages: dict[str, str] = field(default_factory=dict)
    fallback: "SimpleMessageCatalog | None" = None

    def get(
        self,
        key: str,
        default: str | None = None,
        **params: Any,
    ) -> str:
        """Get a translated message.

        Args:
            key: Message key.
            default: Default value if not found.
            **params: Parameters for formatting.

        Returns:
            Translated and formatted message.
        """
        template = self.messages.get(key)

        if template is None and self.fallback:
            return self.fallback.get(key, default, **params)

        if template is None:
            return default or key

        try:
            return template.format(**params)
        except KeyError:
            return template

    def has(self, key: str) -> bool:
        """Check if a message key exists."""
        if key in self.messages:
            return True
        if self.fallback:
            return self.fallback.has(key)
        return False

    def merge(self, other: "SimpleMessageCatalog") -> "SimpleMessageCatalog":
        """Merge with another catalog (other takes precedence).

        Args:
            other: Catalog to merge with.

        Returns:
            New merged catalog.
        """
        return SimpleMessageCatalog(
            locale=self.locale,
            messages={**self.messages, **other.messages},
            fallback=self.fallback,
        )


class TranslationResolver:
    """Resolves translations across multiple catalogs with fallback.

    The resolver tries catalogs in order:
    1. Exact locale match (e.g., "ko-KR")
    2. Language match (e.g., "ko")
    3. Default locale (e.g., "en")

    Example:
        resolver = TranslationResolver()
        resolver.add_catalog(korean_catalog)
        resolver.add_catalog(english_catalog, default=True)

        message = resolver.resolve("report.title", "ko-KR")
    """

    def __init__(self, default_locale: str = "en") -> None:
        """Initialize the resolver.

        Args:
            default_locale: Default locale to use as fallback.
        """
        self._catalogs: dict[str, MessageCatalog] = {}
        self._default_locale = default_locale

    def add_catalog(
        self,
        catalog: MessageCatalog,
        default: bool = False,
    ) -> "TranslationResolver":
        """Add a message catalog.

        Args:
            catalog: Catalog to add.
            default: If True, use this as the default locale.

        Returns:
            Self for chaining.
        """
        self._catalogs[catalog.locale] = catalog
        if default:
            self._default_locale = catalog.locale
        return self

    def resolve(
        self,
        key: str,
        locale: str,
        default: str | None = None,
        **params: Any,
    ) -> str:
        """Resolve a translation.

        Args:
            key: Message key.
            locale: Target locale.
            default: Default value if not found.
            **params: Format parameters.

        Returns:
            Resolved message.
        """
        # Try exact locale
        if locale in self._catalogs:
            result = self._catalogs[locale].get(key, None, **params)
            if result != key:  # Found a translation
                return result

        # Try language only (e.g., "ko" from "ko-KR")
        lang = locale.split("-")[0].split("_")[0]
        if lang in self._catalogs:
            result = self._catalogs[lang].get(key, None, **params)
            if result != key:
                return result

        # Try default locale
        if self._default_locale in self._catalogs:
            return self._catalogs[self._default_locale].get(
                key, default or key, **params
            )

        return default or key

    def get_catalog(self, locale: str) -> MessageCatalog | None:
        """Get a catalog by locale.

        Args:
            locale: Locale code.

        Returns:
            Catalog if found, None otherwise.
        """
        return self._catalogs.get(locale)

    def list_locales(self) -> list[str]:
        """List available locales.

        Returns:
            List of locale codes.
        """
        return list(self._catalogs.keys())


class I18nTransformer(BaseTransformer):
    """Internationalization transformer.

    This transformer processes the report context and translates
    all TranslatableString instances using the configured catalogs.

    Example:
        # With a simple catalog
        catalog = SimpleMessageCatalog(
            locale="ko",
            messages={"report.title": "데이터 품질 리포트"}
        )
        transformer = I18nTransformer(catalog=catalog)

        # With a resolver for multiple locales
        resolver = TranslationResolver()
        resolver.add_catalog(korean_catalog)
        resolver.add_catalog(english_catalog, default=True)
        transformer = I18nTransformer(resolver=resolver)
    """

    def __init__(
        self,
        catalog: MessageCatalog | None = None,
        resolver: TranslationResolver | None = None,
        locale: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the i18n transformer.

        Args:
            catalog: Single message catalog to use.
            resolver: Translation resolver for multiple catalogs.
            locale: Override locale (uses context locale if None).
            name: Optional transformer name.
        """
        super().__init__(name=name or "I18nTransformer")
        self._catalog = catalog
        self._resolver = resolver
        self._override_locale = locale

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        """Apply internationalization to the context.

        Args:
            ctx: Input context.

        Returns:
            Transformed context with translations applied.
        """
        locale = self._override_locale or ctx.locale

        # Update locale in context if overridden
        if self._override_locale and ctx.locale != self._override_locale:
            ctx = ctx.with_locale(self._override_locale)

        # Translate the data
        translated_data = self._translate_data(ctx.data, locale)

        return ctx.with_data(translated_data)

    def _translate_data(self, data: ReportData, locale: str) -> ReportData:
        """Translate all translatable content in the data.

        Args:
            data: Report data to translate.
            locale: Target locale.

        Returns:
            New ReportData with translations applied.
        """
        # Translate metadata
        translated_metadata = self._translate_dict(data.metadata, locale)

        # Translate sections
        translated_sections = {}
        for name, section in data.sections.items():
            translated_sections[name] = self._translate_dict(section, locale)

        # Translate alerts
        translated_alerts = [
            self._translate_dict(alert, locale)
            for alert in data.alerts
        ]

        # Translate recommendations
        translated_recommendations = [
            self._translate_value(rec, locale)
            for rec in data.recommendations
        ]

        return ReportData(
            raw=data.raw,  # Keep raw data unchanged
            sections=translated_sections,
            metadata=translated_metadata,
            alerts=translated_alerts,
            recommendations=translated_recommendations,
            charts=data.charts,  # Charts are typically not translated
            tables=data.tables,
        )

    def _translate_dict(self, d: dict[str, Any], locale: str) -> dict[str, Any]:
        """Recursively translate a dictionary.

        Args:
            d: Dictionary to translate.
            locale: Target locale.

        Returns:
            New dictionary with translations applied.
        """
        result = {}
        for key, value in d.items():
            result[key] = self._translate_value(value, locale)
        return result

    def _translate_value(self, value: Any, locale: str) -> Any:
        """Translate a single value.

        Args:
            value: Value to translate.
            locale: Target locale.

        Returns:
            Translated value.
        """
        if isinstance(value, TranslatableString):
            return self._get_translation(value.key, locale, value.params, value.default)

        if isinstance(value, dict):
            return self._translate_dict(value, locale)

        if isinstance(value, list):
            return [self._translate_value(v, locale) for v in value]

        if isinstance(value, tuple):
            return tuple(self._translate_value(v, locale) for v in value)

        return value

    def _get_translation(
        self,
        key: str,
        locale: str,
        params: dict[str, Any],
        default: str | None,
    ) -> str:
        """Get a translation from catalog or resolver.

        Args:
            key: Message key.
            locale: Target locale.
            params: Format parameters.
            default: Default value.

        Returns:
            Translated message.
        """
        if self._resolver:
            return self._resolver.resolve(key, locale, default, **params)

        if self._catalog:
            return self._catalog.get(key, default, **params)

        return default or key


def load_catalog_from_yaml(path: Path, locale: str) -> SimpleMessageCatalog:
    """Load a message catalog from a YAML file.

    Args:
        path: Path to YAML file.
        locale: Locale code for this catalog.

    Returns:
        Loaded catalog.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML catalog loading")

    with open(path, "r", encoding="utf-8") as f:
        messages = yaml.safe_load(f) or {}

    # Flatten nested structure
    flat_messages = _flatten_dict(messages)

    return SimpleMessageCatalog(locale=locale, messages=flat_messages)


def load_catalog_from_json(path: Path, locale: str) -> SimpleMessageCatalog:
    """Load a message catalog from a JSON file.

    Args:
        path: Path to JSON file.
        locale: Locale code for this catalog.

    Returns:
        Loaded catalog.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Flatten nested structure
    flat_messages = _flatten_dict(messages)

    return SimpleMessageCatalog(locale=locale, messages=flat_messages)


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested dictionary into dot-notation keys.

    Args:
        d: Dictionary to flatten.
        prefix: Current key prefix.

    Returns:
        Flattened dictionary with string values.
    """
    result = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, full_key))
        else:
            result[full_key] = str(value)
    return result
