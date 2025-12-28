"""Dynamic Catalog Loading and Context-Based Messages.

This module provides lazy loading of message catalogs and context-aware
message resolution for enterprise i18n needs.

Features:
- Lazy loading (only load catalogs when needed)
- Multiple storage backends (filesystem, package resources, URL)
- Caching with TTL and LRU eviction
- Namespace support for modular catalogs
- Context-based message selection
- Message inheritance and composition

Usage:
    from truthound.validators.i18n.loader import (
        CatalogManager,
        MessageContext,
        ContextResolver,
    )

    # Create manager with lazy loading
    manager = CatalogManager(
        base_path=Path("locales"),
        lazy=True,
    )

    # Load catalog on demand
    catalog = manager.get_catalog(LocaleInfo.parse("ko"))

    # Context-based resolution
    resolver = ContextResolver(manager)
    message = resolver.resolve(
        key="validation.failed",
        locale=LocaleInfo.parse("ko"),
        context=MessageContext.FORMAL,
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

from truthound.validators.i18n.protocols import (
    BaseCatalogLoader,
    LocaleInfo,
    MessageContext,
    MessageResolver,
    PluralizedMessage,
    ResolvedMessage,
)
from truthound.validators.i18n.plural import get_plural_category, PluralCategory


logger = logging.getLogger(__name__)


# ==============================================================================
# Cache Implementation
# ==============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata.

    Attributes:
        value: Cached value
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        ttl: Time-to-live in seconds
    """
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    ttl: float = 3600.0  # 1 hour default

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update last access time."""
        self.accessed_at = time.time()


class LRUCache:
    """Thread-safe LRU cache with TTL support.

    Implements a Least Recently Used cache with optional
    time-based expiration for entries.
    """

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: float = 3600.0,
    ) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (default from constructor)
        """
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                ttl=ttl or self.default_ttl,
            )

    def delete(self, key: str) -> None:
        """Delete entry from cache.

        Args:
            key: Cache key
        """
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired]
            for key in expired:
                del self._cache[key]
            return len(expired)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired


# ==============================================================================
# Catalog Storage Backends
# ==============================================================================

class CatalogStorageBackend:
    """Base class for catalog storage backends."""

    def load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        """Load catalog from storage.

        Args:
            locale: Target locale
            namespace: Optional namespace

        Returns:
            Message dictionary
        """
        raise NotImplementedError

    def save(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
        namespace: str | None = None,
    ) -> None:
        """Save catalog to storage.

        Args:
            locale: Target locale
            catalog: Message dictionary
            namespace: Optional namespace
        """
        raise NotImplementedError

    def exists(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> bool:
        """Check if catalog exists.

        Args:
            locale: Target locale
            namespace: Optional namespace

        Returns:
            True if catalog exists
        """
        raise NotImplementedError

    def list_locales(self, namespace: str | None = None) -> list[LocaleInfo]:
        """List available locales.

        Args:
            namespace: Optional namespace filter

        Returns:
            List of available locales
        """
        raise NotImplementedError


class FileSystemStorage(CatalogStorageBackend):
    """Filesystem-based catalog storage.

    Stores catalogs as JSON files in a directory structure:
    base_path/
      en/
        validators.json
        errors.json
      ko/
        validators.json
        errors.json
    """

    def __init__(
        self,
        base_path: Path | str,
        file_extension: str = ".json",
        encoding: str = "utf-8",
    ) -> None:
        """Initialize filesystem storage.

        Args:
            base_path: Base directory for catalogs
            file_extension: File extension
            encoding: File encoding
        """
        self.base_path = Path(base_path)
        self.file_extension = file_extension
        self.encoding = encoding

    def _get_path(self, locale: LocaleInfo, namespace: str | None) -> Path:
        """Get path for a catalog file."""
        locale_dir = self.base_path / locale.tag.replace("-", "_")
        filename = f"{namespace or 'messages'}{self.file_extension}"
        return locale_dir / filename

    def load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        path = self._get_path(locale, namespace)

        if not path.exists():
            # Try fallback paths
            fallback_paths = [
                self.base_path / locale.language / f"{namespace or 'messages'}{self.file_extension}",
                self.base_path / f"{locale.language}{self.file_extension}",
            ]
            for fallback in fallback_paths:
                if fallback.exists():
                    path = fallback
                    break
            else:
                return {}

        try:
            with open(path, "r", encoding=self.encoding) as f:
                data = json.load(f)

            # Handle nested JSON structure
            if "messages" in data:
                return data["messages"]
            return data

        except Exception as e:
            logger.error(f"Failed to load catalog from {path}: {e}")
            return {}

    def save(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
        namespace: str | None = None,
    ) -> None:
        path = self._get_path(locale, namespace)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding=self.encoding) as f:
                json.dump(
                    {"locale": locale.tag, "messages": catalog},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save catalog to {path}: {e}")

    def exists(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> bool:
        path = self._get_path(locale, namespace)
        return path.exists()

    def list_locales(self, namespace: str | None = None) -> list[LocaleInfo]:
        if not self.base_path.exists():
            return []

        locales = []
        for item in self.base_path.iterdir():
            if item.is_dir():
                locale_tag = item.name.replace("_", "-")
                try:
                    locale = LocaleInfo.parse(locale_tag)
                    if namespace is None or self.exists(locale, namespace):
                        locales.append(locale)
                except Exception:
                    continue

        return locales


class MemoryStorage(CatalogStorageBackend):
    """In-memory catalog storage for testing."""

    def __init__(self) -> None:
        self._catalogs: dict[str, dict[str, str]] = {}

    def _key(self, locale: LocaleInfo, namespace: str | None) -> str:
        return f"{locale.tag}:{namespace or 'default'}"

    def load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        return self._catalogs.get(self._key(locale, namespace), {})

    def save(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
        namespace: str | None = None,
    ) -> None:
        self._catalogs[self._key(locale, namespace)] = catalog.copy()

    def exists(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> bool:
        return self._key(locale, namespace) in self._catalogs

    def list_locales(self, namespace: str | None = None) -> list[LocaleInfo]:
        locales = []
        ns = namespace or "default"
        for key in self._catalogs:
            parts = key.split(":")
            if len(parts) == 2 and parts[1] == ns:
                try:
                    locales.append(LocaleInfo.parse(parts[0]))
                except Exception:
                    continue
        return locales


# ==============================================================================
# Catalog Manager
# ==============================================================================

class CatalogManager(BaseCatalogLoader):
    """Manager for message catalog loading with lazy loading support.

    Features:
    - Lazy loading of catalogs on first access
    - Caching with configurable TTL
    - Multiple storage backend support
    - Namespace support for modular catalogs
    - Automatic fallback chain resolution

    Example:
        manager = CatalogManager(
            base_path=Path("locales"),
            lazy=True,
            cache_ttl=3600,
        )

        # First access loads the catalog
        catalog = manager.get_catalog(LocaleInfo.parse("ko"))

        # Subsequent accesses use cache
        catalog = manager.get_catalog(LocaleInfo.parse("ko"))

        # Preload all catalogs
        manager.preload(["en", "ko", "ja"])
    """

    def __init__(
        self,
        base_path: Path | str | None = None,
        storage: CatalogStorageBackend | None = None,
        lazy: bool = True,
        cache_ttl: float = 3600.0,
        cache_size: int = 50,
        fallback_locale: str = "en",
    ) -> None:
        """Initialize catalog manager.

        Args:
            base_path: Path for filesystem storage
            storage: Custom storage backend
            lazy: Enable lazy loading
            cache_ttl: Cache TTL in seconds
            cache_size: Maximum cached catalogs
            fallback_locale: Fallback locale code
        """
        super().__init__()

        if storage:
            self.storage = storage
        elif base_path:
            self.storage = FileSystemStorage(base_path)
        else:
            self.storage = MemoryStorage()

        self.lazy = lazy
        self.fallback_locale = LocaleInfo.parse(fallback_locale)
        self._cache = LRUCache(max_size=cache_size, default_ttl=cache_ttl)
        self._loading_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def _get_loading_lock(self, key: str) -> threading.Lock:
        """Get or create a loading lock for a catalog."""
        with self._lock:
            if key not in self._loading_locks:
                self._loading_locks[key] = threading.Lock()
            return self._loading_locks[key]

    def _do_load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        """Load catalog from storage."""
        return self.storage.load(locale, namespace)

    def get_catalog(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
        with_fallback: bool = True,
    ) -> dict[str, str]:
        """Get catalog for a locale.

        Args:
            locale: Target locale
            namespace: Optional namespace
            with_fallback: Include fallback locale messages

        Returns:
            Message dictionary
        """
        cache_key = f"{locale.tag}:{namespace or 'default'}"

        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Load with lock to prevent duplicate loads
        lock = self._get_loading_lock(cache_key)
        with lock:
            # Double-check after acquiring lock
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            # Load catalog
            catalog = self._do_load(locale, namespace)

            # Apply fallback if needed
            if with_fallback and locale != self.fallback_locale:
                fallback = self._do_load(self.fallback_locale, namespace)
                catalog = {**fallback, **catalog}

            # Cache and return
            self._cache.set(cache_key, catalog)
            return catalog

    def load(
        self,
        locale: LocaleInfo,
        namespace: str | None = None,
    ) -> dict[str, str]:
        """Load catalog (implementation of BaseCatalogLoader)."""
        return self.get_catalog(locale, namespace)

    def preload(
        self,
        locales: list[str | LocaleInfo],
        namespaces: list[str] | None = None,
    ) -> None:
        """Preload catalogs for multiple locales.

        Args:
            locales: List of locales to preload
            namespaces: Optional list of namespaces
        """
        namespaces = namespaces or [None]  # type: ignore

        for locale in locales:
            if isinstance(locale, str):
                locale = LocaleInfo.parse(locale)
            for namespace in namespaces:
                self.get_catalog(locale, namespace)

    def reload(
        self,
        locale: LocaleInfo | None = None,
        namespace: str | None = None,
    ) -> None:
        """Reload catalogs from storage.

        Args:
            locale: Specific locale to reload (or all if None)
            namespace: Specific namespace to reload (or all if None)
        """
        if locale and namespace:
            cache_key = f"{locale.tag}:{namespace}"
            self._cache.delete(cache_key)
            self.get_catalog(locale, namespace)
        elif locale:
            # Reload all namespaces for locale
            for key in list(self._loaded_catalogs.keys()):
                if key.startswith(locale.tag):
                    self._cache.delete(key)
        else:
            # Reload all
            self._cache.clear()

    def get_available_locales(self) -> list[LocaleInfo]:
        """Get list of available locales."""
        return self.storage.list_locales()

    def add_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
        namespace: str | None = None,
    ) -> None:
        """Add a catalog programmatically.

        Args:
            locale: Target locale
            catalog: Message dictionary
            namespace: Optional namespace
        """
        self.storage.save(locale, catalog, namespace)
        cache_key = f"{locale.tag}:{namespace or 'default'}"
        self._cache.delete(cache_key)


# ==============================================================================
# Context-Based Message Resolution
# ==============================================================================

@dataclass
class ContextualMessage:
    """Message with context variants.

    Attributes:
        key: Message key
        default: Default message template
        contexts: Context-specific variants
    """
    key: str
    default: str
    contexts: dict[MessageContext, str] = field(default_factory=dict)

    def get(self, context: MessageContext | None = None) -> str:
        """Get message for context.

        Args:
            context: Message context

        Returns:
            Appropriate message template
        """
        if context and context in self.contexts:
            return self.contexts[context]
        return self.default


class ContextResolver(MessageResolver):
    """Context-aware message resolver.

    Resolves messages considering:
    - User context (formal, informal, technical, etc.)
    - Pluralization
    - Locale fallback
    - Message inheritance

    Example:
        resolver = ContextResolver(catalog_manager)

        # Resolve with context
        result = resolver.resolve(
            key="greeting",
            locale=LocaleInfo.parse("ko"),
            context=MessageContext.FORMAL,
            name="김철수",
        )
        # -> "김철수님, 안녕하십니까"

        # Resolve with plural
        result = resolver.resolve_plural(
            key="file_count",
            count=5,
            locale=LocaleInfo.parse("ru"),
        )
        # -> "5 файлов"
    """

    def __init__(
        self,
        catalog_manager: CatalogManager,
        context_separator: str = "@",
        plural_separator: str = "#",
    ) -> None:
        """Initialize resolver.

        Args:
            catalog_manager: Catalog manager instance
            context_separator: Separator for context keys (key@formal)
            plural_separator: Separator for plural keys (key#one)
        """
        self.catalog_manager = catalog_manager
        self.context_separator = context_separator
        self.plural_separator = plural_separator

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
        catalog = self.catalog_manager.get_catalog(locale)
        fallback_used = False

        # Try context-specific key first
        if context:
            context_key = f"{key}{self.context_separator}{context.value}"
            if context_key in catalog:
                template = catalog[context_key]
            elif key in catalog:
                template = catalog[key]
                fallback_used = True
            else:
                template = f"[{key}]"
                fallback_used = True
        else:
            template = catalog.get(key, f"[{key}]")
            fallback_used = key not in catalog

        # Format message
        try:
            message = template.format(**params)
        except KeyError as e:
            message = f"{template} (missing: {e})"

        return ResolvedMessage(
            key=key,
            message=message,
            locale=locale,
            context=context,
            fallback=fallback_used,
        )

    def resolve_plural(
        self,
        key: str,
        count: float | int,
        locale: LocaleInfo,
        context: MessageContext | None = None,
        **params: Any,
    ) -> PluralizedMessage:
        """Resolve a pluralized message.

        Looks for plural forms using the pattern:
        - key#zero, key#one, key#two, key#few, key#many, key#other

        Args:
            key: Base message key
            count: Number for pluralization
            locale: Target locale
            context: Message context
            **params: Additional format parameters

        Returns:
            Pluralized message
        """
        catalog = self.catalog_manager.get_catalog(locale)
        category = get_plural_category(count, locale)

        # Build key variants
        if context:
            base_key = f"{key}{self.context_separator}{context.value}"
        else:
            base_key = key

        # Try plural forms
        plural_key = f"{base_key}{self.plural_separator}{category.value}"
        if plural_key in catalog:
            template = catalog[plural_key]
        elif f"{base_key}{self.plural_separator}other" in catalog:
            template = catalog[f"{base_key}{self.plural_separator}other"]
        elif base_key in catalog:
            template = catalog[base_key]
        else:
            template = f"[{key}]"

        # Format with count
        try:
            message = template.format(count=count, **params)
        except KeyError as e:
            message = f"{template} (missing: {e})"

        return PluralizedMessage(
            message=message,
            count=count,
            category=category,
        )

    def get_contextual_message(
        self,
        key: str,
        locale: LocaleInfo,
    ) -> ContextualMessage:
        """Get all context variants for a message.

        Args:
            key: Message key
            locale: Target locale

        Returns:
            ContextualMessage with all variants
        """
        catalog = self.catalog_manager.get_catalog(locale)

        # Get default
        default = catalog.get(key, f"[{key}]")

        # Get context variants
        contexts: dict[MessageContext, str] = {}
        for ctx in MessageContext:
            context_key = f"{key}{self.context_separator}{ctx.value}"
            if context_key in catalog:
                contexts[ctx] = catalog[context_key]

        return ContextualMessage(key=key, default=default, contexts=contexts)


# ==============================================================================
# Message Composition
# ==============================================================================

class MessageComposer:
    """Compose messages from multiple sources.

    Supports message inheritance and composition patterns.

    Example:
        composer = MessageComposer(resolver)

        # Compose from parts
        message = composer.compose(
            template="Result: {status} - {details}",
            parts={
                "status": ("status.success", {}),
                "details": ("details.count", {"count": 5}),
            },
            locale=LocaleInfo.parse("en"),
        )
    """

    def __init__(self, resolver: ContextResolver) -> None:
        """Initialize composer.

        Args:
            resolver: Context resolver instance
        """
        self.resolver = resolver

    def compose(
        self,
        template: str,
        parts: dict[str, tuple[str, dict[str, Any]]],
        locale: LocaleInfo,
        context: MessageContext | None = None,
    ) -> str:
        """Compose a message from parts.

        Args:
            template: Template with {placeholders}
            parts: Dictionary of placeholder -> (key, params) tuples
            locale: Target locale
            context: Message context

        Returns:
            Composed message
        """
        resolved_parts = {}
        for name, (key, params) in parts.items():
            result = self.resolver.resolve(key, locale, context, **params)
            resolved_parts[name] = result.message

        try:
            return template.format(**resolved_parts)
        except KeyError as e:
            return f"{template} (missing part: {e})"

    def compose_list(
        self,
        items: list[tuple[str, dict[str, Any]]],
        locale: LocaleInfo,
        separator: str = ", ",
        final_separator: str | None = None,
        context: MessageContext | None = None,
    ) -> str:
        """Compose a list of messages.

        Args:
            items: List of (key, params) tuples
            locale: Target locale
            separator: Item separator
            final_separator: Separator before last item (e.g., " and ")
            context: Message context

        Returns:
            Composed list string
        """
        resolved = []
        for key, params in items:
            result = self.resolver.resolve(key, locale, context, **params)
            resolved.append(result.message)

        if not resolved:
            return ""

        if len(resolved) == 1:
            return resolved[0]

        if final_separator:
            return f"{separator.join(resolved[:-1])}{final_separator}{resolved[-1]}"
        return separator.join(resolved)


# ==============================================================================
# Global Instances and Factory Functions
# ==============================================================================

_catalog_manager: CatalogManager | None = None
_context_resolver: ContextResolver | None = None


def get_catalog_manager(
    base_path: Path | str | None = None,
    **kwargs: Any,
) -> CatalogManager:
    """Get or create the global catalog manager.

    Args:
        base_path: Path for filesystem storage
        **kwargs: Additional configuration

    Returns:
        CatalogManager instance
    """
    global _catalog_manager

    if _catalog_manager is None:
        _catalog_manager = CatalogManager(base_path=base_path, **kwargs)

    return _catalog_manager


def get_context_resolver() -> ContextResolver:
    """Get or create the global context resolver.

    Returns:
        ContextResolver instance
    """
    global _context_resolver

    if _context_resolver is None:
        _context_resolver = ContextResolver(get_catalog_manager())

    return _context_resolver


def resolve_message(
    key: str,
    locale: str | LocaleInfo,
    context: MessageContext | None = None,
    **params: Any,
) -> str:
    """Resolve a message (convenience function).

    Args:
        key: Message key
        locale: Target locale
        context: Message context
        **params: Format parameters

    Returns:
        Resolved message string
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)

    resolver = get_context_resolver()
    result = resolver.resolve(key, locale, context, **params)
    return result.message


def resolve_plural_message(
    key: str,
    count: float | int,
    locale: str | LocaleInfo,
    context: MessageContext | None = None,
    **params: Any,
) -> str:
    """Resolve a pluralized message (convenience function).

    Args:
        key: Message key
        count: Number for pluralization
        locale: Target locale
        context: Message context
        **params: Additional format parameters

    Returns:
        Pluralized message string
    """
    if isinstance(locale, str):
        locale = LocaleInfo.parse(locale)

    resolver = get_context_resolver()
    result = resolver.resolve_plural(key, count, locale, context, **params)
    return result.message
