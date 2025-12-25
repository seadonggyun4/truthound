"""Internationalization (i18n) System for Error Messages.

This module provides a comprehensive internationalization system for
error messages with:
- Protocol-based message providers
- Locale management and fallback
- Message catalogs with YAML/JSON support
- Placeholder formatting with type safety
- Error code registry
- Context-aware message resolution

Key features:
- Lazy loading of message catalogs
- Hierarchical locale fallback (ko_KR -> ko -> en)
- Pluralization support
- Named and positional placeholders
- Custom formatters for complex types
- Thread-safe locale management

Example:
    from truthound.profiler.i18n import (
        I18n,
        MessageCode,
        get_message,
        set_locale,
    )

    # Set locale
    set_locale("ko")

    # Get localized message
    msg = get_message(MessageCode.ANALYSIS_FAILED, column="email")
    # -> "분석 실패: email"

    # Or use I18n directly
    i18n = I18n.get_instance()
    msg = i18n.t("error.analysis.failed", column="email")
"""

from __future__ import annotations

import json
import locale
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger("truthound.i18n")


# =============================================================================
# Message Codes (Error Code Registry)
# =============================================================================


class MessageCode(str, Enum):
    """Standardized message codes for all error types.

    Using codes ensures consistency and enables reliable internationalization.
    Format: CATEGORY_SUBCATEGORY_DESCRIPTION

    Categories:
    - ERR: General errors
    - ANALYSIS: Analysis-related errors
    - PATTERN: Pattern matching errors
    - TYPE: Type inference errors
    - IO: Input/output errors
    - MEMORY: Memory errors
    - TIMEOUT: Timeout errors
    - VALIDATION: Validation errors
    - CONFIG: Configuration errors
    - CACHE: Cache-related errors
    """

    # General errors
    ERR_UNKNOWN = "err.unknown"
    ERR_INTERNAL = "err.internal"
    ERR_NOT_IMPLEMENTED = "err.not_implemented"

    # Analysis errors
    ANALYSIS_FAILED = "analysis.failed"
    ANALYSIS_COLUMN_FAILED = "analysis.column_failed"
    ANALYSIS_TABLE_FAILED = "analysis.table_failed"
    ANALYSIS_EMPTY_DATA = "analysis.empty_data"
    ANALYSIS_SKIPPED = "analysis.skipped"

    # Pattern errors
    PATTERN_INVALID = "pattern.invalid"
    PATTERN_NOT_FOUND = "pattern.not_found"
    PATTERN_COMPILE_FAILED = "pattern.compile_failed"
    PATTERN_MATCH_FAILED = "pattern.match_failed"
    PATTERN_TOO_SLOW = "pattern.too_slow"

    # Type inference errors
    TYPE_INFERENCE_FAILED = "type.inference_failed"
    TYPE_AMBIGUOUS = "type.ambiguous"
    TYPE_UNSUPPORTED = "type.unsupported"
    TYPE_CAST_FAILED = "type.cast_failed"

    # IO errors
    IO_FILE_NOT_FOUND = "io.file_not_found"
    IO_PERMISSION_DENIED = "io.permission_denied"
    IO_READ_FAILED = "io.read_failed"
    IO_WRITE_FAILED = "io.write_failed"
    IO_INVALID_FORMAT = "io.invalid_format"
    IO_ENCODING_ERROR = "io.encoding_error"

    # Memory errors
    MEMORY_EXCEEDED = "memory.exceeded"
    MEMORY_ALLOCATION_FAILED = "memory.allocation_failed"
    MEMORY_LIMIT_WARNING = "memory.limit_warning"

    # Timeout errors
    TIMEOUT_EXCEEDED = "timeout.exceeded"
    TIMEOUT_COLUMN = "timeout.column"
    TIMEOUT_OPERATION = "timeout.operation"

    # Validation errors
    VALIDATION_FAILED = "validation.failed"
    VALIDATION_SCHEMA = "validation.schema"
    VALIDATION_CONSTRAINT = "validation.constraint"
    VALIDATION_REQUIRED = "validation.required"
    VALIDATION_TYPE = "validation.type"
    VALIDATION_RANGE = "validation.range"
    VALIDATION_FORMAT = "validation.format"

    # Configuration errors
    CONFIG_INVALID = "config.invalid"
    CONFIG_MISSING = "config.missing"
    CONFIG_TYPE_MISMATCH = "config.type_mismatch"

    # Cache errors
    CACHE_MISS = "cache.miss"
    CACHE_EXPIRED = "cache.expired"
    CACHE_INVALID = "cache.invalid"
    CACHE_CONNECTION_FAILED = "cache.connection_failed"

    # Progress messages (not errors)
    PROGRESS_START = "progress.start"
    PROGRESS_COLUMN = "progress.column"
    PROGRESS_COMPLETE = "progress.complete"
    PROGRESS_FAILED = "progress.failed"

    # Validation messages
    RULE_GENERATED = "rule.generated"
    RULE_SKIPPED = "rule.skipped"
    SUITE_GENERATED = "suite.generated"


# =============================================================================
# Locale Management
# =============================================================================


class LocaleInfo:
    """Information about a locale."""

    def __init__(
        self,
        code: str,
        name: str = "",
        native_name: str = "",
        direction: str = "ltr",
    ):
        """Initialize locale info.

        Args:
            code: Locale code (e.g., "ko_KR", "en_US")
            name: English name
            native_name: Native name
            direction: Text direction ("ltr" or "rtl")
        """
        self.code = code
        self.name = name or code
        self.native_name = native_name or name or code
        self.direction = direction

        # Parse language and region
        parts = code.replace("-", "_").split("_")
        self.language = parts[0].lower()
        self.region = parts[1].upper() if len(parts) > 1 else ""

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"LocaleInfo({self.code!r})"

    @classmethod
    def from_system(cls) -> "LocaleInfo":
        """Create LocaleInfo from system locale."""
        try:
            # Try to get locale from environment
            system_locale = os.environ.get("LANG", "").split(".")[0]
            if not system_locale:
                system_locale = os.environ.get("LC_ALL", "").split(".")[0]
            if not system_locale:
                # Fallback to locale module
                try:
                    system_locale = locale.getlocale()[0] or "en_US"
                except Exception:
                    system_locale = "en_US"
        except Exception:
            system_locale = "en_US"
        return cls(system_locale or "en_US")


# Built-in locale definitions
BUILTIN_LOCALES: dict[str, LocaleInfo] = {
    "en": LocaleInfo("en", "English", "English"),
    "en_US": LocaleInfo("en_US", "English (US)", "English (US)"),
    "en_GB": LocaleInfo("en_GB", "English (UK)", "English (UK)"),
    "ko": LocaleInfo("ko", "Korean", "한국어"),
    "ko_KR": LocaleInfo("ko_KR", "Korean (Korea)", "한국어 (대한민국)"),
    "ja": LocaleInfo("ja", "Japanese", "日本語"),
    "ja_JP": LocaleInfo("ja_JP", "Japanese (Japan)", "日本語 (日本)"),
    "zh": LocaleInfo("zh", "Chinese", "中文"),
    "zh_CN": LocaleInfo("zh_CN", "Chinese (Simplified)", "简体中文"),
    "zh_TW": LocaleInfo("zh_TW", "Chinese (Traditional)", "繁體中文"),
    "de": LocaleInfo("de", "German", "Deutsch"),
    "de_DE": LocaleInfo("de_DE", "German (Germany)", "Deutsch (Deutschland)"),
    "fr": LocaleInfo("fr", "French", "Français"),
    "fr_FR": LocaleInfo("fr_FR", "French (France)", "Français (France)"),
    "es": LocaleInfo("es", "Spanish", "Español"),
    "es_ES": LocaleInfo("es_ES", "Spanish (Spain)", "Español (España)"),
}


class LocaleManager:
    """Thread-safe locale management.

    Manages the current locale and provides fallback chain resolution.
    """

    def __init__(
        self,
        default_locale: str = "en",
        fallback_locale: str = "en",
    ):
        """Initialize locale manager.

        Args:
            default_locale: Default locale code
            fallback_locale: Ultimate fallback locale
        """
        self._default = default_locale
        self._fallback = fallback_locale
        self._current = threading.local()
        self._locales: dict[str, LocaleInfo] = dict(BUILTIN_LOCALES)
        self._lock = threading.RLock()

    @property
    def current(self) -> str:
        """Get current locale for this thread."""
        return getattr(self._current, "locale", self._default)

    @current.setter
    def current(self, value: str) -> None:
        """Set current locale for this thread."""
        self._current.locale = value

    @property
    def default(self) -> str:
        """Get default locale."""
        return self._default

    @property
    def fallback(self) -> str:
        """Get fallback locale."""
        return self._fallback

    def set_locale(self, locale_code: str) -> None:
        """Set the current locale.

        Args:
            locale_code: Locale code (e.g., "ko", "ko_KR")
        """
        self.current = locale_code

    def get_locale(self) -> str:
        """Get the current locale."""
        return self.current

    def get_locale_info(self, locale_code: str | None = None) -> LocaleInfo:
        """Get locale info.

        Args:
            locale_code: Locale code (default: current)

        Returns:
            LocaleInfo for the locale
        """
        code = locale_code or self.current
        if code in self._locales:
            return self._locales[code]
        return LocaleInfo(code)

    def get_fallback_chain(self, locale_code: str | None = None) -> list[str]:
        """Get the fallback chain for a locale.

        The chain goes from specific to general:
        ko_KR -> ko -> en (fallback)

        Args:
            locale_code: Starting locale (default: current)

        Returns:
            List of locale codes to try in order
        """
        code = locale_code or self.current
        info = self.get_locale_info(code)

        chain = [code]

        # Add language without region if different
        if info.region and info.language not in chain:
            chain.append(info.language)

        # Add default if different
        if self._default not in chain:
            chain.append(self._default)

        # Add fallback if different
        if self._fallback not in chain:
            chain.append(self._fallback)

        return chain

    def register_locale(self, locale_info: LocaleInfo) -> None:
        """Register a new locale.

        Args:
            locale_info: Locale information
        """
        with self._lock:
            self._locales[locale_info.code] = locale_info

    def list_locales(self) -> list[str]:
        """List all registered locales."""
        return sorted(self._locales.keys())


# Global locale manager instance
_locale_manager = LocaleManager()


def set_locale(locale_code: str) -> None:
    """Set the current locale globally."""
    _locale_manager.set_locale(locale_code)


def get_locale() -> str:
    """Get the current locale."""
    return _locale_manager.get_locale()


# =============================================================================
# Message Catalog
# =============================================================================


@dataclass
class MessageEntry:
    """A single message entry in the catalog.

    Supports simple strings or complex pluralized forms.
    """

    key: str
    value: str | dict[str, str]  # String or {zero, one, other, ...}
    description: str = ""
    placeholders: tuple[str, ...] = ()

    def get_form(self, count: int | None = None) -> str:
        """Get the appropriate form based on count.

        Args:
            count: Count for pluralization

        Returns:
            Message string
        """
        if isinstance(self.value, str):
            return self.value

        # Pluralization
        if count is None:
            return self.value.get("other", list(self.value.values())[0])

        if count == 0 and "zero" in self.value:
            return self.value["zero"]
        elif count == 1 and "one" in self.value:
            return self.value["one"]
        elif count == 2 and "two" in self.value:
            return self.value["two"]
        else:
            return self.value.get("other", self.value.get("one", ""))


class MessageCatalog:
    """Catalog of messages for a single locale.

    Messages are organized hierarchically using dot notation:
    error.analysis.failed -> {error: {analysis: {failed: "..."}}}
    """

    def __init__(
        self,
        locale_code: str,
        messages: dict[str, Any] | None = None,
    ):
        """Initialize message catalog.

        Args:
            locale_code: Locale this catalog is for
            messages: Initial messages
        """
        self.locale_code = locale_code
        self._messages: dict[str, MessageEntry] = {}
        self._loaded = False

        if messages:
            self._load_dict(messages)

    def _load_dict(self, data: dict[str, Any], prefix: str = "") -> None:
        """Load messages from a dictionary.

        Args:
            data: Message dictionary
            prefix: Key prefix for nested messages
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Check if it's a pluralized form or nested
                if any(k in value for k in ("zero", "one", "two", "other")):
                    # Pluralized form
                    self._messages[full_key] = MessageEntry(
                        key=full_key,
                        value=value,
                    )
                else:
                    # Nested messages
                    self._load_dict(value, full_key)
            else:
                # Simple string
                self._messages[full_key] = MessageEntry(
                    key=full_key,
                    value=str(value),
                )

    def get(self, key: str, count: int | None = None) -> str | None:
        """Get a message by key.

        Args:
            key: Message key (dot notation)
            count: Count for pluralization

        Returns:
            Message string or None
        """
        entry = self._messages.get(key)
        if entry:
            return entry.get_form(count)
        return None

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._messages

    def keys(self) -> list[str]:
        """Get all message keys."""
        return list(self._messages.keys())

    def __len__(self) -> int:
        return len(self._messages)

    def __contains__(self, key: str) -> bool:
        return self.has(key)


# =============================================================================
# Message Loaders
# =============================================================================


@runtime_checkable
class MessageLoader(Protocol):
    """Protocol for loading message catalogs."""

    def load(self, locale_code: str) -> MessageCatalog | None:
        """Load a message catalog for a locale.

        Args:
            locale_code: Locale to load

        Returns:
            MessageCatalog or None if not found
        """
        ...

    def supports(self, locale_code: str) -> bool:
        """Check if this loader can load a locale.

        Args:
            locale_code: Locale to check

        Returns:
            True if supported
        """
        ...


class DictMessageLoader:
    """Loads messages from a dictionary."""

    def __init__(self, messages: dict[str, dict[str, Any]]):
        """Initialize with messages.

        Args:
            messages: Dict mapping locale codes to message dicts
        """
        self._messages = messages

    def load(self, locale_code: str) -> MessageCatalog | None:
        if locale_code in self._messages:
            return MessageCatalog(locale_code, self._messages[locale_code])
        return None

    def supports(self, locale_code: str) -> bool:
        return locale_code in self._messages


class FileMessageLoader:
    """Loads messages from files (JSON or YAML).

    File naming convention:
    - messages_{locale}.json (e.g., messages_ko.json)
    - messages_{locale}.yaml (e.g., messages_ko.yaml)
    """

    def __init__(
        self,
        directory: str | Path,
        filename_pattern: str = "messages_{locale}",
        extensions: tuple[str, ...] = (".json", ".yaml", ".yml"),
    ):
        """Initialize file loader.

        Args:
            directory: Directory containing message files
            filename_pattern: Filename pattern with {locale} placeholder
            extensions: File extensions to try
        """
        self.directory = Path(directory)
        self.filename_pattern = filename_pattern
        self.extensions = extensions

    def _get_file_path(self, locale_code: str) -> Path | None:
        """Get file path for a locale."""
        base_name = self.filename_pattern.format(locale=locale_code)
        for ext in self.extensions:
            path = self.directory / f"{base_name}{ext}"
            if path.exists():
                return path
        return None

    def load(self, locale_code: str) -> MessageCatalog | None:
        path = self._get_file_path(locale_code)
        if not path:
            return None

        try:
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except ImportError:
                    logger.warning("YAML support requires PyYAML package")
                    return None
            else:
                return None

            return MessageCatalog(locale_code, data)

        except Exception as e:
            logger.warning(f"Failed to load messages for {locale_code}: {e}")
            return None

    def supports(self, locale_code: str) -> bool:
        return self._get_file_path(locale_code) is not None


# =============================================================================
# Message Formatter
# =============================================================================


class PlaceholderFormatter:
    """Formats messages with placeholders.

    Supports:
    - Named placeholders: {name}, {column}
    - Indexed placeholders: {0}, {1}
    - Format specs: {count:,d}, {ratio:.2%}
    - Custom formatters for complex types
    """

    def __init__(self):
        self._type_formatters: dict[type, Callable[[Any], str]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default type formatters."""
        self._type_formatters[datetime] = lambda d: d.isoformat()
        self._type_formatters[timedelta] = self._format_timedelta
        self._type_formatters[Path] = str

    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            mins = total_seconds // 60
            secs = total_seconds % 60
            return f"{mins}m {secs}s" if secs else f"{mins}m"
        else:
            hours = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            return f"{hours}h {mins}m" if mins else f"{hours}h"

    def register_formatter(
        self,
        type_: type,
        formatter: Callable[[Any], str],
    ) -> None:
        """Register a custom type formatter.

        Args:
            type_: Type to format
            formatter: Formatter function
        """
        self._type_formatters[type_] = formatter

    def format(
        self,
        template: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Format a template with placeholders.

        Args:
            template: Template string
            *args: Positional arguments
            **kwargs: Named arguments

        Returns:
            Formatted string
        """
        # Pre-process values through type formatters
        processed_kwargs = {}
        for key, value in kwargs.items():
            value_type = type(value)
            if value_type in self._type_formatters:
                processed_kwargs[key] = self._type_formatters[value_type](value)
            else:
                processed_kwargs[key] = value

        processed_args = []
        for value in args:
            value_type = type(value)
            if value_type in self._type_formatters:
                processed_args.append(self._type_formatters[value_type](value))
            else:
                processed_args.append(value)

        try:
            # Try str.format first
            return template.format(*processed_args, **processed_kwargs)
        except (KeyError, IndexError):
            # Fall back to Template for simpler substitution
            try:
                t = Template(template)
                return t.safe_substitute(processed_kwargs)
            except Exception:
                return template


# =============================================================================
# Main I18n Class
# =============================================================================


class I18n:
    """Main internationalization interface.

    Provides message resolution with locale fallback and formatting.

    Example:
        i18n = I18n.get_instance()

        # Set locale
        i18n.set_locale("ko")

        # Get message
        msg = i18n.t("error.analysis.failed", column="email")

        # Or with message code
        msg = i18n.t(MessageCode.ANALYSIS_FAILED, column="email")
    """

    _instance: ClassVar["I18n | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        locale_manager: LocaleManager | None = None,
        loaders: Sequence[MessageLoader] | None = None,
        formatter: PlaceholderFormatter | None = None,
    ):
        """Initialize I18n.

        Args:
            locale_manager: Locale manager
            loaders: Message loaders
            formatter: Placeholder formatter
        """
        self._locale_manager = locale_manager or _locale_manager
        self._loaders: list[MessageLoader] = list(loaders or [])
        self._formatter = formatter or PlaceholderFormatter()
        self._catalogs: dict[str, MessageCatalog] = {}
        self._catalog_lock = threading.RLock()

        # Add default English messages
        self._add_default_messages()

    @classmethod
    def get_instance(cls) -> "I18n":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _add_default_messages(self) -> None:
        """Add default English messages."""
        default_messages = {
            "en": {
                "err": {
                    "unknown": "Unknown error occurred",
                    "internal": "Internal error: {message}",
                    "not_implemented": "{feature} is not implemented",
                },
                "analysis": {
                    "failed": "Analysis failed for column '{column}'",
                    "column_failed": "Column analysis failed: {column} - {reason}",
                    "table_failed": "Table analysis failed: {table}",
                    "empty_data": "Cannot analyze empty dataset",
                    "skipped": "Analysis skipped for column '{column}': {reason}",
                },
                "pattern": {
                    "invalid": "Invalid pattern: {pattern}",
                    "not_found": "Pattern not found: {pattern}",
                    "compile_failed": "Failed to compile pattern '{pattern}': {error}",
                    "match_failed": "Pattern matching failed for column '{column}'",
                    "too_slow": "Pattern matching too slow for '{column}' (>{timeout}s)",
                },
                "type": {
                    "inference_failed": "Type inference failed for column '{column}'",
                    "ambiguous": "Ambiguous type for column '{column}': {types}",
                    "unsupported": "Unsupported data type: {dtype}",
                    "cast_failed": "Failed to cast '{value}' to {target_type}",
                },
                "io": {
                    "file_not_found": "File not found: {path}",
                    "permission_denied": "Permission denied: {path}",
                    "read_failed": "Failed to read file: {path}",
                    "write_failed": "Failed to write file: {path}",
                    "invalid_format": "Invalid file format: {path}",
                    "encoding_error": "Encoding error in file: {path}",
                },
                "memory": {
                    "exceeded": "Memory limit exceeded: {used} > {limit}",
                    "allocation_failed": "Memory allocation failed: {size}",
                    "limit_warning": "Approaching memory limit: {usage:.1%}",
                },
                "timeout": {
                    "exceeded": "Operation timed out after {seconds}s",
                    "column": "Timeout profiling column '{column}' after {seconds}s",
                    "operation": "Operation '{operation}' timed out",
                },
                "validation": {
                    "failed": "Validation failed: {message}",
                    "schema": "Schema validation failed: {field}",
                    "constraint": "Constraint violation: {constraint}",
                    "required": "Required field missing: {field}",
                    "type": "Type mismatch for '{field}': expected {expected}, got {actual}",
                    "range": "Value out of range for '{field}': {value} not in [{min}, {max}]",
                    "format": "Invalid format for '{field}': {value}",
                },
                "config": {
                    "invalid": "Invalid configuration: {message}",
                    "missing": "Missing configuration: {key}",
                    "type_mismatch": "Configuration type mismatch for '{key}': expected {expected}",
                },
                "cache": {
                    "miss": "Cache miss for key: {key}",
                    "expired": "Cache entry expired: {key}",
                    "invalid": "Invalid cache entry: {key}",
                    "connection_failed": "Cache connection failed: {error}",
                },
                "progress": {
                    "start": "Starting profiling: {table}",
                    "column": "Profiling column: {column} ({progress:.1%})",
                    "complete": "Profiling complete in {duration}",
                    "failed": "Profiling failed: {error}",
                },
                "rule": {
                    "generated": "Generated {count} validation rules",
                    "skipped": "Skipped rule generation for '{column}': {reason}",
                },
                "suite": {
                    "generated": "Generated validation suite with {rule_count} rules",
                },
            },
            "ko": {
                "err": {
                    "unknown": "알 수 없는 오류가 발생했습니다",
                    "internal": "내부 오류: {message}",
                    "not_implemented": "{feature} 기능이 구현되지 않았습니다",
                },
                "analysis": {
                    "failed": "'{column}' 컬럼 분석에 실패했습니다",
                    "column_failed": "컬럼 분석 실패: {column} - {reason}",
                    "table_failed": "테이블 분석 실패: {table}",
                    "empty_data": "빈 데이터셋은 분석할 수 없습니다",
                    "skipped": "'{column}' 컬럼 분석을 건너뜁니다: {reason}",
                },
                "pattern": {
                    "invalid": "잘못된 패턴: {pattern}",
                    "not_found": "패턴을 찾을 수 없음: {pattern}",
                    "compile_failed": "'{pattern}' 패턴 컴파일 실패: {error}",
                    "match_failed": "'{column}' 컬럼의 패턴 매칭 실패",
                    "too_slow": "'{column}'의 패턴 매칭이 너무 느립니다 (>{timeout}초)",
                },
                "type": {
                    "inference_failed": "'{column}' 컬럼의 타입 추론 실패",
                    "ambiguous": "'{column}' 컬럼의 타입이 모호함: {types}",
                    "unsupported": "지원하지 않는 데이터 타입: {dtype}",
                    "cast_failed": "'{value}'을(를) {target_type}으로 변환할 수 없습니다",
                },
                "io": {
                    "file_not_found": "파일을 찾을 수 없음: {path}",
                    "permission_denied": "접근 권한 없음: {path}",
                    "read_failed": "파일 읽기 실패: {path}",
                    "write_failed": "파일 쓰기 실패: {path}",
                    "invalid_format": "잘못된 파일 형식: {path}",
                    "encoding_error": "파일 인코딩 오류: {path}",
                },
                "memory": {
                    "exceeded": "메모리 한도 초과: {used} > {limit}",
                    "allocation_failed": "메모리 할당 실패: {size}",
                    "limit_warning": "메모리 한도에 근접: {usage:.1%}",
                },
                "timeout": {
                    "exceeded": "{seconds}초 후 작업 시간 초과",
                    "column": "'{column}' 컬럼 프로파일링 시간 초과 ({seconds}초)",
                    "operation": "'{operation}' 작업 시간 초과",
                },
                "validation": {
                    "failed": "검증 실패: {message}",
                    "schema": "스키마 검증 실패: {field}",
                    "constraint": "제약 조건 위반: {constraint}",
                    "required": "필수 필드 누락: {field}",
                    "type": "'{field}'의 타입 불일치: {expected} 예상, {actual} 발견",
                    "range": "'{field}' 값이 범위를 벗어남: {value}이(가) [{min}, {max}]에 없음",
                    "format": "'{field}'의 형식이 잘못됨: {value}",
                },
                "config": {
                    "invalid": "잘못된 구성: {message}",
                    "missing": "구성 누락: {key}",
                    "type_mismatch": "'{key}'의 구성 타입 불일치: {expected} 예상",
                },
                "cache": {
                    "miss": "캐시 미스: {key}",
                    "expired": "캐시 만료됨: {key}",
                    "invalid": "잘못된 캐시 항목: {key}",
                    "connection_failed": "캐시 연결 실패: {error}",
                },
                "progress": {
                    "start": "프로파일링 시작: {table}",
                    "column": "컬럼 프로파일링 중: {column} ({progress:.1%})",
                    "complete": "프로파일링 완료: {duration}",
                    "failed": "프로파일링 실패: {error}",
                },
                "rule": {
                    "generated": "{count}개의 검증 규칙 생성됨",
                    "skipped": "'{column}'의 규칙 생성 건너뜀: {reason}",
                },
                "suite": {
                    "generated": "{rule_count}개 규칙으로 검증 스위트 생성됨",
                },
            },
            "ja": {
                "err": {
                    "unknown": "不明なエラーが発生しました",
                    "internal": "内部エラー: {message}",
                    "not_implemented": "{feature}は実装されていません",
                },
                "analysis": {
                    "failed": "カラム'{column}'の分析に失敗しました",
                    "column_failed": "カラム分析失敗: {column} - {reason}",
                    "table_failed": "テーブル分析失敗: {table}",
                    "empty_data": "空のデータセットは分析できません",
                    "skipped": "カラム'{column}'の分析をスキップ: {reason}",
                },
                "pattern": {
                    "invalid": "無効なパターン: {pattern}",
                    "not_found": "パターンが見つかりません: {pattern}",
                    "compile_failed": "パターン'{pattern}'のコンパイル失敗: {error}",
                    "match_failed": "カラム'{column}'のパターンマッチング失敗",
                    "too_slow": "'{column}'のパターンマッチングが遅すぎます (>{timeout}秒)",
                },
                "type": {
                    "inference_failed": "カラム'{column}'の型推論失敗",
                    "ambiguous": "カラム'{column}'の型が曖昧: {types}",
                    "unsupported": "サポートされていないデータ型: {dtype}",
                    "cast_failed": "'{value}'を{target_type}に変換できません",
                },
                "io": {
                    "file_not_found": "ファイルが見つかりません: {path}",
                    "permission_denied": "アクセス権限がありません: {path}",
                    "read_failed": "ファイル読み込み失敗: {path}",
                    "write_failed": "ファイル書き込み失敗: {path}",
                    "invalid_format": "無効なファイル形式: {path}",
                    "encoding_error": "ファイルエンコーディングエラー: {path}",
                },
                "memory": {
                    "exceeded": "メモリ制限超過: {used} > {limit}",
                    "allocation_failed": "メモリ割り当て失敗: {size}",
                    "limit_warning": "メモリ制限に近づいています: {usage:.1%}",
                },
                "timeout": {
                    "exceeded": "{seconds}秒後にタイムアウト",
                    "column": "カラム'{column}'のプロファイリングがタイムアウト ({seconds}秒)",
                    "operation": "操作'{operation}'がタイムアウト",
                },
                "validation": {
                    "failed": "検証失敗: {message}",
                    "schema": "スキーマ検証失敗: {field}",
                    "constraint": "制約違反: {constraint}",
                    "required": "必須フィールドが不足: {field}",
                    "type": "'{field}'の型不一致: {expected}が期待されましたが{actual}でした",
                    "range": "'{field}'の値が範囲外: {value}は[{min}, {max}]にありません",
                    "format": "'{field}'の形式が無効: {value}",
                },
                "config": {
                    "invalid": "無効な設定: {message}",
                    "missing": "設定が不足: {key}",
                    "type_mismatch": "'{key}'の設定型不一致: {expected}が期待されました",
                },
                "cache": {
                    "miss": "キャッシュミス: {key}",
                    "expired": "キャッシュ期限切れ: {key}",
                    "invalid": "無効なキャッシュエントリ: {key}",
                    "connection_failed": "キャッシュ接続失敗: {error}",
                },
                "progress": {
                    "start": "プロファイリング開始: {table}",
                    "column": "カラムプロファイリング中: {column} ({progress:.1%})",
                    "complete": "プロファイリング完了: {duration}",
                    "failed": "プロファイリング失敗: {error}",
                },
                "rule": {
                    "generated": "{count}個の検証ルールを生成しました",
                    "skipped": "'{column}'のルール生成をスキップ: {reason}",
                },
                "suite": {
                    "generated": "{rule_count}個のルールで検証スイートを生成しました",
                },
            },
        }

        loader = DictMessageLoader(default_messages)
        self._loaders.insert(0, loader)

    def add_loader(self, loader: MessageLoader) -> None:
        """Add a message loader.

        Args:
            loader: Message loader to add
        """
        self._loaders.append(loader)

    def set_locale(self, locale_code: str) -> None:
        """Set the current locale.

        Args:
            locale_code: Locale code
        """
        self._locale_manager.set_locale(locale_code)

    def get_locale(self) -> str:
        """Get the current locale."""
        return self._locale_manager.get_locale()

    def _get_catalog(self, locale_code: str) -> MessageCatalog | None:
        """Get or load a message catalog.

        Args:
            locale_code: Locale code

        Returns:
            MessageCatalog or None
        """
        with self._catalog_lock:
            if locale_code in self._catalogs:
                return self._catalogs[locale_code]

            # Try loaders in order
            for loader in self._loaders:
                try:
                    if loader.supports(locale_code):
                        catalog = loader.load(locale_code)
                        if catalog:
                            self._catalogs[locale_code] = catalog
                            return catalog
                except Exception as e:
                    logger.debug(f"Loader failed for {locale_code}: {e}")

            return None

    def t(
        self,
        key: str | MessageCode,
        *args: Any,
        count: int | None = None,
        locale: str | None = None,
        default: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Translate a message key.

        Args:
            key: Message key or MessageCode
            *args: Positional format arguments
            count: Count for pluralization
            locale: Override locale
            default: Default message if key not found
            **kwargs: Named format arguments

        Returns:
            Translated and formatted message
        """
        # Convert MessageCode to string
        if isinstance(key, MessageCode):
            key = key.value

        # Get fallback chain
        target_locale = locale or self._locale_manager.current
        fallback_chain = self._locale_manager.get_fallback_chain(target_locale)

        # Try each locale in chain
        message = None
        for loc in fallback_chain:
            catalog = self._get_catalog(loc)
            if catalog:
                message = catalog.get(key, count)
                if message:
                    break

        # Use default if not found
        if message is None:
            if default is not None:
                message = default
            else:
                # Return key as fallback
                logger.warning(f"Message not found: {key} (locale: {target_locale})")
                return key

        # Format message
        try:
            return self._formatter.format(message, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to format message '{key}': {e}")
            return message

    def has(self, key: str | MessageCode, locale: str | None = None) -> bool:
        """Check if a message key exists.

        Args:
            key: Message key or MessageCode
            locale: Locale to check

        Returns:
            True if key exists
        """
        if isinstance(key, MessageCode):
            key = key.value

        target_locale = locale or self._locale_manager.current
        fallback_chain = self._locale_manager.get_fallback_chain(target_locale)

        for loc in fallback_chain:
            catalog = self._get_catalog(loc)
            if catalog and catalog.has(key):
                return True

        return False

    def list_keys(self, locale: str | None = None) -> list[str]:
        """List all available message keys.

        Args:
            locale: Locale (default: current)

        Returns:
            List of keys
        """
        target_locale = locale or self._locale_manager.current
        catalog = self._get_catalog(target_locale)
        if catalog:
            return catalog.keys()
        return []


# =============================================================================
# Internationalized Error Classes
# =============================================================================


class I18nError(Exception):
    """Base exception with i18n support.

    Automatically translates error messages based on the current locale.
    """

    def __init__(
        self,
        code: MessageCode,
        *,
        default: str | None = None,
        locale: str | None = None,
        **kwargs: Any,
    ):
        """Initialize i18n error.

        Args:
            code: Message code
            default: Default message
            locale: Override locale
            **kwargs: Message format arguments
        """
        self.code = code
        self.locale = locale
        self.format_args = kwargs
        self._default = default

        # Get translated message
        i18n = I18n.get_instance()
        message = i18n.t(code, locale=locale, default=default, **kwargs)

        super().__init__(message)

    @property
    def message_code(self) -> str:
        """Get the message code."""
        return self.code.value

    def get_message(self, locale: str | None = None) -> str:
        """Get message in a specific locale.

        Args:
            locale: Target locale

        Returns:
            Translated message
        """
        i18n = I18n.get_instance()
        return i18n.t(
            self.code,
            locale=locale or self.locale,
            default=self._default,
            **self.format_args,
        )


class I18nAnalysisError(I18nError):
    """Analysis error with i18n support."""

    def __init__(
        self,
        code: MessageCode = MessageCode.ANALYSIS_FAILED,
        **kwargs: Any,
    ):
        super().__init__(code, **kwargs)


class I18nPatternError(I18nError):
    """Pattern error with i18n support."""

    def __init__(
        self,
        code: MessageCode = MessageCode.PATTERN_INVALID,
        **kwargs: Any,
    ):
        super().__init__(code, **kwargs)


class I18nTypeError(I18nError):
    """Type inference error with i18n support."""

    def __init__(
        self,
        code: MessageCode = MessageCode.TYPE_INFERENCE_FAILED,
        **kwargs: Any,
    ):
        super().__init__(code, **kwargs)


class I18nIOError(I18nError):
    """IO error with i18n support."""

    def __init__(
        self,
        code: MessageCode = MessageCode.IO_READ_FAILED,
        **kwargs: Any,
    ):
        super().__init__(code, **kwargs)


class I18nTimeoutError(I18nError):
    """Timeout error with i18n support."""

    def __init__(
        self,
        code: MessageCode = MessageCode.TIMEOUT_EXCEEDED,
        **kwargs: Any,
    ):
        super().__init__(code, **kwargs)


class I18nValidationError(I18nError):
    """Validation error with i18n support."""

    def __init__(
        self,
        code: MessageCode = MessageCode.VALIDATION_FAILED,
        **kwargs: Any,
    ):
        super().__init__(code, **kwargs)


# =============================================================================
# Message Catalog Registry
# =============================================================================


class MessageCatalogRegistry:
    """Registry for message catalogs.

    Allows registration of custom message catalogs at runtime.
    """

    _instance: ClassVar["MessageCatalogRegistry | None"] = None

    def __init__(self):
        self._catalogs: dict[str, MessageCatalog] = {}
        self._loaders: list[MessageLoader] = []
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "MessageCatalogRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_catalog(
        self,
        locale_code: str,
        catalog: MessageCatalog,
    ) -> None:
        """Register a message catalog.

        Args:
            locale_code: Locale code
            catalog: Message catalog
        """
        with self._lock:
            self._catalogs[locale_code] = catalog

    def register_messages(
        self,
        locale_code: str,
        messages: dict[str, Any],
    ) -> None:
        """Register messages from a dictionary.

        Args:
            locale_code: Locale code
            messages: Messages dictionary
        """
        catalog = MessageCatalog(locale_code, messages)
        self.register_catalog(locale_code, catalog)

    def register_loader(self, loader: MessageLoader) -> None:
        """Register a message loader.

        Args:
            loader: Message loader
        """
        with self._lock:
            self._loaders.append(loader)

    def get_catalog(self, locale_code: str) -> MessageCatalog | None:
        """Get a catalog by locale.

        Args:
            locale_code: Locale code

        Returns:
            MessageCatalog or None
        """
        with self._lock:
            if locale_code in self._catalogs:
                return self._catalogs[locale_code]

            for loader in self._loaders:
                if loader.supports(locale_code):
                    catalog = loader.load(locale_code)
                    if catalog:
                        self._catalogs[locale_code] = catalog
                        return catalog

            return None

    def list_locales(self) -> list[str]:
        """List registered locales."""
        return list(self._catalogs.keys())


# =============================================================================
# Convenience Functions
# =============================================================================


def get_message(
    code: MessageCode,
    *args: Any,
    locale: str | None = None,
    **kwargs: Any,
) -> str:
    """Get a translated message.

    Args:
        code: Message code
        *args: Positional format arguments
        locale: Override locale
        **kwargs: Named format arguments

    Returns:
        Translated message
    """
    return I18n.get_instance().t(code, *args, locale=locale, **kwargs)


def t(
    key: str | MessageCode,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Shorthand for translate.

    Args:
        key: Message key or code
        *args: Format arguments
        **kwargs: Named arguments

    Returns:
        Translated message
    """
    return I18n.get_instance().t(key, *args, **kwargs)


def register_messages(
    locale_code: str,
    messages: dict[str, Any],
) -> None:
    """Register messages for a locale.

    Args:
        locale_code: Locale code
        messages: Messages dictionary
    """
    registry = MessageCatalogRegistry.get_instance()
    registry.register_messages(locale_code, messages)


def load_messages_from_file(
    path: str | Path,
    locale_code: str | None = None,
) -> None:
    """Load messages from a file.

    Args:
        path: Path to message file
        locale_code: Override locale code (default: from filename)
    """
    path = Path(path)

    # Infer locale from filename if not provided
    if locale_code is None:
        match = re.match(r"messages_(\w+)\.", path.name)
        if match:
            locale_code = match.group(1)
        else:
            locale_code = path.stem

    # Load based on extension
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    elif path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                messages = yaml.safe_load(f)
        except ImportError:
            raise ImportError("YAML support requires PyYAML package")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    register_messages(locale_code, messages)


def create_message_loader(
    directory: str | Path,
    pattern: str = "messages_{locale}",
) -> FileMessageLoader:
    """Create a file message loader.

    Args:
        directory: Directory with message files
        pattern: Filename pattern

    Returns:
        FileMessageLoader
    """
    return FileMessageLoader(directory, pattern)


# =============================================================================
# Context Manager for Locale
# =============================================================================


class locale_context:
    """Context manager for temporary locale change.

    Example:
        with locale_context("ko"):
            msg = get_message(MessageCode.ANALYSIS_FAILED, column="email")
        # Original locale restored
    """

    def __init__(self, locale_code: str):
        self.locale_code = locale_code
        self._previous: str | None = None

    def __enter__(self) -> "locale_context":
        self._previous = get_locale()
        set_locale(self.locale_code)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._previous is not None:
            set_locale(self._previous)


# =============================================================================
# Presets
# =============================================================================


class I18nPresets:
    """Pre-configured I18n setups."""

    @staticmethod
    def minimal() -> I18n:
        """Minimal setup with English only."""
        return I18n()

    @staticmethod
    def with_file_loader(directory: str | Path) -> I18n:
        """Setup with file loader."""
        loader = FileMessageLoader(directory)
        i18n = I18n()
        i18n.add_loader(loader)
        return i18n

    @staticmethod
    def auto_detect_locale() -> I18n:
        """Setup with auto-detected system locale."""
        i18n = I18n()
        locale_info = LocaleInfo.from_system()
        i18n.set_locale(locale_info.code)
        return i18n
