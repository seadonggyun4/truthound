"""Locale loader for external translation files.

This module provides utilities for loading translations
from JSON, YAML, and other external sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from truthound.datadocs.i18n.catalog import ReportCatalog, register_catalog


class LocaleLoader:
    """Loader for external locale files.

    Supports JSON and YAML formats for loading custom translations.

    Example:
        loader = LocaleLoader()

        # Load single file
        loader.load_file(Path("locales/fr.json"))

        # Load directory of locale files
        loader.load_directory(Path("locales/"))
    """

    def __init__(self, auto_register: bool = True) -> None:
        """Initialize loader.

        Args:
            auto_register: Automatically register loaded catalogs.
        """
        self._auto_register = auto_register
        self._catalogs: dict[str, ReportCatalog] = {}

    def load_file(self, path: Path) -> ReportCatalog:
        """Load a locale file.

        Args:
            path: Path to locale file (JSON or YAML).

        Returns:
            Loaded ReportCatalog.

        Raises:
            ValueError: If file format is unsupported.
            FileNotFoundError: If file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Locale file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".json":
            catalog = self._load_json(path)
        elif suffix in (".yaml", ".yml"):
            catalog = self._load_yaml(path)
        else:
            raise ValueError(f"Unsupported locale file format: {suffix}")

        self._catalogs[catalog.locale] = catalog

        if self._auto_register:
            register_catalog(catalog)

        return catalog

    def load_directory(
        self,
        directory: Path,
        pattern: str = "*.json",
    ) -> dict[str, ReportCatalog]:
        """Load all locale files from a directory.

        Args:
            directory: Directory containing locale files.
            pattern: Glob pattern for files.

        Returns:
            Dictionary of locale to catalog.
        """
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        catalogs = {}
        for file_path in directory.glob(pattern):
            try:
                catalog = self.load_file(file_path)
                catalogs[catalog.locale] = catalog
            except Exception:
                # Skip invalid files
                pass

        return catalogs

    def _load_json(self, path: Path) -> ReportCatalog:
        """Load JSON locale file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_locale_data(data, path)

    def _load_yaml(self, path: Path) -> ReportCatalog:
        """Load YAML locale file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML locale files. "
                "Install with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_locale_data(data, path)

    def _parse_locale_data(
        self,
        data: dict[str, Any],
        path: Path,
    ) -> ReportCatalog:
        """Parse locale data into a catalog.

        Supports both flat and nested message structures.

        Args:
            data: Parsed locale data.
            path: Source file path (for locale inference).

        Returns:
            ReportCatalog.
        """
        # Extract locale
        locale = data.get("locale")
        if not locale:
            # Infer from filename (e.g., "ko.json" -> "ko")
            locale = path.stem

        # Extract messages
        messages = data.get("messages", {})
        if not messages:
            # Try flat structure (key: value at root level)
            messages = {
                k: v for k, v in data.items()
                if k not in ("locale", "metadata") and isinstance(v, str)
            }

        # Flatten nested messages
        if any(isinstance(v, dict) for v in messages.values()):
            messages = self._flatten_messages(messages)

        # Extract metadata
        metadata = data.get("metadata", {})
        if "name" not in metadata:
            metadata["name"] = locale
        if "source" not in metadata:
            metadata["source"] = str(path)

        return ReportCatalog(
            locale=locale,
            messages=messages,
            metadata=metadata,
        )

    def _flatten_messages(
        self,
        nested: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, str]:
        """Flatten nested message structure.

        Args:
            nested: Nested message dictionary.
            prefix: Key prefix.

        Returns:
            Flat message dictionary.

        Example:
            Input:  {"report": {"title": "..."}}
            Output: {"report.title": "..."}
        """
        flat = {}

        for key, value in nested.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_messages(value, full_key))
            elif isinstance(value, str):
                flat[full_key] = value

        return flat

    def get_catalogs(self) -> dict[str, ReportCatalog]:
        """Get all loaded catalogs."""
        return self._catalogs.copy()

    def get(self, locale: str) -> ReportCatalog | None:
        """Get a loaded catalog by locale."""
        return self._catalogs.get(locale)


def load_locale_from_file(path: Path | str) -> ReportCatalog:
    """Load a locale from a file.

    Args:
        path: Path to locale file.

    Returns:
        Loaded catalog.
    """
    loader = LocaleLoader(auto_register=True)
    return loader.load_file(Path(path))


def load_locale_from_dict(
    locale: str,
    messages: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> ReportCatalog:
    """Create and register a catalog from a dictionary.

    Args:
        locale: Locale code.
        messages: Message dictionary.
        metadata: Optional metadata.

    Returns:
        Created catalog.
    """
    catalog = ReportCatalog.from_dict(locale, messages, metadata)
    register_catalog(catalog)
    return catalog


def load_locales_from_directory(
    directory: Path | str,
    pattern: str = "*.json",
) -> dict[str, ReportCatalog]:
    """Load all locales from a directory.

    Args:
        directory: Directory path.
        pattern: File pattern.

    Returns:
        Dictionary of loaded catalogs.
    """
    loader = LocaleLoader(auto_register=True)
    return loader.load_directory(Path(directory), pattern)
