"""Validator registry for automatic discovery and registration."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from truthound.validators.base import Validator


class ValidatorRegistry:
    """Singleton registry for validator auto-discovery and lookup."""

    _instance: "ValidatorRegistry | None" = None

    def __new__(cls) -> "ValidatorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._validators = {}
            cls._instance._categories = {}
            cls._instance._initialized = False
        return cls._instance

    def _discover_validators(self) -> None:
        """Auto-discover validators from category packages."""
        if self._initialized:
            return

        categories = [
            "schema",
            "completeness",
            "uniqueness",
            "distribution",
            "string",
            "datetime",
            "aggregate",
        ]

        for category in categories:
            self._load_category(category)

        self._initialized = True

    def _load_category(self, category: str) -> None:
        """Load all validators from a category module."""
        try:
            module = importlib.import_module(f"truthound.validators.{category}")
            # Get all exported validators from __all__
            for name in getattr(module, "__all__", []):
                cls = getattr(module, name, None)
                if cls and hasattr(cls, "name") and hasattr(cls, "validate"):
                    self.register(cls)
        except ImportError:
            pass

    def register(self, validator_cls: type["Validator"]) -> None:
        """Register a validator class."""
        name = getattr(validator_cls, "name", validator_cls.__name__.lower())
        category = getattr(validator_cls, "category", "general")

        self._validators[name] = validator_cls

        if category not in self._categories:
            self._categories[category] = {}
        self._categories[category][name] = validator_cls

    def get(self, name: str) -> type["Validator"]:
        """Get a validator class by name."""
        self._discover_validators()
        if name not in self._validators:
            available = ", ".join(sorted(self._validators.keys()))
            raise ValueError(f"Unknown validator: {name}. Available: {available}")
        return self._validators[name]

    def get_by_category(self, category: str) -> dict[str, type["Validator"]]:
        """Get all validators in a category."""
        self._discover_validators()
        return self._categories.get(category, {}).copy()

    def list_all(self) -> dict[str, type["Validator"]]:
        """List all registered validators."""
        self._discover_validators()
        return self._validators.copy()

    def list_categories(self) -> list[str]:
        """List all categories."""
        self._discover_validators()
        return list(self._categories.keys())

    def __iter__(self) -> Iterator[tuple[str, type["Validator"]]]:
        self._discover_validators()
        return iter(self._validators.items())

    def __contains__(self, name: str) -> bool:
        self._discover_validators()
        return name in self._validators

    def __len__(self) -> int:
        self._discover_validators()
        return len(self._validators)


# Singleton instance
registry = ValidatorRegistry()


def register_validator(cls: type["Validator"]) -> type["Validator"]:
    """Decorator to register a validator class."""
    registry.register(cls)
    return cls
