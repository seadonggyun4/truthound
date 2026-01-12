"""Validator registry for automatic discovery and registration.

This module provides lazy loading of validators to improve startup performance.
Validators are loaded on-demand when first accessed, rather than all at once.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import TYPE_CHECKING, Any, Iterator

from truthound.validators._lazy import (
    CATEGORY_MODULES,
    VALIDATOR_IMPORT_MAP,
    ValidatorImportMetrics,
)

if TYPE_CHECKING:
    from truthound.validators.base import Validator

logger = logging.getLogger(__name__)


class ValidatorRegistry:
    """Singleton registry for validator auto-discovery and lookup.

    This registry uses lazy loading to defer validator imports until they are
    actually needed, significantly improving startup performance when only a
    subset of validators is used.

    Features:
    - Lazy loading: Validators are imported only when accessed
    - Category-based discovery: Load validators by category
    - Custom registration: Register custom validator classes
    - Metrics tracking: Track import performance
    """

    _instance: "ValidatorRegistry | None" = None

    def __new__(cls) -> "ValidatorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._validators: dict[str, type["Validator"]] = {}
            cls._instance._categories: dict[str, dict[str, type["Validator"]]] = {}
            cls._instance._loaded_categories: set[str] = set()
            cls._instance._all_loaded: bool = False
            cls._instance._metrics = ValidatorImportMetrics()
            cls._instance._loaded_modules: dict[str, Any] = {}
        return cls._instance

    def _get_validator_lazy(self, name: str) -> type["Validator"] | None:
        """Load a single validator lazily by name.

        Args:
            name: Validator class name.

        Returns:
            Validator class or None if not found.
        """
        # Check if already loaded
        if name in self._validators:
            return self._validators[name]

        # Check if it's in the import map
        if name not in VALIDATOR_IMPORT_MAP:
            return None

        module_path = VALIDATOR_IMPORT_MAP[name]

        try:
            start_time = time.perf_counter()

            # Load the module if not cached
            if module_path not in self._loaded_modules:
                self._loaded_modules[module_path] = importlib.import_module(module_path)

            module = self._loaded_modules[module_path]
            validator_cls = getattr(module, name, None)

            if validator_cls and hasattr(validator_cls, "validate"):
                # Register it
                self._register_internal(validator_cls)

                duration = time.perf_counter() - start_time
                self._metrics.record_load(name, duration)

                logger.debug(
                    f"Lazy loaded validator '{name}' from '{module_path}' "
                    f"in {duration*1000:.2f}ms"
                )

                return validator_cls

        except (ImportError, AttributeError) as e:
            self._metrics.record_failure(name)
            logger.warning(f"Failed to load validator '{name}': {e}")

        return None

    def _load_category(self, category: str) -> None:
        """Load all validators from a category.

        Args:
            category: Category name to load.
        """
        if category in self._loaded_categories:
            return

        if category not in CATEGORY_MODULES:
            return

        start_time = time.perf_counter()
        loaded_count = 0

        for module_path in CATEGORY_MODULES[category]:
            try:
                if module_path not in self._loaded_modules:
                    self._loaded_modules[module_path] = importlib.import_module(module_path)

                module = self._loaded_modules[module_path]

                # Get all exported classes
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue

                    cls = getattr(module, attr_name, None)
                    if (
                        cls
                        and isinstance(cls, type)
                        and hasattr(cls, "name")
                        and hasattr(cls, "validate")
                    ):
                        self._register_internal(cls)
                        loaded_count += 1

            except ImportError as e:
                logger.warning(f"Failed to load module '{module_path}': {e}")

        self._loaded_categories.add(category)

        duration = time.perf_counter() - start_time
        logger.debug(
            f"Loaded category '{category}' ({loaded_count} validators) "
            f"in {duration*1000:.2f}ms"
        )

    def _load_all_categories(self) -> None:
        """Load all validator categories."""
        if self._all_loaded:
            return

        for category in CATEGORY_MODULES:
            self._load_category(category)

        self._all_loaded = True

    def _register_internal(self, validator_cls: type["Validator"]) -> None:
        """Internal registration without triggering discovery."""
        name = getattr(validator_cls, "name", validator_cls.__name__.lower())
        category = getattr(validator_cls, "category", "general")

        self._validators[name] = validator_cls

        if category not in self._categories:
            self._categories[category] = {}
        self._categories[category][name] = validator_cls

    def register(self, validator_cls: type["Validator"]) -> None:
        """Register a validator class.

        Args:
            validator_cls: Validator class to register.
        """
        self._register_internal(validator_cls)

    def get(self, name: str) -> type["Validator"]:
        """Get a validator class by name.

        This method uses lazy loading - the validator is only imported
        when first requested.

        Args:
            name: Validator name.

        Returns:
            Validator class.

        Raises:
            ValueError: If validator name is not found.
        """
        self._metrics.record_access(name)

        # Check already loaded
        if name in self._validators:
            return self._validators[name]

        # Try lazy loading
        validator_cls = self._get_validator_lazy(name)
        if validator_cls:
            return validator_cls

        # Fall back to loading all and checking
        self._load_all_categories()

        if name not in self._validators:
            available = ", ".join(sorted(self._validators.keys()))
            raise ValueError(f"Unknown validator: {name}. Available: {available}")

        return self._validators[name]

    def get_by_category(self, category: str) -> dict[str, type["Validator"]]:
        """Get all validators in a category.

        This loads only the specified category, not all validators.

        Args:
            category: Category name.

        Returns:
            Dictionary of validator name to class.
        """
        self._load_category(category)
        return self._categories.get(category, {}).copy()

    def list_all(self) -> dict[str, type["Validator"]]:
        """List all registered validators.

        Note: This loads all validators, which defeats lazy loading.
        Use get() for individual validators when possible.

        Returns:
            Dictionary of all validators.
        """
        self._load_all_categories()
        return self._validators.copy()

    def list_categories(self) -> list[str]:
        """List all validator categories.

        Returns:
            List of category names.
        """
        return list(CATEGORY_MODULES.keys())

    def get_metrics(self) -> dict[str, Any]:
        """Get registry metrics summary.

        Returns:
            Metrics dictionary.
        """
        return self._metrics.get_summary()

    def is_loaded(self, name: str) -> bool:
        """Check if a validator is already loaded.

        Args:
            name: Validator name.

        Returns:
            True if loaded, False otherwise.
        """
        return name in self._validators

    def is_available(self, name: str) -> bool:
        """Check if a validator is available (can be loaded).

        Args:
            name: Validator name.

        Returns:
            True if available, False otherwise.
        """
        return name in self._validators or name in VALIDATOR_IMPORT_MAP

    def preload(self, *names: str) -> None:
        """Preload specific validators.

        Args:
            names: Validator names to preload.
        """
        for name in names:
            if name not in self._validators and name in VALIDATOR_IMPORT_MAP:
                self._get_validator_lazy(name)

    def preload_category(self, category: str) -> None:
        """Preload all validators in a category.

        Args:
            category: Category name.
        """
        self._load_category(category)

    def __iter__(self) -> Iterator[tuple[str, type["Validator"]]]:
        """Iterate over all validators (loads all)."""
        self._load_all_categories()
        return iter(self._validators.items())

    def __contains__(self, name: str) -> bool:
        """Check if a validator is available."""
        return self.is_available(name)

    def __len__(self) -> int:
        """Get count of available validators."""
        return len(self._validators) + len(
            set(VALIDATOR_IMPORT_MAP.keys()) - set(self._validators.keys())
        )

    def __getitem__(self, name: str) -> type["Validator"]:
        """Get validator by name using index notation."""
        return self.get(name)


# Singleton instance
registry = ValidatorRegistry()


def register_validator(cls: type["Validator"]) -> type["Validator"]:
    """Decorator to register a validator class.

    Args:
        cls: Validator class to register.

    Returns:
        The same validator class (for decorator chaining).
    """
    registry.register(cls)
    return cls
