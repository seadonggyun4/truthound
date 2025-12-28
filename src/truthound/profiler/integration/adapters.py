"""Adapters for converting generated rules to validators.

This module provides the bridge between profiler-generated rules
and the validator system, enabling automatic conversion.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.profiler.generators.base import GeneratedRule
    from truthound.validators.base import Validator

logger = logging.getLogger(__name__)


class RuleToValidatorAdapter(ABC):
    """Abstract base class for rule-to-validator adapters.

    Each adapter handles conversion of specific rule types to validators.

    Example:
        class NullCheckAdapter(RuleToValidatorAdapter):
            def supports(self, rule: GeneratedRule) -> bool:
                return rule.validator_class == "NullValidator"

            def convert(self, rule: GeneratedRule) -> Validator:
                from truthound.validators import NullValidator
                return NullValidator(columns=rule.columns, **rule.parameters)
    """

    @abstractmethod
    def supports(self, rule: "GeneratedRule") -> bool:
        """Check if this adapter supports the given rule."""
        ...

    @abstractmethod
    def convert(self, rule: "GeneratedRule") -> "Validator":
        """Convert a generated rule to a validator."""
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority for adapter selection (higher = preferred)."""
        ...


class DefaultRuleAdapter(RuleToValidatorAdapter):
    """Default adapter using validator class name resolution.

    This adapter attempts to import the validator class by name
    and instantiate it with the rule's parameters.
    """

    @property
    def priority(self) -> int:
        return 0  # Lowest priority, fallback

    def supports(self, rule: "GeneratedRule") -> bool:
        """Supports any rule with a validator_class."""
        return bool(rule.validator_class)

    def convert(self, rule: "GeneratedRule") -> "Validator":
        """Convert by importing and instantiating the validator class."""
        from truthound.validators import get_validator

        validator_name = rule.validator_class
        params = dict(rule.parameters)

        if rule.columns:
            params["columns"] = list(rule.columns)
        if rule.mostly is not None:
            params["mostly"] = rule.mostly

        try:
            # Try to get validator by name
            validator_cls = get_validator(validator_name)
            if validator_cls is None:
                raise ValueError(f"Unknown validator: {validator_name}")
            return validator_cls(**params)
        except Exception as e:
            logger.error(f"Failed to create validator from rule {rule.name}: {e}")
            raise ValueError(f"Cannot create validator from rule: {e}") from e


class DynamicImportAdapter(RuleToValidatorAdapter):
    """Adapter that dynamically imports validator classes."""

    @property
    def priority(self) -> int:
        return 10

    def supports(self, rule: "GeneratedRule") -> bool:
        return bool(rule.validator_class)

    def convert(self, rule: "GeneratedRule") -> "Validator":
        """Convert by dynamically importing the validator class."""
        import importlib

        validator_name = rule.validator_class
        params = dict(rule.parameters)

        if rule.columns:
            params["columns"] = list(rule.columns)
        if rule.mostly is not None:
            params["mostly"] = rule.mostly

        # Try common module paths
        module_paths = [
            f"truthound.validators.{validator_name.lower()}",
            "truthound.validators",
            f"truthound.validators.builtin.{validator_name.lower()}",
        ]

        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, validator_name):
                    validator_cls = getattr(module, validator_name)
                    return validator_cls(**params)
            except ImportError:
                continue

        raise ValueError(f"Cannot find validator class: {validator_name}")


class ValidatorRegistry:
    """Registry for rule-to-validator adapters.

    Manages a collection of adapters and selects the best one
    for converting each rule.

    Example:
        registry = ValidatorRegistry()
        registry.register(MyCustomAdapter())

        validator = registry.create_validator(rule)
    """

    _instance: "ValidatorRegistry | None" = None

    def __init__(self) -> None:
        """Initialize the registry."""
        self._adapters: list[RuleToValidatorAdapter] = []
        self._custom_factories: dict[str, Callable[["GeneratedRule"], "Validator"]] = {}

        # Register default adapters
        self._adapters.append(DynamicImportAdapter())
        self._adapters.append(DefaultRuleAdapter())

    @classmethod
    def get_instance(cls) -> "ValidatorRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = ValidatorRegistry()
        return cls._instance

    def register(self, adapter: RuleToValidatorAdapter) -> None:
        """Register an adapter.

        Args:
            adapter: The adapter to register.
        """
        self._adapters.append(adapter)
        # Keep sorted by priority (descending)
        self._adapters.sort(key=lambda a: a.priority, reverse=True)

    def unregister(self, adapter: RuleToValidatorAdapter) -> bool:
        """Unregister an adapter.

        Args:
            adapter: The adapter to remove.

        Returns:
            True if the adapter was found and removed.
        """
        try:
            self._adapters.remove(adapter)
            return True
        except ValueError:
            return False

    def register_factory(
        self,
        validator_class: str,
        factory: Callable[["GeneratedRule"], "Validator"],
    ) -> None:
        """Register a factory function for a specific validator class.

        Args:
            validator_class: The validator class name.
            factory: Factory function that creates the validator.
        """
        self._custom_factories[validator_class] = factory

    def create_validator(self, rule: "GeneratedRule") -> "Validator":
        """Create a validator from a rule.

        Args:
            rule: The rule to convert.

        Returns:
            A Validator instance.

        Raises:
            ValueError: If no adapter can handle the rule.
        """
        # Check custom factories first
        if rule.validator_class in self._custom_factories:
            return self._custom_factories[rule.validator_class](rule)

        # Try adapters in priority order
        for adapter in self._adapters:
            if adapter.supports(rule):
                try:
                    return adapter.convert(rule)
                except Exception as e:
                    logger.debug(
                        f"Adapter {adapter.__class__.__name__} failed for rule "
                        f"{rule.name}: {e}"
                    )
                    continue

        raise ValueError(
            f"No adapter can create validator for rule: {rule.name} "
            f"(class: {rule.validator_class})"
        )

    def can_create(self, rule: "GeneratedRule") -> bool:
        """Check if any adapter can create a validator for the rule.

        Args:
            rule: The rule to check.

        Returns:
            True if a validator can be created.
        """
        if rule.validator_class in self._custom_factories:
            return True
        return any(adapter.supports(rule) for adapter in self._adapters)

    def list_adapters(self) -> list[tuple[str, int]]:
        """List registered adapters with their priorities.

        Returns:
            List of (adapter_name, priority) tuples.
        """
        return [
            (adapter.__class__.__name__, adapter.priority)
            for adapter in self._adapters
        ]


# Global registry instance
_registry = ValidatorRegistry()


def create_validator_from_rule(rule: "GeneratedRule") -> "Validator":
    """Create a validator from a generated rule.

    This is a convenience function using the global registry.

    Args:
        rule: The generated rule.

    Returns:
        A Validator instance.

    Example:
        rule = GeneratedRule(
            name="check_nulls",
            validator_class="NullValidator",
            columns=["id", "name"],
        )
        validator = create_validator_from_rule(rule)
    """
    return _registry.create_validator(rule)


def register_rule_adapter(adapter: RuleToValidatorAdapter) -> None:
    """Register an adapter with the global registry.

    Args:
        adapter: The adapter to register.
    """
    _registry.register(adapter)


def register_validator_factory(
    validator_class: str,
    factory: Callable[["GeneratedRule"], "Validator"],
) -> None:
    """Register a validator factory with the global registry.

    Args:
        validator_class: The validator class name.
        factory: Factory function that creates the validator.

    Example:
        def create_custom_validator(rule):
            return CustomValidator(**rule.parameters)

        register_validator_factory("CustomValidator", create_custom_validator)
    """
    _registry.register_factory(validator_class, factory)
