"""Adapters for converting generated rules to validators.

This module provides the bridge between profiler-generated rules
and the validator system, enabling automatic conversion.

Example:
    from truthound.profiler.integration.adapters import (
        create_validator_from_rule,
        ValidatorRegistry,
    )

    # Create validator from rule
    validator = create_validator_from_rule(rule)

    # Or use registry directly
    registry = ValidatorRegistry.get_instance()
    validator = registry.create_validator(rule)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from truthound.profiler.integration.naming import resolve_validator_name

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

    Supports multiple naming conventions:
    - PascalCase: "ColumnTypeValidator" → "column_type"
    - snake_case: "column_type" → "column_type"
    - kebab-case: "column-type" → "column_type"
    """

    # Parameter transformations: maps (validator_name, param_name) -> transformer
    # Transformer receives (param_value, rule) and returns transformed params dict
    PARAMETER_TRANSFORMERS: dict[
        tuple[str, str],
        Callable[..., dict[str, Any]]
    ] = {}

    @classmethod
    def _init_transformers(cls) -> None:
        """Initialize parameter transformers for known validators."""
        if cls.PARAMETER_TRANSFORMERS:
            return  # Already initialized

        # ColumnTypeValidator: expected_type (single) -> expected_types (dict)
        def transform_column_type(
            value: Any,
            rule: "GeneratedRule",
        ) -> dict[str, Any]:
            """Transform expected_type to expected_types dict."""
            if rule.columns:
                # Build dict mapping each column to the expected type
                return {"expected_types": {col: value for col in rule.columns}}
            return {"expected_types": {"__default__": value}}

        cls.PARAMETER_TRANSFORMERS[("column_type", "expected_type")] = transform_column_type

    @property
    def priority(self) -> int:
        return 0  # Lowest priority, fallback

    def supports(self, rule: "GeneratedRule") -> bool:
        """Supports any rule with a validator_class."""
        return bool(rule.validator_class)

    def _transform_parameters(
        self,
        validator_name: str,
        params: dict[str, Any],
        rule: "GeneratedRule",
    ) -> dict[str, Any]:
        """Transform parameters based on validator-specific rules.

        Args:
            validator_name: Canonical validator name (e.g., "column_type")
            params: Original parameters from the rule
            rule: The original rule (for access to columns, etc.)

        Returns:
            Transformed parameters dict
        """
        self._init_transformers()

        transformed = {}
        processed_keys: set[str] = set()

        for param_key, param_value in params.items():
            transformer_key = (validator_name, param_key)
            if transformer_key in self.PARAMETER_TRANSFORMERS:
                transformer = self.PARAMETER_TRANSFORMERS[transformer_key]
                transformed.update(transformer(param_value, rule))
                processed_keys.add(param_key)
            else:
                transformed[param_key] = param_value

        return transformed

    def convert(self, rule: "GeneratedRule") -> "Validator":
        """Convert by importing and instantiating the validator class."""
        import inspect
        from truthound.validators import get_validator

        validator_name = rule.validator_class
        params = dict(rule.parameters)

        # Resolve name to canonical form first (needed for parameter transformation)
        canonical_name = resolve_validator_name(validator_name)

        # Transform parameters based on validator-specific rules
        params = self._transform_parameters(canonical_name, params, rule)

        # Get validator class first to check its signature
        validator_cls = get_validator(canonical_name)
        if validator_cls is None:
            raise ValueError(f"Unknown validator: {canonical_name}")

        # Check if validator expects 'column' (singular) vs 'columns' (plural)
        try:
            sig = inspect.signature(validator_cls.__init__)
            validator_params = list(sig.parameters.keys())
            expects_column = "column" in validator_params
            expects_columns = "columns" in validator_params
        except (ValueError, TypeError):
            expects_column = False
            expects_columns = False

        # Add columns only if not already consumed by transformer
        if rule.columns and "columns" not in params and "column" not in params:
            # Check if validator uses expected_types dict (columns already embedded)
            if "expected_types" not in params:
                if expects_column and not expects_columns and len(rule.columns) == 1:
                    # Single-column validator: use 'column' parameter
                    params["column"] = rule.columns[0]
                else:
                    # Multi-column validator or kwargs-based: use 'columns'
                    params["columns"] = list(rule.columns)
        if rule.mostly is not None:
            params["mostly"] = rule.mostly

        try:
            return validator_cls(**params)
        except Exception as e:
            logger.error(f"Failed to create validator from rule {rule.name}: {e}")
            raise ValueError(f"Cannot create validator from rule: {e}") from e


class DynamicImportAdapter(RuleToValidatorAdapter):
    """Adapter that dynamically imports validator classes.

    Attempts to import validators by their class name from common paths.
    Supports both PascalCase class names and snake_case module names.
    """

    @property
    def priority(self) -> int:
        return 10

    def supports(self, rule: "GeneratedRule") -> bool:
        return bool(rule.validator_class)

    def convert(self, rule: "GeneratedRule") -> "Validator":
        """Convert by dynamically importing the validator class."""
        import importlib
        import inspect

        validator_name = rule.validator_class
        params = dict(rule.parameters)

        # Resolve to canonical name for module lookup
        canonical_name = resolve_validator_name(validator_name)

        # Apply parameter transformations (reuse DefaultRuleAdapter's transformers)
        DefaultRuleAdapter._init_transformers()
        transformed_params: dict[str, Any] = {}
        for param_key, param_value in params.items():
            transformer_key = (canonical_name, param_key)
            if transformer_key in DefaultRuleAdapter.PARAMETER_TRANSFORMERS:
                transformer = DefaultRuleAdapter.PARAMETER_TRANSFORMERS[transformer_key]
                transformed_params.update(transformer(param_value, rule))
            else:
                transformed_params[param_key] = param_value
        params = transformed_params

        # Try common module paths
        module_paths = [
            f"truthound.validators.{canonical_name}",
            "truthound.validators",
            f"truthound.validators.builtin.{canonical_name}",
        ]

        # Class name variations to try
        pascal_name = ''.join(word.capitalize() for word in canonical_name.split('_'))
        class_names = [validator_name, pascal_name, pascal_name + "Validator"]

        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                for class_name in class_names:
                    if hasattr(module, class_name):
                        validator_cls = getattr(module, class_name)

                        # Check if validator expects 'column' (singular) vs 'columns' (plural)
                        try:
                            sig = inspect.signature(validator_cls.__init__)
                            validator_params = list(sig.parameters.keys())
                            expects_column = "column" in validator_params
                            expects_columns = "columns" in validator_params
                        except (ValueError, TypeError):
                            expects_column = False
                            expects_columns = False

                        # Add columns only if not already consumed by transformer
                        final_params = dict(params)
                        if rule.columns and "columns" not in final_params and "column" not in final_params:
                            if "expected_types" not in final_params:
                                if expects_column and not expects_columns and len(rule.columns) == 1:
                                    # Single-column validator: use 'column' parameter
                                    final_params["column"] = rule.columns[0]
                                else:
                                    # Multi-column validator or kwargs-based: use 'columns'
                                    final_params["columns"] = list(rule.columns)
                        if rule.mostly is not None:
                            final_params["mostly"] = rule.mostly

                        return validator_cls(**final_params)
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
