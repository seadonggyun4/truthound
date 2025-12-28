"""Decorators for custom validator registration and metadata.

This module provides decorators that simplify validator development by:
- Automatic registration in the global validator registry
- Metadata annotation for documentation and discovery
- Deprecation warnings for legacy validators
- Validation of class structure at decoration time

Example:
    @custom_validator(
        name="email_format",
        category="string",
        description="Validates email address format",
        tags=["format", "string", "email"],
    )
    class EmailFormatValidator(Validator, StringValidatorMixin):
        def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
            ...
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

# Global validator registry
_VALIDATOR_REGISTRY: dict[str, type] = {}
_VALIDATOR_METADATA: dict[str, "ValidatorMeta"] = {}

T = TypeVar("T")


@dataclass(frozen=True)
class ValidatorMeta:
    """Metadata for a registered validator.

    Attributes:
        name: Unique validator name
        category: Validator category (schema, string, numeric, etc.)
        description: Human-readable description
        version: Semantic version string
        author: Author name or email
        tags: List of tags for discovery
        deprecated: Whether the validator is deprecated
        deprecated_message: Message to show when deprecated
        replacement: Name of replacement validator if deprecated
        examples: List of usage examples
        config_schema: JSON schema for validator config
    """

    name: str
    category: str = "general"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    deprecated: bool = False
    deprecated_message: str = ""
    replacement: str = ""
    examples: tuple[str, ...] = field(default_factory=tuple)
    config_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": list(self.tags),
            "deprecated": self.deprecated,
            "deprecated_message": self.deprecated_message,
            "replacement": self.replacement,
            "examples": list(self.examples),
            "config_schema": self.config_schema,
        }


def custom_validator(
    name: str,
    category: str = "custom",
    description: str = "",
    version: str = "1.0.0",
    author: str = "",
    tags: list[str] | None = None,
    examples: list[str] | None = None,
    config_schema: dict[str, Any] | None = None,
    auto_register: bool = True,
) -> Callable[[type[T]], type[T]]:
    """Decorator to define and register a custom validator.

    This is the primary decorator for creating custom validators. It:
    1. Sets the `name` and `category` class attributes
    2. Stores metadata for documentation
    3. Optionally registers in the global registry

    Args:
        name: Unique validator name (used in config and CLI)
        category: Category for grouping (schema, string, numeric, etc.)
        description: Human-readable description
        version: Semantic version (for plugin compatibility)
        author: Author name or email
        tags: List of tags for filtering and discovery
        examples: Usage examples for documentation
        config_schema: JSON schema for validator configuration
        auto_register: Whether to auto-register (default True)

    Returns:
        Decorated class with name/category set and registered

    Example:
        @custom_validator(
            name="percentage_range",
            category="numeric",
            description="Validates values are valid percentages (0-100)",
            tags=["numeric", "range", "percentage"],
        )
        class PercentageValidator(Validator, NumericValidatorMixin):
            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                issues = []
                for col in self._get_numeric_columns(lf):
                    # Check values are in [0, 100]
                    ...
                return issues

    Note:
        The validator name must be unique. If a validator with the same
        name is already registered, a ValueError will be raised.
    """

    def decorator(cls: type[T]) -> type[T]:
        # Validate the class has required methods
        if not hasattr(cls, "validate"):
            raise TypeError(
                f"Validator class {cls.__name__} must have a 'validate' method"
            )

        # Set class attributes
        cls.name = name  # type: ignore
        cls.category = category  # type: ignore

        # Store metadata
        meta = ValidatorMeta(
            name=name,
            category=category,
            description=description or cls.__doc__ or "",
            version=version,
            author=author,
            tags=tuple(tags or []),
            examples=tuple(examples or []),
            config_schema=config_schema,
        )
        _VALIDATOR_METADATA[name] = meta

        # Store reference to metadata on class
        cls._validator_meta = meta  # type: ignore

        # Register if requested
        if auto_register:
            if name in _VALIDATOR_REGISTRY:
                raise ValueError(
                    f"Validator '{name}' is already registered. "
                    f"Use a different name or set auto_register=False."
                )
            _VALIDATOR_REGISTRY[name] = cls

        return cls

    return decorator


def register_validator(cls: type[T]) -> type[T]:
    """Simple decorator to register an existing validator class.

    Use this when you have a validator class that already has `name`
    and `category` attributes set, and you just want to register it.

    Args:
        cls: Validator class to register

    Returns:
        The same class, now registered

    Example:
        @register_validator
        class MyValidator(Validator):
            name = "my_validator"
            category = "custom"

            def validate(self, lf):
                ...
    """
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(
            f"Validator class {cls.__name__} must have a 'name' attribute"
        )

    if name in _VALIDATOR_REGISTRY:
        raise ValueError(f"Validator '{name}' is already registered")

    _VALIDATOR_REGISTRY[name] = cls

    # Create basic metadata if not present
    if name not in _VALIDATOR_METADATA:
        _VALIDATOR_METADATA[name] = ValidatorMeta(
            name=name,
            category=getattr(cls, "category", "general"),
            description=cls.__doc__ or "",
        )

    return cls


def validator_metadata(
    description: str = "",
    version: str = "1.0.0",
    author: str = "",
    tags: list[str] | None = None,
    examples: list[str] | None = None,
    config_schema: dict[str, Any] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to add metadata to an existing validator.

    Use this in combination with @register_validator when you want
    to add detailed metadata to a validator that's already defined.

    Args:
        description: Human-readable description
        version: Semantic version string
        author: Author name or email
        tags: List of tags for discovery
        examples: Usage examples
        config_schema: JSON schema for config

    Returns:
        Decorator function

    Example:
        @validator_metadata(
            description="Validates phone number format",
            tags=["string", "phone", "format"],
            examples=[
                "PhoneValidator(format='US')",
                "PhoneValidator(format='KR')",
            ],
        )
        @register_validator
        class PhoneValidator(Validator):
            name = "phone"
            category = "string"
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        name = getattr(cls, "name", cls.__name__)
        category = getattr(cls, "category", "general")

        meta = ValidatorMeta(
            name=name,
            category=category,
            description=description or cls.__doc__ or "",
            version=version,
            author=author,
            tags=tuple(tags or []),
            examples=tuple(examples or []),
            config_schema=config_schema,
        )
        _VALIDATOR_METADATA[name] = meta
        cls._validator_meta = meta  # type: ignore

        return cls

    return decorator


def deprecated_validator(
    message: str = "",
    replacement: str = "",
    remove_in_version: str = "",
) -> Callable[[type[T]], type[T]]:
    """Mark a validator as deprecated.

    When the deprecated validator is instantiated, a deprecation warning
    will be issued. Use this to guide users to newer alternatives.

    Args:
        message: Custom deprecation message
        replacement: Name of the replacement validator
        remove_in_version: Version when this will be removed

    Returns:
        Decorator function

    Example:
        @deprecated_validator(
            message="Use 'email_v2' for RFC 5322 compliance",
            replacement="email_v2",
            remove_in_version="2.0.0",
        )
        class OldEmailValidator(Validator):
            name = "email_v1"
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # Build warning message
            name = getattr(cls, "name", cls.__name__)
            warn_msg = f"Validator '{name}' is deprecated."

            if message:
                warn_msg += f" {message}"
            if replacement:
                warn_msg += f" Use '{replacement}' instead."
            if remove_in_version:
                warn_msg += f" Will be removed in version {remove_in_version}."

            warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init  # type: ignore

        # Update metadata if exists
        name = getattr(cls, "name", cls.__name__)
        if name in _VALIDATOR_METADATA:
            old_meta = _VALIDATOR_METADATA[name]
            _VALIDATOR_METADATA[name] = ValidatorMeta(
                name=old_meta.name,
                category=old_meta.category,
                description=old_meta.description,
                version=old_meta.version,
                author=old_meta.author,
                tags=old_meta.tags,
                deprecated=True,
                deprecated_message=message,
                replacement=replacement,
                examples=old_meta.examples,
                config_schema=old_meta.config_schema,
            )

        return cls

    return decorator


# ============================================================================
# Registry Access Functions
# ============================================================================


def get_registered_validators() -> dict[str, type]:
    """Get all registered validators.

    Returns:
        Dictionary mapping validator names to classes
    """
    return _VALIDATOR_REGISTRY.copy()


def get_validator_by_name(name: str) -> type | None:
    """Get a validator class by name.

    Args:
        name: Validator name

    Returns:
        Validator class or None if not found
    """
    return _VALIDATOR_REGISTRY.get(name)


def get_validator_metadata(name: str) -> ValidatorMeta | None:
    """Get metadata for a validator.

    Args:
        name: Validator name

    Returns:
        ValidatorMeta or None if not found
    """
    return _VALIDATOR_METADATA.get(name)


def get_validators_by_category(category: str) -> list[type]:
    """Get all validators in a category.

    Args:
        category: Category name

    Returns:
        List of validator classes
    """
    return [
        cls
        for name, cls in _VALIDATOR_REGISTRY.items()
        if getattr(cls, "category", "") == category
    ]


def get_validators_by_tag(tag: str) -> list[type]:
    """Get all validators with a specific tag.

    Args:
        tag: Tag to search for

    Returns:
        List of validator classes
    """
    result = []
    for name, cls in _VALIDATOR_REGISTRY.items():
        meta = _VALIDATOR_METADATA.get(name)
        if meta and tag in meta.tags:
            result.append(cls)
    return result


def list_validator_categories() -> list[str]:
    """Get all unique validator categories.

    Returns:
        Sorted list of category names
    """
    categories = {
        getattr(cls, "category", "general")
        for cls in _VALIDATOR_REGISTRY.values()
    }
    return sorted(categories)


def list_validator_tags() -> list[str]:
    """Get all unique validator tags.

    Returns:
        Sorted list of tag names
    """
    tags: set[str] = set()
    for meta in _VALIDATOR_METADATA.values():
        tags.update(meta.tags)
    return sorted(tags)


def unregister_validator(name: str) -> bool:
    """Unregister a validator by name.

    Useful for testing or replacing validators at runtime.

    Args:
        name: Validator name to unregister

    Returns:
        True if validator was unregistered, False if not found
    """
    if name in _VALIDATOR_REGISTRY:
        del _VALIDATOR_REGISTRY[name]
        if name in _VALIDATOR_METADATA:
            del _VALIDATOR_METADATA[name]
        return True
    return False


def clear_registry() -> None:
    """Clear all registered validators.

    WARNING: This is primarily for testing. Use with caution.
    """
    _VALIDATOR_REGISTRY.clear()
    _VALIDATOR_METADATA.clear()
