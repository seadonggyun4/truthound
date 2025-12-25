"""Base classes for rule generators.

This module provides the foundational abstractions for the rule
generation system, enabling automatic creation of validation rules
from profile results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.profiler.base import (
        ColumnProfile,
        TableProfile,
        Strictness,
    )
    from truthound.validators.base import Validator


class RuleCategory(str, Enum):
    """Categories of generated rules."""

    SCHEMA = "schema"
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    FORMAT = "format"
    DISTRIBUTION = "distribution"
    PATTERN = "pattern"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"
    ANOMALY = "anomaly"


class RuleConfidence(str, Enum):
    """Confidence level in the generated rule."""

    LOW = "low"       # Rule might produce false positives
    MEDIUM = "medium" # Reasonable confidence
    HIGH = "high"     # High confidence rule


@dataclass(frozen=True)
class GeneratedRule:
    """Represents a generated validation rule.

    This is an immutable data structure that captures all information
    about a generated rule before it's converted to a Validator.
    """

    # Rule identification
    name: str
    validator_class: str  # Fully qualified class name or short name
    category: RuleCategory

    # Configuration
    parameters: dict[str, Any] = field(default_factory=dict)
    columns: tuple[str, ...] = field(default_factory=tuple)

    # Metadata
    confidence: RuleConfidence = RuleConfidence.MEDIUM
    description: str = ""
    rationale: str = ""  # Why this rule was generated

    # For mostly parameter (0.0 to 1.0)
    mostly: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "validator_class": self.validator_class,
            "category": self.category.value,
            "parameters": self.parameters,
            "columns": list(self.columns),
            "confidence": self.confidence.value,
            "description": self.description,
            "rationale": self.rationale,
            "mostly": self.mostly,
        }

    def to_validator_config(self) -> dict[str, Any]:
        """Convert to validator configuration dict.

        This can be used to instantiate the validator:
            config = rule.to_validator_config()
            validator = ValidatorClass(**config)
        """
        config = dict(self.parameters)
        if self.columns:
            config["columns"] = list(self.columns)
        if self.mostly is not None:
            config["mostly"] = self.mostly
        return config


class RuleGenerator(ABC):
    """Abstract base class for rule generators.

    Rule generators analyze profile results and produce validation
    rules appropriate for the data characteristics.

    Example:
        class CustomGenerator(RuleGenerator):
            name = "custom"
            categories = {RuleCategory.SCHEMA}

            def generate(self, profile, strictness):
                rules = []
                # Analysis logic...
                return rules
    """

    name: str = "base"
    description: str = "Base rule generator"
    categories: set[RuleCategory] = set()
    priority: int = 0  # Higher priority generators run first

    def __init__(self, **kwargs: Any):
        """Initialize the generator with optional configuration."""
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        profile: TableProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate validation rules from a table profile.

        Args:
            profile: Complete table profile
            strictness: How strict the generated rules should be

        Returns:
            List of generated rules
        """
        pass

    def generate_for_column(
        self,
        column: ColumnProfile,
        strictness: Strictness,
    ) -> list[GeneratedRule]:
        """Generate rules for a single column.

        Override this for column-level generators.

        Args:
            column: Column profile
            strictness: Strictness level

        Returns:
            List of generated rules
        """
        return []

    def supports_category(self, category: RuleCategory) -> bool:
        """Check if this generator produces rules in the given category."""
        return category in self.categories


class RuleGeneratorRegistry:
    """Registry for rule generators.

    Allows dynamic registration and lookup of generators.
    """

    def __init__(self) -> None:
        self._generators: dict[str, type[RuleGenerator]] = {}

    def register(
        self,
        generator_class: type[RuleGenerator],
        name: str | None = None,
    ) -> None:
        """Register a generator class."""
        key = name or generator_class.name
        self._generators[key] = generator_class

    def get(self, name: str) -> type[RuleGenerator]:
        """Get a registered generator by name."""
        if name not in self._generators:
            raise KeyError(
                f"Generator '{name}' not found. "
                f"Available: {list(self._generators.keys())}"
            )
        return self._generators[name]

    def list_all(self) -> dict[str, type[RuleGenerator]]:
        """List all registered generators."""
        return dict(self._generators)

    def get_by_category(
        self,
        category: RuleCategory,
    ) -> dict[str, type[RuleGenerator]]:
        """Get generators that support a specific category."""
        return {
            name: gen for name, gen in self._generators.items()
            if category in gen.categories
        }

    def create_all(self, **kwargs: Any) -> list[RuleGenerator]:
        """Create instances of all registered generators."""
        instances = [gen(**kwargs) for gen in self._generators.values()]
        # Sort by priority (highest first)
        instances.sort(key=lambda g: -g.priority)
        return instances


# Global registry instance
rule_generator_registry = RuleGeneratorRegistry()


def register_generator(
    name: str | None = None,
) -> Callable[[type[RuleGenerator]], type[RuleGenerator]]:
    """Decorator to register a rule generator.

    Example:
        @register_generator("custom")
        class CustomGenerator(RuleGenerator):
            ...
    """
    def decorator(cls: type[RuleGenerator]) -> type[RuleGenerator]:
        rule_generator_registry.register(cls, name)
        return cls
    return decorator


# =============================================================================
# Helper Classes
# =============================================================================


@dataclass
class StrictnessThresholds:
    """Thresholds that vary by strictness level.

    This provides a structured way to define thresholds
    that change based on the strictness setting.
    """

    # Completeness thresholds (max allowed null ratio)
    null_ratio_loose: float = 0.10
    null_ratio_medium: float = 0.05
    null_ratio_strict: float = 0.01

    # Uniqueness thresholds (min required unique ratio)
    unique_ratio_loose: float = 0.90
    unique_ratio_medium: float = 0.95
    unique_ratio_strict: float = 0.99

    # Pattern match thresholds
    pattern_match_loose: float = 0.90
    pattern_match_medium: float = 0.95
    pattern_match_strict: float = 0.99

    # Range tolerance (% outside observed range allowed)
    range_tolerance_loose: float = 0.10
    range_tolerance_medium: float = 0.05
    range_tolerance_strict: float = 0.01

    def get_null_threshold(self, strictness: Strictness) -> float:
        from truthound.profiler.base import Strictness
        mapping = {
            Strictness.LOOSE: self.null_ratio_loose,
            Strictness.MEDIUM: self.null_ratio_medium,
            Strictness.STRICT: self.null_ratio_strict,
        }
        return mapping[strictness]

    def get_unique_threshold(self, strictness: Strictness) -> float:
        from truthound.profiler.base import Strictness
        mapping = {
            Strictness.LOOSE: self.unique_ratio_loose,
            Strictness.MEDIUM: self.unique_ratio_medium,
            Strictness.STRICT: self.unique_ratio_strict,
        }
        return mapping[strictness]

    def get_pattern_threshold(self, strictness: Strictness) -> float:
        from truthound.profiler.base import Strictness
        mapping = {
            Strictness.LOOSE: self.pattern_match_loose,
            Strictness.MEDIUM: self.pattern_match_medium,
            Strictness.STRICT: self.pattern_match_strict,
        }
        return mapping[strictness]

    def get_range_tolerance(self, strictness: Strictness) -> float:
        from truthound.profiler.base import Strictness
        mapping = {
            Strictness.LOOSE: self.range_tolerance_loose,
            Strictness.MEDIUM: self.range_tolerance_medium,
            Strictness.STRICT: self.range_tolerance_strict,
        }
        return mapping[strictness]


# Default thresholds
DEFAULT_THRESHOLDS = StrictnessThresholds()


class RuleBuilder:
    """Fluent builder for creating GeneratedRule instances.

    Example:
        rule = (RuleBuilder("email_check")
            .validator("EmailValidator")
            .category(RuleCategory.FORMAT)
            .columns("email", "alt_email")
            .confidence(RuleConfidence.HIGH)
            .description("Validates email format")
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._validator_class = ""
        self._category = RuleCategory.SCHEMA
        self._parameters: dict[str, Any] = {}
        self._columns: list[str] = []
        self._confidence = RuleConfidence.MEDIUM
        self._description = ""
        self._rationale = ""
        self._mostly: float | None = None

    def validator(self, class_name: str) -> RuleBuilder:
        self._validator_class = class_name
        return self

    def category(self, cat: RuleCategory) -> RuleBuilder:
        self._category = cat
        return self

    def param(self, key: str, value: Any) -> RuleBuilder:
        self._parameters[key] = value
        return self

    def params(self, **kwargs: Any) -> RuleBuilder:
        self._parameters.update(kwargs)
        return self

    def columns(self, *cols: str) -> RuleBuilder:
        self._columns.extend(cols)
        return self

    def column(self, col: str) -> RuleBuilder:
        self._columns.append(col)
        return self

    def confidence(self, conf: RuleConfidence) -> RuleBuilder:
        self._confidence = conf
        return self

    def description(self, desc: str) -> RuleBuilder:
        self._description = desc
        return self

    def rationale(self, rat: str) -> RuleBuilder:
        self._rationale = rat
        return self

    def mostly(self, value: float) -> RuleBuilder:
        self._mostly = value
        return self

    def build(self) -> GeneratedRule:
        return GeneratedRule(
            name=self._name,
            validator_class=self._validator_class,
            category=self._category,
            parameters=self._parameters,
            columns=tuple(self._columns),
            confidence=self._confidence,
            description=self._description,
            rationale=self._rationale,
            mostly=self._mostly,
        )
