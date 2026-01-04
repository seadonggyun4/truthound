"""Rule generators for automatic validation suite creation.

This package provides generators that analyze profile results
and automatically create appropriate validation rules.
"""

from truthound.profiler.generators.base import (
    RuleGenerator,
    GeneratedRule,
    RuleGeneratorRegistry,
    rule_generator_registry,
    register_generator,
)
from truthound.profiler.generators.schema_rules import SchemaRuleGenerator
from truthound.profiler.generators.stats_rules import StatsRuleGenerator
from truthound.profiler.generators.pattern_rules import PatternRuleGenerator
from truthound.profiler.generators.ml_rules import MLRuleGenerator
from truthound.profiler.generators.suite_generator import (
    ValidationSuiteGenerator,
    ValidationSuite,
    ProfileAdapter,
    ProfileInput,
    generate_suite,
    save_suite,
    load_suite,
)

__all__ = [
    # Base
    "RuleGenerator",
    "GeneratedRule",
    "RuleGeneratorRegistry",
    "rule_generator_registry",
    "register_generator",
    # Generators
    "SchemaRuleGenerator",
    "StatsRuleGenerator",
    "PatternRuleGenerator",
    "MLRuleGenerator",
    # Suite
    "ValidationSuiteGenerator",
    "ValidationSuite",
    "ProfileAdapter",
    "ProfileInput",
    "generate_suite",
    "save_suite",
    "load_suite",
]
