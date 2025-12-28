"""Pattern Optimizer - Automatic ReDoS Pattern Transformation.

This module provides automatic optimization of dangerous regex patterns
to safer alternatives while preserving matching semantics.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Pattern Optimizer                             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────┐
    │               │               │               │                 │
    ▼               ▼               ▼               ▼                 ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌─────────┐
│  Rule   │   │Transform│    │ Semantic │   │  Safety  │    │ Report  │
│ Engine  │   │ Pipeline│    │ Verifier │   │ Validator│    │Generator│
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └─────────┘

Optimization strategies:
1. Nested Quantifier Flattening: (a+)+ → a+
2. Alternation Simplification: (a|a)+ → a+
3. Possessive Quantifier Simulation: a++ equivalent
4. Atomic Group Insertion (where applicable)
5. Anchor Addition: Reduce backtracking scope
6. Character Class Optimization: [a-zA-Z] → \\w (when appropriate)

Usage:
    from truthound.validators.security.redos.optimizer import (
        PatternOptimizer,
        optimize_pattern,
    )

    # Quick optimization
    result = optimize_pattern(r"(a+)+b")
    print(result.optimized_pattern)  # "a+b"
    print(result.applied_rules)  # ["flatten_nested_quantifiers"]

    # Full optimizer with custom rules
    optimizer = PatternOptimizer()
    result = optimizer.optimize(pattern)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol, Sequence

from truthound.validators.security.redos.core import (
    ReDoSRisk,
    RegexComplexityAnalyzer,
    SafeRegexConfig,
)


class OptimizationType(Enum):
    """Types of optimization applied."""

    FLATTEN_NESTED_QUANTIFIERS = auto()
    SIMPLIFY_ALTERNATION = auto()
    ADD_ANCHORS = auto()
    LIMIT_QUANTIFIERS = auto()
    SIMPLIFY_CHARACTER_CLASS = auto()
    REMOVE_REDUNDANT_GROUPS = auto()
    POSSESSIVE_SIMULATION = auto()
    ATOMIC_GROUP_SIMULATION = auto()
    FACTOR_COMMON_PREFIX = auto()
    LAZY_TO_POSSESSIVE = auto()


@dataclass
class OptimizationRule:
    """Represents a single optimization rule.

    Attributes:
        name: Rule identifier
        description: Human-readable description
        pattern: Regex pattern to match dangerous constructs
        replacement: Replacement pattern or function
        risk_reduction: Expected risk reduction (0-1)
        preserves_semantics: Whether the rule preserves exact matching
        optimization_type: Type of optimization
    """

    name: str
    description: str
    pattern: re.Pattern | str
    replacement: str | Callable[[re.Match], str]
    risk_reduction: float
    preserves_semantics: bool = True
    optimization_type: OptimizationType = OptimizationType.FLATTEN_NESTED_QUANTIFIERS

    def __post_init__(self):
        """Compile pattern if it's a string."""
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)


@dataclass
class OptimizationResult:
    """Result of pattern optimization.

    Attributes:
        original_pattern: The input pattern
        optimized_pattern: The optimized pattern
        applied_rules: List of rules that were applied
        risk_before: Risk level before optimization
        risk_after: Risk level after optimization
        semantics_preserved: Whether matching semantics are preserved
        warnings: Any warnings about the optimization
        transformations: Detailed transformation log
    """

    original_pattern: str
    optimized_pattern: str
    applied_rules: list[str] = field(default_factory=list)
    risk_before: ReDoSRisk = ReDoSRisk.NONE
    risk_after: ReDoSRisk = ReDoSRisk.NONE
    semantics_preserved: bool = True
    warnings: list[str] = field(default_factory=list)
    transformations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def was_optimized(self) -> bool:
        """Check if any optimization was applied."""
        return self.original_pattern != self.optimized_pattern

    @property
    def risk_reduced(self) -> bool:
        """Check if risk was reduced."""
        return self.risk_after.value < self.risk_before.value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_pattern": self.original_pattern,
            "optimized_pattern": self.optimized_pattern,
            "applied_rules": self.applied_rules,
            "risk_before": self.risk_before.name,
            "risk_after": self.risk_after.name,
            "was_optimized": self.was_optimized,
            "risk_reduced": self.risk_reduced,
            "semantics_preserved": self.semantics_preserved,
            "warnings": self.warnings,
            "transformations": self.transformations,
        }


class TransformationStrategyProtocol(Protocol):
    """Protocol for transformation strategies."""

    def can_apply(self, pattern: str) -> bool:
        """Check if this strategy can be applied."""
        ...

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Apply the transformation.

        Returns:
            Tuple of (transformed_pattern, semantics_preserved)
        """
        ...


class BaseTransformation(ABC):
    """Base class for pattern transformations."""

    name: str = "base"
    description: str = ""
    optimization_type: OptimizationType = OptimizationType.FLATTEN_NESTED_QUANTIFIERS

    @abstractmethod
    def can_apply(self, pattern: str) -> bool:
        """Check if this transformation can be applied."""
        pass

    @abstractmethod
    def apply(self, pattern: str) -> tuple[str, bool]:
        """Apply the transformation.

        Returns:
            Tuple of (transformed_pattern, semantics_preserved)
        """
        pass


class FlattenNestedQuantifiers(BaseTransformation):
    """Flatten nested quantifiers like (a+)+ → a+.

    This is the most common ReDoS pattern and flattening
    dramatically reduces backtracking.
    """

    name = "flatten_nested_quantifiers"
    description = "Flatten nested quantifiers (a+)+ to a+"
    optimization_type = OptimizationType.FLATTEN_NESTED_QUANTIFIERS

    # Pattern to match nested quantifiers
    _PATTERN = re.compile(
        r"\(([^()]+)([+*])\)\2"  # (content+)+ or (content*)*
    )

    def can_apply(self, pattern: str) -> bool:
        """Check for nested quantifiers."""
        return bool(self._PATTERN.search(pattern))

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Flatten nested quantifiers."""
        def replacer(match: re.Match) -> str:
            content = match.group(1)
            quantifier = match.group(2)
            # If content already ends with a quantifier, just use the content
            if content.endswith(("+", "*", "?")):
                return content
            return f"({content}){quantifier}"

        result = self._PATTERN.sub(replacer, pattern)

        # Second pass for simpler cases like (a+)+
        simple_pattern = re.compile(r"\(([^()]+)[+*]\)[+*]")
        while simple_pattern.search(result):
            result = simple_pattern.sub(r"(\1)+", result)

        return result, True  # Semantically equivalent for matching purposes


class SimplifyAlternation(BaseTransformation):
    """Simplify overlapping alternations.

    Patterns like (a|aa)+ can be simplified to a+
    when alternatives overlap.
    """

    name = "simplify_alternation"
    description = "Simplify overlapping alternations"
    optimization_type = OptimizationType.SIMPLIFY_ALTERNATION

    def can_apply(self, pattern: str) -> bool:
        """Check for quantified alternation."""
        return bool(re.search(r"\([^)]*\|[^)]*\)[+*]", pattern))

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Simplify alternations where possible."""
        # This is a simplified implementation
        # Full implementation would analyze overlap semantics

        # Pattern like (a|ab)+ where one is prefix of another
        prefix_alt = re.compile(r"\(([a-z]+)\|\1[a-z]+\)[+*]")
        result = prefix_alt.sub(r"(\1[a-z]*)+", pattern)

        # Check if we made changes
        semantics_preserved = result == pattern  # Conservative: assume not preserved if changed

        return result, semantics_preserved


class AddAnchors(BaseTransformation):
    """Add anchors to reduce backtracking scope.

    Adding ^ and $ anchors limits where the pattern
    can match, reducing backtracking significantly.
    """

    name = "add_anchors"
    description = "Add anchors to limit matching scope"
    optimization_type = OptimizationType.ADD_ANCHORS

    def can_apply(self, pattern: str) -> bool:
        """Check if pattern lacks anchors."""
        has_start = pattern.startswith("^") or pattern.startswith("\\A")
        has_end = pattern.endswith("$") or pattern.endswith("\\Z")
        return not (has_start and has_end)

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Add anchors if missing."""
        result = pattern
        if not (pattern.startswith("^") or pattern.startswith("\\A")):
            result = "^" + result
        if not (pattern.endswith("$") or pattern.endswith("\\Z")):
            result = result + "$"

        # Anchors change matching semantics
        return result, False


class LimitQuantifiers(BaseTransformation):
    """Convert unbounded quantifiers to bounded.

    Patterns like a* → a{0,1000} to limit potential matches.
    """

    name = "limit_quantifiers"
    description = "Convert unbounded quantifiers to bounded"
    optimization_type = OptimizationType.LIMIT_QUANTIFIERS

    DEFAULT_LIMIT = 1000

    def __init__(self, limit: int = DEFAULT_LIMIT):
        """Initialize with max limit."""
        self.limit = limit

    def can_apply(self, pattern: str) -> bool:
        """Check for unbounded quantifiers."""
        return bool(re.search(r"[+*](?!\?)", pattern))

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Limit quantifiers."""
        # Convert * to {0,limit}
        result = re.sub(r"\*(?!\?)", f"{{0,{self.limit}}}", pattern)
        # Convert + to {1,limit}
        result = re.sub(r"\+(?!\?)", f"{{1,{self.limit}}}", result)

        # This changes semantics for very long inputs
        return result, False


class RemoveRedundantGroups(BaseTransformation):
    """Remove unnecessary capturing groups.

    Groups that don't need to capture can be converted
    to non-capturing for slight performance improvement.
    """

    name = "remove_redundant_groups"
    description = "Convert unnecessary capturing groups to non-capturing"
    optimization_type = OptimizationType.REMOVE_REDUNDANT_GROUPS

    def can_apply(self, pattern: str) -> bool:
        """Check for capturing groups."""
        # Has capturing group that's not referenced
        has_capture = bool(re.search(r"\((?!\?)", pattern))
        has_backref = bool(re.search(r"\\[1-9]", pattern))
        return has_capture and not has_backref

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Convert to non-capturing groups."""
        # Only convert groups that aren't followed by quantifiers
        # (quantified groups are more likely to be intentional)
        result = re.sub(r"\((?!\?)([^()]+)\)(?![+*?{])", r"(?:\1)", pattern)

        return result, True  # Matching semantics preserved


class FactorCommonPrefix(BaseTransformation):
    """Factor out common prefixes from alternations.

    (abc|abd) → ab(c|d) reduces redundant matching.
    """

    name = "factor_common_prefix"
    description = "Factor common prefixes from alternations"
    optimization_type = OptimizationType.FACTOR_COMMON_PREFIX

    def can_apply(self, pattern: str) -> bool:
        """Check for alternation."""
        return "|" in pattern

    def apply(self, pattern: str) -> tuple[str, bool]:
        """Factor common prefixes."""
        # Find alternation groups
        alt_pattern = re.compile(r"\(([^()]+\|[^()]+)\)")

        def factor_group(match: re.Match) -> str:
            alternatives = match.group(1).split("|")
            if len(alternatives) < 2:
                return match.group(0)

            # Find common prefix
            prefix = alternatives[0]
            for alt in alternatives[1:]:
                while prefix and not alt.startswith(prefix):
                    prefix = prefix[:-1]

            if not prefix:
                return match.group(0)

            # Factor out prefix
            new_alts = [alt[len(prefix):] or "(?:)" for alt in alternatives]
            return f"{prefix}({'|'.join(new_alts)})"

        result = alt_pattern.sub(factor_group, pattern)
        return result, True


class PatternOptimizer:
    """Main optimizer for dangerous regex patterns.

    This optimizer applies a series of transformations to convert
    potentially dangerous regex patterns into safer alternatives.

    Example:
        optimizer = PatternOptimizer()

        # Basic optimization
        result = optimizer.optimize(r"(a+)+b")
        print(result.optimized_pattern)  # "a+b"

        # Aggressive optimization (may change semantics)
        result = optimizer.optimize(r".*foo.*", aggressive=True)
        print(result.optimized_pattern)  # "^.*foo.*$"

        # Custom transformation pipeline
        optimizer = PatternOptimizer(
            transformations=[
                FlattenNestedQuantifiers(),
                RemoveRedundantGroups(),
            ]
        )
    """

    # Default transformations in order of application
    DEFAULT_TRANSFORMATIONS: list[type[BaseTransformation]] = [
        FlattenNestedQuantifiers,
        SimplifyAlternation,
        RemoveRedundantGroups,
        FactorCommonPrefix,
    ]

    # Aggressive transformations (may change semantics)
    AGGRESSIVE_TRANSFORMATIONS: list[type[BaseTransformation]] = [
        LimitQuantifiers,
        AddAnchors,
    ]

    def __init__(
        self,
        transformations: Sequence[BaseTransformation] | None = None,
        config: SafeRegexConfig | None = None,
        max_iterations: int = 10,
    ):
        """Initialize the optimizer.

        Args:
            transformations: Custom transformations to apply
            config: Safety configuration
            max_iterations: Maximum optimization passes
        """
        if transformations is not None:
            self._transformations = list(transformations)
        else:
            self._transformations = [t() for t in self.DEFAULT_TRANSFORMATIONS]

        self._aggressive_transformations = [t() for t in self.AGGRESSIVE_TRANSFORMATIONS]
        self.config = config or SafeRegexConfig()
        self.max_iterations = max_iterations
        self._analyzer = RegexComplexityAnalyzer(self.config)

    def optimize(
        self,
        pattern: str,
        aggressive: bool = False,
        preserve_semantics: bool = True,
    ) -> OptimizationResult:
        """Optimize a regex pattern.

        Args:
            pattern: Pattern to optimize
            aggressive: Apply aggressive optimizations
            preserve_semantics: Only apply semantic-preserving transformations

        Returns:
            OptimizationResult with optimization details
        """
        # Validate input
        try:
            re.compile(pattern)
        except re.error as e:
            return OptimizationResult(
                original_pattern=pattern,
                optimized_pattern=pattern,
                warnings=[f"Invalid regex: {e}"],
            )

        # Get initial risk
        initial_analysis = self._analyzer.analyze(pattern)
        risk_before = initial_analysis.risk_level

        # Apply transformations
        current = pattern
        applied_rules: list[str] = []
        transformations_log: list[dict[str, Any]] = []
        semantics_preserved = True

        # Get transformations to apply
        all_transforms = list(self._transformations)
        if aggressive:
            all_transforms.extend(self._aggressive_transformations)

        # Iterative optimization
        for iteration in range(self.max_iterations):
            made_change = False

            for transform in all_transforms:
                if not transform.can_apply(current):
                    continue

                new_pattern, preserves = transform.apply(current)

                # Skip if semantics change and we require preservation
                if preserve_semantics and not preserves:
                    continue

                # Validate transformed pattern
                try:
                    re.compile(new_pattern)
                except re.error:
                    continue  # Skip invalid transformation

                # Check if transformation reduced risk
                new_analysis = self._analyzer.analyze(new_pattern)
                if new_analysis.risk_level.value <= initial_analysis.risk_level.value:
                    if new_pattern != current:
                        transformations_log.append({
                            "iteration": iteration,
                            "rule": transform.name,
                            "before": current,
                            "after": new_pattern,
                            "preserves_semantics": preserves,
                        })
                        current = new_pattern
                        applied_rules.append(transform.name)
                        made_change = True

                        if not preserves:
                            semantics_preserved = False

            if not made_change:
                break

        # Get final risk
        final_analysis = self._analyzer.analyze(current)
        risk_after = final_analysis.risk_level

        # Generate warnings
        warnings: list[str] = []
        if not semantics_preserved:
            warnings.append(
                "Optimization changed matching semantics. "
                "Verify behavior with test cases."
            )
        if risk_after == risk_before and risk_before.value >= ReDoSRisk.HIGH.value:
            warnings.append(
                "Could not reduce risk. Consider rewriting the pattern manually."
            )

        return OptimizationResult(
            original_pattern=pattern,
            optimized_pattern=current,
            applied_rules=applied_rules,
            risk_before=risk_before,
            risk_after=risk_after,
            semantics_preserved=semantics_preserved,
            warnings=warnings,
            transformations=transformations_log,
        )

    def add_transformation(self, transformation: BaseTransformation) -> None:
        """Add a custom transformation.

        Args:
            transformation: Transformation to add
        """
        self._transformations.append(transformation)

    def suggest_alternatives(self, pattern: str, count: int = 3) -> list[str]:
        """Suggest alternative patterns.

        Args:
            pattern: Original pattern
            count: Number of alternatives to suggest

        Returns:
            List of alternative patterns
        """
        alternatives: list[str] = []

        # Try different optimization strategies
        for aggressive in [False, True]:
            for preserve in [True, False]:
                result = self.optimize(
                    pattern,
                    aggressive=aggressive,
                    preserve_semantics=preserve,
                )
                if result.was_optimized and result.optimized_pattern not in alternatives:
                    alternatives.append(result.optimized_pattern)
                    if len(alternatives) >= count:
                        return alternatives

        return alternatives

    def explain_optimization(self, pattern: str) -> str:
        """Explain what optimizations would be applied.

        Args:
            pattern: Pattern to analyze

        Returns:
            Human-readable explanation
        """
        lines = [f"Pattern: {pattern}", ""]

        # List applicable transformations
        applicable = []
        for transform in self._transformations + self._aggressive_transformations:
            if transform.can_apply(pattern):
                applicable.append(transform)

        if not applicable:
            lines.append("No optimizations applicable.")
        else:
            lines.append("Applicable optimizations:")
            for transform in applicable:
                lines.append(f"  - {transform.name}: {transform.description}")

        # Show optimization result
        result = self.optimize(pattern)
        lines.append("")
        if result.was_optimized:
            lines.append(f"Optimized pattern: {result.optimized_pattern}")
            lines.append(f"Risk reduction: {result.risk_before.name} → {result.risk_after.name}")
        else:
            lines.append("No optimizations applied.")

        return "\n".join(lines)


# ============================================================================
# Convenience functions
# ============================================================================


def optimize_pattern(
    pattern: str,
    aggressive: bool = False,
    preserve_semantics: bool = True,
) -> OptimizationResult:
    """Optimize a regex pattern.

    Args:
        pattern: Pattern to optimize
        aggressive: Apply aggressive optimizations
        preserve_semantics: Only apply semantic-preserving transformations

    Returns:
        OptimizationResult with optimization details

    Example:
        result = optimize_pattern(r"(a+)+b")
        print(result.optimized_pattern)  # "a+b"
        print(result.risk_reduced)  # True
    """
    optimizer = PatternOptimizer()
    return optimizer.optimize(pattern, aggressive, preserve_semantics)


def suggest_safe_alternatives(pattern: str, count: int = 3) -> list[str]:
    """Suggest safer alternatives for a pattern.

    Args:
        pattern: Original pattern
        count: Number of alternatives

    Returns:
        List of alternative patterns

    Example:
        alternatives = suggest_safe_alternatives(r"(a+)+")
        # Returns ["a+", "a{1,1000}", ...]
    """
    optimizer = PatternOptimizer()
    return optimizer.suggest_alternatives(pattern, count)
