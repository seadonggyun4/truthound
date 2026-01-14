"""Rule combinators for composing complex routing conditions.

This module provides logical combinators for building complex rules:
- AllOf (AND): All rules must match
- AnyOf (OR): At least one rule must match
- NotRule (NOT): Inverts a rule's result
- AtLeast: At least N rules must match
- Exactly: Exactly N rules must match

These combinators can be nested to create arbitrarily complex conditions.

Example:
    >>> from truthound.checkpoint.routing import AllOf, AnyOf, NotRule
    >>> from truthound.checkpoint.routing.rules import SeverityRule, TagRule
    >>>
    >>> # Complex rule: critical issues in prod, or high issues outside business hours
    >>> rule = AnyOf([
    ...     AllOf([
    ...         SeverityRule(min_severity="critical"),
    ...         TagRule(tags={"env": "prod"}),
    ...     ]),
    ...     AllOf([
    ...         SeverityRule(min_severity="high"),
    ...         NotRule(TimeWindowRule(start_time="09:00", end_time="17:00")),
    ...     ]),
    ... ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.checkpoint.routing.base import RouteContext, RoutingRule


@dataclass
class AllOf:
    """Logical AND combinator - all rules must match.

    Evaluates rules in order and short-circuits on first failure.

    Attributes:
        rules: List of rules that must all match
        description_separator: Separator for combining rule descriptions

    Example:
        >>> rule = AllOf([
        ...     SeverityRule(min_severity="critical"),
        ...     TagRule(tags={"env": "prod"}),
        ...     StatusRule(statuses=["failure"]),
        ... ])
        >>> # Matches only when ALL conditions are true
    """

    rules: list["RoutingRule"] = field(default_factory=list)
    description_separator: str = " AND "

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate all rules with short-circuit AND logic.

        Args:
            context: The routing context.

        Returns:
            True if all rules match, False otherwise.
        """
        if not self.rules:
            return True

        for rule in self.rules:
            if not rule.evaluate(context):
                return False

        return True

    def add(self, rule: "RoutingRule") -> "AllOf":
        """Add a rule to the combinator.

        Args:
            rule: Rule to add.

        Returns:
            Self for chaining.
        """
        self.rules.append(rule)
        return self

    @property
    def description(self) -> str:
        """Get combined description of all rules."""
        if not self.rules:
            return "always (empty AllOf)"

        descriptions = [f"({r.description})" for r in self.rules]
        return self.description_separator.join(descriptions)

    def __len__(self) -> int:
        return len(self.rules)


@dataclass
class AnyOf:
    """Logical OR combinator - at least one rule must match.

    Evaluates rules in order and short-circuits on first match.

    Attributes:
        rules: List of rules where at least one must match
        description_separator: Separator for combining rule descriptions

    Example:
        >>> rule = AnyOf([
        ...     SeverityRule(min_severity="critical"),
        ...     TagRule(tags={"escalate": "true"}),
        ... ])
        >>> # Matches when ANY condition is true
    """

    rules: list["RoutingRule"] = field(default_factory=list)
    description_separator: str = " OR "

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate rules with short-circuit OR logic.

        Args:
            context: The routing context.

        Returns:
            True if any rule matches, False otherwise.
        """
        if not self.rules:
            return False

        for rule in self.rules:
            if rule.evaluate(context):
                return True

        return False

    def add(self, rule: "RoutingRule") -> "AnyOf":
        """Add a rule to the combinator.

        Args:
            rule: Rule to add.

        Returns:
            Self for chaining.
        """
        self.rules.append(rule)
        return self

    @property
    def description(self) -> str:
        """Get combined description of all rules."""
        if not self.rules:
            return "never (empty AnyOf)"

        descriptions = [f"({r.description})" for r in self.rules]
        return self.description_separator.join(descriptions)

    def __len__(self) -> int:
        return len(self.rules)


@dataclass
class NotRule:
    """Logical NOT combinator - inverts a rule's result.

    Attributes:
        rule: The rule to invert

    Example:
        >>> # Match when NOT in production
        >>> rule = NotRule(TagRule(tags={"env": "prod"}))
        >>>
        >>> # Match outside business hours
        >>> rule = NotRule(TimeWindowRule(start_time="09:00", end_time="17:00"))
    """

    rule: "RoutingRule"

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate the rule and invert the result.

        Args:
            context: The routing context.

        Returns:
            Inverted result of the wrapped rule.
        """
        return not self.rule.evaluate(context)

    @property
    def description(self) -> str:
        """Get description with NOT prefix."""
        return f"NOT ({self.rule.description})"


@dataclass
class AtLeast:
    """Combinator requiring at least N rules to match.

    Useful when you want flexible matching, e.g., "at least 2 of these
    3 conditions must be true".

    Attributes:
        rules: List of rules to evaluate
        count: Minimum number that must match

    Example:
        >>> # Match if at least 2 of these 3 conditions are true
        >>> rule = AtLeast(
        ...     rules=[
        ...         SeverityRule(min_severity="critical"),
        ...         TagRule(tags={"env": "prod"}),
        ...         IssueCountRule(min_issues=10),
        ...     ],
        ...     count=2,
        ... )
    """

    rules: list["RoutingRule"] = field(default_factory=list)
    count: int = 1

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if at least N rules match.

        Args:
            context: The routing context.

        Returns:
            True if at least count rules match.
        """
        if not self.rules:
            return self.count <= 0

        matches = 0
        for rule in self.rules:
            if rule.evaluate(context):
                matches += 1
                if matches >= self.count:
                    return True  # Short-circuit

        return matches >= self.count

    def add(self, rule: "RoutingRule") -> "AtLeast":
        """Add a rule to the combinator.

        Args:
            rule: Rule to add.

        Returns:
            Self for chaining.
        """
        self.rules.append(rule)
        return self

    @property
    def description(self) -> str:
        """Get combined description."""
        if not self.rules:
            return f"at least {self.count} of (empty)"

        descriptions = [r.description for r in self.rules]
        return f"at least {self.count} of: [{', '.join(descriptions)}]"

    def __len__(self) -> int:
        return len(self.rules)


@dataclass
class Exactly:
    """Combinator requiring exactly N rules to match.

    Attributes:
        rules: List of rules to evaluate
        count: Exact number that must match

    Example:
        >>> # Match if exactly 1 of these conditions is true
        >>> rule = Exactly(
        ...     rules=[
        ...         StatusRule(statuses=["success"]),
        ...         StatusRule(statuses=["warning"]),
        ...         StatusRule(statuses=["failure"]),
        ...     ],
        ...     count=1,
        ... )
    """

    rules: list["RoutingRule"] = field(default_factory=list)
    count: int = 1

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if exactly N rules match.

        Args:
            context: The routing context.

        Returns:
            True if exactly count rules match.
        """
        if not self.rules:
            return self.count == 0

        matches = sum(1 for rule in self.rules if rule.evaluate(context))
        return matches == self.count

    def add(self, rule: "RoutingRule") -> "Exactly":
        """Add a rule to the combinator.

        Args:
            rule: Rule to add.

        Returns:
            Self for chaining.
        """
        self.rules.append(rule)
        return self

    @property
    def description(self) -> str:
        """Get combined description."""
        if not self.rules:
            return f"exactly {self.count} of (empty)"

        descriptions = [r.description for r in self.rules]
        return f"exactly {self.count} of: [{', '.join(descriptions)}]"

    def __len__(self) -> int:
        return len(self.rules)


@dataclass
class NoneOf:
    """Combinator requiring no rules to match (equivalent to NOT(AnyOf)).

    Attributes:
        rules: List of rules where none should match

    Example:
        >>> # Match if none of these are true (success in non-prod)
        >>> rule = NoneOf([
        ...     SeverityRule(min_severity="high"),
        ...     TagRule(tags={"env": "prod"}),
        ... ])
    """

    rules: list["RoutingRule"] = field(default_factory=list)

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if no rules match.

        Args:
            context: The routing context.

        Returns:
            True if no rules match.
        """
        for rule in self.rules:
            if rule.evaluate(context):
                return False
        return True

    def add(self, rule: "RoutingRule") -> "NoneOf":
        """Add a rule to the combinator.

        Args:
            rule: Rule to add.

        Returns:
            Self for chaining.
        """
        self.rules.append(rule)
        return self

    @property
    def description(self) -> str:
        """Get combined description."""
        if not self.rules:
            return "always (empty NoneOf)"

        descriptions = [r.description for r in self.rules]
        return f"none of: [{', '.join(descriptions)}]"

    def __len__(self) -> int:
        return len(self.rules)


@dataclass
class Conditional:
    """Conditional rule evaluation: if A then B, else C.

    Attributes:
        condition: The condition to check
        if_true: Rule to evaluate if condition is true
        if_false: Rule to evaluate if condition is false

    Example:
        >>> # If production, require critical; otherwise require high
        >>> rule = Conditional(
        ...     condition=TagRule(tags={"env": "prod"}),
        ...     if_true=SeverityRule(min_severity="critical"),
        ...     if_false=SeverityRule(min_severity="high"),
        ... )
    """

    condition: "RoutingRule"
    if_true: "RoutingRule"
    if_false: "RoutingRule | None" = None

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate condition and appropriate branch.

        Args:
            context: The routing context.

        Returns:
            Result of the appropriate branch evaluation.
        """
        if self.condition.evaluate(context):
            return self.if_true.evaluate(context)
        elif self.if_false is not None:
            return self.if_false.evaluate(context)
        else:
            return False

    @property
    def description(self) -> str:
        """Get combined description."""
        parts = [
            f"if ({self.condition.description})",
            f"then ({self.if_true.description})",
        ]
        if self.if_false:
            parts.append(f"else ({self.if_false.description})")
        return " ".join(parts)


def all_of(*rules: "RoutingRule") -> AllOf:
    """Create an AllOf combinator from rules.

    Convenience function for creating AllOf with positional arguments.

    Args:
        *rules: Rules that must all match.

    Returns:
        AllOf combinator.

    Example:
        >>> rule = all_of(
        ...     SeverityRule(min_severity="critical"),
        ...     TagRule(tags={"env": "prod"}),
        ... )
    """
    return AllOf(rules=list(rules))


def any_of(*rules: "RoutingRule") -> AnyOf:
    """Create an AnyOf combinator from rules.

    Convenience function for creating AnyOf with positional arguments.

    Args:
        *rules: Rules where at least one must match.

    Returns:
        AnyOf combinator.

    Example:
        >>> rule = any_of(
        ...     SeverityRule(min_severity="critical"),
        ...     ErrorRule(),
        ... )
    """
    return AnyOf(rules=list(rules))


def not_rule(rule: "RoutingRule") -> NotRule:
    """Create a NotRule wrapper.

    Convenience function for creating NotRule.

    Args:
        rule: Rule to negate.

    Returns:
        NotRule wrapper.

    Example:
        >>> rule = not_rule(TagRule(tags={"env": "prod"}))
    """
    return NotRule(rule=rule)
