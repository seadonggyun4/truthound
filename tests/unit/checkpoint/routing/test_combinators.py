"""Tests for rule combinators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest

from truthound.checkpoint.routing.base import RouteContext
from truthound.checkpoint.routing.combinators import (
    AllOf,
    AnyOf,
    AtLeast,
    Conditional,
    Exactly,
    NoneOf,
    NotRule,
    all_of,
    any_of,
    not_rule,
)
from truthound.checkpoint.routing.rules import (
    IssueCountRule,
    SeverityRule,
    StatusRule,
    TagRule,
)


def create_context(**kwargs) -> RouteContext:
    """Create test context."""
    defaults = {
        "checkpoint_name": "test",
        "run_id": "run_123",
        "status": "failure",
        "data_asset": "data.csv",
        "run_time": datetime.now(),
        "total_issues": 10,
        "critical_issues": 2,
        "high_issues": 3,
        "medium_issues": 3,
        "low_issues": 2,
        "info_issues": 0,
        "pass_rate": 85.0,
        "tags": {"env": "prod", "team": "data"},
        "metadata": {},
    }
    defaults.update(kwargs)
    return RouteContext(**defaults)


@dataclass
class MockRule:
    """Simple mock rule."""

    result: bool = True

    def evaluate(self, ctx: RouteContext) -> bool:
        return self.result

    @property
    def description(self) -> str:
        return f"mock({self.result})"


class TestAllOf:
    """Tests for AllOf combinator."""

    def test_empty_all_of(self):
        """Empty AllOf should match."""
        rule = AllOf()
        ctx = create_context()

        assert rule.evaluate(ctx) is True

    def test_single_rule_matches(self):
        """Single matching rule."""
        rule = AllOf([MockRule(True)])
        ctx = create_context()

        assert rule.evaluate(ctx) is True

    def test_single_rule_not_matches(self):
        """Single non-matching rule."""
        rule = AllOf([MockRule(False)])
        ctx = create_context()

        assert rule.evaluate(ctx) is False

    def test_all_match(self):
        """All rules match."""
        rule = AllOf([MockRule(True), MockRule(True), MockRule(True)])
        ctx = create_context()

        assert rule.evaluate(ctx) is True

    def test_one_fails(self):
        """One rule fails."""
        rule = AllOf([MockRule(True), MockRule(False), MockRule(True)])
        ctx = create_context()

        assert rule.evaluate(ctx) is False

    def test_short_circuit(self):
        """Short-circuits on first failure."""
        # Track evaluation order
        evaluations = []

        class TrackedRule:
            def __init__(self, result, name):
                self.result = result
                self.name = name

            def evaluate(self, ctx):
                evaluations.append(self.name)
                return self.result

            @property
            def description(self):
                return self.name

        rule = AllOf([
            TrackedRule(True, "r1"),
            TrackedRule(False, "r2"),
            TrackedRule(True, "r3"),  # Should not be evaluated
        ])

        rule.evaluate(create_context())

        assert evaluations == ["r1", "r2"]  # r3 not evaluated

    def test_add_rule(self):
        """Test adding rules fluently."""
        rule = AllOf()
        rule.add(MockRule(True)).add(MockRule(True))

        assert len(rule) == 2
        assert rule.evaluate(create_context()) is True

    def test_with_real_rules(self):
        """Test with real rule implementations."""
        rule = AllOf([
            SeverityRule(min_severity="critical"),
            StatusRule(statuses=["failure"]),
        ])

        ctx = create_context(critical_issues=1, status="failure")
        assert rule.evaluate(ctx) is True

        ctx2 = create_context(critical_issues=0, status="failure")
        assert rule.evaluate(ctx2) is False

    def test_description(self):
        """Test combined description."""
        rule = AllOf([MockRule(True), MockRule(False)])

        assert "AND" in rule.description


class TestAnyOf:
    """Tests for AnyOf combinator."""

    def test_empty_any_of(self):
        """Empty AnyOf should not match."""
        rule = AnyOf()
        ctx = create_context()

        assert rule.evaluate(ctx) is False

    def test_single_rule_matches(self):
        """Single matching rule."""
        rule = AnyOf([MockRule(True)])

        assert rule.evaluate(create_context()) is True

    def test_first_matches(self):
        """First rule matches."""
        rule = AnyOf([MockRule(True), MockRule(False)])

        assert rule.evaluate(create_context()) is True

    def test_last_matches(self):
        """Last rule matches."""
        rule = AnyOf([MockRule(False), MockRule(False), MockRule(True)])

        assert rule.evaluate(create_context()) is True

    def test_none_match(self):
        """No rules match."""
        rule = AnyOf([MockRule(False), MockRule(False)])

        assert rule.evaluate(create_context()) is False

    def test_short_circuit(self):
        """Short-circuits on first match."""
        evaluations = []

        class TrackedRule:
            def __init__(self, result, name):
                self.result = result
                self.name = name

            def evaluate(self, ctx):
                evaluations.append(self.name)
                return self.result

            @property
            def description(self):
                return self.name

        rule = AnyOf([
            TrackedRule(False, "r1"),
            TrackedRule(True, "r2"),
            TrackedRule(True, "r3"),
        ])

        rule.evaluate(create_context())

        assert evaluations == ["r1", "r2"]

    def test_description(self):
        """Test combined description."""
        # Need at least 2 rules to see OR separator
        rule = AnyOf([MockRule(True), MockRule(False)])

        assert "OR" in rule.description


class TestNotRule:
    """Tests for NotRule combinator."""

    def test_inverts_true(self):
        """Inverts True to False."""
        rule = NotRule(MockRule(True))

        assert rule.evaluate(create_context()) is False

    def test_inverts_false(self):
        """Inverts False to True."""
        rule = NotRule(MockRule(False))

        assert rule.evaluate(create_context()) is True

    def test_with_real_rule(self):
        """Test with real rule."""
        # Not in production
        rule = NotRule(TagRule(tags={"env": "prod"}))

        ctx1 = create_context(tags={"env": "staging"})
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(tags={"env": "prod"})
        assert rule.evaluate(ctx2) is False

    def test_description(self):
        """Test description includes NOT."""
        rule = NotRule(MockRule(True))

        assert "NOT" in rule.description


class TestAtLeast:
    """Tests for AtLeast combinator."""

    def test_empty_at_least_zero(self):
        """Empty AtLeast with count=0 matches."""
        rule = AtLeast(rules=[], count=0)

        assert rule.evaluate(create_context()) is True

    def test_at_least_one(self):
        """At least one must match."""
        rule = AtLeast([MockRule(True), MockRule(False), MockRule(False)], count=1)

        assert rule.evaluate(create_context()) is True

    def test_at_least_two(self):
        """At least two must match."""
        rule1 = AtLeast([MockRule(True), MockRule(True), MockRule(False)], count=2)
        assert rule1.evaluate(create_context()) is True

        rule2 = AtLeast([MockRule(True), MockRule(False), MockRule(False)], count=2)
        assert rule2.evaluate(create_context()) is False

    def test_short_circuit(self):
        """Short-circuits when count reached."""
        evaluations = []

        class TrackedRule:
            def __init__(self, result, name):
                self.result = result
                self.name = name

            def evaluate(self, ctx):
                evaluations.append(self.name)
                return self.result

            @property
            def description(self):
                return self.name

        rule = AtLeast([
            TrackedRule(True, "r1"),
            TrackedRule(True, "r2"),
            TrackedRule(True, "r3"),
        ], count=2)

        rule.evaluate(create_context())

        assert evaluations == ["r1", "r2"]

    def test_description(self):
        """Test description."""
        rule = AtLeast([MockRule(True)], count=2)

        assert "at least 2" in rule.description


class TestExactly:
    """Tests for Exactly combinator."""

    def test_exactly_zero(self):
        """Exactly zero must match."""
        rule = Exactly([MockRule(False), MockRule(False)], count=0)

        assert rule.evaluate(create_context()) is True

    def test_exactly_one(self):
        """Exactly one must match."""
        rule1 = Exactly([MockRule(True), MockRule(False), MockRule(False)], count=1)
        assert rule1.evaluate(create_context()) is True

        rule2 = Exactly([MockRule(True), MockRule(True), MockRule(False)], count=1)
        assert rule2.evaluate(create_context()) is False

    def test_exactly_all(self):
        """Exactly all must match."""
        rule = Exactly([MockRule(True), MockRule(True)], count=2)

        assert rule.evaluate(create_context()) is True

    def test_description(self):
        """Test description."""
        rule = Exactly([MockRule(True)], count=1)

        assert "exactly 1" in rule.description


class TestNoneOf:
    """Tests for NoneOf combinator."""

    def test_empty_none_of(self):
        """Empty NoneOf matches."""
        rule = NoneOf()

        assert rule.evaluate(create_context()) is True

    def test_none_match(self):
        """When no rules match, NoneOf matches."""
        rule = NoneOf([MockRule(False), MockRule(False)])

        assert rule.evaluate(create_context()) is True

    def test_one_matches(self):
        """When any rule matches, NoneOf fails."""
        rule = NoneOf([MockRule(False), MockRule(True)])

        assert rule.evaluate(create_context()) is False

    def test_description(self):
        """Test description."""
        rule = NoneOf([MockRule(True)])

        assert "none of" in rule.description


class TestConditional:
    """Tests for Conditional combinator."""

    def test_condition_true(self):
        """When condition is True, evaluate if_true branch."""
        rule = Conditional(
            condition=MockRule(True),
            if_true=MockRule(True),
            if_false=MockRule(False),
        )

        assert rule.evaluate(create_context()) is True

    def test_condition_false_with_else(self):
        """When condition is False, evaluate if_false branch."""
        rule = Conditional(
            condition=MockRule(False),
            if_true=MockRule(True),
            if_false=MockRule(True),
        )

        assert rule.evaluate(create_context()) is True

    def test_condition_false_no_else(self):
        """When condition is False with no else, return False."""
        rule = Conditional(
            condition=MockRule(False),
            if_true=MockRule(True),
        )

        assert rule.evaluate(create_context()) is False

    def test_with_real_rules(self):
        """Test with real rules."""
        # If production, require critical; otherwise require high
        rule = Conditional(
            condition=TagRule(tags={"env": "prod"}),
            if_true=SeverityRule(min_severity="critical"),
            if_false=SeverityRule(min_severity="high"),
        )

        # Prod with critical issues
        ctx1 = create_context(tags={"env": "prod"}, critical_issues=1)
        assert rule.evaluate(ctx1) is True

        # Prod without critical issues
        ctx2 = create_context(tags={"env": "prod"}, critical_issues=0)
        assert rule.evaluate(ctx2) is False

        # Dev with high issues
        ctx3 = create_context(tags={"env": "dev"}, high_issues=1)
        assert rule.evaluate(ctx3) is True

    def test_description(self):
        """Test description."""
        rule = Conditional(
            condition=MockRule(True),
            if_true=MockRule(True),
        )

        assert "if" in rule.description
        assert "then" in rule.description


class TestNestedCombinators:
    """Tests for nested combinator patterns."""

    def test_all_of_any_of(self):
        """AllOf containing AnyOf."""
        rule = AllOf([
            AnyOf([MockRule(True), MockRule(False)]),
            AnyOf([MockRule(False), MockRule(True)]),
        ])

        assert rule.evaluate(create_context()) is True

    def test_any_of_all_of(self):
        """AnyOf containing AllOf."""
        rule = AnyOf([
            AllOf([MockRule(True), MockRule(False)]),  # False
            AllOf([MockRule(True), MockRule(True)]),   # True
        ])

        assert rule.evaluate(create_context()) is True

    def test_double_negation(self):
        """Double negation."""
        rule = NotRule(NotRule(MockRule(True)))

        assert rule.evaluate(create_context()) is True

    def test_complex_nesting(self):
        """Complex nested structure."""
        # (critical AND prod) OR (high AND NOT prod)
        rule = AnyOf([
            AllOf([
                SeverityRule(min_severity="critical"),
                TagRule(tags={"env": "prod"}),
            ]),
            AllOf([
                SeverityRule(min_severity="high"),
                NotRule(TagRule(tags={"env": "prod"})),
            ]),
        ])

        # Critical in prod
        ctx1 = create_context(critical_issues=1, tags={"env": "prod"})
        assert rule.evaluate(ctx1) is True

        # High in staging
        ctx2 = create_context(critical_issues=0, high_issues=1, tags={"env": "staging"})
        assert rule.evaluate(ctx2) is True

        # High in prod (no critical)
        ctx3 = create_context(critical_issues=0, high_issues=1, tags={"env": "prod"})
        assert rule.evaluate(ctx3) is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_all_of_function(self):
        """Test all_of convenience function."""
        rule = all_of(MockRule(True), MockRule(True))

        assert isinstance(rule, AllOf)
        assert rule.evaluate(create_context()) is True

    def test_any_of_function(self):
        """Test any_of convenience function."""
        rule = any_of(MockRule(False), MockRule(True))

        assert isinstance(rule, AnyOf)
        assert rule.evaluate(create_context()) is True

    def test_not_rule_function(self):
        """Test not_rule convenience function."""
        rule = not_rule(MockRule(True))

        assert isinstance(rule, NotRule)
        assert rule.evaluate(create_context()) is False
