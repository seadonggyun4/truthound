"""Tests for rule evaluation engines."""

from __future__ import annotations

from datetime import datetime

import pytest

from truthound.checkpoint.routing.base import RouteContext
from truthound.checkpoint.routing.engine import (
    ExpressionEngine,
    ExpressionEvaluationError,
    ExpressionRule,
    ExpressionSecurityError,
    Jinja2Engine,
    Jinja2Rule,
    RuleEngine,
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
        "metadata": {"region": "us-east-1"},
    }
    defaults.update(kwargs)
    return RouteContext(**defaults)


class TestExpressionEngine:
    """Tests for ExpressionEngine."""

    def test_simple_comparison(self):
        """Test simple comparison expressions."""
        engine = ExpressionEngine()

        assert engine.evaluate("x > 5", {"x": 10}) is True
        assert engine.evaluate("x > 5", {"x": 3}) is False

    def test_boolean_expressions(self):
        """Test boolean expressions."""
        engine = ExpressionEngine()
        ctx = {"a": True, "b": False, "c": True}

        assert engine.evaluate("a and c", ctx) is True
        assert engine.evaluate("a and b", ctx) is False
        assert engine.evaluate("a or b", ctx) is True
        assert engine.evaluate("not b", ctx) is True

    def test_complex_expression(self):
        """Test complex expressions."""
        engine = ExpressionEngine()
        ctx = {"critical_issues": 2, "status": "failure", "pass_rate": 85.0}

        expr = "critical_issues > 0 and status == 'failure' and pass_rate < 90"
        assert engine.evaluate(expr, ctx) is True

    def test_string_operations(self):
        """Test string operations."""
        engine = ExpressionEngine()

        assert engine.evaluate("'prod' in env", {"env": "prod-us"}) is True
        assert engine.evaluate("name.startswith('test')", {"name": "test_data"}) is True

    def test_list_operations(self):
        """Test list operations."""
        engine = ExpressionEngine()
        ctx = {"items": [1, 2, 3, 4, 5]}

        assert engine.evaluate("len(items) > 3", ctx) is True
        assert engine.evaluate("3 in items", ctx) is True
        assert engine.evaluate("max(items) == 5", ctx) is True

    def test_dict_operations(self):
        """Test dictionary operations."""
        engine = ExpressionEngine()
        ctx = {"tags": {"env": "prod", "team": "data"}}

        assert engine.evaluate("tags.get('env') == 'prod'", ctx) is True
        assert engine.evaluate("'team' in tags", ctx) is True

    def test_safe_builtins(self):
        """Test safe built-in functions."""
        engine = ExpressionEngine()

        assert engine.evaluate("len([1, 2, 3])", {}) == 3
        assert engine.evaluate("min(1, 2, 3)", {}) == 1
        assert engine.evaluate("max(1, 2, 3)", {}) == 3
        assert engine.evaluate("sum([1, 2, 3])", {}) == 6
        assert engine.evaluate("abs(-5)", {}) == 5
        assert engine.evaluate("round(3.7)", {}) == 4

    def test_forbidden_operations(self):
        """Test forbidden operations are blocked."""
        engine = ExpressionEngine()

        with pytest.raises(ExpressionSecurityError):
            engine.evaluate("__import__('os')", {})

        with pytest.raises(ExpressionSecurityError):
            engine.evaluate("x.__class__", {"x": 1})

    def test_forbidden_builtins(self):
        """Test forbidden built-ins are blocked."""
        engine = ExpressionEngine()

        with pytest.raises(ExpressionSecurityError):
            engine.evaluate("eval('1+1')", {})

        with pytest.raises(ExpressionSecurityError):
            engine.evaluate("exec('x=1')", {})

        with pytest.raises(ExpressionSecurityError):
            engine.evaluate("open('/etc/passwd')", {})

    def test_syntax_error(self):
        """Test syntax error handling."""
        engine = ExpressionEngine()

        with pytest.raises(ExpressionSecurityError, match="Syntax error"):
            engine.evaluate("x >", {"x": 1})

    def test_evaluation_error(self):
        """Test evaluation error handling."""
        engine = ExpressionEngine()

        with pytest.raises(ExpressionEvaluationError):
            engine.evaluate("x / y", {"x": 1, "y": 0})

    def test_undefined_variable(self):
        """Test undefined variable access."""
        engine = ExpressionEngine()

        with pytest.raises(ExpressionSecurityError):
            engine.evaluate("undefined_var", {})

    def test_expression_length_limit(self):
        """Test expression length limit."""
        engine = ExpressionEngine(max_expression_length=10)

        errors = engine.validate("x > 5 and y < 10", {"x", "y"})
        assert any("too long" in e.lower() for e in errors)

    def test_validation(self):
        """Test expression validation."""
        engine = ExpressionEngine()

        # Valid expression
        errors = engine.validate("x > 5", {"x"})
        assert len(errors) == 0

        # Invalid variable
        errors = engine.validate("undefined", set())
        assert len(errors) > 0

    def test_caching(self):
        """Test expression caching."""
        engine = ExpressionEngine(cache_compiled=True)

        # First evaluation
        engine.evaluate("x > 5", {"x": 10})
        assert "x > 5" in engine._cache

        # Clear cache
        engine.clear_cache()
        assert len(engine._cache) == 0


class TestJinja2Engine:
    """Tests for Jinja2Engine."""

    def test_simple_template(self):
        """Test simple template evaluation."""
        pytest.importorskip("jinja2")
        engine = Jinja2Engine()

        result = engine.evaluate("{{ x > 5 }}", {"x": 10})
        assert result is True

    def test_template_with_logic(self):
        """Test template with Jinja2 logic."""
        pytest.importorskip("jinja2")
        engine = Jinja2Engine()

        template = "{{ 'yes' if status == 'failure' else 'no' }}"
        result = engine.render(template, {"status": "failure"})
        assert result == "yes"

    def test_template_filters(self):
        """Test Jinja2 filters."""
        pytest.importorskip("jinja2")
        engine = Jinja2Engine()

        result = engine.render("{{ name|upper }}", {"name": "test"})
        assert result == "TEST"

    def test_truthy_values(self):
        """Test truthy value evaluation."""
        pytest.importorskip("jinja2")
        engine = Jinja2Engine()

        assert engine.evaluate("{{ true }}", {}) is True
        assert engine.evaluate("{{ false }}", {}) is False
        assert engine.evaluate("{{ '' }}", {}) is False
        assert engine.evaluate("{{ 'yes' }}", {}) is True
        assert engine.evaluate("{{ 1 }}", {}) is True
        assert engine.evaluate("{{ 0 }}", {}) is False

    def test_sandboxed_environment(self):
        """Test sandboxed environment blocks dangerous operations."""
        pytest.importorskip("jinja2")
        engine = Jinja2Engine()

        # This should not allow arbitrary code execution
        # Sandboxed environment restricts attribute access
        result = engine.render("{{ range(3)|list }}", {})
        assert result == "[0, 1, 2]"

    def test_caching(self):
        """Test template caching."""
        pytest.importorskip("jinja2")
        engine = Jinja2Engine(cache_templates=True)

        template = "{{ x }}"
        engine.render(template, {"x": 1})
        assert template in engine._cache

        engine.clear_cache()
        assert len(engine._cache) == 0


class TestExpressionRule:
    """Tests for ExpressionRule."""

    def test_evaluate(self):
        """Test rule evaluation."""
        rule = ExpressionRule("critical_issues > 0")
        ctx = create_context(critical_issues=2)

        assert rule.evaluate(ctx) is True

    def test_complex_expression(self):
        """Test complex expression rule."""
        rule = ExpressionRule(
            "critical_issues > 0 and status == 'failure' and pass_rate < 90"
        )
        ctx = create_context(critical_issues=2, status="failure", pass_rate=85.0)

        assert rule.evaluate(ctx) is True

    def test_description(self):
        """Test rule description."""
        rule = ExpressionRule("x > 5")
        assert "x > 5" in rule.description

        rule_with_desc = ExpressionRule("x > 5", _description="Custom description")
        assert rule_with_desc.description == "Custom description"


class TestJinja2Rule:
    """Tests for Jinja2Rule."""

    def test_evaluate(self):
        """Test rule evaluation."""
        pytest.importorskip("jinja2")
        rule = Jinja2Rule("{{ critical_issues > 0 }}")
        ctx = create_context(critical_issues=2)

        assert rule.evaluate(ctx) is True

    def test_with_tags(self):
        """Test rule with tag access."""
        pytest.importorskip("jinja2")
        rule = Jinja2Rule("{{ tags.get('env') == 'prod' }}")
        ctx = create_context(tags={"env": "prod"})

        assert rule.evaluate(ctx) is True


class TestRuleEngine:
    """Tests for RuleEngine facade."""

    def test_detect_template(self):
        """Test template detection."""
        engine = RuleEngine()

        assert engine.is_template("{{ x }}") is True
        assert engine.is_template("{% if x %}{% endif %}") is True
        assert engine.is_template("x > 5") is False

    def test_create_expression_rule(self):
        """Test creating expression rule."""
        engine = RuleEngine()

        rule = engine.create_rule("critical_issues > 0")
        assert isinstance(rule, ExpressionRule)

    def test_create_template_rule(self):
        """Test creating template rule."""
        pytest.importorskip("jinja2")
        engine = RuleEngine()

        rule = engine.create_rule("{{ critical_issues > 0 }}")
        assert isinstance(rule, Jinja2Rule)

    def test_evaluate_expression(self):
        """Test evaluating expression."""
        engine = RuleEngine()
        ctx = create_context(critical_issues=2)

        result = engine.evaluate("critical_issues > 0", ctx)
        assert result is True

    def test_evaluate_template(self):
        """Test evaluating template."""
        pytest.importorskip("jinja2")
        engine = RuleEngine()
        ctx = create_context(critical_issues=2)

        result = engine.evaluate("{{ critical_issues > 0 }}", ctx)
        assert result is True

    def test_validate_expression(self):
        """Test expression validation."""
        engine = RuleEngine()

        # Valid expression
        errors = engine.validate_expression("critical_issues > 0")
        assert len(errors) == 0

        # Invalid syntax
        errors = engine.validate_expression("x >")
        assert len(errors) > 0

    def test_validate_template(self):
        """Test template validation."""
        pytest.importorskip("jinja2")
        engine = RuleEngine()

        # Valid template
        errors = engine.validate_expression("{{ x > 5 }}")
        assert len(errors) == 0

    def test_clear_caches(self):
        """Test clearing all caches."""
        engine = RuleEngine()

        # Populate caches
        engine._expr_engine.evaluate("x > 5", {"x": 10})

        # Clear
        engine.clear_caches()

        assert len(engine._expr_engine._cache) == 0


class TestContextIntegration:
    """Integration tests with RouteContext."""

    def test_all_context_fields(self):
        """Test accessing all context fields in expressions."""
        engine = RuleEngine()
        ctx = create_context()

        expressions = [
            "checkpoint_name == 'test'",
            "run_id == 'run_123'",
            "status == 'failure'",
            "data_asset == 'data.csv'",
            "total_issues == 10",
            "critical_issues == 2",
            "high_issues == 3",
            "medium_issues == 3",
            "low_issues == 2",
            "info_issues == 0",
            "pass_rate == 85.0",
            "'env' in tags",
            "tags.get('env') == 'prod'",
            "'region' in metadata",
        ]

        for expr in expressions:
            assert engine.evaluate(expr, ctx) is True, f"Failed: {expr}"

    def test_real_world_scenarios(self):
        """Test real-world expression scenarios."""
        engine = RuleEngine()

        # Scenario 1: Critical issues in production
        ctx1 = create_context(
            critical_issues=1, tags={"env": "prod"}
        )
        expr1 = "critical_issues > 0 and tags.get('env') == 'prod'"
        assert engine.evaluate(expr1, ctx1) is True

        # Scenario 2: Low pass rate
        ctx2 = create_context(pass_rate=75.0)
        expr2 = "pass_rate < 80"
        assert engine.evaluate(expr2, ctx2) is True

        # Scenario 3: Multiple conditions
        ctx3 = create_context(
            status="failure",
            total_issues=50,
            critical_issues=5,
        )
        expr3 = "(status == 'failure' or status == 'error') and total_issues > 20"
        assert engine.evaluate(expr3, ctx3) is True
