"""Tests for routing configuration parser."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from truthound.checkpoint.routing.base import RouteMode, RoutePriority
from truthound.checkpoint.routing.combinators import AllOf, AnyOf, NotRule
from truthound.checkpoint.routing.config import (
    ActionFactory,
    ConfigurationError,
    RouteConfigParser,
    RuleFactory,
)
from truthound.checkpoint.routing.rules import (
    AlwaysRule,
    IssueCountRule,
    SeverityRule,
    StatusRule,
    TagRule,
)


class TestRuleFactory:
    """Tests for RuleFactory."""

    def test_create_always_rule(self):
        """Test creating AlwaysRule."""
        factory = RuleFactory()
        rule = factory.create({"type": "always"})

        assert isinstance(rule, AlwaysRule)

    def test_create_severity_rule(self):
        """Test creating SeverityRule."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "severity",
            "min_severity": "critical",
            "min_count": 2,
        })

        assert isinstance(rule, SeverityRule)
        assert rule.min_severity == "critical"
        assert rule.min_count == 2

    def test_create_issue_count_rule(self):
        """Test creating IssueCountRule."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "issue_count",
            "min_issues": 5,
            "max_issues": 20,
        })

        assert isinstance(rule, IssueCountRule)
        assert rule.min_issues == 5
        assert rule.max_issues == 20

    def test_create_status_rule(self):
        """Test creating StatusRule."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "status",
            "statuses": ["failure", "error"],
        })

        assert isinstance(rule, StatusRule)
        assert "failure" in rule.statuses

    def test_create_tag_rule(self):
        """Test creating TagRule."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "tag",
            "tags": {"env": "prod"},
            "match_all": True,
        })

        assert isinstance(rule, TagRule)
        assert rule.tags == {"env": "prod"}

    def test_create_all_of(self):
        """Test creating AllOf combinator."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "all_of",
            "rules": [
                {"type": "severity", "min_severity": "critical"},
                {"type": "status", "statuses": ["failure"]},
            ],
        })

        assert isinstance(rule, AllOf)
        assert len(rule.rules) == 2

    def test_create_any_of(self):
        """Test creating AnyOf combinator."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "any_of",
            "rules": [
                {"type": "severity", "min_severity": "critical"},
                {"type": "severity", "min_severity": "high"},
            ],
        })

        assert isinstance(rule, AnyOf)
        assert len(rule.rules) == 2

    def test_create_not_rule(self):
        """Test creating NotRule combinator."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "not",
            "rule": {"type": "tag", "tags": {"env": "prod"}},
        })

        assert isinstance(rule, NotRule)

    def test_create_expression_rule(self):
        """Test creating expression rule."""
        factory = RuleFactory()
        rule = factory.create({
            "type": "expression",
            "expression": "critical_issues > 0",
        })

        assert rule.expression == "critical_issues > 0"

    def test_expression_shorthand(self):
        """Test expression shorthand (no type needed)."""
        factory = RuleFactory()
        rule = factory.create({
            "expression": "critical_issues > 0",
        })

        assert rule.expression == "critical_issues > 0"

    def test_unknown_rule_type(self):
        """Test unknown rule type raises error."""
        factory = RuleFactory()

        with pytest.raises(ConfigurationError, match="Unknown rule type"):
            factory.create({"type": "unknown_rule"})

    def test_register_custom_rule(self):
        """Test registering custom rule type."""
        factory = RuleFactory()

        class CustomRule:
            def __init__(self, **kwargs):
                self.value = kwargs.get("value")

            def evaluate(self, ctx):
                return True

            @property
            def description(self):
                return "custom"

        factory.register("custom", CustomRule)
        rule = factory.create({"type": "custom", "value": 42})

        assert rule.value == 42


class TestActionFactory:
    """Tests for ActionFactory."""

    def test_env_var_substitution(self):
        """Test environment variable substitution."""
        factory = ActionFactory()

        with patch.dict(os.environ, {"TEST_URL": "https://example.com"}):
            config = factory._substitute_env_vars({
                "url": "${TEST_URL}/webhook",
                "nested": {"key": "${TEST_URL}"},
            })

        assert config["url"] == "https://example.com/webhook"
        assert config["nested"]["key"] == "https://example.com"

    def test_env_var_missing(self):
        """Test missing env var keeps placeholder."""
        factory = ActionFactory()

        config = factory._substitute_env_vars({
            "url": "${MISSING_VAR}",
        })

        assert config["url"] == "${MISSING_VAR}"

    def test_create_webhook_action(self):
        """Test creating webhook action."""
        factory = ActionFactory()

        # Mock the import to avoid actual action creation
        with patch(
            "truthound.checkpoint.routing.config.ActionFactory._create_webhook_action"
        ) as mock:
            mock.return_value = MagicMock()
            factory.create({"type": "webhook", "url": "https://example.com"})
            mock.assert_called_once()

    def test_unknown_action_type(self):
        """Test unknown action type raises error."""
        factory = ActionFactory()

        with pytest.raises(ConfigurationError, match="Unknown action type"):
            factory.create({"type": "unknown_action"})

    def test_missing_type(self):
        """Test missing type raises error."""
        factory = ActionFactory()

        with pytest.raises(ConfigurationError, match="requires 'type'"):
            factory.create({})

    def test_register_custom_action(self):
        """Test registering custom action type."""
        factory = ActionFactory()

        mock_action = MagicMock()
        factory.register("custom", lambda **kwargs: mock_action)

        action = factory.create({"type": "custom", "param": "value"})
        assert action is mock_action


class TestRouteConfigParser:
    """Tests for RouteConfigParser."""

    def test_parse_simple_config(self):
        """Test parsing simple configuration."""
        parser = RouteConfigParser()

        config = {
            "mode": "all_matches",
            "routes": [
                {
                    "name": "critical_alerts",
                    "rule": {"type": "severity", "min_severity": "critical"},
                    "actions": [],
                    "priority": "critical",
                },
            ],
        }

        router = parser.parse(config)

        assert router.mode == RouteMode.ALL_MATCHES
        assert len(router) == 1
        assert router.routes[0].name == "critical_alerts"
        assert router.routes[0].priority == RoutePriority.CRITICAL.value

    def test_parse_with_default_actions(self):
        """Test parsing with default actions."""
        parser = RouteConfigParser()

        # Mock action factory
        mock_action = MagicMock()
        parser.action_factory.create = MagicMock(return_value=mock_action)

        config = {
            "default_actions": [
                {"type": "webhook", "url": "https://example.com"},
            ],
            "routes": [],
        }

        router = parser.parse(config)

        assert len(router._default_actions) == 1

    def test_parse_complex_rules(self):
        """Test parsing complex nested rules."""
        parser = RouteConfigParser()

        config = {
            "routes": [
                {
                    "name": "complex_route",
                    "rule": {
                        "type": "all_of",
                        "rules": [
                            {"type": "severity", "min_severity": "high"},
                            {
                                "type": "any_of",
                                "rules": [
                                    {"type": "tag", "tags": {"env": "prod"}},
                                    {"type": "tag", "tags": {"priority": "high"}},
                                ],
                            },
                        ],
                    },
                    "actions": [],
                },
            ],
        }

        router = parser.parse(config)

        assert len(router) == 1
        route = router.routes[0]
        assert isinstance(route.rule, AllOf)

    def test_parse_file_yaml(self):
        """Test parsing YAML file."""
        pytest.importorskip("yaml")

        yaml_content = """
mode: first_match
routes:
  - name: test_route
    rule:
      type: always
    actions: []
    priority: high
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            parser = RouteConfigParser()
            router = parser.parse_file(f.name)

        os.unlink(f.name)

        assert router.mode == RouteMode.FIRST_MATCH
        assert len(router) == 1

    def test_parse_file_json(self):
        """Test parsing JSON file."""
        json_content = {
            "mode": "all_matches",
            "routes": [
                {
                    "name": "test_route",
                    "rule": {"type": "always"},
                    "actions": [],
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()

            parser = RouteConfigParser()
            router = parser.parse_file(f.name)

        os.unlink(f.name)

        assert len(router) == 1

    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = RouteConfigParser()

        with pytest.raises(ConfigurationError, match="not found"):
            parser.parse_file("/nonexistent/path.yaml")

    def test_parse_unsupported_format(self):
        """Test parsing unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some content")
            f.flush()

            parser = RouteConfigParser()

            with pytest.raises(ConfigurationError, match="Unsupported"):
                parser.parse_file(f.name)

        os.unlink(f.name)

    def test_validate_config(self):
        """Test configuration validation."""
        parser = RouteConfigParser()

        # Valid config
        valid_config = {
            "mode": "all_matches",
            "routes": [
                {
                    "name": "test",
                    "rule": {"type": "always"},
                    "actions": [{"type": "webhook"}],
                },
            ],
        }
        errors = parser.validate_config(valid_config)
        assert len(errors) == 0

        # Invalid mode
        invalid_mode = {
            "mode": "invalid_mode",
            "routes": [],
        }
        errors = parser.validate_config(invalid_mode)
        assert any("Invalid mode" in e for e in errors)

        # Missing route name
        missing_name = {
            "routes": [{"rule": {"type": "always"}, "actions": []}],
        }
        errors = parser.validate_config(missing_name)
        assert any("missing 'name'" in e for e in errors)

        # Missing action type
        missing_action_type = {
            "routes": [
                {"name": "test", "rule": {"type": "always"}, "actions": [{}]}
            ],
        }
        errors = parser.validate_config(missing_action_type)
        assert any("missing 'type'" in e for e in errors)

    def test_parse_yaml_string(self):
        """Test parsing YAML string directly."""
        pytest.importorskip("yaml")

        yaml_content = """
routes:
  - name: test
    rule:
      type: always
    actions: []
"""
        parser = RouteConfigParser()
        router = parser.parse_yaml(yaml_content)

        assert len(router) == 1

    def test_parse_json_string(self):
        """Test parsing JSON string directly."""
        json_content = json.dumps({
            "routes": [{"name": "test", "rule": {"type": "always"}, "actions": []}]
        })

        parser = RouteConfigParser()
        router = parser.parse_json(json_content)

        assert len(router) == 1


class TestFullConfigScenarios:
    """End-to-end configuration scenarios."""

    def test_production_like_config(self):
        """Test production-like configuration."""
        parser = RouteConfigParser()

        config = {
            "mode": "all_matches",
            "routes": [
                # Critical issues in production -> PagerDuty
                {
                    "name": "critical_prod_alerts",
                    "rule": {
                        "type": "all_of",
                        "rules": [
                            {"type": "severity", "min_severity": "critical"},
                            {"type": "tag", "tags": {"env": "prod"}},
                        ],
                    },
                    "actions": [],  # Would have PagerDuty action
                    "priority": "critical",
                },
                # High issues -> Slack
                {
                    "name": "high_severity_alerts",
                    "rule": {
                        "type": "severity",
                        "min_severity": "high",
                    },
                    "actions": [],  # Would have Slack action
                    "priority": "high",
                },
                # All failures -> logging
                {
                    "name": "failure_logging",
                    "rule": {
                        "type": "status",
                        "statuses": ["failure", "error"],
                    },
                    "actions": [],
                    "priority": "low",
                },
            ],
        }

        router = parser.parse(config)

        assert len(router) == 3
        # Check priority ordering (critical first)
        assert router.routes[0].name == "critical_prod_alerts"
        assert router.routes[1].name == "high_severity_alerts"

    def test_expression_based_config(self):
        """Test configuration with expression rules."""
        parser = RouteConfigParser()

        config = {
            "routes": [
                {
                    "name": "complex_condition",
                    "rule": {
                        "expression": "critical_issues > 5 or (high_issues > 10 and pass_rate < 80)",
                    },
                    "actions": [],
                },
            ],
        }

        router = parser.parse(config)

        assert len(router) == 1
