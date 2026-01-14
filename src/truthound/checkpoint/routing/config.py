"""YAML/JSON configuration parser for routing rules.

This module provides configuration-driven route setup:
- RouteConfigParser: Parses YAML/JSON configuration into Route objects
- ActionFactory: Creates action instances from configuration
- RuleFactory: Creates rule instances from configuration

Configuration Format:
    routes:
      - name: critical_alerts
        rule:
          type: all_of
          rules:
            - type: severity
              min_severity: critical
            - type: tag
              tags:
                env: prod
        actions:
          - type: pagerduty
            routing_key: "${PAGERDUTY_KEY}"
        priority: critical
        enabled: true

Example:
    >>> from truthound.checkpoint.routing.config import RouteConfigParser
    >>> parser = RouteConfigParser()
    >>> router = parser.parse_file("routing.yaml")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction
    from truthound.checkpoint.routing.base import ActionRouter, Route, RoutingRule

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


@dataclass
class RuleFactory:
    """Factory for creating routing rules from configuration.

    Supports all built-in rule types and custom rule registration.

    Attributes:
        custom_rules: Dictionary of custom rule type factories
    """

    custom_rules: dict[str, Callable[..., "RoutingRule"]] = field(
        default_factory=dict
    )

    def create(self, config: dict[str, Any]) -> "RoutingRule":
        """Create a rule from configuration.

        Args:
            config: Rule configuration dictionary with 'type' key.

        Returns:
            Configured RoutingRule instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        rule_type = config.get("type", "expression")

        # Handle expression shorthand
        if "expression" in config and rule_type == "expression":
            return self._create_expression_rule(config)

        # Handle template shorthand
        if "template" in config:
            return self._create_template_rule(config)

        # Standard rule types
        factory_map: dict[str, Callable[..., "RoutingRule"]] = {
            "always": self._create_always_rule,
            "never": self._create_never_rule,
            "severity": self._create_severity_rule,
            "issue_count": self._create_issue_count_rule,
            "status": self._create_status_rule,
            "tag": self._create_tag_rule,
            "data_asset": self._create_data_asset_rule,
            "metadata": self._create_metadata_rule,
            "time_window": self._create_time_window_rule,
            "pass_rate": self._create_pass_rate_rule,
            "error": self._create_error_rule,
            "expression": self._create_expression_rule,
            "template": self._create_template_rule,
            # Combinators
            "all_of": self._create_all_of,
            "any_of": self._create_any_of,
            "not": self._create_not,
            "at_least": self._create_at_least,
            "exactly": self._create_exactly,
            "none_of": self._create_none_of,
            "conditional": self._create_conditional,
        }

        # Check custom rules
        if rule_type in self.custom_rules:
            return self.custom_rules[rule_type](**config)

        if rule_type not in factory_map:
            raise ConfigurationError(
                f"Unknown rule type: {rule_type}. "
                f"Available types: {list(factory_map.keys())}"
            )

        return factory_map[rule_type](config)

    def register(
        self,
        rule_type: str,
        factory: Callable[..., "RoutingRule"],
    ) -> "RuleFactory":
        """Register a custom rule type.

        Args:
            rule_type: Type name for the rule.
            factory: Factory function that creates the rule.

        Returns:
            Self for chaining.
        """
        self.custom_rules[rule_type] = factory
        return self

    # Rule creation methods

    def _create_always_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import AlwaysRule

        return AlwaysRule()

    def _create_never_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import NeverRule

        return NeverRule()

    def _create_severity_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import SeverityRule

        return SeverityRule(
            min_severity=config.get("min_severity", "high"),
            max_severity=config.get("max_severity"),
            exact_count=config.get("exact_count"),
            min_count=config.get("min_count", 1),
        )

    def _create_issue_count_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import IssueCountRule

        return IssueCountRule(
            min_issues=config.get("min_issues", 0),
            max_issues=config.get("max_issues"),
            count_type=config.get("count_type", "total"),
        )

    def _create_status_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import StatusRule

        return StatusRule(
            statuses=config.get("statuses", ["failure", "error"]),
            negate=config.get("negate", False),
        )

    def _create_tag_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import TagRule

        return TagRule(
            tags=config.get("tags", {}),
            match_all=config.get("match_all", True),
            negate=config.get("negate", False),
        )

    def _create_data_asset_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import DataAssetRule

        return DataAssetRule(
            pattern=config.get("pattern", "*"),
            is_regex=config.get("is_regex", False),
            case_sensitive=config.get("case_sensitive", True),
        )

    def _create_metadata_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import MetadataRule

        if "key_path" not in config:
            raise ConfigurationError("MetadataRule requires 'key_path'")

        return MetadataRule(
            key_path=config["key_path"],
            expected_value=config.get("expected_value"),
            comparator=config.get("comparator", "eq"),
        )

    def _create_time_window_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import TimeWindowRule

        return TimeWindowRule(
            start_time=config.get("start_time", "00:00"),
            end_time=config.get("end_time", "23:59"),
            days_of_week=config.get("days_of_week"),
            timezone=config.get("timezone", "UTC"),
            use_run_time=config.get("use_run_time", True),
        )

    def _create_pass_rate_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import PassRateRule

        return PassRateRule(
            min_rate=config.get("min_rate", 0.0),
            max_rate=config.get("max_rate", 100.0),
        )

    def _create_error_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.rules import ErrorRule

        return ErrorRule(
            pattern=config.get("pattern"),
            negate=config.get("negate", False),
        )

    def _create_expression_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.engine import ExpressionRule

        expression = config.get("expression", "True")
        return ExpressionRule(
            expression=expression,
            _description=config.get("description"),
        )

    def _create_template_rule(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.engine import Jinja2Rule

        template = config.get("template", "{{ True }}")
        return Jinja2Rule(
            template=template,
            _description=config.get("description"),
        )

    # Combinator creation methods

    def _create_all_of(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import AllOf

        rules = [self.create(r) for r in config.get("rules", [])]
        return AllOf(rules=rules)

    def _create_any_of(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import AnyOf

        rules = [self.create(r) for r in config.get("rules", [])]
        return AnyOf(rules=rules)

    def _create_not(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import NotRule

        rule_config = config.get("rule")
        if not rule_config:
            raise ConfigurationError("NotRule requires 'rule' configuration")

        return NotRule(rule=self.create(rule_config))

    def _create_at_least(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import AtLeast

        rules = [self.create(r) for r in config.get("rules", [])]
        return AtLeast(
            rules=rules,
            count=config.get("count", 1),
        )

    def _create_exactly(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import Exactly

        rules = [self.create(r) for r in config.get("rules", [])]
        return Exactly(
            rules=rules,
            count=config.get("count", 1),
        )

    def _create_none_of(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import NoneOf

        rules = [self.create(r) for r in config.get("rules", [])]
        return NoneOf(rules=rules)

    def _create_conditional(self, config: dict[str, Any]) -> "RoutingRule":
        from truthound.checkpoint.routing.combinators import Conditional

        condition = config.get("condition")
        if_true = config.get("if_true")
        if_false = config.get("if_false")

        if not condition or not if_true:
            raise ConfigurationError(
                "Conditional requires 'condition' and 'if_true'"
            )

        return Conditional(
            condition=self.create(condition),
            if_true=self.create(if_true),
            if_false=self.create(if_false) if if_false else None,
        )


@dataclass
class ActionFactory:
    """Factory for creating actions from configuration.

    Supports all built-in action types and custom action registration.

    Attributes:
        custom_actions: Dictionary of custom action type factories
        env_pattern: Regex pattern for environment variable substitution
    """

    custom_actions: dict[str, Callable[..., "BaseAction[Any]"]] = field(
        default_factory=dict
    )
    env_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\$\{([^}]+)\}"),
        repr=False,
    )

    def create(self, config: dict[str, Any]) -> "BaseAction[Any]":
        """Create an action from configuration.

        Args:
            config: Action configuration dictionary with 'type' key.

        Returns:
            Configured BaseAction instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        action_type = config.get("type")
        if not action_type:
            raise ConfigurationError("Action configuration requires 'type'")

        # Substitute environment variables
        config = self._substitute_env_vars(config)

        # Standard action types
        factory_map: dict[str, Callable[[dict[str, Any]], "BaseAction[Any]"]] = {
            "slack": self._create_slack_action,
            "email": self._create_email_action,
            "pagerduty": self._create_pagerduty_action,
            "webhook": self._create_webhook_action,
            "github": self._create_github_action,
            "teams": self._create_teams_action,
            "opsgenie": self._create_opsgenie_action,
            "discord": self._create_discord_action,
            "telegram": self._create_telegram_action,
            "store_result": self._create_store_result_action,
            "update_docs": self._create_update_docs_action,
            "custom": self._create_custom_action,
        }

        # Check custom actions
        if action_type in self.custom_actions:
            return self.custom_actions[action_type](**config)

        if action_type not in factory_map:
            raise ConfigurationError(
                f"Unknown action type: {action_type}. "
                f"Available types: {list(factory_map.keys())}"
            )

        return factory_map[action_type](config)

    def register(
        self,
        action_type: str,
        factory: Callable[..., "BaseAction[Any]"],
    ) -> "ActionFactory":
        """Register a custom action type.

        Args:
            action_type: Type name for the action.
            factory: Factory function that creates the action.

        Returns:
            Self for chaining.
        """
        self.custom_actions[action_type] = factory
        return self

    def _substitute_env_vars(self, config: dict[str, Any]) -> dict[str, Any]:
        """Recursively substitute environment variables in config."""
        result = {}

        for key, value in config.items():
            if isinstance(value, str):
                result[key] = self.env_pattern.sub(
                    lambda m: os.environ.get(m.group(1), m.group(0)),
                    value,
                )
            elif isinstance(value, dict):
                result[key] = self._substitute_env_vars(value)
            elif isinstance(value, list):
                result[key] = [
                    self._substitute_env_vars(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                result[key] = value

        return result

    # Action creation methods

    def _create_slack_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.slack_notify import SlackNotification

        return SlackNotification(
            webhook_url=config.get("webhook_url", ""),
            channel=config.get("channel"),
            username=config.get("username"),
            icon_emoji=config.get("icon_emoji"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_email_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.email_notify import EmailNotification

        return EmailNotification(
            smtp_host=config.get("smtp_host", "localhost"),
            smtp_port=config.get("smtp_port", 587),
            from_address=config.get("from_address", ""),
            to_addresses=config.get("to_addresses", []),
            subject_template=config.get("subject_template"),
            use_tls=config.get("use_tls", True),
            username=config.get("username"),
            password=config.get("password"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_pagerduty_action(
        self, config: dict[str, Any]
    ) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.pagerduty import PagerDutyAction

        return PagerDutyAction(
            routing_key=config.get("routing_key", ""),
            severity_mapping=config.get("severity_mapping"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_webhook_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.webhook import WebhookAction

        return WebhookAction(
            url=config.get("url", ""),
            method=config.get("method", "POST"),
            headers=config.get("headers"),
            payload_template=config.get("payload_template"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_github_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.github_action import GitHubAction

        return GitHubAction(
            token=config.get("token", ""),
            repository=config.get("repository", ""),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_teams_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.teams_notify import TeamsNotification

        return TeamsNotification(
            webhook_url=config.get("webhook_url", ""),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_opsgenie_action(
        self, config: dict[str, Any]
    ) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction

        return OpsGenieAction(
            api_key=config.get("api_key", ""),
            region=config.get("region", "us"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_discord_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.discord_notify import DiscordNotification

        return DiscordNotification(
            webhook_url=config.get("webhook_url", ""),
            username=config.get("username"),
            avatar_url=config.get("avatar_url"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_telegram_action(
        self, config: dict[str, Any]
    ) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.telegram_notify import TelegramNotification

        return TelegramNotification(
            bot_token=config.get("bot_token", ""),
            chat_id=config.get("chat_id", ""),
            parse_mode=config.get("parse_mode", "HTML"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_store_result_action(
        self, config: dict[str, Any]
    ) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.store_result import StoreValidationResult

        return StoreValidationResult(
            store_path=config.get("store_path", "./results"),
            store_type=config.get("store_type", "filesystem"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_update_docs_action(
        self, config: dict[str, Any]
    ) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.update_docs import UpdateDataDocs

        return UpdateDataDocs(
            docs_path=config.get("docs_path", "./data_docs"),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )

    def _create_custom_action(self, config: dict[str, Any]) -> "BaseAction[Any]":
        from truthound.checkpoint.actions.custom import CustomAction

        return CustomAction(
            callable_path=config.get("callable_path", ""),
            args=config.get("args", []),
            kwargs=config.get("kwargs", {}),
            notify_on=config.get("notify_on", "always"),
            name=config.get("name"),
        )


@dataclass
class RouteConfigParser:
    """Parser for routing configuration files.

    Parses YAML or JSON configuration into ActionRouter with routes.

    Attributes:
        rule_factory: Factory for creating rules
        action_factory: Factory for creating actions
    """

    rule_factory: RuleFactory = field(default_factory=RuleFactory)
    action_factory: ActionFactory = field(default_factory=ActionFactory)

    def parse(self, config: dict[str, Any]) -> "ActionRouter":
        """Parse configuration dictionary into ActionRouter.

        Args:
            config: Configuration dictionary.

        Returns:
            Configured ActionRouter.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        from truthound.checkpoint.routing.base import (
            ActionRouter,
            Route,
            RouteMode,
            RoutePriority,
        )

        # Parse router settings
        mode = RouteMode(config.get("mode", "all_matches"))
        router = ActionRouter(mode=mode)

        # Parse default actions
        default_actions_config = config.get("default_actions", [])
        if default_actions_config:
            default_actions = [
                self.action_factory.create(ac) for ac in default_actions_config
            ]
            router.set_default_actions(default_actions)

        # Parse routes
        routes_config = config.get("routes", [])
        for route_config in routes_config:
            route = self._parse_route(route_config)
            router.add_route(route)

        return router

    def _parse_route(self, config: dict[str, Any]) -> "Route":
        """Parse a single route configuration."""
        from truthound.checkpoint.routing.base import Route, RoutePriority

        name = config.get("name")
        if not name:
            raise ConfigurationError("Route requires 'name'")

        # Parse rule
        rule_config = config.get("rule", {"type": "always"})
        rule = self.rule_factory.create(rule_config)

        # Parse actions
        actions_config = config.get("actions", [])
        actions = [self.action_factory.create(ac) for ac in actions_config]

        # Parse priority
        priority = config.get("priority", "normal")
        if isinstance(priority, str):
            priority_map = {
                "critical": RoutePriority.CRITICAL,
                "high": RoutePriority.HIGH,
                "normal": RoutePriority.NORMAL,
                "low": RoutePriority.LOW,
                "default": RoutePriority.DEFAULT,
            }
            priority = priority_map.get(priority.lower(), RoutePriority.NORMAL)

        return Route(
            name=name,
            rule=rule,
            actions=actions,
            priority=priority,
            enabled=config.get("enabled", True),
            stop_on_match=config.get("stop_on_match", False),
            metadata=config.get("metadata", {}),
        )

    def parse_file(self, path: str | Path) -> "ActionRouter":
        """Parse configuration from a file.

        Supports YAML (.yaml, .yml) and JSON (.json) files.

        Args:
            path: Path to configuration file.

        Returns:
            Configured ActionRouter.

        Raises:
            ConfigurationError: If file cannot be parsed.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                config = yaml.safe_load(content)
            except ImportError:
                raise ConfigurationError(
                    "PyYAML is required for YAML configuration files. "
                    "Install it with: pip install pyyaml"
                )
        elif path.suffix == ".json":
            import json

            config = json.loads(content)
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {path.suffix}"
            )

        return self.parse(config)

    def parse_yaml(self, yaml_content: str) -> "ActionRouter":
        """Parse configuration from YAML string.

        Args:
            yaml_content: YAML configuration string.

        Returns:
            Configured ActionRouter.
        """
        try:
            import yaml

            config = yaml.safe_load(yaml_content)
        except ImportError:
            raise ConfigurationError(
                "PyYAML is required for YAML configuration. "
                "Install it with: pip install pyyaml"
            )

        return self.parse(config)

    def parse_json(self, json_content: str) -> "ActionRouter":
        """Parse configuration from JSON string.

        Args:
            json_content: JSON configuration string.

        Returns:
            Configured ActionRouter.
        """
        import json

        config = json.loads(json_content)
        return self.parse(config)

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration without creating objects.

        Args:
            config: Configuration dictionary.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Validate mode
        mode = config.get("mode", "all_matches")
        valid_modes = ["first_match", "all_matches", "priority_group"]
        if mode not in valid_modes:
            errors.append(f"Invalid mode: {mode}. Must be one of: {valid_modes}")

        # Validate routes
        routes = config.get("routes", [])
        if not isinstance(routes, list):
            errors.append("'routes' must be a list")
        else:
            for i, route in enumerate(routes):
                route_errors = self._validate_route(route, i)
                errors.extend(route_errors)

        return errors

    def _validate_route(
        self, route: dict[str, Any], index: int
    ) -> list[str]:
        """Validate a single route configuration."""
        errors: list[str] = []
        prefix = f"Route {index}"

        if not isinstance(route, dict):
            return [f"{prefix}: must be a dictionary"]

        name = route.get("name")
        if not name:
            errors.append(f"{prefix}: missing 'name'")
        else:
            prefix = f"Route '{name}'"

        # Validate rule
        rule = route.get("rule")
        if not rule:
            errors.append(f"{prefix}: missing 'rule'")
        elif not isinstance(rule, dict):
            errors.append(f"{prefix}: 'rule' must be a dictionary")

        # Validate actions
        actions = route.get("actions", [])
        if not isinstance(actions, list):
            errors.append(f"{prefix}: 'actions' must be a list")
        else:
            for j, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(f"{prefix}: action {j} must be a dictionary")
                elif "type" not in action:
                    errors.append(f"{prefix}: action {j} missing 'type'")

        return errors
