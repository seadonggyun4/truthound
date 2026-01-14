"""Rule-based notification routing system.

This module provides a flexible, extensible routing system for directing
checkpoint results to appropriate notification actions based on configurable rules.

Key Components:
    - RoutingRule: Protocol for rule evaluation
    - RouteCondition: Base condition types (severity, issue count, tags, regex)
    - RuleEngine: Python expression and Jinja2 template evaluation
    - ActionRouter: Main router that orchestrates rule evaluation and action execution
    - RouteConfig: YAML/JSON configuration parser

Example:
    >>> from truthound.checkpoint.routing import ActionRouter, SeverityRoute
    >>> from truthound.checkpoint.actions import SlackNotification, PagerDutyAction
    >>>
    >>> router = ActionRouter()
    >>> router.add_route(
    ...     SeverityRoute(
    ...         min_severity="critical",
    ...         actions=[PagerDutyAction(...)],
    ...     )
    ... )
    >>> router.add_route(
    ...     SeverityRoute(
    ...         min_severity="high",
    ...         actions=[SlackNotification(...)],
    ...     )
    ... )
    >>>
    >>> # Use with checkpoint
    >>> checkpoint = Checkpoint(..., router=router)
"""

from truthound.checkpoint.routing.base import (
    ActionRouter,
    Route,
    RouteContext,
    RoutingResult,
    RoutingRule,
)
from truthound.checkpoint.routing.combinators import (
    AllOf,
    AnyOf,
    NotRule,
)
from truthound.checkpoint.routing.config import RouteConfigParser
from truthound.checkpoint.routing.engine import (
    ExpressionEngine,
    Jinja2Engine,
    RuleEngine,
)
from truthound.checkpoint.routing.rules import (
    AlwaysRule,
    DataAssetRule,
    ErrorRule,
    IssueCountRule,
    MetadataRule,
    NeverRule,
    PassRateRule,
    SeverityRule,
    StatusRule,
    TagRule,
    TimeWindowRule,
)

__all__ = [
    # Base
    "ActionRouter",
    "Route",
    "RouteContext",
    "RoutingResult",
    "RoutingRule",
    # Rules
    "AlwaysRule",
    "DataAssetRule",
    "ErrorRule",
    "IssueCountRule",
    "MetadataRule",
    "NeverRule",
    "PassRateRule",
    "SeverityRule",
    "StatusRule",
    "TagRule",
    "TimeWindowRule",
    # Combinators
    "AllOf",
    "AnyOf",
    "NotRule",
    # Engine
    "ExpressionEngine",
    "Jinja2Engine",
    "RuleEngine",
    # Config
    "RouteConfigParser",
]
