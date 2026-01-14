"""Base classes and protocols for rule-based routing.

This module defines the core abstractions for the routing system:
- RoutingRule: Protocol for evaluating conditions
- Route: Combines a rule with actions
- ActionRouter: Orchestrates routing decisions
- RouteContext: Provides context data for rule evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult, BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult

logger = logging.getLogger(__name__)


class RoutePriority(int, Enum):
    """Priority levels for route evaluation order.

    Higher priority routes are evaluated first.
    """

    CRITICAL = 100
    HIGH = 80
    NORMAL = 50
    LOW = 20
    DEFAULT = 0


class RouteMode(str, Enum):
    """Routing mode for action selection.

    Attributes:
        FIRST_MATCH: Stop at first matching route
        ALL_MATCHES: Execute all matching routes
        PRIORITY_GROUP: Execute all routes in highest priority group
    """

    FIRST_MATCH = "first_match"
    ALL_MATCHES = "all_matches"
    PRIORITY_GROUP = "priority_group"


@dataclass(frozen=True)
class RouteContext:
    """Context data available for rule evaluation.

    Provides a normalized view of checkpoint result data
    that can be used by routing rules.

    Attributes:
        checkpoint_name: Name of the checkpoint
        run_id: Unique run identifier
        status: Checkpoint status (success, failure, error, warning)
        data_asset: Name/path of the validated data
        run_time: When the checkpoint ran
        total_issues: Total number of issues found
        critical_issues: Number of critical severity issues
        high_issues: Number of high severity issues
        medium_issues: Number of medium severity issues
        low_issues: Number of low severity issues
        info_issues: Number of info severity issues
        pass_rate: Percentage of validations that passed
        tags: Tags associated with the checkpoint
        metadata: Additional metadata
        validation_duration_ms: Time spent on validation
        error: Error message if any
    """

    checkpoint_name: str
    run_id: str
    status: str
    data_asset: str
    run_time: datetime
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    info_issues: int = 0
    pass_rate: float = 100.0
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_duration_ms: float = 0.0
    error: str | None = None

    @classmethod
    def from_checkpoint_result(cls, result: "CheckpointResult") -> "RouteContext":
        """Create context from a checkpoint result.

        Args:
            result: The checkpoint result to extract context from.

        Returns:
            RouteContext with extracted data.
        """
        # Extract statistics from validation result
        total_issues = 0
        critical_issues = 0
        high_issues = 0
        medium_issues = 0
        low_issues = 0
        info_issues = 0
        pass_rate = 100.0

        if result.validation_result:
            stats = result.validation_result.statistics
            total_issues = stats.total_issues
            critical_issues = stats.critical_issues
            high_issues = stats.high_issues
            medium_issues = stats.medium_issues
            low_issues = stats.low_issues
            info_issues = getattr(stats, "info_issues", 0)
            pass_rate = stats.pass_rate

        # Get status value
        status_value = (
            result.status.value
            if hasattr(result.status, "value")
            else str(result.status)
        )

        return cls(
            checkpoint_name=result.checkpoint_name,
            run_id=result.run_id,
            status=status_value,
            data_asset=result.data_asset,
            run_time=result.run_time,
            total_issues=total_issues,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            info_issues=info_issues,
            pass_rate=pass_rate,
            tags=result.metadata.get("tags", {}),
            metadata=result.metadata,
            validation_duration_ms=result.duration_ms,
            error=result.error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "run_id": self.run_id,
            "status": self.status,
            "data_asset": self.data_asset,
            "run_time": self.run_time.isoformat(),
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "info_issues": self.info_issues,
            "pass_rate": self.pass_rate,
            "tags": self.tags,
            "metadata": self.metadata,
            "validation_duration_ms": self.validation_duration_ms,
            "error": self.error,
        }


@runtime_checkable
class RoutingRule(Protocol):
    """Protocol for routing rule evaluation.

    Any class implementing this protocol can be used as a routing rule.
    Rules evaluate context data and return True if the route should be taken.

    Example:
        >>> class MyRule:
        ...     def evaluate(self, context: RouteContext) -> bool:
        ...         return context.critical_issues > 0
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Matches when critical issues exist"
    """

    def evaluate(self, context: RouteContext) -> bool:
        """Evaluate the rule against the given context.

        Args:
            context: The routing context to evaluate.

        Returns:
            True if the rule matches, False otherwise.
        """
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the rule."""
        ...


@dataclass
class Route:
    """A route that combines a rule with actions.

    A route defines when (via rule) and what (via actions) should
    be executed for a checkpoint result.

    Attributes:
        name: Human-readable name for this route
        rule: The rule that determines if this route matches
        actions: Actions to execute when the route matches
        priority: Priority for evaluation order (higher = first)
        enabled: Whether this route is active
        stop_on_match: Whether to stop evaluating further routes on match
        metadata: Additional metadata for the route
    """

    name: str
    rule: RoutingRule
    actions: list["BaseAction[Any]"]
    priority: RoutePriority | int = RoutePriority.NORMAL
    enabled: bool = True
    stop_on_match: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize priority to integer."""
        if isinstance(self.priority, RoutePriority):
            self.priority = self.priority.value

    def matches(self, context: RouteContext) -> bool:
        """Check if this route matches the given context.

        Args:
            context: The routing context to check.

        Returns:
            True if the route is enabled and the rule matches.
        """
        if not self.enabled:
            return False

        try:
            return self.rule.evaluate(context)
        except Exception as e:
            logger.warning(
                f"Route '{self.name}' rule evaluation failed: {e}",
                exc_info=True,
            )
            return False

    def __repr__(self) -> str:
        return (
            f"Route(name={self.name!r}, "
            f"rule={self.rule.description!r}, "
            f"actions={len(self.actions)}, "
            f"priority={self.priority})"
        )


@dataclass
class RoutingResult:
    """Result of routing evaluation.

    Attributes:
        matched_routes: Routes that matched the context
        executed_actions: Actions that were executed
        action_results: Results from executed actions
        evaluation_time_ms: Time spent evaluating routes
        context: The context that was evaluated
    """

    matched_routes: list[Route] = field(default_factory=list)
    executed_actions: list["BaseAction[Any]"] = field(default_factory=list)
    action_results: list["ActionResult"] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    context: RouteContext | None = None

    @property
    def has_matches(self) -> bool:
        """Check if any routes matched."""
        return len(self.matched_routes) > 0

    @property
    def all_successful(self) -> bool:
        """Check if all executed actions were successful."""
        return all(r.success for r in self.action_results)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "matched_routes": [r.name for r in self.matched_routes],
            "executed_actions": [a.name for a in self.executed_actions],
            "action_results": [r.to_dict() for r in self.action_results],
            "evaluation_time_ms": self.evaluation_time_ms,
        }


class ActionRouter:
    """Routes checkpoint results to appropriate actions based on rules.

    The ActionRouter is the main orchestrator for rule-based notification
    routing. It evaluates routes against checkpoint results and executes
    matching actions.

    Example:
        >>> router = ActionRouter(mode=RouteMode.ALL_MATCHES)
        >>> router.add_route(Route(
        ...     name="critical_alerts",
        ...     rule=SeverityRule(min_severity="critical"),
        ...     actions=[PagerDutyAction(...)],
        ...     priority=RoutePriority.CRITICAL,
        ... ))
        >>> router.add_route(Route(
        ...     name="high_alerts",
        ...     rule=SeverityRule(min_severity="high"),
        ...     actions=[SlackNotification(...)],
        ...     priority=RoutePriority.HIGH,
        ... ))
        >>>
        >>> result = router.route(checkpoint_result)
        >>> print(f"Matched {len(result.matched_routes)} routes")

    Attributes:
        mode: How to handle multiple matching routes
        default_actions: Actions to execute when no routes match
    """

    def __init__(
        self,
        mode: RouteMode = RouteMode.ALL_MATCHES,
        default_actions: list["BaseAction[Any]"] | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            mode: Routing mode for handling matches
            default_actions: Actions to execute when no routes match
        """
        self._routes: list[Route] = []
        self._mode = mode
        self._default_actions = default_actions or []

    @property
    def routes(self) -> list[Route]:
        """Get all registered routes."""
        return list(self._routes)

    @property
    def mode(self) -> RouteMode:
        """Get the routing mode."""
        return self._mode

    @mode.setter
    def mode(self, value: RouteMode) -> None:
        """Set the routing mode."""
        self._mode = value

    def add_route(self, route: Route) -> "ActionRouter":
        """Add a route to the router.

        Routes are kept sorted by priority (highest first).

        Args:
            route: The route to add.

        Returns:
            Self for chaining.
        """
        self._routes.append(route)
        self._routes.sort(key=lambda r: r.priority, reverse=True)
        return self

    def remove_route(self, name: str) -> bool:
        """Remove a route by name.

        Args:
            name: Name of the route to remove.

        Returns:
            True if a route was removed, False if not found.
        """
        for i, route in enumerate(self._routes):
            if route.name == name:
                del self._routes[i]
                return True
        return False

    def get_route(self, name: str) -> Route | None:
        """Get a route by name.

        Args:
            name: Name of the route to find.

        Returns:
            The route if found, None otherwise.
        """
        for route in self._routes:
            if route.name == name:
                return route
        return None

    def evaluate(self, context: RouteContext) -> list[Route]:
        """Evaluate which routes match the given context.

        This method only evaluates routes without executing actions.

        Args:
            context: The routing context to evaluate.

        Returns:
            List of matching routes.
        """
        matched: list[Route] = []
        highest_priority: int | None = None

        for route in self._routes:
            if not route.matches(context):
                continue

            if self._mode == RouteMode.FIRST_MATCH:
                return [route]

            if self._mode == RouteMode.PRIORITY_GROUP:
                if highest_priority is None:
                    highest_priority = route.priority
                elif route.priority < highest_priority:
                    break  # Routes are sorted by priority
                matched.append(route)
            else:  # ALL_MATCHES
                matched.append(route)
                if route.stop_on_match:
                    break

        return matched

    def route(
        self,
        checkpoint_result: "CheckpointResult",
        execute_actions: bool = True,
    ) -> RoutingResult:
        """Route a checkpoint result to appropriate actions.

        Args:
            checkpoint_result: The result to route.
            execute_actions: Whether to execute matched actions.

        Returns:
            RoutingResult with matched routes and action results.
        """
        import time

        start_time = time.time()

        # Create context from checkpoint result
        context = RouteContext.from_checkpoint_result(checkpoint_result)

        # Evaluate routes
        matched_routes = self.evaluate(context)

        result = RoutingResult(
            matched_routes=matched_routes,
            context=context,
        )

        # Collect actions to execute
        actions_to_execute: list["BaseAction[Any]"] = []

        if matched_routes:
            for route in matched_routes:
                actions_to_execute.extend(route.actions)
        elif self._default_actions:
            actions_to_execute = list(self._default_actions)
            logger.debug("No routes matched, using default actions")

        result.executed_actions = actions_to_execute

        # Execute actions if requested
        if execute_actions and actions_to_execute:
            for action in actions_to_execute:
                try:
                    action_result = action.execute(checkpoint_result)
                    result.action_results.append(action_result)
                except Exception as e:
                    from truthound.checkpoint.actions.base import (
                        ActionResult,
                        ActionStatus,
                    )

                    error_result = ActionResult(
                        action_name=action.name,
                        action_type=action.action_type,
                        status=ActionStatus.ERROR,
                        message="Action execution failed during routing",
                        error=str(e),
                    )
                    result.action_results.append(error_result)
                    logger.error(
                        f"Action '{action.name}' failed during routing: {e}",
                        exc_info=True,
                    )

        result.evaluation_time_ms = (time.time() - start_time) * 1000

        return result

    def clear_routes(self) -> None:
        """Remove all routes."""
        self._routes.clear()

    def set_default_actions(
        self, actions: list["BaseAction[Any]"]
    ) -> "ActionRouter":
        """Set default actions for when no routes match.

        Args:
            actions: Actions to execute as default.

        Returns:
            Self for chaining.
        """
        self._default_actions = actions
        return self

    def __len__(self) -> int:
        return len(self._routes)

    def __repr__(self) -> str:
        return (
            f"ActionRouter(routes={len(self._routes)}, mode={self._mode.value})"
        )
