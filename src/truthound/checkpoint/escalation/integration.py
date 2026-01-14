"""Escalation Policy Integration with Routing System.

This module provides integration between the escalation policy
system and the existing routing infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.escalation.engine import (
    EscalationEngine,
    EscalationEngineConfig,
    NotificationHandler,
)
from truthound.checkpoint.escalation.protocols import (
    EscalationLevel,
    EscalationPolicy,
    EscalationRecord,
    EscalationResult,
    EscalationTarget,
    EscalationTrigger,
    TargetType,
)
from truthound.checkpoint.routing.base import RouteContext, RoutingRule

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult

logger = logging.getLogger(__name__)


@dataclass
class EscalationRuleConfig:
    """Configuration for escalation rule.

    Attributes:
        policy_name: Name of escalation policy to use.
        triggers: Conditions that trigger escalation.
        severity_threshold: Minimum severity to trigger.
        issue_count_threshold: Minimum issue count to trigger.
        status_filter: Status values that trigger escalation.
        cooldown_minutes: Minimum time between triggers for same checkpoint.
        metadata: Additional rule metadata.
    """

    policy_name: str
    triggers: list[EscalationTrigger] = field(
        default_factory=lambda: [EscalationTrigger.UNACKNOWLEDGED]
    )
    severity_threshold: str = "high"
    issue_count_threshold: int = 1
    status_filter: list[str] = field(
        default_factory=lambda: ["failure", "error"]
    )
    cooldown_minutes: int = 15
    metadata: dict[str, Any] = field(default_factory=dict)


class EscalationRule(RoutingRule):
    """Routing rule that triggers escalation policies.

    This rule integrates with the routing system to automatically
    trigger escalations based on checkpoint results.

    Example:
        >>> rule = EscalationRule(
        ...     config=EscalationRuleConfig(
        ...         policy_name="critical_alerts",
        ...         severity_threshold="critical",
        ...     )
        ... )
        >>> router.add_route(Route(
        ...     name="escalate_critical",
        ...     rule=rule,
        ...     actions=[],
        ... ))
    """

    # Class-level severity order
    SEVERITY_ORDER = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "info": 4,
    }

    def __init__(self, config: EscalationRuleConfig) -> None:
        """Initialize escalation rule.

        Args:
            config: Rule configuration.
        """
        self._config = config
        self._last_trigger: dict[str, datetime] = {}

    @property
    def config(self) -> EscalationRuleConfig:
        """Get rule configuration."""
        return self._config

    @property
    def description(self) -> str:
        """Get human-readable description."""
        return (
            f"Escalate to policy '{self._config.policy_name}' "
            f"when severity >= {self._config.severity_threshold}"
        )

    def evaluate(self, context: RouteContext) -> bool:
        """Evaluate if escalation should be triggered.

        Args:
            context: Routing context.

        Returns:
            True if escalation should be triggered.
        """
        # Check status filter
        if context.status not in self._config.status_filter:
            return False

        # Check severity threshold
        max_severity = self._get_max_severity(context)
        if not self._meets_severity_threshold(max_severity):
            return False

        # Check issue count threshold
        if context.total_issues < self._config.issue_count_threshold:
            return False

        # Check cooldown
        if not self._check_cooldown(context.checkpoint_name):
            return False

        return True

    def _get_max_severity(self, context: RouteContext) -> str:
        """Get the maximum severity from context.

        Args:
            context: Routing context.

        Returns:
            Maximum severity level.
        """
        if context.critical_issues > 0:
            return "critical"
        if context.high_issues > 0:
            return "high"
        if context.medium_issues > 0:
            return "medium"
        if context.low_issues > 0:
            return "low"
        return "info"

    def _meets_severity_threshold(self, severity: str) -> bool:
        """Check if severity meets threshold.

        Args:
            severity: Severity to check.

        Returns:
            True if meets threshold.
        """
        threshold_order = self.SEVERITY_ORDER.get(self._config.severity_threshold, 4)
        severity_order = self.SEVERITY_ORDER.get(severity, 4)
        return severity_order <= threshold_order

    def _check_cooldown(self, checkpoint_name: str) -> bool:
        """Check if cooldown has elapsed.

        Args:
            checkpoint_name: Checkpoint name.

        Returns:
            True if cooldown has elapsed.
        """
        if self._config.cooldown_minutes <= 0:
            return True

        last = self._last_trigger.get(checkpoint_name)
        if last is None:
            return True

        cooldown = timedelta(minutes=self._config.cooldown_minutes)
        return datetime.now() - last >= cooldown

    def record_trigger(self, checkpoint_name: str) -> None:
        """Record that escalation was triggered.

        Args:
            checkpoint_name: Checkpoint that triggered.
        """
        self._last_trigger[checkpoint_name] = datetime.now()


class EscalationAction:
    """Action that triggers escalation via the escalation engine.

    This action integrates escalation triggering into the
    routing action system.

    Example:
        >>> action = EscalationAction(
        ...     engine=engine,
        ...     policy_name="critical_alerts",
        ... )
        >>> route = Route(
        ...     name="escalate",
        ...     rule=SeverityRule(min_severity="critical"),
        ...     actions=[action],
        ... )
    """

    def __init__(
        self,
        engine: EscalationEngine,
        policy_name: str,
        name: str = "escalation",
    ) -> None:
        """Initialize escalation action.

        Args:
            engine: Escalation engine to use.
            policy_name: Policy to trigger.
            name: Action name.
        """
        self._engine = engine
        self._policy_name = policy_name
        self._name = name

    @property
    def name(self) -> str:
        """Get action name."""
        return self._name

    @property
    def action_type(self) -> str:
        """Get action type."""
        return "escalation"

    def execute(self, checkpoint_result: "CheckpointResult") -> Any:
        """Execute the escalation action.

        Args:
            checkpoint_result: Checkpoint result to escalate.

        Returns:
            ActionResult-like object.
        """
        # Build incident ID from checkpoint
        incident_id = f"{checkpoint_result.checkpoint_name}:{checkpoint_result.run_id}"

        # Build context
        context = {
            "checkpoint_name": checkpoint_result.checkpoint_name,
            "run_id": checkpoint_result.run_id,
            "status": str(checkpoint_result.status),
            "data_asset": checkpoint_result.data_asset,
            "run_time": checkpoint_result.run_time.isoformat(),
        }

        if checkpoint_result.validation_result:
            stats = checkpoint_result.validation_result.statistics
            context["total_issues"] = stats.total_issues
            context["critical_issues"] = stats.critical_issues
            context["high_issues"] = stats.high_issues

            # Get max severity for policy matching
            if stats.critical_issues > 0:
                context["severity"] = "critical"
            elif stats.high_issues > 0:
                context["severity"] = "high"
            elif stats.medium_issues > 0:
                context["severity"] = "medium"
            else:
                context["severity"] = "low"

        # Trigger escalation asynchronously
        loop = asyncio.get_event_loop()
        try:
            if loop.is_running():
                # Schedule coroutine to run
                future = asyncio.ensure_future(
                    self._engine.trigger(
                        incident_id=incident_id,
                        context=context,
                        policy_name=self._policy_name,
                    )
                )
                # Return placeholder result
                return _create_pending_result(self._name, incident_id)
            else:
                result = loop.run_until_complete(
                    self._engine.trigger(
                        incident_id=incident_id,
                        context=context,
                        policy_name=self._policy_name,
                    )
                )
                return _create_result(self._name, result)
        except Exception as e:
            logger.exception(f"Escalation action failed: {e}")
            return _create_error_result(self._name, str(e))


def _create_pending_result(action_name: str, incident_id: str) -> Any:
    """Create a pending action result."""
    try:
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        return ActionResult(
            action_name=action_name,
            action_type="escalation",
            status=ActionStatus.SUCCESS,
            message=f"Escalation triggered for {incident_id}",
            metadata={"incident_id": incident_id, "async": True},
        )
    except ImportError:
        return {
            "action_name": action_name,
            "status": "success",
            "message": f"Escalation triggered for {incident_id}",
        }


def _create_result(action_name: str, escalation_result: EscalationResult) -> Any:
    """Create an action result from escalation result."""
    try:
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        status = ActionStatus.SUCCESS if escalation_result.success else ActionStatus.ERROR
        return ActionResult(
            action_name=action_name,
            action_type="escalation",
            status=status,
            message=escalation_result.message,
            error=escalation_result.error,
            metadata={
                "record_id": escalation_result.record.id if escalation_result.record else None,
                "level": escalation_result.level,
            },
        )
    except ImportError:
        return {
            "action_name": action_name,
            "status": "success" if escalation_result.success else "error",
            "message": escalation_result.message,
        }


def _create_error_result(action_name: str, error: str) -> Any:
    """Create an error action result."""
    try:
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        return ActionResult(
            action_name=action_name,
            action_type="escalation",
            status=ActionStatus.ERROR,
            message="Escalation action failed",
            error=error,
        )
    except ImportError:
        return {
            "action_name": action_name,
            "status": "error",
            "error": error,
        }


class ActionNotificationAdapter:
    """Adapter to use existing actions as escalation notifications.

    This adapter allows using the existing notification actions
    (Slack, Email, PagerDuty, etc.) with the escalation system.

    Example:
        >>> adapter = ActionNotificationAdapter()
        >>> adapter.register_action("slack", slack_action)
        >>> adapter.register_action("pagerduty", pagerduty_action)
        >>> engine.set_notification_handler(adapter.handle_notification)
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._actions: dict[str, "BaseAction[Any]"] = {}
        self._target_type_mapping: dict[TargetType, str] = {
            TargetType.CHANNEL: "slack",
            TargetType.EMAIL: "email",
            TargetType.WEBHOOK: "webhook",
            TargetType.SCHEDULE: "pagerduty",
        }

    def register_action(self, name: str, action: "BaseAction[Any]") -> None:
        """Register an action for notifications.

        Args:
            name: Action name.
            action: Action instance.
        """
        self._actions[name] = action

    def set_target_type_mapping(
        self,
        target_type: TargetType,
        action_name: str,
    ) -> None:
        """Set which action to use for a target type.

        Args:
            target_type: Target type.
            action_name: Action name to use.
        """
        self._target_type_mapping[target_type] = action_name

    async def handle_notification(
        self,
        record: EscalationRecord,
        level: EscalationLevel,
        targets: list[EscalationTarget],
    ) -> bool:
        """Handle notification for escalation.

        This is the notification handler for the escalation engine.

        Args:
            record: Escalation record.
            level: Current escalation level.
            targets: Targets to notify.

        Returns:
            True if at least one notification succeeded.
        """
        success_count = 0
        checkpoint_result = self._build_mock_checkpoint_result(record, level)

        for target in targets:
            action_name = self._get_action_for_target(target)
            action = self._actions.get(action_name)

            if not action:
                logger.warning(f"No action registered for target type: {target.type}")
                continue

            try:
                result = action.execute(checkpoint_result)
                if hasattr(result, "success") and result.success:
                    success_count += 1
                elif isinstance(result, dict) and result.get("status") == "success":
                    success_count += 1
            except Exception as e:
                logger.exception(f"Notification to {target.identifier} failed: {e}")

        return success_count > 0

    def _get_action_for_target(self, target: EscalationTarget) -> str:
        """Get action name for a target.

        Args:
            target: Target to get action for.

        Returns:
            Action name.
        """
        # Check if target has explicit action in metadata
        if "action" in target.metadata:
            return target.metadata["action"]

        # Use target type mapping
        return self._target_type_mapping.get(target.type, "webhook")

    def _build_mock_checkpoint_result(
        self,
        record: EscalationRecord,
        level: EscalationLevel,
    ) -> Any:
        """Build a mock checkpoint result for action execution.

        Args:
            record: Escalation record.
            level: Escalation level.

        Returns:
            Mock checkpoint result.
        """
        # Try to import and create a proper result
        try:
            from truthound.checkpoint.checkpoint import CheckpointResult, CheckpointStatus

            return CheckpointResult(
                checkpoint_name=record.context.get(
                    "checkpoint_name", f"escalation:{record.incident_id}"
                ),
                run_id=record.id,
                status=CheckpointStatus.FAILURE,
                data_asset=record.context.get("data_asset", ""),
                run_time=record.created_at,
                metadata={
                    "escalation_level": level.level,
                    "escalation_record_id": record.id,
                    "incident_id": record.incident_id,
                    "escalation_count": record.escalation_count,
                    **record.context,
                },
            )
        except ImportError:
            # Return a simple dict fallback
            return {
                "checkpoint_name": record.context.get(
                    "checkpoint_name", f"escalation:{record.incident_id}"
                ),
                "run_id": record.id,
                "status": "failure",
                "escalation_level": level.level,
                "escalation_record_id": record.id,
                "incident_id": record.incident_id,
            }


def create_escalation_route(
    engine: EscalationEngine,
    policy: EscalationPolicy,
    severity_threshold: str = "high",
    status_filter: list[str] | None = None,
) -> Any:
    """Create a route that triggers escalation.

    Convenience function to create a complete escalation route.

    Args:
        engine: Escalation engine.
        policy: Escalation policy.
        severity_threshold: Minimum severity.
        status_filter: Status values that trigger.

    Returns:
        Route configured for escalation.

    Example:
        >>> route = create_escalation_route(
        ...     engine=engine,
        ...     policy=policy,
        ...     severity_threshold="critical",
        ... )
        >>> router.add_route(route)
    """
    from truthound.checkpoint.routing.base import Route, RoutePriority

    rule = EscalationRule(
        EscalationRuleConfig(
            policy_name=policy.name,
            severity_threshold=severity_threshold,
            status_filter=status_filter or ["failure", "error"],
        )
    )

    action = EscalationAction(
        engine=engine,
        policy_name=policy.name,
        name=f"escalate_{policy.name}",
    )

    # Map severity to priority
    priority_map = {
        "critical": RoutePriority.CRITICAL,
        "high": RoutePriority.HIGH,
        "medium": RoutePriority.NORMAL,
        "low": RoutePriority.LOW,
    }

    return Route(
        name=f"escalation_{policy.name}",
        rule=rule,
        actions=[action],
        priority=priority_map.get(severity_threshold, RoutePriority.NORMAL),
        metadata={"policy_name": policy.name},
    )


def setup_escalation_with_existing_actions(
    engine: EscalationEngine,
    actions: dict[str, "BaseAction[Any]"],
) -> ActionNotificationAdapter:
    """Set up escalation to use existing notification actions.

    Args:
        engine: Escalation engine.
        actions: Dictionary of action name to action.

    Returns:
        Configured adapter.

    Example:
        >>> adapter = setup_escalation_with_existing_actions(
        ...     engine=engine,
        ...     actions={
        ...         "slack": slack_action,
        ...         "email": email_action,
        ...         "pagerduty": pagerduty_action,
        ...     },
        ... )
    """
    adapter = ActionNotificationAdapter()

    for name, action in actions.items():
        adapter.register_action(name, action)

    engine.set_notification_handler(adapter.handle_notification)

    return adapter
