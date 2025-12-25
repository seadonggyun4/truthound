"""PagerDuty integration action.

This action creates incidents and events in PagerDuty when checkpoint
validations detect critical issues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class PagerDutyConfig(ActionConfig):
    """Configuration for PagerDuty action.

    Attributes:
        routing_key: PagerDuty Events API v2 routing key (integration key).
        severity: PagerDuty severity level ("critical", "error", "warning", "info").
        auto_severity: Automatically map validation severity to PagerDuty severity.
        component: Component name for the incident.
        group: Logical grouping for the incident.
        class_type: Class/type for the incident.
        custom_details: Additional custom details to include.
        dedup_key_template: Template for deduplication key.
        resolve_on_success: Automatically resolve incidents on success.
        api_endpoint: PagerDuty Events API endpoint.
    """

    routing_key: str = ""
    severity: str = "error"
    auto_severity: bool = True
    component: str = "data-quality"
    group: str = "truthound"
    class_type: str = "validation"
    custom_details: dict[str, Any] = field(default_factory=dict)
    dedup_key_template: str = "{checkpoint}_{data_asset}"
    resolve_on_success: bool = True
    api_endpoint: str = "https://events.pagerduty.com/v2/enqueue"
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE_OR_ERROR


class PagerDutyAction(BaseAction[PagerDutyConfig]):
    """Action to create PagerDuty incidents.

    Creates or resolves incidents in PagerDuty based on validation
    results using the Events API v2.

    Example:
        >>> action = PagerDutyAction(
        ...     routing_key="your-integration-key",
        ...     auto_severity=True,
        ...     resolve_on_success=True,
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "pagerduty"

    @classmethod
    def _default_config(cls) -> PagerDutyConfig:
        return PagerDutyConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Create or resolve PagerDuty incident."""
        import urllib.request
        import urllib.error

        config = self._config

        if not config.routing_key:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No routing key configured",
                error="routing_key is required",
            )

        status = checkpoint_result.status.value

        # Determine if we should trigger or resolve
        if status == "success" and config.resolve_on_success:
            event_action = "resolve"
        elif status in ("failure", "error"):
            event_action = "trigger"
        else:
            # For warning status, still trigger but with lower severity
            event_action = "trigger"

        # Build payload
        payload = self._build_payload(checkpoint_result, event_action)

        # Send to PagerDuty
        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                config.api_endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                response_body = json.loads(response.read().decode("utf-8"))

            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message=f"PagerDuty event sent: {event_action}",
                details={
                    "event_action": event_action,
                    "dedup_key": payload.get("dedup_key"),
                    "response": response_body,
                },
            )

        except urllib.error.URLError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send PagerDuty event",
                error=str(e),
            )

    def _build_payload(
        self,
        checkpoint_result: "CheckpointResult",
        event_action: str,
    ) -> dict[str, Any]:
        """Build PagerDuty Events API v2 payload."""
        config = self._config
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        # Generate dedup key
        dedup_key = config.dedup_key_template.format(
            checkpoint=checkpoint_result.checkpoint_name,
            data_asset=checkpoint_result.data_asset,
            run_id=checkpoint_result.run_id,
        )

        payload: dict[str, Any] = {
            "routing_key": config.routing_key,
            "event_action": event_action,
            "dedup_key": dedup_key,
        }

        if event_action in ("trigger", "acknowledge"):
            # Determine severity
            if config.auto_severity and stats:
                if stats.critical_issues > 0:
                    severity = "critical"
                elif stats.high_issues > 0:
                    severity = "error"
                elif stats.medium_issues > 0:
                    severity = "warning"
                else:
                    severity = "info"
            else:
                severity = config.severity

            # Build summary
            summary = (
                f"Data quality {checkpoint_result.status.value} for "
                f"'{checkpoint_result.checkpoint_name}' on {checkpoint_result.data_asset}"
            )

            if stats and stats.total_issues > 0:
                summary += f" - {stats.total_issues} issues found"
                if stats.critical_issues > 0:
                    summary += f" ({stats.critical_issues} critical)"

            payload["payload"] = {
                "summary": summary[:1024],  # PagerDuty limit
                "severity": severity,
                "source": checkpoint_result.data_asset or "truthound",
                "component": config.component,
                "group": config.group,
                "class": config.class_type,
                "timestamp": checkpoint_result.run_time.isoformat(),
                "custom_details": {
                    "checkpoint": checkpoint_result.checkpoint_name,
                    "run_id": checkpoint_result.run_id,
                    "status": checkpoint_result.status.value,
                    "data_asset": checkpoint_result.data_asset,
                    "statistics": {
                        "total_issues": stats.total_issues if stats else 0,
                        "critical": stats.critical_issues if stats else 0,
                        "high": stats.high_issues if stats else 0,
                        "medium": stats.medium_issues if stats else 0,
                        "low": stats.low_issues if stats else 0,
                        "pass_rate": f"{stats.pass_rate * 100:.1f}%" if stats else "100%",
                    },
                    **config.custom_details,
                },
            }

        return payload

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.routing_key:
            errors.append("routing_key is required")

        if self._config.severity not in ("critical", "error", "warning", "info"):
            errors.append(f"Invalid severity: {self._config.severity}")

        return errors
