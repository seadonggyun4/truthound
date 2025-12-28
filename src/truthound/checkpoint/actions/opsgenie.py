"""OpsGenie integration action.

This module provides comprehensive OpsGenie integration for checkpoint notifications
using the OpsGenie Alert API v2.

Features:
    - Full OpsGenie Alert API v2 support
    - Multiple alert actions (create, close, acknowledge, add note)
    - Flexible responder configuration (users, teams, schedules, escalations)
    - Auto-priority mapping from validation severity
    - Deduplication for alert grouping
    - Tag and custom property support
    - Region-aware API endpoints (US, EU)
    - Extensible payload builder pattern

Example:
    >>> from truthound.checkpoint.actions import OpsGenieAction
    >>>
    >>> action = OpsGenieAction(
    ...     api_key="your-api-key",
    ...     notify_on="failure",
    ...     auto_priority=True,
    ...     responders=[
    ...         {"type": "team", "name": "data-quality-team"},
    ...         {"type": "user", "username": "admin@example.com"},
    ...     ],
    ...     tags=["data-quality", "production"],
    ... )
    >>> result = action.execute(checkpoint_result)

References:
    - OpsGenie Alert API: https://docs.opsgenie.com/docs/alert-api
    - OpsGenie Integrations: https://docs.opsgenie.com/docs/integration-types-to-use
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


# =============================================================================
# Enums and Constants
# =============================================================================


class OpsGenieRegion(str, Enum):
    """OpsGenie API regions."""

    US = "us"
    EU = "eu"

    def __str__(self) -> str:
        return self.value

    @property
    def api_url(self) -> str:
        """Get API URL for this region."""
        if self == OpsGenieRegion.EU:
            return "https://api.eu.opsgenie.com"
        return "https://api.opsgenie.com"


class AlertPriority(str, Enum):
    """OpsGenie alert priority levels.

    P1 is the highest priority, P5 is the lowest.
    """

    P1 = "P1"  # Critical
    P2 = "P2"  # High
    P3 = "P3"  # Moderate
    P4 = "P4"  # Low
    P5 = "P5"  # Informational

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_severity(cls, severity: str) -> "AlertPriority":
        """Map severity string to priority.

        Args:
            severity: Severity level (critical, high, medium, low, info).

        Returns:
            Corresponding AlertPriority.
        """
        mapping = {
            "critical": cls.P1,
            "high": cls.P2,
            "medium": cls.P3,
            "moderate": cls.P3,
            "low": cls.P4,
            "info": cls.P5,
            "informational": cls.P5,
        }
        return mapping.get(severity.lower(), cls.P3)


class AlertAction(str, Enum):
    """OpsGenie alert actions."""

    CREATE = "create"
    CLOSE = "close"
    ACKNOWLEDGE = "acknowledge"
    UNACKNOWLEDGE = "unacknowledge"
    ADD_NOTE = "add_note"
    SNOOZE = "snooze"
    ESCALATE = "escalate"
    ASSIGN = "assign"
    ADD_TAGS = "add_tags"
    REMOVE_TAGS = "remove_tags"
    ADD_DETAILS = "add_details"
    REMOVE_DETAILS = "remove_details"

    def __str__(self) -> str:
        return self.value


class ResponderType(str, Enum):
    """Types of OpsGenie responders."""

    USER = "user"
    TEAM = "team"
    ESCALATION = "escalation"
    SCHEDULE = "schedule"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Responder Configuration
# =============================================================================


@dataclass
class Responder:
    """OpsGenie responder configuration.

    Represents a user, team, escalation, or schedule that should be notified.

    Attributes:
        type: Type of responder (user, team, escalation, schedule).
        id: Responder ID (optional, use either id or name/username).
        name: Team/escalation/schedule name (for non-user types).
        username: User email/username (for user type).
    """

    type: ResponderType | str
    id: str | None = None
    name: str | None = None
    username: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize responder type."""
        if isinstance(self.type, str):
            self.type = ResponderType(self.type.lower())

    def to_dict(self) -> dict[str, str]:
        """Convert to OpsGenie API format.

        Returns:
            Dictionary in OpsGenie responder format.
        """
        result: dict[str, str] = {"type": str(self.type)}

        if self.id:
            result["id"] = self.id
        elif self.type == ResponderType.USER and self.username:
            result["username"] = self.username
        elif self.name:
            result["name"] = self.name

        return result

    @classmethod
    def user(cls, username: str) -> "Responder":
        """Create a user responder.

        Args:
            username: User email/username.

        Returns:
            Responder instance for a user.
        """
        return cls(type=ResponderType.USER, username=username)

    @classmethod
    def team(cls, name: str | None = None, id: str | None = None) -> "Responder":
        """Create a team responder.

        Args:
            name: Team name.
            id: Team ID.

        Returns:
            Responder instance for a team.
        """
        return cls(type=ResponderType.TEAM, name=name, id=id)

    @classmethod
    def escalation(cls, name: str | None = None, id: str | None = None) -> "Responder":
        """Create an escalation responder.

        Args:
            name: Escalation name.
            id: Escalation ID.

        Returns:
            Responder instance for an escalation.
        """
        return cls(type=ResponderType.ESCALATION, name=name, id=id)

    @classmethod
    def schedule(cls, name: str | None = None, id: str | None = None) -> "Responder":
        """Create a schedule responder.

        Args:
            name: Schedule name.
            id: Schedule ID.

        Returns:
            Responder instance for a schedule.
        """
        return cls(type=ResponderType.SCHEDULE, name=name, id=id)


# =============================================================================
# Alert Payload Builder (Builder Pattern)
# =============================================================================


@runtime_checkable
class AlertPayloadComponent(Protocol):
    """Protocol for alert payload components."""

    def apply(self, payload: dict[str, Any]) -> None:
        """Apply this component to the payload."""
        ...


class AlertPayloadBuilder:
    """Fluent builder for creating OpsGenie alert payloads.

    This builder provides a type-safe, fluent API for constructing
    OpsGenie alert payloads with proper validation.

    Example:
        >>> payload = (
        ...     AlertPayloadBuilder()
        ...     .set_message("Data quality check failed")
        ...     .set_priority(AlertPriority.P2)
        ...     .add_responder(Responder.team("data-team"))
        ...     .add_tag("production")
        ...     .set_details({"checkpoint": "daily_check"})
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._message: str = ""
        self._alias: str | None = None
        self._description: str | None = None
        self._responders: list[Responder] = []
        self._visible_to: list[Responder] = []
        self._actions: list[str] = []
        self._tags: list[str] = []
        self._details: dict[str, str] = {}
        self._entity: str | None = None
        self._source: str | None = None
        self._priority: AlertPriority = AlertPriority.P3
        self._user: str | None = None
        self._note: str | None = None

    def set_message(self, message: str) -> "AlertPayloadBuilder":
        """Set the alert message (required, max 130 chars).

        Args:
            message: Alert message.

        Returns:
            Self for method chaining.
        """
        self._message = message[:130]
        return self

    def set_alias(self, alias: str) -> "AlertPayloadBuilder":
        """Set the alert alias for deduplication (max 512 chars).

        Args:
            alias: Unique alias for grouping alerts.

        Returns:
            Self for method chaining.
        """
        self._alias = alias[:512]
        return self

    def set_description(self, description: str) -> "AlertPayloadBuilder":
        """Set the alert description (max 15000 chars).

        Args:
            description: Detailed description.

        Returns:
            Self for method chaining.
        """
        self._description = description[:15000]
        return self

    def add_responder(self, responder: Responder) -> "AlertPayloadBuilder":
        """Add a responder to the alert.

        Args:
            responder: Responder to notify.

        Returns:
            Self for method chaining.
        """
        self._responders.append(responder)
        return self

    def add_responders(self, responders: list[Responder]) -> "AlertPayloadBuilder":
        """Add multiple responders.

        Args:
            responders: List of responders to notify.

        Returns:
            Self for method chaining.
        """
        self._responders.extend(responders)
        return self

    def add_visible_to(self, responder: Responder) -> "AlertPayloadBuilder":
        """Add a responder who can see the alert (without being notified).

        Args:
            responder: Responder who can view the alert.

        Returns:
            Self for method chaining.
        """
        self._visible_to.append(responder)
        return self

    def add_action(self, action: str) -> "AlertPayloadBuilder":
        """Add a custom action button.

        Args:
            action: Action button text.

        Returns:
            Self for method chaining.
        """
        self._actions.append(action)
        return self

    def add_tag(self, tag: str) -> "AlertPayloadBuilder":
        """Add a tag to the alert.

        Args:
            tag: Tag string.

        Returns:
            Self for method chaining.
        """
        self._tags.append(tag)
        return self

    def add_tags(self, tags: list[str]) -> "AlertPayloadBuilder":
        """Add multiple tags.

        Args:
            tags: List of tags.

        Returns:
            Self for method chaining.
        """
        self._tags.extend(tags)
        return self

    def set_details(self, details: dict[str, Any]) -> "AlertPayloadBuilder":
        """Set custom details (key-value pairs, max 8000 chars total).

        Args:
            details: Dictionary of custom details.

        Returns:
            Self for method chaining.
        """
        self._details = {str(k): str(v) for k, v in details.items()}
        return self

    def add_detail(self, key: str, value: Any) -> "AlertPayloadBuilder":
        """Add a single custom detail.

        Args:
            key: Detail key.
            value: Detail value.

        Returns:
            Self for method chaining.
        """
        self._details[str(key)] = str(value)
        return self

    def set_entity(self, entity: str) -> "AlertPayloadBuilder":
        """Set the entity associated with the alert (max 512 chars).

        Args:
            entity: Entity name (e.g., service, host).

        Returns:
            Self for method chaining.
        """
        self._entity = entity[:512]
        return self

    def set_source(self, source: str) -> "AlertPayloadBuilder":
        """Set the alert source (max 100 chars).

        Args:
            source: Source of the alert.

        Returns:
            Self for method chaining.
        """
        self._source = source[:100]
        return self

    def set_priority(self, priority: AlertPriority | str) -> "AlertPayloadBuilder":
        """Set the alert priority.

        Args:
            priority: Priority level (P1-P5).

        Returns:
            Self for method chaining.
        """
        if isinstance(priority, str):
            priority = AlertPriority(priority.upper())
        self._priority = priority
        return self

    def set_user(self, user: str) -> "AlertPayloadBuilder":
        """Set the user who triggered the alert.

        Args:
            user: Username or email.

        Returns:
            Self for method chaining.
        """
        self._user = user
        return self

    def set_note(self, note: str) -> "AlertPayloadBuilder":
        """Set a note to include with the alert.

        Args:
            note: Note text.

        Returns:
            Self for method chaining.
        """
        self._note = note
        return self

    def build(self) -> dict[str, Any]:
        """Build the alert payload.

        Returns:
            Dictionary representing the OpsGenie alert payload.

        Raises:
            ValueError: If required fields are missing.
        """
        if not self._message:
            raise ValueError("Message is required")

        payload: dict[str, Any] = {
            "message": self._message,
            "priority": str(self._priority),
        }

        if self._alias:
            payload["alias"] = self._alias
        if self._description:
            payload["description"] = self._description
        if self._responders:
            payload["responders"] = [r.to_dict() for r in self._responders]
        if self._visible_to:
            payload["visibleTo"] = [r.to_dict() for r in self._visible_to]
        if self._actions:
            payload["actions"] = self._actions
        if self._tags:
            payload["tags"] = self._tags
        if self._details:
            payload["details"] = self._details
        if self._entity:
            payload["entity"] = self._entity
        if self._source:
            payload["source"] = self._source
        if self._user:
            payload["user"] = self._user
        if self._note:
            payload["note"] = self._note

        return payload


# =============================================================================
# Message Templates (Strategy Pattern)
# =============================================================================


class AlertTemplate(ABC):
    """Abstract base class for alert message templates.

    Templates define how checkpoint results are formatted into OpsGenie alerts.
    """

    @abstractmethod
    def build_payload(
        self,
        checkpoint_result: "CheckpointResult",
        config: "OpsGenieConfig",
    ) -> dict[str, Any]:
        """Build alert payload from checkpoint result.

        Args:
            checkpoint_result: The validation result.
            config: OpsGenie configuration.

        Returns:
            Alert payload dictionary.
        """
        pass


class DefaultAlertTemplate(AlertTemplate):
    """Default alert template with comprehensive details."""

    def build_payload(
        self,
        checkpoint_result: "CheckpointResult",
        config: "OpsGenieConfig",
    ) -> dict[str, Any]:
        """Build default alert payload."""
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None
        status = checkpoint_result.status.value

        # Build message
        message = (
            f"[{status.upper()}] {checkpoint_result.checkpoint_name} - "
            f"{checkpoint_result.data_asset}"
        )

        # Build description
        description_parts = [
            f"**Checkpoint**: {checkpoint_result.checkpoint_name}",
            f"**Data Asset**: {checkpoint_result.data_asset}",
            f"**Status**: {status}",
            f"**Run ID**: {checkpoint_result.run_id}",
            f"**Time**: {checkpoint_result.run_time.isoformat()}",
        ]

        if stats:
            description_parts.extend([
                "",
                "**Statistics**:",
                f"- Total Issues: {stats.total_issues}",
                f"- Critical: {stats.critical_issues}",
                f"- High: {stats.high_issues}",
                f"- Medium: {stats.medium_issues}",
                f"- Low: {stats.low_issues}",
                f"- Pass Rate: {stats.pass_rate * 100:.1f}%",
            ])

        description = "\n".join(description_parts)

        # Determine priority
        priority = self._determine_priority(stats, config)

        # Build alias for deduplication
        alias = config.alias_template.format(
            checkpoint=checkpoint_result.checkpoint_name,
            data_asset=checkpoint_result.data_asset,
            run_id=checkpoint_result.run_id,
        )

        # Build payload using builder
        builder = (
            AlertPayloadBuilder()
            .set_message(message)
            .set_alias(alias)
            .set_description(description)
            .set_priority(priority)
            .set_entity(checkpoint_result.data_asset or "truthound")
            .set_source("truthound")
            .add_tags(list(config.tags))
        )

        # Add responders
        for responder in config.responders:
            if isinstance(responder, Responder):
                builder.add_responder(responder)
            elif isinstance(responder, dict):
                builder.add_responder(Responder(**responder))

        # Add custom details
        details = {
            "checkpoint": checkpoint_result.checkpoint_name,
            "data_asset": checkpoint_result.data_asset or "",
            "run_id": checkpoint_result.run_id,
            "status": status,
        }
        if stats:
            details.update({
                "total_issues": str(stats.total_issues),
                "critical_issues": str(stats.critical_issues),
                "high_issues": str(stats.high_issues),
                "medium_issues": str(stats.medium_issues),
                "low_issues": str(stats.low_issues),
                "pass_rate": f"{stats.pass_rate * 100:.1f}%",
            })
        details.update(config.custom_details)
        builder.set_details(details)

        return builder.build()

    def _determine_priority(
        self,
        stats: Any,
        config: "OpsGenieConfig",
    ) -> AlertPriority:
        """Determine alert priority based on validation statistics."""
        if not config.auto_priority or not stats:
            return config.priority

        if stats.critical_issues > 0:
            return AlertPriority.P1
        elif stats.high_issues > 0:
            return AlertPriority.P2
        elif stats.medium_issues > 0:
            return AlertPriority.P3
        elif stats.low_issues > 0:
            return AlertPriority.P4
        else:
            return AlertPriority.P5


class MinimalAlertTemplate(AlertTemplate):
    """Minimal alert template with only essential information."""

    def build_payload(
        self,
        checkpoint_result: "CheckpointResult",
        config: "OpsGenieConfig",
    ) -> dict[str, Any]:
        """Build minimal alert payload."""
        status = checkpoint_result.status.value
        stats = checkpoint_result.validation_result.statistics if checkpoint_result.validation_result else None

        message = f"[{status.upper()}] {checkpoint_result.checkpoint_name}"
        if stats and stats.total_issues > 0:
            message += f" ({stats.total_issues} issues)"

        alias = config.alias_template.format(
            checkpoint=checkpoint_result.checkpoint_name,
            data_asset=checkpoint_result.data_asset or "",
            run_id=checkpoint_result.run_id,
        )

        builder = (
            AlertPayloadBuilder()
            .set_message(message)
            .set_alias(alias)
            .set_priority(config.priority)
            .set_source("truthound")
            .add_tags(list(config.tags))
        )

        for responder in config.responders:
            if isinstance(responder, Responder):
                builder.add_responder(responder)
            elif isinstance(responder, dict):
                builder.add_responder(Responder(**responder))

        return builder.build()


class DetailedAlertTemplate(AlertTemplate):
    """Detailed alert template with extended information and recommendations."""

    def build_payload(
        self,
        checkpoint_result: "CheckpointResult",
        config: "OpsGenieConfig",
    ) -> dict[str, Any]:
        """Build detailed alert payload."""
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None
        status = checkpoint_result.status.value

        # Build message
        message = f"Data Quality Alert: {checkpoint_result.checkpoint_name}"

        # Build comprehensive description
        description_parts = [
            "# Data Quality Validation Alert",
            "",
            "## Overview",
            f"- **Checkpoint**: {checkpoint_result.checkpoint_name}",
            f"- **Data Asset**: {checkpoint_result.data_asset}",
            f"- **Status**: {status.upper()}",
            f"- **Run ID**: {checkpoint_result.run_id}",
            f"- **Timestamp**: {checkpoint_result.run_time.isoformat()}",
        ]

        if stats:
            description_parts.extend([
                "",
                "## Validation Statistics",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total Issues | {stats.total_issues} |",
                f"| Critical | {stats.critical_issues} |",
                f"| High | {stats.high_issues} |",
                f"| Medium | {stats.medium_issues} |",
                f"| Low | {stats.low_issues} |",
                f"| Pass Rate | {stats.pass_rate * 100:.1f}% |",
            ])

        # Add failed validations if available
        if validation and validation.results:
            failed_results = [r for r in validation.results if not r.passed]
            if failed_results:
                description_parts.extend([
                    "",
                    "## Failed Validations",
                ])
                for i, result in enumerate(failed_results[:10], 1):
                    description_parts.append(
                        f"{i}. **{result.validator_name}** on `{result.column}`: {result.message}"
                    )
                if len(failed_results) > 10:
                    description_parts.append(f"... and {len(failed_results) - 10} more")

        description_parts.extend([
            "",
            "## Recommended Actions",
            "1. Review the validation results in detail",
            "2. Investigate the root cause of failures",
            "3. Apply necessary data fixes",
            "4. Re-run validation to confirm resolution",
        ])

        description = "\n".join(description_parts)

        # Determine priority
        priority = self._determine_priority(stats, config)

        alias = config.alias_template.format(
            checkpoint=checkpoint_result.checkpoint_name,
            data_asset=checkpoint_result.data_asset or "",
            run_id=checkpoint_result.run_id,
        )

        builder = (
            AlertPayloadBuilder()
            .set_message(message)
            .set_alias(alias)
            .set_description(description)
            .set_priority(priority)
            .set_entity(checkpoint_result.data_asset or "truthound")
            .set_source("truthound")
            .add_tags(list(config.tags))
            .add_action("View Dashboard")
            .add_action("Acknowledge")
            .add_action("Escalate")
        )

        for responder in config.responders:
            if isinstance(responder, Responder):
                builder.add_responder(responder)
            elif isinstance(responder, dict):
                builder.add_responder(Responder(**responder))

        # Extended details
        details = {
            "checkpoint": checkpoint_result.checkpoint_name,
            "data_asset": checkpoint_result.data_asset or "",
            "run_id": checkpoint_result.run_id,
            "status": status,
            "environment": config.custom_details.get("environment", "unknown"),
        }
        if stats:
            details.update({
                "total_issues": str(stats.total_issues),
                "critical_issues": str(stats.critical_issues),
                "pass_rate": f"{stats.pass_rate * 100:.1f}%",
            })
        details.update(config.custom_details)
        builder.set_details(details)

        return builder.build()

    def _determine_priority(
        self,
        stats: Any,
        config: "OpsGenieConfig",
    ) -> AlertPriority:
        """Determine alert priority based on validation statistics."""
        if not config.auto_priority or not stats:
            return config.priority

        if stats.critical_issues > 0:
            return AlertPriority.P1
        elif stats.high_issues > 0:
            return AlertPriority.P2
        elif stats.medium_issues > 0:
            return AlertPriority.P3
        else:
            return AlertPriority.P4


# Template Registry
_TEMPLATE_REGISTRY: dict[str, type[AlertTemplate]] = {
    "default": DefaultAlertTemplate,
    "minimal": MinimalAlertTemplate,
    "detailed": DetailedAlertTemplate,
}


def register_template(name: str, template_class: type[AlertTemplate]) -> None:
    """Register a custom alert template.

    Args:
        name: Template name for lookup.
        template_class: AlertTemplate subclass.
    """
    _TEMPLATE_REGISTRY[name.lower()] = template_class


def get_template(name: str) -> AlertTemplate:
    """Get a template instance by name.

    Args:
        name: Template name.

    Returns:
        AlertTemplate instance.

    Raises:
        ValueError: If template is not found.
    """
    template_class = _TEMPLATE_REGISTRY.get(name.lower())
    if not template_class:
        available = ", ".join(_TEMPLATE_REGISTRY.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return template_class()


# =============================================================================
# HTTP Client Abstraction
# =============================================================================


class OpsGenieHTTPClient:
    """HTTP client for OpsGenie API calls.

    This abstraction allows for easier testing and customization.
    """

    def __init__(
        self,
        api_key: str,
        region: OpsGenieRegion = OpsGenieRegion.US,
        timeout: int = 30,
        proxy: str | None = None,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            api_key: OpsGenie API key.
            region: API region (US or EU).
            timeout: Request timeout in seconds.
            proxy: Optional proxy URL.
            verify_ssl: Whether to verify SSL certificates.
        """
        self._api_key = api_key
        self._region = region
        self._timeout = timeout
        self._proxy = proxy
        self._verify_ssl = verify_ssl

    @property
    def base_url(self) -> str:
        """Get the base API URL."""
        return self._region.api_url

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"GenieKey {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Truthound/1.0",
        }

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send POST request to OpsGenie API.

        Args:
            endpoint: API endpoint path.
            payload: Request payload.

        Returns:
            Response data.

        Raises:
            OpsGenieAPIError: If the request fails.
        """
        import urllib.request
        import urllib.error
        import ssl

        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        headers = self._get_headers()

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        # Configure SSL
        context: ssl.SSLContext | None = None
        if not self._verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Configure proxy
        if self._proxy:
            proxy_handler = urllib.request.ProxyHandler({"https": self._proxy})
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()

        try:
            with opener.open(request, timeout=self._timeout, context=context) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body) if response_body else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise OpsGenieAPIError(
                f"HTTP {e.code}: {e.reason}",
                status_code=e.code,
                response_body=error_body,
            ) from e
        except urllib.error.URLError as e:
            raise OpsGenieAPIError(f"URL Error: {e.reason}") from e


class OpsGenieAPIError(Exception):
    """Exception for OpsGenie API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Error message.
            status_code: HTTP status code.
            response_body: Raw response body.
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OpsGenieConfig(ActionConfig):
    """Configuration for OpsGenie action.

    Attributes:
        api_key: OpsGenie API key (required).
        region: API region (us, eu).
        priority: Default alert priority (P1-P5).
        auto_priority: Automatically map validation severity to priority.
        responders: List of responders to notify.
        visible_to: List of responders who can view but not be notified.
        tags: Tags to attach to alerts.
        alias_template: Template for generating alert aliases (dedup).
        close_on_success: Close matching alerts on validation success.
        acknowledge_on_warning: Acknowledge alerts on warning status.
        template: Alert template to use (default, minimal, detailed).
        custom_template: Custom AlertTemplate instance.
        custom_details: Additional details to include in alerts.
        source: Source identifier for alerts.
        entity: Entity associated with alerts.
        actions: Custom action buttons to include.
        proxy: Optional proxy URL.
        verify_ssl: Whether to verify SSL certificates.
    """

    api_key: str = ""
    region: OpsGenieRegion | str = OpsGenieRegion.US
    priority: AlertPriority | str = AlertPriority.P3
    auto_priority: bool = True
    responders: list[Responder | dict[str, str]] = field(default_factory=list)
    visible_to: list[Responder | dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=lambda: ["truthound", "data-quality"])
    alias_template: str = "truthound_{checkpoint}_{data_asset}"
    close_on_success: bool = True
    acknowledge_on_warning: bool = False
    template: str = "default"
    custom_template: AlertTemplate | None = None
    custom_details: dict[str, str] = field(default_factory=dict)
    source: str = "truthound"
    entity: str | None = None
    actions: list[str] = field(default_factory=list)
    proxy: str | None = None
    verify_ssl: bool = True
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE_OR_ERROR

    def __post_init__(self) -> None:
        """Normalize configuration values."""
        super().__post_init__()

        if isinstance(self.region, str):
            self.region = OpsGenieRegion(self.region.lower())
        if isinstance(self.priority, str):
            self.priority = AlertPriority(self.priority.upper())


# =============================================================================
# Main Action Class
# =============================================================================


class OpsGenieAction(BaseAction[OpsGenieConfig]):
    """Action to create and manage OpsGenie alerts.

    Creates, closes, or acknowledges alerts in OpsGenie based on validation
    results using the Alert API v2.

    Features:
        - Full OpsGenie Alert API v2 support
        - Multiple alert actions (create, close, acknowledge)
        - Flexible responder configuration
        - Auto-priority mapping from validation severity
        - Deduplication for alert grouping
        - Extensible template system

    Example:
        >>> # Basic usage
        >>> action = OpsGenieAction(
        ...     api_key="your-api-key",
        ...     notify_on="failure",
        ... )
        >>> result = action.execute(checkpoint_result)

        >>> # With team responders
        >>> action = OpsGenieAction(
        ...     api_key="your-api-key",
        ...     responders=[
        ...         Responder.team("data-quality-team"),
        ...         Responder.user("admin@example.com"),
        ...     ],
        ...     auto_priority=True,
        ...     tags=["production", "critical-pipeline"],
        ... )

        >>> # Using custom template
        >>> action = OpsGenieAction(
        ...     api_key="your-api-key",
        ...     template="detailed",
        ...     close_on_success=True,
        ... )
    """

    action_type = "opsgenie"

    def __init__(
        self,
        config: OpsGenieConfig | None = None,
        client: OpsGenieHTTPClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the action.

        Args:
            config: OpsGenie configuration.
            client: Optional HTTP client (for testing).
            **kwargs: Additional configuration options.
        """
        super().__init__(config, **kwargs)
        self._client = client

    @classmethod
    def _default_config(cls) -> OpsGenieConfig:
        """Create default configuration."""
        return OpsGenieConfig()

    def _get_client(self) -> OpsGenieHTTPClient:
        """Get or create HTTP client."""
        if self._client:
            return self._client

        config = self._config
        return OpsGenieHTTPClient(
            api_key=config.api_key,
            region=config.region if isinstance(config.region, OpsGenieRegion) else OpsGenieRegion(config.region),
            timeout=config.timeout_seconds,
            proxy=config.proxy,
            verify_ssl=config.verify_ssl,
        )

    def _get_template(self) -> AlertTemplate:
        """Get the configured alert template."""
        if self._config.custom_template:
            return self._config.custom_template
        return get_template(self._config.template)

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute the OpsGenie action."""
        config = self._config

        if not config.api_key:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No API key configured",
                error="api_key is required",
            )

        status = checkpoint_result.status.value
        client = self._get_client()

        try:
            # Determine action based on status
            if status == "success" and config.close_on_success:
                return self._close_alert(checkpoint_result, client)
            elif status == "warning" and config.acknowledge_on_warning:
                return self._acknowledge_alert(checkpoint_result, client)
            elif status in ("failure", "error"):
                return self._create_alert(checkpoint_result, client)
            else:
                # Create alert for other statuses
                return self._create_alert(checkpoint_result, client)

        except OpsGenieAPIError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="OpsGenie API error",
                error=str(e),
                details={
                    "status_code": e.status_code,
                    "response": e.response_body,
                },
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send OpsGenie alert",
                error=str(e),
            )

    def _create_alert(
        self,
        checkpoint_result: "CheckpointResult",
        client: OpsGenieHTTPClient,
    ) -> ActionResult:
        """Create a new alert."""
        template = self._get_template()
        payload = template.build_payload(checkpoint_result, self._config)

        response = client.post("/v2/alerts", payload)

        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message="OpsGenie alert created",
            details={
                "action": "create",
                "alias": payload.get("alias"),
                "priority": payload.get("priority"),
                "request_id": response.get("requestId"),
                "result": response.get("result"),
            },
        )

    def _close_alert(
        self,
        checkpoint_result: "CheckpointResult",
        client: OpsGenieHTTPClient,
    ) -> ActionResult:
        """Close an existing alert."""
        alias = self._config.alias_template.format(
            checkpoint=checkpoint_result.checkpoint_name,
            data_asset=checkpoint_result.data_asset or "",
            run_id=checkpoint_result.run_id,
        )

        payload = {
            "source": self._config.source,
            "user": "truthound",
            "note": f"Validation succeeded at {checkpoint_result.run_time.isoformat()}",
        }

        # Use alias identifier
        endpoint = f"/v2/alerts/{alias}/close?identifierType=alias"
        response = client.post(endpoint, payload)

        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message="OpsGenie alert closed",
            details={
                "action": "close",
                "alias": alias,
                "request_id": response.get("requestId"),
            },
        )

    def _acknowledge_alert(
        self,
        checkpoint_result: "CheckpointResult",
        client: OpsGenieHTTPClient,
    ) -> ActionResult:
        """Acknowledge an existing alert."""
        alias = self._config.alias_template.format(
            checkpoint=checkpoint_result.checkpoint_name,
            data_asset=checkpoint_result.data_asset or "",
            run_id=checkpoint_result.run_id,
        )

        payload = {
            "source": self._config.source,
            "user": "truthound",
            "note": f"Validation warning at {checkpoint_result.run_time.isoformat()}",
        }

        endpoint = f"/v2/alerts/{alias}/acknowledge?identifierType=alias"
        response = client.post(endpoint, payload)

        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message="OpsGenie alert acknowledged",
            details={
                "action": "acknowledge",
                "alias": alias,
                "request_id": response.get("requestId"),
            },
        )

    def validate_config(self) -> list[str]:
        """Validate the configuration."""
        errors = []

        if not self._config.api_key:
            errors.append("api_key is required")

        priority = self._config.priority
        if isinstance(priority, str):
            try:
                AlertPriority(priority.upper())
            except ValueError:
                errors.append(f"Invalid priority: {priority}. Must be P1-P5.")

        region = self._config.region
        if isinstance(region, str):
            try:
                OpsGenieRegion(region.lower())
            except ValueError:
                errors.append(f"Invalid region: {region}. Must be 'us' or 'eu'.")

        # Validate responders
        for i, responder in enumerate(self._config.responders):
            if isinstance(responder, dict):
                if "type" not in responder:
                    errors.append(f"Responder {i}: 'type' is required")
                elif responder["type"] not in ("user", "team", "escalation", "schedule"):
                    errors.append(f"Responder {i}: Invalid type '{responder['type']}'")

        return errors


# =============================================================================
# Convenience Functions
# =============================================================================


def create_opsgenie_action(
    api_key: str,
    *,
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE_OR_ERROR,
    region: OpsGenieRegion | str = OpsGenieRegion.US,
    priority: AlertPriority | str = AlertPriority.P3,
    auto_priority: bool = True,
    template: str = "default",
    **kwargs: Any,
) -> OpsGenieAction:
    """Create an OpsGenie action with common settings.

    Args:
        api_key: OpsGenie API key.
        notify_on: When to send notifications.
        region: API region.
        priority: Default priority.
        auto_priority: Auto-map severity to priority.
        template: Alert template to use.
        **kwargs: Additional configuration options.

    Returns:
        Configured OpsGenieAction instance.
    """
    return OpsGenieAction(
        api_key=api_key,
        notify_on=notify_on,
        region=region,
        priority=priority,
        auto_priority=auto_priority,
        template=template,
        **kwargs,
    )


def create_critical_alert(
    api_key: str,
    *,
    responders: list[Responder | dict[str, str]] | None = None,
    tags: list[str] | None = None,
    region: OpsGenieRegion | str = OpsGenieRegion.US,
    **kwargs: Any,
) -> OpsGenieAction:
    """Create an action for critical alerts only.

    This action only triggers on failures and errors with P1 priority.

    Args:
        api_key: OpsGenie API key.
        responders: List of responders to notify.
        tags: Tags to attach to alerts.
        region: API region.
        **kwargs: Additional configuration options.

    Returns:
        Configured OpsGenieAction for critical alerts.
    """
    default_tags = ["truthound", "critical", "data-quality"]
    return OpsGenieAction(
        api_key=api_key,
        notify_on=NotifyCondition.FAILURE_OR_ERROR,
        region=region,
        priority=AlertPriority.P1,
        auto_priority=False,
        responders=responders or [],
        tags=tags or default_tags,
        template="detailed",
        close_on_success=True,
        **kwargs,
    )


def create_team_alert(
    api_key: str,
    team_name: str,
    *,
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE_OR_ERROR,
    region: OpsGenieRegion | str = OpsGenieRegion.US,
    auto_priority: bool = True,
    **kwargs: Any,
) -> OpsGenieAction:
    """Create an action that alerts a specific team.

    Args:
        api_key: OpsGenie API key.
        team_name: Name of the team to notify.
        notify_on: When to send notifications.
        region: API region.
        auto_priority: Auto-map severity to priority.
        **kwargs: Additional configuration options.

    Returns:
        Configured OpsGenieAction for team alerts.
    """
    return OpsGenieAction(
        api_key=api_key,
        notify_on=notify_on,
        region=region,
        auto_priority=auto_priority,
        responders=[Responder.team(team_name)],
        **kwargs,
    )


def create_escalation_alert(
    api_key: str,
    escalation_name: str,
    *,
    region: OpsGenieRegion | str = OpsGenieRegion.US,
    **kwargs: Any,
) -> OpsGenieAction:
    """Create an action that uses an escalation policy.

    Args:
        api_key: OpsGenie API key.
        escalation_name: Name of the escalation policy.
        region: API region.
        **kwargs: Additional configuration options.

    Returns:
        Configured OpsGenieAction with escalation.
    """
    return OpsGenieAction(
        api_key=api_key,
        notify_on=NotifyCondition.FAILURE_OR_ERROR,
        region=region,
        priority=AlertPriority.P1,
        auto_priority=False,
        responders=[Responder.escalation(escalation_name)],
        template="detailed",
        close_on_success=True,
        **kwargs,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "OpsGenieRegion",
    "AlertPriority",
    "AlertAction",
    "ResponderType",
    # Responder
    "Responder",
    # Builder
    "AlertPayloadBuilder",
    # Templates
    "AlertTemplate",
    "DefaultAlertTemplate",
    "MinimalAlertTemplate",
    "DetailedAlertTemplate",
    "register_template",
    "get_template",
    # HTTP
    "OpsGenieHTTPClient",
    "OpsGenieAPIError",
    # Config & Action
    "OpsGenieConfig",
    "OpsGenieAction",
    # Factory functions
    "create_opsgenie_action",
    "create_critical_alert",
    "create_team_alert",
    "create_escalation_alert",
]
