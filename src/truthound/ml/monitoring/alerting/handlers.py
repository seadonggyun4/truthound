"""Alert handlers for sending notifications.

Provides handlers for various notification channels:
- Slack: Send to Slack channels
- PagerDuty: Create incidents
- Webhook: Send to custom endpoints
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any
import json
import logging

from truthound.ml.monitoring.protocols import IAlertHandler, Alert, AlertSeverity


logger = logging.getLogger(__name__)


class AlertHandler(ABC, IAlertHandler):
    """Base class for alert handlers."""

    pass


@dataclass
class SlackConfig:
    """Configuration for Slack handler.

    Attributes:
        webhook_url: Slack webhook URL
        channel: Default channel
        username: Bot username
        icon_emoji: Bot icon emoji
        mention_on_critical: Users to mention on critical alerts
    """

    webhook_url: str
    channel: str | None = None
    username: str = "Truthound Monitor"
    icon_emoji: str = ":robot_face:"
    mention_on_critical: list[str] | None = None


class SlackAlertHandler(AlertHandler):
    """Sends alerts to Slack.

    Example:
        >>> handler = SlackAlertHandler(SlackConfig(
        ...     webhook_url="https://hooks.slack.com/...",
        ...     channel="#alerts",
        ... ))
        >>> await handler.handle(alert)
    """

    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ffcc00",
        AlertSeverity.ERROR: "#ff6600",
        AlertSeverity.CRITICAL: "#ff0000",
    }

    def __init__(self, config: SlackConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "slack"

    async def handle(self, alert: Alert) -> bool:
        """Send alert to Slack.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for Slack handler. "
                "Install with: pip install aiohttp"
            )

        payload = self._build_payload(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    async def resolve(self, alert: Alert) -> bool:
        """Send resolution notification to Slack.

        Args:
            alert: Resolved alert

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp
        except ImportError:
            return False

        payload = self._build_resolve_payload(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Slack resolution: {e}")
            return False

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build Slack message payload."""
        color = self.SEVERITY_COLORS.get(alert.severity, "#808080")

        # Build mentions
        mentions = ""
        if alert.severity == AlertSeverity.CRITICAL and self._config.mention_on_critical:
            mentions = " ".join(f"<@{u}>" for u in self._config.mention_on_critical)
            mentions = f"\n{mentions}"

        attachments = [
            {
                "color": color,
                "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                "text": alert.message + mentions,
                "fields": [
                    {"title": "Model", "value": alert.model_id, "short": True},
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Time", "value": alert.triggered_at.isoformat(), "short": True},
                ],
                "footer": "Truthound ML Monitor",
                "ts": int(alert.triggered_at.timestamp()),
            }
        ]

        payload: dict[str, Any] = {
            "username": self._config.username,
            "icon_emoji": self._config.icon_emoji,
            "attachments": attachments,
        }

        if self._config.channel:
            payload["channel"] = self._config.channel

        return payload

    def _build_resolve_payload(self, alert: Alert) -> dict[str, Any]:
        """Build Slack resolve message payload."""
        return {
            "username": self._config.username,
            "icon_emoji": self._config.icon_emoji,
            "attachments": [
                {
                    "color": "#36a64f",
                    "title": f"[RESOLVED] {alert.rule_name}",
                    "text": f"Alert for {alert.model_id} has been resolved.",
                    "footer": "Truthound ML Monitor",
                }
            ],
        }


@dataclass
class PagerDutyConfig:
    """Configuration for PagerDuty handler.

    Attributes:
        routing_key: PagerDuty routing key (integration key)
        service_name: Service name
        severity_mapping: Map alert severity to PagerDuty severity
    """

    routing_key: str
    service_name: str = "Truthound"
    severity_mapping: dict[AlertSeverity, str] | None = None


class PagerDutyAlertHandler(AlertHandler):
    """Sends alerts to PagerDuty.

    Example:
        >>> handler = PagerDutyAlertHandler(PagerDutyConfig(
        ...     routing_key="your-routing-key",
        ... ))
        >>> await handler.handle(alert)
    """

    DEFAULT_SEVERITY_MAPPING = {
        AlertSeverity.INFO: "info",
        AlertSeverity.WARNING: "warning",
        AlertSeverity.ERROR: "error",
        AlertSeverity.CRITICAL: "critical",
    }

    EVENTS_API = "https://events.pagerduty.com/v2/enqueue"

    def __init__(self, config: PagerDutyConfig):
        self._config = config
        self._severity_mapping = config.severity_mapping or self.DEFAULT_SEVERITY_MAPPING
        self._dedup_keys: dict[str, str] = {}  # alert_id -> dedup_key

    @property
    def name(self) -> str:
        return "pagerduty"

    async def handle(self, alert: Alert) -> bool:
        """Create PagerDuty incident.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for PagerDuty handler. "
                "Install with: pip install aiohttp"
            )

        dedup_key = f"truthound-{alert.model_id}-{alert.rule_name}"
        self._dedup_keys[alert.alert_id] = dedup_key

        payload = {
            "routing_key": self._config.routing_key,
            "event_action": "trigger",
            "dedup_key": dedup_key,
            "payload": {
                "summary": f"[{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}",
                "source": self._config.service_name,
                "severity": self._severity_mapping.get(alert.severity, "warning"),
                "timestamp": alert.triggered_at.isoformat(),
                "custom_details": {
                    "model_id": alert.model_id,
                    "rule_name": alert.rule_name,
                    "metrics": alert.metrics.to_dict(),
                },
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.EVENTS_API,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    return response.status == 202
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    async def resolve(self, alert: Alert) -> bool:
        """Resolve PagerDuty incident.

        Args:
            alert: Resolved alert

        Returns:
            True if resolved successfully
        """
        try:
            import aiohttp
        except ImportError:
            return False

        dedup_key = self._dedup_keys.get(alert.alert_id)
        if not dedup_key:
            return False

        payload = {
            "routing_key": self._config.routing_key,
            "event_action": "resolve",
            "dedup_key": dedup_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.EVENTS_API,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 202:
                        del self._dedup_keys[alert.alert_id]
                        return True
                    return False
        except Exception as e:
            logger.error(f"Failed to resolve PagerDuty incident: {e}")
            return False


@dataclass
class WebhookConfig:
    """Configuration for webhook handler.

    Attributes:
        url: Webhook URL
        method: HTTP method
        headers: Additional headers
        timeout_seconds: Request timeout
    """

    url: str
    method: str = "POST"
    headers: dict[str, str] | None = None
    timeout_seconds: int = 30


class WebhookAlertHandler(AlertHandler):
    """Sends alerts to a webhook endpoint.

    Example:
        >>> handler = WebhookAlertHandler(WebhookConfig(
        ...     url="https://example.com/alerts",
        ... ))
        >>> await handler.handle(alert)
    """

    def __init__(self, config: WebhookConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "webhook"

    async def handle(self, alert: Alert) -> bool:
        """Send alert to webhook.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for webhook handler. "
                "Install with: pip install aiohttp"
            )

        headers = {"Content-Type": "application/json"}
        if self._config.headers:
            headers.update(self._config.headers)

        payload = {
            "event_type": "alert.triggered",
            "alert": alert.to_dict(),
        }

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
                async with session.request(
                    self._config.method,
                    self._config.url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    return 200 <= response.status < 300
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    async def resolve(self, alert: Alert) -> bool:
        """Send resolution to webhook.

        Args:
            alert: Resolved alert

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp
        except ImportError:
            return False

        headers = {"Content-Type": "application/json"}
        if self._config.headers:
            headers.update(self._config.headers)

        payload = {
            "event_type": "alert.resolved",
            "alert": alert.to_dict(),
        }

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
                async with session.request(
                    self._config.method,
                    self._config.url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    return 200 <= response.status < 300
        except Exception as e:
            logger.error(f"Failed to send webhook resolution: {e}")
            return False
