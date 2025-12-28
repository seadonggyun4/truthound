"""Alerting system for model monitoring.

Provides rule-based alerting with multiple handlers:
- Slack: Send alerts to Slack channels
- PagerDuty: Trigger incidents
- Webhook: Send to custom endpoints
"""

from truthound.ml.monitoring.alerting.rules import (
    AlertRule,
    ThresholdRule,
    AnomalyRule,
    TrendRule,
    RuleEngine,
)
from truthound.ml.monitoring.alerting.handlers import (
    SlackAlertHandler,
    PagerDutyAlertHandler,
    WebhookAlertHandler,
)

__all__ = [
    # Rules
    "AlertRule",
    "ThresholdRule",
    "AnomalyRule",
    "TrendRule",
    "RuleEngine",
    # Handlers
    "SlackAlertHandler",
    "PagerDutyAlertHandler",
    "WebhookAlertHandler",
]
