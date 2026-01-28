"""Enhanced breaking change alerting with context.

This module extends the basic alerting system with rich context,
impact analysis, and integration with notification systems.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from truthound.profiler.evolution.changes import (
    ChangeType,
    ChangeSeverity,
    CompatibilityLevel,
    SchemaChange,
    SchemaChangeSummary,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Impact Analysis
# =============================================================================


class ImpactScope(str, Enum):
    """Scope of a breaking change's impact."""

    LOCAL = "local"  # Affects only this dataset
    DOWNSTREAM = "downstream"  # Affects downstream consumers
    PIPELINE = "pipeline"  # Affects entire pipeline
    ORGANIZATION = "organization"  # Affects multiple teams


class ImpactCategory(str, Enum):
    """Category of impact."""

    DATA_LOSS = "data_loss"  # Potential data loss
    TYPE_MISMATCH = "type_mismatch"  # Type incompatibility
    NULL_VIOLATION = "null_violation"  # Null constraint violation
    QUERY_FAILURE = "query_failure"  # Queries will fail
    PERFORMANCE = "performance"  # Performance degradation
    SCHEMA_DRIFT = "schema_drift"  # Schema inconsistency


@dataclass
class ImpactAssessment:
    """Assessment of a breaking change's impact.

    Attributes:
        scope: Scope of the impact.
        category: Category of impact.
        affected_consumers: List of affected downstream consumers.
        affected_queries: Queries that may fail.
        data_risk_level: Risk level for data integrity (1-5).
        estimated_fix_effort: Estimated effort to fix (hours).
        recommendations: List of recommended actions.
    """

    scope: ImpactScope
    category: ImpactCategory
    affected_consumers: list[str] = field(default_factory=list)
    affected_queries: list[str] = field(default_factory=list)
    data_risk_level: int = 1  # 1=low, 5=critical
    estimated_fix_effort: float = 0.0  # hours
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope.value,
            "category": self.category.value,
            "affected_consumers": self.affected_consumers,
            "affected_queries": self.affected_queries,
            "data_risk_level": self.data_risk_level,
            "estimated_fix_effort": self.estimated_fix_effort,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Enhanced Alert
# =============================================================================


@dataclass
class BreakingChangeAlert:
    """Enhanced alert for breaking schema changes.

    Provides rich context including impact analysis, affected systems,
    and actionable recommendations.

    Attributes:
        alert_id: Unique alert identifier.
        title: Alert title.
        description: Detailed description.
        severity: Alert severity.
        changes: Breaking changes in this alert.
        impact: Impact assessment.
        source_info: Information about the data source.
        detected_at: When the change was detected.
        acknowledged_at: When the alert was acknowledged.
        resolved_at: When the issue was resolved.
        metadata: Additional metadata.
    """

    alert_id: str
    title: str
    description: str
    severity: ChangeSeverity
    changes: list[SchemaChange] = field(default_factory=list)
    impact: ImpactAssessment | None = None
    source_info: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "changes": [c.to_dict() for c in self.changes],
            "impact": self.impact.to_dict() if self.impact else None,
            "source_info": self.source_info,
            "detected_at": self.detected_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BreakingChangeAlert":
        """Create from dictionary."""
        impact = None
        if data.get("impact"):
            impact = ImpactAssessment(
                scope=ImpactScope(data["impact"]["scope"]),
                category=ImpactCategory(data["impact"]["category"]),
                affected_consumers=data["impact"].get("affected_consumers", []),
                affected_queries=data["impact"].get("affected_queries", []),
                data_risk_level=data["impact"].get("data_risk_level", 1),
                estimated_fix_effort=data["impact"].get("estimated_fix_effort", 0.0),
                recommendations=data["impact"].get("recommendations", []),
            )

        return cls(
            alert_id=data["alert_id"],
            title=data["title"],
            description=data["description"],
            severity=ChangeSeverity(data["severity"]),
            changes=[SchemaChange.from_dict(c) for c in data.get("changes", [])],
            impact=impact,
            source_info=data.get("source_info", {}),
            detected_at=datetime.fromisoformat(data["detected_at"]),
            acknowledged_at=(
                datetime.fromisoformat(data["acknowledged_at"])
                if data.get("acknowledged_at")
                else None
            ),
            resolved_at=(
                datetime.fromisoformat(data["resolved_at"])
                if data.get("resolved_at")
                else None
            ),
            metadata=data.get("metadata", {}),
        )

    def acknowledge(self) -> None:
        """Mark the alert as acknowledged."""
        self.acknowledged_at = datetime.now()

    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved_at = datetime.now()

    @property
    def is_acknowledged(self) -> bool:
        """Check if alert is acknowledged."""
        return self.acknowledged_at is not None

    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None

    @property
    def status(self) -> str:
        """Get current alert status."""
        if self.resolved_at:
            return "resolved"
        if self.acknowledged_at:
            return "acknowledged"
        return "open"

    def format_slack_message(self) -> dict[str, Any]:
        """Format alert for Slack notification.

        Returns:
            Slack message payload dict.
        """
        color = {
            ChangeSeverity.CRITICAL: "#FF0000",
            ChangeSeverity.WARNING: "#FFA500",
            ChangeSeverity.INFO: "#0000FF",
        }.get(self.severity, "#808080")

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🚨 {self.title}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": self.description},
            },
        ]

        # Add changes
        if self.changes:
            changes_text = "\n".join(
                f"• {'🔴' if c.breaking else '🟡'} {c.description}"
                for c in self.changes[:5]
            )
            if len(self.changes) > 5:
                changes_text += f"\n_...and {len(self.changes) - 5} more_"

            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Changes:*\n{changes_text}"},
            })

        # Add impact
        if self.impact:
            impact_text = (
                f"*Scope:* {self.impact.scope.value}\n"
                f"*Category:* {self.impact.category.value}\n"
                f"*Risk Level:* {'🔴' * self.impact.data_risk_level}"
            )
            if self.impact.recommendations:
                impact_text += "\n*Recommendations:*\n" + "\n".join(
                    f"• {r}" for r in self.impact.recommendations[:3]
                )
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": impact_text},
            })

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }

    def format_email(self) -> tuple[str, str]:
        """Format alert for email notification.

        Returns:
            Tuple of (subject, html_body).
        """
        subject = f"[{self.severity.value.upper()}] {self.title}"

        body = f"""
        <html>
        <body>
        <h2>{self.title}</h2>
        <p><strong>Severity:</strong> {self.severity.value}</p>
        <p><strong>Detected:</strong> {self.detected_at.isoformat()}</p>

        <h3>Description</h3>
        <p>{self.description}</p>

        <h3>Changes</h3>
        <ul>
        """

        for change in self.changes:
            prefix = "🔴 BREAKING: " if change.breaking else "🟡 "
            body += f"<li>{prefix}{change.description}</li>"
            if change.migration_hint:
                body += f"<br><em>Hint: {change.migration_hint}</em>"

        body += "</ul>"

        if self.impact:
            body += f"""
            <h3>Impact Assessment</h3>
            <ul>
            <li><strong>Scope:</strong> {self.impact.scope.value}</li>
            <li><strong>Category:</strong> {self.impact.category.value}</li>
            <li><strong>Risk Level:</strong> {self.impact.data_risk_level}/5</li>
            </ul>
            """

            if self.impact.recommendations:
                body += "<h4>Recommendations</h4><ul>"
                for rec in self.impact.recommendations:
                    body += f"<li>{rec}</li>"
                body += "</ul>"

        body += """
        </body>
        </html>
        """

        return subject, body


# =============================================================================
# Impact Analyzer
# =============================================================================


class ImpactAnalyzer:
    """Analyzes the impact of schema changes.

    Provides automated impact assessment based on change types
    and configured consumer mappings.

    Example:
        analyzer = ImpactAnalyzer()
        analyzer.register_consumer("reporting_dashboard", ["orders", "customers"])

        changes = [ColumnRemovedChange("customer_id", "Int64")]
        impact = analyzer.analyze(changes, source="orders")
    """

    def __init__(self):
        """Initialize the analyzer."""
        # Map of consumer name -> list of source tables it depends on
        self._consumers: dict[str, list[str]] = {}
        # Map of source table -> list of common queries
        self._queries: dict[str, list[str]] = {}

    def register_consumer(
        self,
        consumer: str,
        depends_on: list[str],
    ) -> None:
        """Register a downstream consumer.

        Args:
            consumer: Consumer name (e.g., "reporting_dashboard").
            depends_on: List of source tables the consumer depends on.
        """
        self._consumers[consumer] = depends_on

    def register_query(self, source: str, query: str) -> None:
        """Register a common query for a source.

        Args:
            source: Source table name.
            query: Query string or description.
        """
        if source not in self._queries:
            self._queries[source] = []
        self._queries[source].append(query)

    def analyze(
        self,
        changes: list[SchemaChange],
        source: str = "",
    ) -> ImpactAssessment:
        """Analyze impact of schema changes.

        Args:
            changes: List of schema changes.
            source: Source table/dataset name.

        Returns:
            ImpactAssessment with analysis results.
        """
        # Determine primary category
        category = self._determine_category(changes)

        # Find affected consumers
        affected_consumers = self._find_affected_consumers(source)

        # Find affected queries
        affected_queries = self._find_affected_queries(changes, source)

        # Determine scope
        scope = self._determine_scope(affected_consumers)

        # Calculate risk level
        risk_level = self._calculate_risk_level(changes, affected_consumers)

        # Generate recommendations
        recommendations = self._generate_recommendations(changes, category)

        # Estimate fix effort
        fix_effort = self._estimate_fix_effort(changes, affected_consumers)

        return ImpactAssessment(
            scope=scope,
            category=category,
            affected_consumers=affected_consumers,
            affected_queries=affected_queries,
            data_risk_level=risk_level,
            estimated_fix_effort=fix_effort,
            recommendations=recommendations,
        )

    def _determine_category(self, changes: list[SchemaChange]) -> ImpactCategory:
        """Determine the primary impact category."""
        has_removal = any(c.change_type == ChangeType.COLUMN_REMOVED for c in changes)
        has_type_change = any(c.change_type == ChangeType.TYPE_CHANGED for c in changes)
        has_null_change = any(c.change_type == ChangeType.NULLABLE_CHANGED for c in changes)

        if has_removal:
            return ImpactCategory.DATA_LOSS
        if has_type_change:
            return ImpactCategory.TYPE_MISMATCH
        if has_null_change:
            return ImpactCategory.NULL_VIOLATION

        return ImpactCategory.SCHEMA_DRIFT

    def _find_affected_consumers(self, source: str) -> list[str]:
        """Find consumers affected by changes to a source."""
        if not source:
            return []

        affected = []
        for consumer, depends_on in self._consumers.items():
            if source in depends_on:
                affected.append(consumer)

        return affected

    def _find_affected_queries(
        self,
        changes: list[SchemaChange],
        source: str,
    ) -> list[str]:
        """Find queries that may be affected."""
        affected = []

        # Get column names involved in changes
        changed_columns = {c.column for c in changes if c.column}

        # Check registered queries
        queries = self._queries.get(source, [])
        for query in queries:
            # Simple heuristic: check if any changed column is mentioned
            query_lower = query.lower()
            for col in changed_columns:
                if col.lower() in query_lower:
                    affected.append(query)
                    break

        return affected

    def _determine_scope(self, affected_consumers: list[str]) -> ImpactScope:
        """Determine the scope of impact."""
        if len(affected_consumers) == 0:
            return ImpactScope.LOCAL
        if len(affected_consumers) <= 2:
            return ImpactScope.DOWNSTREAM
        if len(affected_consumers) <= 5:
            return ImpactScope.PIPELINE
        return ImpactScope.ORGANIZATION

    def _calculate_risk_level(
        self,
        changes: list[SchemaChange],
        affected_consumers: list[str],
    ) -> int:
        """Calculate risk level (1-5)."""
        risk = 1

        # Breaking changes increase risk
        breaking_count = sum(1 for c in changes if c.breaking)
        risk += min(2, breaking_count)

        # More affected consumers increase risk
        risk += min(2, len(affected_consumers) // 2)

        # Column removal is high risk
        if any(c.change_type == ChangeType.COLUMN_REMOVED for c in changes):
            risk += 1

        return min(5, risk)

    def _generate_recommendations(
        self,
        changes: list[SchemaChange],
        category: ImpactCategory,
    ) -> list[str]:
        """Generate recommendations based on changes."""
        recommendations = []

        # Category-specific recommendations
        if category == ImpactCategory.DATA_LOSS:
            recommendations.append(
                "Review data retention policies before removing columns"
            )
            recommendations.append(
                "Consider adding column deprecation warning before removal"
            )
        elif category == ImpactCategory.TYPE_MISMATCH:
            recommendations.append(
                "Update downstream consumers to handle new data types"
            )
            recommendations.append("Test data conversion thoroughly before deployment")
        elif category == ImpactCategory.NULL_VIOLATION:
            recommendations.append(
                "Ensure all existing records have values for newly required columns"
            )
            recommendations.append("Add default values during migration if possible")

        # Change-specific recommendations
        for change in changes:
            if change.migration_hint:
                recommendations.append(change.migration_hint)

        return list(dict.fromkeys(recommendations))  # Remove duplicates

    def _estimate_fix_effort(
        self,
        changes: list[SchemaChange],
        affected_consumers: list[str],
    ) -> float:
        """Estimate effort to fix (in hours)."""
        base_effort = len(changes) * 0.5  # 30 min per change
        consumer_effort = len(affected_consumers) * 1.0  # 1 hour per consumer

        # Breaking changes take longer
        breaking_effort = sum(1.0 for c in changes if c.breaking)

        return base_effort + consumer_effort + breaking_effort


# =============================================================================
# Alert Manager
# =============================================================================


@runtime_checkable
class AlertNotifier(Protocol):
    """Protocol for alert notification handlers."""

    @abstractmethod
    def send(self, alert: BreakingChangeAlert) -> bool:
        """Send an alert notification.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        ...


class BreakingChangeAlertManager:
    """Manager for breaking change alerts.

    Coordinates alert creation, notification, and tracking.

    Example:
        manager = BreakingChangeAlertManager(
            impact_analyzer=ImpactAnalyzer(),
            notifiers=[SlackNotifier(), EmailNotifier()],
        )

        # Create and send alert
        alert = manager.create_alert(
            changes=[ColumnRemovedChange("customer_id", "Int64")],
            source="orders",
        )

        # Track alert history
        history = manager.get_alert_history()
    """

    def __init__(
        self,
        impact_analyzer: ImpactAnalyzer | None = None,
        notifiers: list[AlertNotifier] | None = None,
        alert_storage_path: str | Path | None = None,
    ):
        """Initialize the manager.

        Args:
            impact_analyzer: Analyzer for impact assessment.
            notifiers: List of notification handlers.
            alert_storage_path: Path to store alert history (optional).
        """
        self._impact_analyzer = impact_analyzer or ImpactAnalyzer()
        self._notifiers = notifiers or []
        self._storage_path = Path(alert_storage_path) if alert_storage_path else None
        self._alerts: list[BreakingChangeAlert] = []
        self._alert_counter = 0

        if self._storage_path:
            self._load_alerts()

    def _load_alerts(self) -> None:
        """Load alerts from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                self._alerts = [BreakingChangeAlert.from_dict(a) for a in data]
                self._alert_counter = len(self._alerts)
        except Exception as e:
            logger.warning(f"Failed to load alerts: {e}")

    def _save_alerts(self) -> None:
        """Save alerts to storage."""
        if not self._storage_path:
            return

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, "w") as f:
                json.dump([a.to_dict() for a in self._alerts], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def add_notifier(self, notifier: AlertNotifier) -> None:
        """Add a notification handler."""
        self._notifiers.append(notifier)

    def create_alert(
        self,
        changes: list[SchemaChange],
        source: str = "",
        title: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        notify: bool = True,
    ) -> BreakingChangeAlert:
        """Create a breaking change alert.

        Args:
            changes: List of schema changes.
            source: Source dataset/table name.
            title: Alert title (auto-generated if not provided).
            description: Alert description (auto-generated if not provided).
            metadata: Additional metadata.
            notify: Whether to send notifications.

        Returns:
            Created BreakingChangeAlert.
        """
        # Filter to breaking changes only
        breaking_changes = [c for c in changes if c.breaking]
        if not breaking_changes:
            breaking_changes = changes

        # Generate alert ID
        self._alert_counter += 1
        alert_id = f"ALERT-{self._alert_counter:06d}"

        # Determine severity
        severity = max(
            (c.severity for c in breaking_changes),
            key=lambda s: ["info", "warning", "critical"].index(s.value),
            default=ChangeSeverity.WARNING,
        )

        # Generate title and description
        if not title:
            title = self._generate_title(breaking_changes, source)
        if not description:
            description = self._generate_description(breaking_changes)

        # Analyze impact
        impact = self._impact_analyzer.analyze(breaking_changes, source)

        # Create alert
        alert = BreakingChangeAlert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            changes=breaking_changes,
            impact=impact,
            source_info={"source": source} if source else {},
            metadata=metadata or {},
        )

        # Store alert
        self._alerts.append(alert)
        self._save_alerts()

        # Send notifications
        if notify:
            self._send_notifications(alert)

        return alert

    def _generate_title(
        self,
        changes: list[SchemaChange],
        source: str,
    ) -> str:
        """Generate alert title."""
        source_text = f" in {source}" if source else ""
        return f"Breaking Schema Changes Detected{source_text} ({len(changes)} changes)"

    def _generate_description(self, changes: list[SchemaChange]) -> str:
        """Generate alert description."""
        lines = ["The following breaking changes have been detected:"]
        for change in changes[:5]:
            lines.append(f"- {change.description}")
        if len(changes) > 5:
            lines.append(f"- ...and {len(changes) - 5} more")
        return "\n".join(lines)

    def _send_notifications(self, alert: BreakingChangeAlert) -> None:
        """Send notifications for an alert."""
        for notifier in self._notifiers:
            try:
                notifier.send(alert)
            except Exception as e:
                logger.error(f"Failed to send notification via {notifier}: {e}")

    def get_alert(self, alert_id: str) -> BreakingChangeAlert | None:
        """Get an alert by ID."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                return alert
        return None

    def get_alert_history(
        self,
        limit: int | None = None,
        status: str | None = None,
        severity: ChangeSeverity | None = None,
    ) -> list[BreakingChangeAlert]:
        """Get alert history.

        Args:
            limit: Maximum number of alerts.
            status: Filter by status ("open", "acknowledged", "resolved").
            severity: Filter by severity.

        Returns:
            List of alerts, most recent first.
        """
        alerts = list(reversed(self._alerts))

        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if limit:
            alerts = alerts[:limit]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge.

        Returns:
            True if acknowledged.
        """
        alert = self.get_alert(alert_id)
        if alert:
            alert.acknowledge()
            self._save_alerts()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID to resolve.

        Returns:
            True if resolved.
        """
        alert = self.get_alert(alert_id)
        if alert:
            alert.resolve()
            self._save_alerts()
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        total = len(self._alerts)
        open_count = sum(1 for a in self._alerts if a.status == "open")
        acknowledged = sum(1 for a in self._alerts if a.status == "acknowledged")
        resolved = sum(1 for a in self._alerts if a.status == "resolved")

        by_severity = {}
        for severity in ChangeSeverity:
            by_severity[severity.value] = sum(
                1 for a in self._alerts if a.severity == severity
            )

        return {
            "total": total,
            "open": open_count,
            "acknowledged": acknowledged,
            "resolved": resolved,
            "by_severity": by_severity,
        }


# =============================================================================
# Notifier Implementations
# =============================================================================


class CallbackNotifier:
    """Simple callback-based notifier."""

    def __init__(self, callback: Callable[[BreakingChangeAlert], None]):
        """Initialize with callback.

        Args:
            callback: Function to call with alert.
        """
        self._callback = callback

    def send(self, alert: BreakingChangeAlert) -> bool:
        """Send via callback."""
        try:
            self._callback(alert)
            return True
        except Exception as e:
            logger.error(f"Callback notifier failed: {e}")
            return False


class FileNotifier:
    """Writes alerts to a file (useful for testing/logging)."""

    def __init__(self, path: str | Path, format: str = "json"):
        """Initialize with file path.

        Args:
            path: Path to write alerts.
            format: Output format ("json" or "text").
        """
        self._path = Path(path)
        self._format = format

    def send(self, alert: BreakingChangeAlert) -> bool:
        """Write alert to file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._path, "a") as f:
                if self._format == "json":
                    f.write(json.dumps(alert.to_dict()) + "\n")
                else:
                    f.write(f"[{alert.detected_at}] {alert.title}\n")
                    f.write(f"  Severity: {alert.severity.value}\n")
                    for change in alert.changes:
                        f.write(f"  - {change.description}\n")
                    f.write("\n")

            return True
        except Exception as e:
            logger.error(f"File notifier failed: {e}")
            return False
