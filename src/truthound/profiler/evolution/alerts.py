"""Schema change alerting system.

This module provides alerting capabilities for schema changes,
integrating with various notification systems.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from truthound.profiler.evolution.protocols import ChangeNotifier
from truthound.profiler.evolution.changes import (
    ChangeSeverity,
    SchemaChange,
    SchemaChangeSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class SchemaChangeAlert:
    """Alert for schema changes.

    Attributes:
        title: Alert title.
        body: Alert body/description.
        severity: Severity level.
        changes: List of changes in the alert.
        timestamp: When the alert was created.
        metadata: Additional metadata.
    """

    title: str
    body: str
    severity: ChangeSeverity
    changes: list[SchemaChange] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "body": self.body,
            "severity": self.severity.value,
            "changes": [c.to_dict() for c in self.changes],
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AlertHandler(ChangeNotifier, ABC):
    """Abstract base class for alert handlers."""

    def __init__(
        self,
        min_severity: ChangeSeverity = ChangeSeverity.WARNING,
    ):
        """Initialize the handler.

        Args:
            min_severity: Minimum severity to handle.
        """
        self._min_severity = min_severity

    @property
    def min_severity(self) -> ChangeSeverity:
        """Get minimum severity."""
        return self._min_severity

    def supports_severity(self, severity: str) -> bool:
        """Check if this handler supports the severity."""
        severity_order = {
            ChangeSeverity.INFO.value: 0,
            ChangeSeverity.WARNING.value: 1,
            ChangeSeverity.CRITICAL.value: 2,
        }
        min_level = severity_order.get(self._min_severity.value, 0)
        given_level = severity_order.get(severity, 0)
        return given_level >= min_level


class ConsoleAlertHandler(AlertHandler):
    """Alert handler that prints to console.

    Useful for development and testing.
    """

    def __init__(
        self,
        min_severity: ChangeSeverity = ChangeSeverity.INFO,
        use_colors: bool = True,
    ):
        """Initialize handler.

        Args:
            min_severity: Minimum severity to display.
            use_colors: Whether to use ANSI colors.
        """
        super().__init__(min_severity)
        self._use_colors = use_colors

    def notify(
        self,
        changes: list[SchemaChange],
        severity: str,
    ) -> None:
        """Print changes to console."""
        if not self.supports_severity(severity):
            return

        # Format output
        color_codes = {
            "info": "\033[94m",  # Blue
            "warning": "\033[93m",  # Yellow
            "critical": "\033[91m",  # Red
            "reset": "\033[0m",
        }

        color = color_codes.get(severity, "")
        reset = color_codes["reset"] if self._use_colors else ""

        if not self._use_colors:
            color = ""

        print(f"\n{color}{'='*60}")
        print(f"SCHEMA CHANGES DETECTED - Severity: {severity.upper()}")
        print(f"{'='*60}{reset}\n")

        for change in changes:
            prefix = "[BREAKING] " if change.breaking else ""
            print(f"{color}{prefix}{change.description}{reset}")
            if change.migration_hint:
                print(f"  Hint: {change.migration_hint}")
            print()


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs changes.

    Integrates with Python logging.
    """

    def __init__(
        self,
        logger_name: str = "truthound.schema_evolution",
        min_severity: ChangeSeverity = ChangeSeverity.WARNING,
    ):
        """Initialize handler.

        Args:
            logger_name: Name of the logger to use.
            min_severity: Minimum severity to log.
        """
        super().__init__(min_severity)
        self._logger = logging.getLogger(logger_name)

    def notify(
        self,
        changes: list[SchemaChange],
        severity: str,
    ) -> None:
        """Log changes."""
        if not self.supports_severity(severity):
            return

        log_levels = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "critical": logging.CRITICAL,
        }
        level = log_levels.get(severity, logging.WARNING)

        summary = SchemaChangeSummary.from_changes(changes)

        self._logger.log(
            level,
            f"Schema changes detected: {summary.total_changes} total, "
            f"{summary.breaking_changes} breaking"
        )

        for change in changes:
            prefix = "[BREAKING] " if change.breaking else ""
            self._logger.log(level, f"  {prefix}{change.description}")


class CallbackAlertHandler(AlertHandler):
    """Alert handler that calls a custom callback.

    Enables integration with any notification system.
    """

    def __init__(
        self,
        callback: Callable[[list[SchemaChange], str], None],
        min_severity: ChangeSeverity = ChangeSeverity.WARNING,
    ):
        """Initialize handler.

        Args:
            callback: Function to call with changes and severity.
            min_severity: Minimum severity to handle.
        """
        super().__init__(min_severity)
        self._callback = callback

    def notify(
        self,
        changes: list[SchemaChange],
        severity: str,
    ) -> None:
        """Call the callback."""
        if not self.supports_severity(severity):
            return

        try:
            self._callback(changes, severity)
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")


class SchemaChangeAlertManager:
    """Manager for schema change alerts.

    Coordinates multiple alert handlers and provides a unified
    interface for sending alerts.

    Example:
        manager = SchemaChangeAlertManager(handlers=[
            ConsoleAlertHandler(),
            LoggingAlertHandler(),
        ])

        # Send alert for all changes
        manager.alert(changes)

        # Only alert for breaking changes
        manager.alert_if_breaking(changes)
    """

    def __init__(
        self,
        handlers: list[AlertHandler] | None = None,
        default_metadata: dict[str, Any] | None = None,
    ):
        """Initialize the manager.

        Args:
            handlers: List of alert handlers.
            default_metadata: Default metadata to include in alerts.
        """
        self._handlers = handlers or []
        self._default_metadata = default_metadata or {}
        self._alert_history: list[SchemaChangeAlert] = []

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler.

        Args:
            handler: Handler to add.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: AlertHandler) -> bool:
        """Remove an alert handler.

        Args:
            handler: Handler to remove.

        Returns:
            True if removed, False if not found.
        """
        try:
            self._handlers.remove(handler)
            return True
        except ValueError:
            return False

    def alert(
        self,
        changes: list[SchemaChange],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SchemaChangeAlert | None:
        """Send an alert for changes.

        Args:
            changes: Changes to alert about.
            title: Optional alert title.
            metadata: Additional metadata.

        Returns:
            The alert that was sent, or None if no handlers.
        """
        if not changes:
            return None

        # Determine severity
        severity = self._get_highest_severity(changes)

        # Create alert
        alert = SchemaChangeAlert(
            title=title or f"Schema Changes Detected ({len(changes)} changes)",
            body=self._format_changes(changes),
            severity=severity,
            changes=changes,
            metadata={**self._default_metadata, **(metadata or {})},
        )

        # Send to handlers
        for handler in self._handlers:
            try:
                handler.notify(changes, severity.value)
            except Exception as e:
                logger.error(f"Alert handler {handler.__class__.__name__} failed: {e}")

        self._alert_history.append(alert)
        return alert

    def alert_if_breaking(
        self,
        changes: list[SchemaChange],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SchemaChangeAlert | None:
        """Send an alert only if there are breaking changes.

        Args:
            changes: Changes to check.
            title: Optional alert title.
            metadata: Additional metadata.

        Returns:
            The alert that was sent, or None if no breaking changes.
        """
        breaking = [c for c in changes if c.breaking]
        if not breaking:
            return None

        return self.alert(
            breaking,
            title=title or f"Breaking Schema Changes Detected ({len(breaking)} changes)",
            metadata=metadata,
        )

    def get_alert_history(
        self,
        limit: int | None = None,
    ) -> list[SchemaChangeAlert]:
        """Get alert history.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of past alerts, most recent first.
        """
        history = list(reversed(self._alert_history))
        if limit:
            history = history[:limit]
        return history

    def _get_highest_severity(
        self,
        changes: list[SchemaChange],
    ) -> ChangeSeverity:
        """Get the highest severity among changes."""
        severity_order = [
            ChangeSeverity.INFO,
            ChangeSeverity.WARNING,
            ChangeSeverity.CRITICAL,
        ]

        highest = ChangeSeverity.INFO
        for change in changes:
            if severity_order.index(change.severity) > severity_order.index(highest):
                highest = change.severity

        return highest

    def _format_changes(self, changes: list[SchemaChange]) -> str:
        """Format changes for alert body."""
        lines = []
        for change in changes:
            prefix = "[BREAKING] " if change.breaking else ""
            lines.append(f"{prefix}{change.description}")
            if change.migration_hint:
                lines.append(f"  Hint: {change.migration_hint}")
        return "\n".join(lines)


def create_alert_manager(
    console: bool = True,
    logging_enabled: bool = True,
    min_severity: str = "warning",
    custom_callback: Callable[[list[SchemaChange], str], None] | None = None,
) -> SchemaChangeAlertManager:
    """Factory function for creating alert managers.

    Args:
        console: Enable console output.
        logging_enabled: Enable logging.
        min_severity: Minimum severity level.
        custom_callback: Optional custom callback.

    Returns:
        Configured alert manager.
    """
    severity = ChangeSeverity(min_severity)
    handlers: list[AlertHandler] = []

    if console:
        handlers.append(ConsoleAlertHandler(min_severity=severity))

    if logging_enabled:
        handlers.append(LoggingAlertHandler(min_severity=severity))

    if custom_callback:
        handlers.append(CallbackAlertHandler(custom_callback, min_severity=severity))

    return SchemaChangeAlertManager(handlers=handlers)
