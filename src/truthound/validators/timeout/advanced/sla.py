"""SLA monitoring and alerting for timeout operations.

This module provides comprehensive SLA monitoring:
- Define SLA targets for operations
- Track violation rates
- Send alerts when SLAs are breached
- Generate compliance reports

Example:
    from truthound.validators.timeout.advanced.sla import (
        SLAMonitor,
        SLADefinition,
        check_sla_compliance,
    )

    # Define SLA
    sla = SLADefinition(
        name="validation_latency",
        target_ms=1000,
        target_percentile=0.99,
        max_violation_rate=0.01,
    )

    # Create monitor
    monitor = SLAMonitor()
    monitor.register_sla(sla)

    # Record operations
    monitor.record("validation_latency", 500)

    # Check compliance
    metrics = monitor.get_metrics("validation_latency")
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Deque

logger = logging.getLogger(__name__)


class SLASeverity(str, Enum):
    """Severity levels for SLA violations."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SLAStatus(str, Enum):
    """Status of SLA compliance."""

    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class SLADefinition:
    """Definition of an SLA.

    Attributes:
        name: SLA name/identifier
        target_ms: Target latency in milliseconds
        target_percentile: Percentile to measure (e.g., 0.99 for P99)
        max_violation_rate: Maximum acceptable violation rate (0.0-1.0)
        window_seconds: Time window for measurement
        severity: Severity of violations
        description: Human-readable description
    """

    name: str
    target_ms: float
    target_percentile: float = 0.99
    max_violation_rate: float = 0.01
    window_seconds: float = 3600.0  # 1 hour
    severity: SLASeverity = SLASeverity.WARNING
    description: str = ""

    def __post_init__(self) -> None:
        """Validate SLA definition."""
        if not 0 < self.target_percentile <= 1:
            raise ValueError(f"target_percentile must be in (0, 1], got {self.target_percentile}")
        if not 0 <= self.max_violation_rate <= 1:
            raise ValueError(f"max_violation_rate must be in [0, 1], got {self.max_violation_rate}")


@dataclass
class SLAViolation:
    """Record of an SLA violation.

    Attributes:
        sla_name: Name of violated SLA
        actual_ms: Actual latency
        target_ms: Target latency
        violation_ratio: How much over target (actual/target)
        timestamp: When violation occurred
        metadata: Additional context
    """

    sla_name: str
    actual_ms: float
    target_ms: float
    violation_ratio: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate violation ratio."""
        if self.target_ms > 0:
            self.violation_ratio = self.actual_ms / self.target_ms


@dataclass
class SLAMetrics:
    """Metrics for an SLA.

    Attributes:
        name: SLA name
        status: Current compliance status
        total_requests: Total requests in window
        violations: Number of violations
        violation_rate: Violation rate (0.0-1.0)
        p50_ms: 50th percentile latency
        p90_ms: 90th percentile latency
        p99_ms: 99th percentile latency
        avg_ms: Average latency
        min_ms: Minimum latency
        max_ms: Maximum latency
        window_start: Start of measurement window
        window_end: End of measurement window
    """

    name: str
    status: SLAStatus = SLAStatus.UNKNOWN
    total_requests: int = 0
    violations: int = 0
    violation_rate: float = 0.0
    p50_ms: float | None = None
    p90_ms: float | None = None
    p99_ms: float | None = None
    avg_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "total_requests": self.total_requests,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
        }


@dataclass
class SLAAlert:
    """Alert for SLA violation.

    Attributes:
        sla_name: Name of SLA
        severity: Alert severity
        message: Alert message
        metrics: Current metrics
        violations: Recent violations
        timestamp: When alert was generated
    """

    sla_name: str
    severity: SLASeverity
    message: str
    metrics: SLAMetrics | None = None
    violations: list[SLAViolation] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sla_name": self.sla_name,
            "severity": self.severity.value,
            "message": self.message,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "violation_count": len(self.violations),
            "timestamp": self.timestamp.isoformat(),
        }


class AlertChannel(ABC):
    """Base class for alert channels."""

    @abstractmethod
    def send_alert(self, alert: SLAAlert) -> bool:
        """Send an alert.

        Args:
            alert: Alert to send

        Returns:
            True if alert was sent successfully
        """
        pass


class LoggingAlertChannel(AlertChannel):
    """Alert channel that logs alerts."""

    def __init__(self, logger_name: str = "sla_alerts"):
        """Initialize logging channel.

        Args:
            logger_name: Logger name to use
        """
        self._logger = logging.getLogger(logger_name)

    def send_alert(self, alert: SLAAlert) -> bool:
        """Log the alert."""
        level = {
            SLASeverity.INFO: logging.INFO,
            SLASeverity.WARNING: logging.WARNING,
            SLASeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        self._logger.log(level, f"SLA Alert: {alert.message}")
        return True


class CallbackAlertChannel(AlertChannel):
    """Alert channel that calls a callback function."""

    def __init__(self, callback: Callable[[SLAAlert], None]):
        """Initialize callback channel.

        Args:
            callback: Function to call with alert
        """
        self._callback = callback

    def send_alert(self, alert: SLAAlert) -> bool:
        """Call the callback."""
        try:
            self._callback(alert)
            return True
        except Exception as e:
            logger.warning(f"Alert callback failed: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Alert channel that posts to a webhook URL."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        """Initialize webhook channel.

        Args:
            url: Webhook URL
            headers: HTTP headers
            timeout: Request timeout
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def send_alert(self, alert: SLAAlert) -> bool:
        """Post alert to webhook."""
        import json
        import urllib.request
        import urllib.error

        try:
            data = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers=self.headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout):
                return True

        except urllib.error.URLError as e:
            logger.warning(f"Webhook alert failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Webhook alert failed: {e}")
            return False


@dataclass
class LatencyRecord:
    """Record of a latency measurement."""

    latency_ms: float
    timestamp: datetime
    exceeded_target: bool = False


class SLAMonitor:
    """Monitor for SLA compliance.

    Tracks latency metrics and detects SLA violations.

    Example:
        monitor = SLAMonitor()

        # Register SLA
        monitor.register_sla(SLADefinition(
            name="api_latency",
            target_ms=100,
            target_percentile=0.99,
        ))

        # Record operations
        monitor.record("api_latency", 50)

        # Get metrics
        metrics = monitor.get_metrics("api_latency")
    """

    def __init__(
        self,
        alert_channels: list[AlertChannel] | None = None,
        check_interval_seconds: float = 60.0,
    ):
        """Initialize SLA monitor.

        Args:
            alert_channels: Channels to send alerts to
            check_interval_seconds: How often to check for violations
        """
        self._slas: dict[str, SLADefinition] = {}
        self._records: dict[str, Deque[LatencyRecord]] = {}
        self._violations: dict[str, list[SLAViolation]] = {}
        self._alert_channels = alert_channels or [LoggingAlertChannel()]
        self._check_interval = check_interval_seconds
        self._last_alert: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._alert_cooldown_seconds = 300.0  # 5 minutes

    def register_sla(self, sla: SLADefinition) -> None:
        """Register an SLA.

        Args:
            sla: SLA definition
        """
        with self._lock:
            self._slas[sla.name] = sla
            max_records = int(sla.window_seconds * 10)  # Assume ~10 req/sec
            self._records[sla.name] = deque(maxlen=max(max_records, 1000))
            self._violations[sla.name] = []

    def unregister_sla(self, name: str) -> None:
        """Unregister an SLA.

        Args:
            name: SLA name
        """
        with self._lock:
            self._slas.pop(name, None)
            self._records.pop(name, None)
            self._violations.pop(name, None)

    def record(
        self,
        sla_name: str,
        latency_ms: float,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SLAViolation | None:
        """Record a latency measurement.

        Args:
            sla_name: SLA name
            latency_ms: Latency in milliseconds
            timestamp: Measurement timestamp
            metadata: Additional metadata

        Returns:
            SLAViolation if target was exceeded, None otherwise
        """
        with self._lock:
            sla = self._slas.get(sla_name)
            if not sla:
                return None

            ts = timestamp or datetime.now(timezone.utc)
            exceeded = latency_ms > sla.target_ms

            record = LatencyRecord(
                latency_ms=latency_ms,
                timestamp=ts,
                exceeded_target=exceeded,
            )
            self._records[sla_name].append(record)

            if exceeded:
                violation = SLAViolation(
                    sla_name=sla_name,
                    actual_ms=latency_ms,
                    target_ms=sla.target_ms,
                    timestamp=ts,
                    metadata=metadata or {},
                )
                self._violations[sla_name].append(violation)

                # Trim violations list
                if len(self._violations[sla_name]) > 1000:
                    self._violations[sla_name] = self._violations[sla_name][-1000:]

                # Check if we should alert
                self._check_and_alert(sla_name)

                return violation

            return None

    def _check_and_alert(self, sla_name: str) -> None:
        """Check if alert should be sent.

        Args:
            sla_name: SLA name to check
        """
        sla = self._slas.get(sla_name)
        if not sla:
            return

        # Check cooldown
        last_alert = self._last_alert.get(sla_name)
        if last_alert:
            elapsed = (datetime.now(timezone.utc) - last_alert).total_seconds()
            if elapsed < self._alert_cooldown_seconds:
                return

        metrics = self._calculate_metrics(sla_name)
        if metrics.status == SLAStatus.VIOLATED:
            alert = SLAAlert(
                sla_name=sla_name,
                severity=sla.severity,
                message=f"SLA '{sla_name}' violated: {metrics.violation_rate:.1%} violation rate "
                        f"(threshold: {sla.max_violation_rate:.1%})",
                metrics=metrics,
                violations=self._violations.get(sla_name, [])[-10:],
            )

            self._send_alert(alert)
            self._last_alert[sla_name] = datetime.now(timezone.utc)

    def _send_alert(self, alert: SLAAlert) -> None:
        """Send alert to all channels.

        Args:
            alert: Alert to send
        """
        for channel in self._alert_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logger.warning(f"Failed to send alert via {channel}: {e}")

    def _calculate_metrics(self, sla_name: str) -> SLAMetrics:
        """Calculate metrics for an SLA.

        Args:
            sla_name: SLA name

        Returns:
            SLAMetrics
        """
        sla = self._slas.get(sla_name)
        records = self._records.get(sla_name)

        if not sla or not records:
            return SLAMetrics(name=sla_name, status=SLAStatus.UNKNOWN)

        # Filter to window
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=sla.window_seconds)

        in_window = [r for r in records if r.timestamp >= window_start]

        if not in_window:
            return SLAMetrics(
                name=sla_name,
                status=SLAStatus.UNKNOWN,
                window_start=window_start,
                window_end=now,
            )

        latencies = [r.latency_ms for r in in_window]
        violations = sum(1 for r in in_window if r.exceeded_target)
        violation_rate = violations / len(in_window)

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def get_percentile(p: float) -> float:
            idx = int(n * p)
            idx = max(0, min(idx, n - 1))
            return sorted_latencies[idx]

        # Determine status
        target_percentile_value = get_percentile(sla.target_percentile)
        if violation_rate > sla.max_violation_rate:
            status = SLAStatus.VIOLATED
        elif violation_rate > sla.max_violation_rate * 0.8:
            status = SLAStatus.AT_RISK
        else:
            status = SLAStatus.COMPLIANT

        return SLAMetrics(
            name=sla_name,
            status=status,
            total_requests=len(in_window),
            violations=violations,
            violation_rate=violation_rate,
            p50_ms=get_percentile(0.5),
            p90_ms=get_percentile(0.9),
            p99_ms=get_percentile(0.99),
            avg_ms=statistics.mean(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            window_start=window_start,
            window_end=now,
        )

    def get_metrics(self, sla_name: str) -> SLAMetrics:
        """Get metrics for an SLA.

        Args:
            sla_name: SLA name

        Returns:
            SLAMetrics
        """
        with self._lock:
            return self._calculate_metrics(sla_name)

    def get_all_metrics(self) -> dict[str, SLAMetrics]:
        """Get metrics for all SLAs.

        Returns:
            Dictionary of SLA name to metrics
        """
        with self._lock:
            return {name: self._calculate_metrics(name) for name in self._slas}

    def get_violations(
        self,
        sla_name: str,
        limit: int = 100,
    ) -> list[SLAViolation]:
        """Get recent violations for an SLA.

        Args:
            sla_name: SLA name
            limit: Maximum violations to return

        Returns:
            List of violations
        """
        with self._lock:
            violations = self._violations.get(sla_name, [])
            return violations[-limit:]

    def check_compliance(self, sla_name: str) -> bool:
        """Check if SLA is currently compliant.

        Args:
            sla_name: SLA name

        Returns:
            True if compliant
        """
        metrics = self.get_metrics(sla_name)
        return metrics.status == SLAStatus.COMPLIANT

    def add_alert_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel.

        Args:
            channel: Channel to add
        """
        self._alert_channels.append(channel)

    def clear(self, sla_name: str | None = None) -> None:
        """Clear recorded data.

        Args:
            sla_name: SLA to clear (None = all)
        """
        with self._lock:
            if sla_name:
                if sla_name in self._records:
                    self._records[sla_name].clear()
                if sla_name in self._violations:
                    self._violations[sla_name].clear()
            else:
                for records in self._records.values():
                    records.clear()
                for violations in self._violations.values():
                    violations.clear()


# Module-level monitor
_default_monitor: SLAMonitor | None = None


def check_sla_compliance(sla_name: str) -> bool:
    """Check SLA compliance using default monitor.

    Args:
        sla_name: SLA name

    Returns:
        True if compliant
    """
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = SLAMonitor()
    return _default_monitor.check_compliance(sla_name)


def create_sla_monitor(
    alert_channels: list[AlertChannel] | None = None,
) -> SLAMonitor:
    """Create an SLA monitor.

    Args:
        alert_channels: Alert channels to use

    Returns:
        SLAMonitor
    """
    return SLAMonitor(alert_channels)
