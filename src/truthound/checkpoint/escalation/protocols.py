"""Escalation Policy Protocols and Core Types.

This module defines the core protocols and data types for the
multi-level escalation policy system.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol, Sequence, runtime_checkable


class TargetType(str, Enum):
    """Type of escalation target."""

    USER = "user"
    TEAM = "team"
    CHANNEL = "channel"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EMAIL = "email"
    PHONE = "phone"
    CUSTOM = "custom"


class EscalationTrigger(str, Enum):
    """Trigger conditions for escalation."""

    UNACKNOWLEDGED = "unacknowledged"  # Alert not acknowledged within timeout
    UNRESOLVED = "unresolved"  # Alert not resolved within timeout
    SEVERITY_UPGRADE = "severity_upgrade"  # Severity increased
    REPEATED_FAILURE = "repeated_failure"  # Same issue occurred again
    THRESHOLD_BREACH = "threshold_breach"  # Metric threshold exceeded
    MANUAL = "manual"  # Manual escalation request
    SCHEDULED = "scheduled"  # Scheduled escalation
    CUSTOM = "custom"  # Custom trigger condition


@dataclass(frozen=True)
class EscalationTarget:
    """Target for escalation notification.

    Represents a recipient or destination for escalation alerts.

    Attributes:
        type: Type of target.
        identifier: Unique identifier (user_id, team_id, channel, etc.).
        name: Human-readable name.
        priority: Priority for this target (lower = higher priority).
        metadata: Additional target-specific configuration.

    Example:
        >>> target = EscalationTarget(
        ...     type=TargetType.USER,
        ...     identifier="user-123",
        ...     name="John Doe",
        ...     priority=1,
        ... )
    """

    type: TargetType
    identifier: str
    name: str = ""
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "identifier": self.identifier,
            "name": self.name,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationTarget:
        """Create from dictionary."""
        return cls(
            type=TargetType(data["type"]),
            identifier=data["identifier"],
            name=data.get("name", ""),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def user(cls, identifier: str, name: str = "", **kwargs: Any) -> EscalationTarget:
        """Create a user target."""
        return cls(type=TargetType.USER, identifier=identifier, name=name, **kwargs)

    @classmethod
    def team(cls, identifier: str, name: str = "", **kwargs: Any) -> EscalationTarget:
        """Create a team target."""
        return cls(type=TargetType.TEAM, identifier=identifier, name=name, **kwargs)

    @classmethod
    def channel(cls, identifier: str, name: str = "", **kwargs: Any) -> EscalationTarget:
        """Create a channel target."""
        return cls(type=TargetType.CHANNEL, identifier=identifier, name=name, **kwargs)

    @classmethod
    def schedule(cls, identifier: str, name: str = "", **kwargs: Any) -> EscalationTarget:
        """Create an on-call schedule target."""
        return cls(type=TargetType.SCHEDULE, identifier=identifier, name=name, **kwargs)

    @classmethod
    def webhook(cls, url: str, name: str = "", **kwargs: Any) -> EscalationTarget:
        """Create a webhook target."""
        return cls(type=TargetType.WEBHOOK, identifier=url, name=name, **kwargs)

    @classmethod
    def email(cls, address: str, name: str = "", **kwargs: Any) -> EscalationTarget:
        """Create an email target."""
        return cls(type=TargetType.EMAIL, identifier=address, name=name, **kwargs)


@dataclass
class EscalationLevel:
    """Definition of an escalation level.

    Each level defines who gets notified and when.

    Attributes:
        level: Level number (1 = first level, higher = more escalated).
        delay_minutes: Minutes to wait before escalating to this level.
        targets: List of notification targets for this level.
        repeat_count: Number of times to repeat notification (0 = once).
        repeat_interval_minutes: Minutes between repeat notifications.
        require_ack: Whether acknowledgment is required before next level.
        auto_resolve_minutes: Auto-resolve after this many minutes (0 = never).
        conditions: Additional conditions for this level to activate.

    Example:
        >>> level = EscalationLevel(
        ...     level=1,
        ...     delay_minutes=0,
        ...     targets=[EscalationTarget.user("lead-123", "Team Lead")],
        ...     repeat_count=2,
        ...     repeat_interval_minutes=5,
        ... )
    """

    level: int
    delay_minutes: int = 0
    targets: list[EscalationTarget] = field(default_factory=list)
    repeat_count: int = 0
    repeat_interval_minutes: int = 5
    require_ack: bool = True
    auto_resolve_minutes: int = 0
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delay(self) -> timedelta:
        """Get delay as timedelta."""
        return timedelta(minutes=self.delay_minutes)

    @property
    def repeat_interval(self) -> timedelta:
        """Get repeat interval as timedelta."""
        return timedelta(minutes=self.repeat_interval_minutes)

    @property
    def auto_resolve_timeout(self) -> timedelta | None:
        """Get auto-resolve timeout as timedelta."""
        if self.auto_resolve_minutes <= 0:
            return None
        return timedelta(minutes=self.auto_resolve_minutes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "delay_minutes": self.delay_minutes,
            "targets": [t.to_dict() for t in self.targets],
            "repeat_count": self.repeat_count,
            "repeat_interval_minutes": self.repeat_interval_minutes,
            "require_ack": self.require_ack,
            "auto_resolve_minutes": self.auto_resolve_minutes,
            "conditions": self.conditions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationLevel:
        """Create from dictionary."""
        return cls(
            level=data["level"],
            delay_minutes=data.get("delay_minutes", 0),
            targets=[EscalationTarget.from_dict(t) for t in data.get("targets", [])],
            repeat_count=data.get("repeat_count", 0),
            repeat_interval_minutes=data.get("repeat_interval_minutes", 5),
            require_ack=data.get("require_ack", True),
            auto_resolve_minutes=data.get("auto_resolve_minutes", 0),
            conditions=data.get("conditions", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EscalationPolicy:
    """Escalation policy definition.

    Defines the complete escalation chain and behavior.

    Attributes:
        name: Unique policy name.
        description: Human-readable description.
        levels: Ordered list of escalation levels.
        enabled: Whether the policy is active.
        triggers: Conditions that trigger this policy.
        severity_filter: Minimum severity to trigger (critical > high > medium > low > info).
        max_escalations: Maximum number of escalations per incident.
        cooldown_minutes: Cooldown period between escalations for same incident.
        business_hours_only: Only escalate during business hours.
        timezone: Timezone for business hours calculation.
        metadata: Additional policy metadata.

    Example:
        >>> policy = EscalationPolicy(
        ...     name="production_critical",
        ...     description="Critical production alerts",
        ...     levels=[
        ...         EscalationLevel(level=1, delay_minutes=0, targets=[...]),
        ...         EscalationLevel(level=2, delay_minutes=15, targets=[...]),
        ...     ],
        ...     severity_filter=["critical", "high"],
        ... )
    """

    name: str
    description: str = ""
    levels: list[EscalationLevel] = field(default_factory=list)
    enabled: bool = True
    triggers: list[EscalationTrigger] = field(
        default_factory=lambda: [EscalationTrigger.UNACKNOWLEDGED]
    )
    severity_filter: list[str] = field(
        default_factory=lambda: ["critical", "high", "medium", "low", "info"]
    )
    max_escalations: int = 0  # 0 = unlimited
    cooldown_minutes: int = 0
    business_hours_only: bool = False
    business_hours_start: int = 9  # 9 AM
    business_hours_end: int = 18  # 6 PM
    business_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    timezone: str = "UTC"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cooldown(self) -> timedelta:
        """Get cooldown as timedelta."""
        return timedelta(minutes=self.cooldown_minutes)

    @property
    def max_level(self) -> int:
        """Get the maximum escalation level."""
        if not self.levels:
            return 0
        return max(level.level for level in self.levels)

    def get_level(self, level_num: int) -> EscalationLevel | None:
        """Get escalation level by number."""
        for level in self.levels:
            if level.level == level_num:
                return level
        return None

    def get_next_level(self, current_level: int) -> EscalationLevel | None:
        """Get the next escalation level."""
        sorted_levels = sorted(self.levels, key=lambda x: x.level)
        for level in sorted_levels:
            if level.level > current_level:
                return level
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "levels": [level.to_dict() for level in self.levels],
            "enabled": self.enabled,
            "triggers": [t.value for t in self.triggers],
            "severity_filter": self.severity_filter,
            "max_escalations": self.max_escalations,
            "cooldown_minutes": self.cooldown_minutes,
            "business_hours_only": self.business_hours_only,
            "business_hours_start": self.business_hours_start,
            "business_hours_end": self.business_hours_end,
            "business_days": self.business_days,
            "timezone": self.timezone,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationPolicy:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            levels=[EscalationLevel.from_dict(l) for l in data.get("levels", [])],
            enabled=data.get("enabled", True),
            triggers=[EscalationTrigger(t) for t in data.get("triggers", ["unacknowledged"])],
            severity_filter=data.get(
                "severity_filter", ["critical", "high", "medium", "low", "info"]
            ),
            max_escalations=data.get("max_escalations", 0),
            cooldown_minutes=data.get("cooldown_minutes", 0),
            business_hours_only=data.get("business_hours_only", False),
            business_hours_start=data.get("business_hours_start", 9),
            business_hours_end=data.get("business_hours_end", 18),
            business_days=data.get("business_days", [0, 1, 2, 3, 4]),
            timezone=data.get("timezone", "UTC"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EscalationPolicyConfig:
    """Configuration for the escalation policy system.

    Attributes:
        default_policy: Default policy name when none specified.
        policies: Map of policy names to policies.
        global_enabled: Global enable/disable switch.
        store_type: Storage backend type (memory, redis, sqlite).
        scheduler_type: Scheduler type (apscheduler, custom).
        max_concurrent_escalations: Maximum concurrent active escalations.
        cleanup_interval_minutes: Interval for expired record cleanup.
        metrics_enabled: Enable Prometheus metrics.
    """

    default_policy: str = "default"
    policies: dict[str, EscalationPolicy] = field(default_factory=dict)
    global_enabled: bool = True
    store_type: str = "memory"
    store_config: dict[str, Any] = field(default_factory=dict)
    scheduler_type: str = "apscheduler"
    scheduler_config: dict[str, Any] = field(default_factory=dict)
    max_concurrent_escalations: int = 1000
    cleanup_interval_minutes: int = 60
    metrics_enabled: bool = True

    def get_policy(self, name: str | None = None) -> EscalationPolicy | None:
        """Get a policy by name or the default."""
        policy_name = name or self.default_policy
        return self.policies.get(policy_name)

    def add_policy(self, policy: EscalationPolicy) -> None:
        """Add a policy."""
        self.policies[policy.name] = policy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_policy": self.default_policy,
            "policies": {name: p.to_dict() for name, p in self.policies.items()},
            "global_enabled": self.global_enabled,
            "store_type": self.store_type,
            "store_config": self.store_config,
            "scheduler_type": self.scheduler_type,
            "scheduler_config": self.scheduler_config,
            "max_concurrent_escalations": self.max_concurrent_escalations,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "metrics_enabled": self.metrics_enabled,
        }


@dataclass
class EscalationRecord:
    """Record of an active or completed escalation.

    Tracks the state and history of an escalation process.

    Attributes:
        id: Unique record identifier.
        incident_id: External incident/alert identifier.
        policy_name: Name of the escalation policy.
        current_level: Current escalation level.
        state: Current escalation state.
        created_at: When the escalation started.
        updated_at: Last state update timestamp.
        acknowledged_at: When acknowledged (if applicable).
        acknowledged_by: Who acknowledged (if applicable).
        resolved_at: When resolved (if applicable).
        resolved_by: Who resolved (if applicable).
        next_escalation_at: When next escalation is scheduled.
        escalation_count: Number of escalations performed.
        notification_count: Number of notifications sent.
        history: List of state transitions.
        context: Original trigger context.
        metadata: Additional record metadata.
    """

    id: str
    incident_id: str
    policy_name: str
    current_level: int = 1
    state: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    next_escalation_at: datetime | None = None
    escalation_count: int = 0
    notification_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def generate_id(cls, incident_id: str, policy_name: str) -> str:
        """Generate a unique record ID."""
        components = {
            "incident_id": incident_id,
            "policy_name": policy_name,
            "timestamp": datetime.now().isoformat(),
        }
        canonical = json.dumps(components, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    @classmethod
    def create(
        cls,
        incident_id: str,
        policy_name: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EscalationRecord:
        """Create a new escalation record."""
        record_id = cls.generate_id(incident_id, policy_name)
        return cls(
            id=record_id,
            incident_id=incident_id,
            policy_name=policy_name,
            context=context or {},
            **kwargs,
        )

    @property
    def is_active(self) -> bool:
        """Check if escalation is still active."""
        return self.state in ("pending", "active", "escalating")

    @property
    def is_acknowledged(self) -> bool:
        """Check if escalation has been acknowledged."""
        return self.acknowledged_at is not None

    @property
    def is_resolved(self) -> bool:
        """Check if escalation has been resolved."""
        return self.resolved_at is not None

    @property
    def duration(self) -> timedelta:
        """Get duration from creation to now or resolution."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.created_at

    def add_history_event(
        self,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the history."""
        self.history.append(
            {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            }
        )
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "policy_name": self.policy_name,
            "current_level": self.current_level,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat()
            if self.acknowledged_at
            else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "next_escalation_at": self.next_escalation_at.isoformat()
            if self.next_escalation_at
            else None,
            "escalation_count": self.escalation_count,
            "notification_count": self.notification_count,
            "history": self.history,
            "context": self.context,
            "metadata": self.metadata,
            "is_active": self.is_active,
            "is_acknowledged": self.is_acknowledged,
            "is_resolved": self.is_resolved,
            "duration_seconds": self.duration.total_seconds(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationRecord:
        """Create from dictionary."""
        record = cls(
            id=data["id"],
            incident_id=data["incident_id"],
            policy_name=data["policy_name"],
            current_level=data.get("current_level", 1),
            state=data.get("state", "pending"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            escalation_count=data.get("escalation_count", 0),
            notification_count=data.get("notification_count", 0),
            history=data.get("history", []),
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
        )
        if data.get("acknowledged_at"):
            record.acknowledged_at = datetime.fromisoformat(data["acknowledged_at"])
            record.acknowledged_by = data.get("acknowledged_by")
        if data.get("resolved_at"):
            record.resolved_at = datetime.fromisoformat(data["resolved_at"])
            record.resolved_by = data.get("resolved_by")
        if data.get("next_escalation_at"):
            record.next_escalation_at = datetime.fromisoformat(data["next_escalation_at"])
        return record

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EscalationRecord):
            return self.id == other.id
        return False


@dataclass
class EscalationResult:
    """Result of an escalation operation.

    Attributes:
        success: Whether the operation succeeded.
        record: The escalation record.
        action: The action performed.
        level: The escalation level involved.
        targets_notified: List of targets that were notified.
        message: Human-readable result message.
        error: Error message if failed.
        metadata: Additional result metadata.
    """

    success: bool
    record: EscalationRecord | None = None
    action: str = ""
    level: int = 0
    targets_notified: list[EscalationTarget] = field(default_factory=list)
    message: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        record: EscalationRecord,
        action: str,
        targets: list[EscalationTarget] | None = None,
        message: str = "",
    ) -> EscalationResult:
        """Create a success result."""
        return cls(
            success=True,
            record=record,
            action=action,
            level=record.current_level,
            targets_notified=targets or [],
            message=message or f"Escalation {action} completed successfully",
        )

    @classmethod
    def failure_result(
        cls,
        record: EscalationRecord | None,
        action: str,
        error: str,
    ) -> EscalationResult:
        """Create a failure result."""
        return cls(
            success=False,
            record=record,
            action=action,
            level=record.current_level if record else 0,
            message=f"Escalation {action} failed",
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "record": self.record.to_dict() if self.record else None,
            "action": self.action,
            "level": self.level,
            "targets_notified": [t.to_dict() for t in self.targets_notified],
            "message": self.message,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class EscalationStats:
    """Statistics for the escalation system.

    Attributes:
        total_escalations: Total escalations triggered.
        active_escalations: Currently active escalations.
        acknowledged_count: Number acknowledged.
        resolved_count: Number resolved.
        timed_out_count: Number that timed out.
        cancelled_count: Number cancelled.
        avg_time_to_acknowledge: Average time to acknowledge.
        avg_time_to_resolve: Average time to resolve.
        escalations_by_level: Count by escalation level.
        escalations_by_policy: Count by policy name.
    """

    total_escalations: int = 0
    active_escalations: int = 0
    acknowledged_count: int = 0
    resolved_count: int = 0
    timed_out_count: int = 0
    cancelled_count: int = 0
    avg_time_to_acknowledge_seconds: float = 0.0
    avg_time_to_resolve_seconds: float = 0.0
    escalations_by_level: dict[int, int] = field(default_factory=dict)
    escalations_by_policy: dict[str, int] = field(default_factory=dict)
    notifications_sent: int = 0

    @property
    def acknowledgment_rate(self) -> float:
        """Calculate acknowledgment rate as percentage."""
        if self.total_escalations == 0:
            return 0.0
        return (self.acknowledged_count / self.total_escalations) * 100

    @property
    def resolution_rate(self) -> float:
        """Calculate resolution rate as percentage."""
        if self.total_escalations == 0:
            return 0.0
        return (self.resolved_count / self.total_escalations) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_escalations": self.total_escalations,
            "active_escalations": self.active_escalations,
            "acknowledged_count": self.acknowledged_count,
            "resolved_count": self.resolved_count,
            "timed_out_count": self.timed_out_count,
            "cancelled_count": self.cancelled_count,
            "acknowledgment_rate": round(self.acknowledgment_rate, 2),
            "resolution_rate": round(self.resolution_rate, 2),
            "avg_time_to_acknowledge_seconds": round(
                self.avg_time_to_acknowledge_seconds, 2
            ),
            "avg_time_to_resolve_seconds": round(self.avg_time_to_resolve_seconds, 2),
            "escalations_by_level": self.escalations_by_level,
            "escalations_by_policy": self.escalations_by_policy,
            "notifications_sent": self.notifications_sent,
        }


@runtime_checkable
class EscalationStoreProtocol(Protocol):
    """Protocol for escalation record storage backends.

    Implementations must provide thread-safe storage for
    escalation records with query and lifecycle support.
    """

    def get(self, record_id: str) -> EscalationRecord | None:
        """Get an escalation record by ID.

        Args:
            record_id: The record ID.

        Returns:
            The record if found, None otherwise.
        """
        ...

    def get_by_incident(self, incident_id: str) -> list[EscalationRecord]:
        """Get all records for an incident.

        Args:
            incident_id: The incident ID.

        Returns:
            List of records for the incident.
        """
        ...

    def get_active(self, policy_name: str | None = None) -> list[EscalationRecord]:
        """Get all active escalations.

        Args:
            policy_name: Optional policy filter.

        Returns:
            List of active records.
        """
        ...

    def get_pending_escalations(self, before: datetime) -> list[EscalationRecord]:
        """Get records with pending escalations before a timestamp.

        Args:
            before: Timestamp threshold.

        Returns:
            List of records needing escalation.
        """
        ...

    def save(self, record: EscalationRecord) -> EscalationRecord:
        """Save an escalation record.

        Args:
            record: The record to save.

        Returns:
            The saved record.
        """
        ...

    def delete(self, record_id: str) -> bool:
        """Delete a record.

        Args:
            record_id: The record ID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def cleanup_resolved(self, older_than: timedelta) -> int:
        """Remove resolved records older than threshold.

        Args:
            older_than: Age threshold.

        Returns:
            Number of records removed.
        """
        ...

    def get_stats(self) -> EscalationStats:
        """Get store statistics.

        Returns:
            Current statistics.
        """
        ...

    def clear(self) -> None:
        """Clear all records."""
        ...


class BaseEscalationStore(ABC):
    """Abstract base class for escalation store implementations."""

    def __init__(self, name: str = "base"):
        """Initialize base store.

        Args:
            name: Unique name for this store.
        """
        self._name = name
        self._stats = EscalationStats()

    @property
    def name(self) -> str:
        """Get store name."""
        return self._name

    @abstractmethod
    def get(self, record_id: str) -> EscalationRecord | None:
        """Get an escalation record by ID."""
        pass

    @abstractmethod
    def get_by_incident(self, incident_id: str) -> list[EscalationRecord]:
        """Get all records for an incident."""
        pass

    @abstractmethod
    def get_active(self, policy_name: str | None = None) -> list[EscalationRecord]:
        """Get all active escalations."""
        pass

    @abstractmethod
    def get_pending_escalations(self, before: datetime) -> list[EscalationRecord]:
        """Get records with pending escalations before a timestamp."""
        pass

    @abstractmethod
    def save(self, record: EscalationRecord) -> EscalationRecord:
        """Save an escalation record."""
        pass

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        pass

    @abstractmethod
    def cleanup_resolved(self, older_than: timedelta) -> int:
        """Remove resolved records older than threshold."""
        pass

    def get_stats(self) -> EscalationStats:
        """Get store statistics."""
        return self._stats

    @abstractmethod
    def clear(self) -> None:
        """Clear all records."""
        pass

    def _update_stats_on_save(self, record: EscalationRecord, is_new: bool) -> None:
        """Update statistics when saving a record."""
        if is_new:
            self._stats.total_escalations += 1
            policy_count = self._stats.escalations_by_policy.get(record.policy_name, 0)
            self._stats.escalations_by_policy[record.policy_name] = policy_count + 1

        if record.is_active:
            pass  # Active count updated in get_active
        elif record.state == "acknowledged":
            self._stats.acknowledged_count += 1
        elif record.state == "resolved":
            self._stats.resolved_count += 1
        elif record.state == "timed_out":
            self._stats.timed_out_count += 1
        elif record.state == "cancelled":
            self._stats.cancelled_count += 1
