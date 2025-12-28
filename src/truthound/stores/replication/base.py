"""Base classes and configuration for replication.

This module defines the data structures and configurations used
for cross-region replication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from truthound.stores.base import ValidationStore


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


class ReplicationMode(str, Enum):
    """Replication modes."""

    SYNC = "sync"  # Synchronous - wait for all replicas
    ASYNC = "async"  # Asynchronous - fire and forget
    SEMI_SYNC = "semi_sync"  # Wait for at least one replica


class ReadPreference(str, Enum):
    """Read preference for replicated stores."""

    PRIMARY = "primary"  # Always read from primary
    SECONDARY = "secondary"  # Prefer secondary replicas
    NEAREST = "nearest"  # Read from nearest replica
    ANY = "any"  # Read from any available


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""

    LAST_WRITE_WINS = "last_write_wins"  # Most recent write wins
    FIRST_WRITE_WINS = "first_write_wins"  # First write wins
    PRIMARY_WINS = "primary_wins"  # Primary always wins
    MERGE = "merge"  # Attempt to merge (custom logic)
    MANUAL = "manual"  # Require manual resolution


class ReplicaHealth(str, Enum):
    """Health status of a replica."""

    HEALTHY = "healthy"  # Replica is up and synchronized
    DEGRADED = "degraded"  # Replica is up but lagging
    UNHEALTHY = "unhealthy"  # Replica is down or unreachable
    UNKNOWN = "unknown"  # Health status unknown


class ReplicaState(str, Enum):
    """State of a replica."""

    ACTIVE = "active"  # Actively receiving writes
    PAUSED = "paused"  # Temporarily paused
    SYNCING = "syncing"  # Catching up after failure
    FAILED = "failed"  # Replication failed
    REMOVED = "removed"  # Removed from replication


@dataclass
class ReplicaTarget(Generic[ConfigT]):
    """A replication target.

    Attributes:
        name: Unique name for this replica.
        store: The storage backend for this replica.
        region: Geographic region of the replica.
        priority: Priority for failover (lower is higher priority).
        is_read_replica: Whether this is a read-only replica.
        sync_timeout_seconds: Timeout for sync operations.
        max_retry_attempts: Maximum retry attempts.
        retry_delay_seconds: Base delay between retries.
    """

    name: str
    store: "ValidationStore[ConfigT]"
    region: str
    priority: int = 1
    is_read_replica: bool = False
    sync_timeout_seconds: float = 30.0
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    health: ReplicaHealth = ReplicaHealth.UNKNOWN
    state: ReplicaState = ReplicaState.ACTIVE
    last_sync_time: datetime | None = None
    replication_lag_ms: float = 0.0

    def mark_healthy(self) -> None:
        """Mark replica as healthy."""
        self.health = ReplicaHealth.HEALTHY
        self.state = ReplicaState.ACTIVE
        self.last_sync_time = datetime.now()

    def mark_degraded(self, lag_ms: float) -> None:
        """Mark replica as degraded."""
        self.health = ReplicaHealth.DEGRADED
        self.replication_lag_ms = lag_ms

    def mark_unhealthy(self) -> None:
        """Mark replica as unhealthy."""
        self.health = ReplicaHealth.UNHEALTHY
        self.state = ReplicaState.FAILED

    def mark_syncing(self) -> None:
        """Mark replica as syncing."""
        self.state = ReplicaState.SYNCING

    def pause(self) -> None:
        """Pause replication to this replica."""
        self.state = ReplicaState.PAUSED

    def resume(self) -> None:
        """Resume replication to this replica."""
        if self.state == ReplicaState.PAUSED:
            self.state = ReplicaState.ACTIVE


@dataclass
class ReplicationConfig:
    """Configuration for replication.

    Attributes:
        mode: Replication mode (sync, async, semi_sync).
        targets: List of replication targets.
        read_preference: Where to read from.
        conflict_resolution: How to resolve conflicts.
        min_sync_replicas: Minimum replicas for semi-sync.
        enable_health_checks: Enable replica health monitoring.
        health_check_interval_seconds: Health check interval.
        max_replication_lag_ms: Maximum acceptable lag.
        enable_failover: Enable automatic failover.
        enable_metrics: Collect replication metrics.
    """

    mode: ReplicationMode = ReplicationMode.ASYNC
    targets: list[ReplicaTarget] = field(default_factory=list)
    read_preference: ReadPreference = ReadPreference.PRIMARY
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
    min_sync_replicas: int = 1
    enable_health_checks: bool = True
    health_check_interval_seconds: float = 30.0
    max_replication_lag_ms: float = 5000.0
    enable_failover: bool = True
    enable_metrics: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if self.mode == ReplicationMode.SEMI_SYNC:
            if self.min_sync_replicas < 1:
                raise ValueError("min_sync_replicas must be >= 1 for semi_sync mode")
            if self.min_sync_replicas > len(self.targets):
                raise ValueError(
                    "min_sync_replicas cannot exceed number of targets"
                )
        if self.health_check_interval_seconds <= 0:
            raise ValueError("health_check_interval_seconds must be positive")
        if self.max_replication_lag_ms < 0:
            raise ValueError("max_replication_lag_ms must be non-negative")


@dataclass
class ReplicationMetrics:
    """Metrics for replication operations.

    Attributes:
        writes_to_primary: Total writes to primary.
        writes_replicated: Total writes successfully replicated.
        writes_failed: Total replication failures.
        conflicts_detected: Number of conflicts detected.
        conflicts_resolved: Number of conflicts resolved.
        failovers: Number of failover events.
        current_lag_ms: Current replication lag per target.
        total_replication_time_ms: Total time spent replicating.
    """

    writes_to_primary: int = 0
    writes_replicated: int = 0
    writes_failed: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    failovers: int = 0
    current_lag_ms: dict[str, float] = field(default_factory=dict)
    total_replication_time_ms: float = 0.0
    _replica_success_count: dict[str, int] = field(default_factory=dict)
    _replica_failure_count: dict[str, int] = field(default_factory=dict)

    def record_primary_write(self) -> None:
        """Record a write to primary."""
        self.writes_to_primary += 1

    def record_replication_success(
        self, target_name: str, time_ms: float
    ) -> None:
        """Record successful replication."""
        self.writes_replicated += 1
        self.total_replication_time_ms += time_ms
        self._replica_success_count[target_name] = (
            self._replica_success_count.get(target_name, 0) + 1
        )

    def record_replication_failure(self, target_name: str) -> None:
        """Record replication failure."""
        self.writes_failed += 1
        self._replica_failure_count[target_name] = (
            self._replica_failure_count.get(target_name, 0) + 1
        )

    def record_conflict(self, resolved: bool = True) -> None:
        """Record a conflict."""
        self.conflicts_detected += 1
        if resolved:
            self.conflicts_resolved += 1

    def record_failover(self) -> None:
        """Record a failover event."""
        self.failovers += 1

    def update_lag(self, target_name: str, lag_ms: float) -> None:
        """Update replication lag for a target."""
        self.current_lag_ms[target_name] = lag_ms

    def get_replica_success_rate(self, target_name: str) -> float:
        """Get success rate for a replica."""
        success = self._replica_success_count.get(target_name, 0)
        failure = self._replica_failure_count.get(target_name, 0)
        total = success + failure
        if total == 0:
            return 100.0
        return (success / total) * 100

    def get_average_replication_time(self) -> float:
        """Get average replication time in ms."""
        if self.writes_replicated == 0:
            return 0.0
        return self.total_replication_time_ms / self.writes_replicated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "writes_to_primary": self.writes_to_primary,
            "writes_replicated": self.writes_replicated,
            "writes_failed": self.writes_failed,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "failovers": self.failovers,
            "current_lag_ms": self.current_lag_ms,
            "average_replication_time_ms": self.get_average_replication_time(),
            "replica_success_rates": {
                name: self.get_replica_success_rate(name)
                for name in self._replica_success_count
            },
        }
