"""Replication monitoring and event handling.

This module provides monitoring capabilities for replication,
including health checks and event emission.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from truthound.stores.replication.base import (
    ReplicaHealth,
    ReplicaTarget,
    ReplicationConfig,
    ReplicationMetrics,
)
from truthound.stores.replication.syncer import ReplicationSyncer


ConfigT = TypeVar("ConfigT")


class ReplicationEventType(str, Enum):
    """Types of replication events."""

    REPLICA_HEALTHY = "replica_healthy"
    REPLICA_DEGRADED = "replica_degraded"
    REPLICA_UNHEALTHY = "replica_unhealthy"
    REPLICATION_SUCCESS = "replication_success"
    REPLICATION_FAILURE = "replication_failure"
    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    FAILOVER_TRIGGERED = "failover_triggered"
    LAG_EXCEEDED = "lag_exceeded"
    CONFLICT_DETECTED = "conflict_detected"


@dataclass
class ReplicationEvent:
    """Replication event.

    Attributes:
        event_type: Type of the event.
        replica_name: Name of the affected replica.
        timestamp: When the event occurred.
        details: Additional event details.
    """

    event_type: ReplicationEventType
    replica_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "replica_name": self.replica_name,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


EventHandler = Callable[[ReplicationEvent], None]


class ReplicationMonitor(Generic[ConfigT]):
    """Monitor for replication health and events.

    Monitors replica health and emits events for status changes,
    enabling alerting and automatic failover.

    Example:
        >>> monitor = ReplicationMonitor(config, syncer)
        >>>
        >>> @monitor.on_event
        >>> def handle_event(event: ReplicationEvent):
        ...     if event.event_type == ReplicationEventType.REPLICA_UNHEALTHY:
        ...         alert(f"Replica {event.replica_name} is unhealthy!")
        >>>
        >>> async with monitor:
        ...     # Monitor runs health checks in background
        ...     await asyncio.sleep(300)
    """

    def __init__(
        self,
        config: ReplicationConfig,
        syncer: ReplicationSyncer,
    ) -> None:
        """Initialize replication monitor.

        Args:
            config: Replication configuration.
            syncer: Replication syncer for health checks.
        """
        self._config = config
        self._syncer = syncer
        self._handlers: list[EventHandler] = []
        self._event_buffer: list[ReplicationEvent] = []
        self._running = False
        self._health_check_task: asyncio.Task | None = None
        self._previous_health: dict[str, ReplicaHealth] = {}

    @property
    def events(self) -> list[ReplicationEvent]:
        """Get buffered events."""
        return self._event_buffer.copy()

    def on_event(self, handler: EventHandler) -> EventHandler:
        """Decorator to register event handler.

        Args:
            handler: Function to handle events.

        Returns:
            The handler function.
        """
        self._handlers.append(handler)
        return handler

    def add_handler(self, handler: EventHandler) -> None:
        """Add event handler."""
        self._handlers.append(handler)

    def _emit_event(self, event: ReplicationEvent) -> None:
        """Emit event to all handlers."""
        # Buffer event
        self._event_buffer.append(event)
        if len(self._event_buffer) > 1000:
            self._event_buffer.pop(0)

        # Call handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass

    async def _check_replica_health(
        self,
        target: ReplicaTarget[ConfigT],
    ) -> None:
        """Check health of a single replica."""
        health = await self._syncer.check_health(target)
        previous = self._previous_health.get(target.name, ReplicaHealth.UNKNOWN)

        if health != previous:
            if health == ReplicaHealth.HEALTHY:
                self._emit_event(
                    ReplicationEvent(
                        event_type=ReplicationEventType.REPLICA_HEALTHY,
                        replica_name=target.name,
                        details={"previous_health": previous.value},
                    )
                )
            elif health == ReplicaHealth.DEGRADED:
                self._emit_event(
                    ReplicationEvent(
                        event_type=ReplicationEventType.REPLICA_DEGRADED,
                        replica_name=target.name,
                        details={
                            "lag_ms": target.replication_lag_ms,
                            "previous_health": previous.value,
                        },
                    )
                )
            elif health == ReplicaHealth.UNHEALTHY:
                self._emit_event(
                    ReplicationEvent(
                        event_type=ReplicationEventType.REPLICA_UNHEALTHY,
                        replica_name=target.name,
                        details={"previous_health": previous.value},
                    )
                )

            self._previous_health[target.name] = health

        # Check replication lag
        if target.replication_lag_ms > self._config.max_replication_lag_ms:
            self._emit_event(
                ReplicationEvent(
                    event_type=ReplicationEventType.LAG_EXCEEDED,
                    replica_name=target.name,
                    details={
                        "lag_ms": target.replication_lag_ms,
                        "max_lag_ms": self._config.max_replication_lag_ms,
                    },
                )
            )

    async def _health_check_loop(self) -> None:
        """Background task for health checks."""
        while self._running:
            try:
                # Check all replicas
                for target in self._config.targets:
                    await self._check_replica_health(target)

                await asyncio.sleep(self._config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return

        self._running = True

        if self._config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

    async def __aenter__(self) -> "ReplicationMonitor[ConfigT]":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary of all replicas."""
        return {
            "targets": [
                {
                    "name": target.name,
                    "region": target.region,
                    "health": target.health.value,
                    "state": target.state.value,
                    "lag_ms": target.replication_lag_ms,
                    "last_sync": (
                        target.last_sync_time.isoformat()
                        if target.last_sync_time
                        else None
                    ),
                }
                for target in self._config.targets
            ],
            "healthy_count": sum(
                1 for t in self._config.targets
                if t.health == ReplicaHealth.HEALTHY
            ),
            "total_count": len(self._config.targets),
        }

    def clear_events(self) -> None:
        """Clear event buffer."""
        self._event_buffer.clear()
