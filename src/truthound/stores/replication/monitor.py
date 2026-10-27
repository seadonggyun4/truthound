"""Replication monitoring and event handling.

This module provides monitoring capabilities for replication,
including health checks and event emission.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.stores.replication.base import (
    ReplicaHealth,
    ReplicaTarget,
    ReplicationConfig,
)

if TYPE_CHECKING:
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
        config.validate()
        self._config = config
        self._syncer = syncer
        self._handlers: list[EventHandler] = []
        self._event_buffer: list[ReplicationEvent] = []
        self._running = False
        self._health_check_task: asyncio.Task | None = None
        self._previous_health: dict[str, ReplicaHealth] = {}
        self._errors: list[dict[str, Any]] = []
        self._monitor_failures = 0
        self._handler_failures = 0

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

    def _record_error(self, phase: str, replica: str, handler_index: int | None = None) -> None:
        self._errors.append(
            {
                "phase": phase,
                "replica": replica,
                "handler_index": handler_index,
                "timestamp": datetime.now().isoformat(),
            }
        )
        del self._errors[:-100]
        if phase == "handler":
            self._handler_failures += 1
        else:
            self._monitor_failures += 1

    async def _emit_event(self, event: ReplicationEvent) -> None:
        """Emit event to all handlers."""
        # Buffer event
        self._event_buffer.append(event)
        if len(self._event_buffer) > 1000:
            self._event_buffer.pop(0)

        failures = []
        for index, handler in enumerate(tuple(self._handlers)):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                self._record_error("handler", event.replica_name, index)
                failures.append(error)
        if failures:
            raise ExceptionGroup("Replication event handlers failed", failures)

    async def _check_replica_health(
        self,
        target: ReplicaTarget[ConfigT],
    ) -> None:
        """Check health of a single replica."""
        try:
            health = await self._syncer.check_health(target)
        except Exception:
            target.mark_unhealthy()
            raise
        previous = self._previous_health.get(target.name, ReplicaHealth.UNKNOWN)

        if health != previous:
            self._previous_health[target.name] = health
            if health == ReplicaHealth.HEALTHY:
                await self._emit_event(
                    ReplicationEvent(
                        event_type=ReplicationEventType.REPLICA_HEALTHY,
                        replica_name=target.name,
                        details={"previous_health": previous.value},
                    )
                )
            elif health == ReplicaHealth.DEGRADED:
                await self._emit_event(
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
                await self._emit_event(
                    ReplicationEvent(
                        event_type=ReplicationEventType.REPLICA_UNHEALTHY,
                        replica_name=target.name,
                        details={"previous_health": previous.value},
                    )
                )

            self._previous_health[target.name] = health

        # Check replication lag
        if target.replication_lag_ms > self._config.max_replication_lag_ms:
            await self._emit_event(
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
        try:
            while self._running:
                for target in tuple(self._config.targets):
                    if not self._running:
                        break
                    try:
                        await self._check_replica_health(target)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        self._record_error("health_check", target.name)
                # Failed checks still honor the interval and do not starve peers.
                if self._running:
                    await asyncio.sleep(self._config.health_check_interval_seconds)
        finally:
            self._running = False

    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return
        self._config.validate()

        if self._config.enable_health_checks:
            self._running = True
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
            finally:
                self._health_check_task = None

    async def __aenter__(self) -> ReplicationMonitor[ConfigT]:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary of all replicas."""
        return {
            "running": self._running,
            "monitor_failures": self._monitor_failures,
            "handler_failures": self._handler_failures,
            "errors": [dict(error) for error in self._errors],
            "targets": [
                {
                    "name": target.name,
                    "region": target.region,
                    "health": target.health.value,
                    "state": target.state.value,
                    "lag_ms": target.replication_lag_ms,
                    "last_sync": (
                        target.last_sync_time.isoformat() if target.last_sync_time else None
                    ),
                }
                for target in self._config.targets
            ],
            "healthy_count": sum(
                1 for t in self._config.targets if t.health == ReplicaHealth.HEALTHY
            ),
            "total_count": len(self._config.targets),
        }

    def clear_events(self) -> None:
        """Clear event buffer."""
        self._event_buffer.clear()
