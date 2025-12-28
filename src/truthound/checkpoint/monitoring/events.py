"""Event system for monitoring.

Provides a lightweight event bus for publishing and subscribing
to monitoring events across the system.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable
from threading import Lock

from truthound.checkpoint.monitoring.protocols import (
    MonitoringEvent,
    MonitoringEventType,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


# Type aliases for callbacks
SyncCallback = Callable[[MonitoringEvent], None]
AsyncCallback = Callable[[MonitoringEvent], Awaitable[None]]
EventCallback = SyncCallback | AsyncCallback


@dataclass
class Subscription:
    """Represents a subscription to events.

    Attributes:
        subscriber_id: Unique subscriber identifier.
        event_types: Event types to subscribe to.
        callback: Callback function.
        is_async: Whether callback is async.
        filter_fn: Optional filter function.
        created_at: When subscription was created.
    """

    subscriber_id: str
    event_types: frozenset[MonitoringEventType]
    callback: EventCallback
    is_async: bool = False
    filter_fn: Callable[[MonitoringEvent], bool] | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def matches(self, event: MonitoringEvent) -> bool:
        """Check if subscription matches event."""
        # Check event type
        if event.event_type not in self.event_types:
            return False

        # Apply filter if present
        if self.filter_fn and not self.filter_fn(event):
            return False

        return True


class EventEmitter:
    """Mixin class for objects that emit events.

    Usage:
        >>> class MyCollector(EventEmitter):
        ...     async def collect(self):
        ...         data = await self._do_collect()
        ...         self.emit(MonitoringEventType.METRICS_COLLECTED, {"data": data})
    """

    def __init__(self) -> None:
        self._event_bus: EventBus | None = None
        self._event_source: str = self.__class__.__name__

    def set_event_bus(self, bus: EventBus) -> None:
        """Set the event bus for publishing events."""
        self._event_bus = bus

    def set_event_source(self, source: str) -> None:
        """Set the source name for events."""
        self._event_source = source

    def emit(
        self,
        event_type: MonitoringEventType,
        data: dict[str, Any] | None = None,
        severity: AlertSeverity = AlertSeverity.INFO,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Emit an event.

        Args:
            event_type: Type of event.
            data: Event data.
            severity: Alert severity.
            labels: Additional labels.
        """
        if self._event_bus is None:
            return

        event = MonitoringEvent(
            event_type=event_type,
            source=self._event_source,
            data=data or {},
            severity=severity,
            labels=labels or {},
        )
        self._event_bus.publish(event)

    async def emit_async(
        self,
        event_type: MonitoringEventType,
        data: dict[str, Any] | None = None,
        severity: AlertSeverity = AlertSeverity.INFO,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Emit an event asynchronously.

        Args:
            event_type: Type of event.
            data: Event data.
            severity: Alert severity.
            labels: Additional labels.
        """
        if self._event_bus is None:
            return

        event = MonitoringEvent(
            event_type=event_type,
            source=self._event_source,
            data=data or {},
            severity=severity,
            labels=labels or {},
        )
        await self._event_bus.publish_async(event)


class EventBus:
    """Central event bus for monitoring events.

    Thread-safe event bus that supports both sync and async callbacks.

    Example:
        >>> bus = EventBus()
        >>>
        >>> def on_task_completed(event):
        ...     print(f"Task completed: {event.data}")
        >>>
        >>> bus.subscribe(
        ...     "my_subscriber",
        ...     [MonitoringEventType.TASK_COMPLETED],
        ...     on_task_completed,
        ... )
        >>>
        >>> bus.publish(MonitoringEvent(
        ...     event_type=MonitoringEventType.TASK_COMPLETED,
        ...     source="collector",
        ...     data={"task_id": "123"},
        ... ))
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, Subscription] = {}
        self._by_event_type: dict[MonitoringEventType, set[str]] = defaultdict(set)
        self._lock = Lock()
        self._event_history: list[MonitoringEvent] = []
        self._max_history_size = 1000
        self._async_queue: asyncio.Queue[MonitoringEvent] | None = None
        self._running = False

    def subscribe(
        self,
        subscriber_id: str,
        event_types: list[MonitoringEventType] | MonitoringEventType,
        callback: EventCallback,
        filter_fn: Callable[[MonitoringEvent], bool] | None = None,
    ) -> str:
        """Subscribe to events.

        Args:
            subscriber_id: Unique subscriber ID.
            event_types: Event types to subscribe to.
            callback: Callback function (sync or async).
            filter_fn: Optional filter function.

        Returns:
            Subscription ID.
        """
        if isinstance(event_types, MonitoringEventType):
            event_types = [event_types]

        is_async = asyncio.iscoroutinefunction(callback)
        subscription = Subscription(
            subscriber_id=subscriber_id,
            event_types=frozenset(event_types),
            callback=callback,
            is_async=is_async,
            filter_fn=filter_fn,
        )

        with self._lock:
            self._subscriptions[subscriber_id] = subscription
            for event_type in event_types:
                self._by_event_type[event_type].add(subscriber_id)

        logger.debug(f"Subscribed {subscriber_id} to {event_types}")
        return subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscriber_id: Subscriber to remove.

        Returns:
            True if unsubscribed, False if not found.
        """
        with self._lock:
            subscription = self._subscriptions.pop(subscriber_id, None)
            if subscription is None:
                return False

            for event_type in subscription.event_types:
                self._by_event_type[event_type].discard(subscriber_id)

        logger.debug(f"Unsubscribed {subscriber_id}")
        return True

    def publish(self, event: MonitoringEvent) -> int:
        """Publish an event synchronously.

        Args:
            event: Event to publish.

        Returns:
            Number of callbacks invoked.
        """
        self._add_to_history(event)

        # Get matching subscriptions
        with self._lock:
            subscriber_ids = self._by_event_type.get(event.event_type, set()).copy()
            subscriptions = [
                self._subscriptions[sid]
                for sid in subscriber_ids
                if sid in self._subscriptions
            ]

        invoked = 0
        for subscription in subscriptions:
            if not subscription.matches(event):
                continue

            try:
                if subscription.is_async:
                    # Schedule async callback
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(subscription.callback(event))  # type: ignore
                    except RuntimeError:
                        # No running loop, skip async callbacks
                        logger.warning(
                            f"Cannot invoke async callback for {subscription.subscriber_id}: "
                            "no running event loop"
                        )
                        continue
                else:
                    subscription.callback(event)
                invoked += 1
            except Exception as e:
                logger.error(
                    f"Error invoking callback for {subscription.subscriber_id}: {e}"
                )

        return invoked

    async def publish_async(self, event: MonitoringEvent) -> int:
        """Publish an event asynchronously.

        Args:
            event: Event to publish.

        Returns:
            Number of callbacks invoked.
        """
        self._add_to_history(event)

        # Get matching subscriptions
        with self._lock:
            subscriber_ids = self._by_event_type.get(event.event_type, set()).copy()
            subscriptions = [
                self._subscriptions[sid]
                for sid in subscriber_ids
                if sid in self._subscriptions
            ]

        tasks = []
        sync_count = 0

        for subscription in subscriptions:
            if not subscription.matches(event):
                continue

            try:
                if subscription.is_async:
                    tasks.append(subscription.callback(event))  # type: ignore
                else:
                    subscription.callback(event)
                    sync_count += 1
            except Exception as e:
                logger.error(
                    f"Error invoking callback for {subscription.subscriber_id}: {e}"
                )

        # Await all async callbacks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Async callback error: {result}")

        return sync_count + len(tasks)

    def _add_to_history(self, event: MonitoringEvent) -> None:
        """Add event to history, pruning if necessary."""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size:]

    def get_history(
        self,
        event_types: list[MonitoringEventType] | None = None,
        limit: int = 100,
    ) -> list[MonitoringEvent]:
        """Get event history.

        Args:
            event_types: Optional filter by event types.
            limit: Maximum events to return.

        Returns:
            List of recent events.
        """
        with self._lock:
            events = self._event_history.copy()

        if event_types:
            event_type_set = set(event_types)
            events = [e for e in events if e.event_type in event_type_set]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()

    @property
    def subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self._subscriptions)

    def get_subscriptions(self) -> list[dict[str, Any]]:
        """Get list of active subscriptions."""
        with self._lock:
            return [
                {
                    "subscriber_id": sub.subscriber_id,
                    "event_types": [et.value for et in sub.event_types],
                    "is_async": sub.is_async,
                    "created_at": sub.created_at.isoformat(),
                }
                for sub in self._subscriptions.values()
            ]


# Global event bus instance
_event_bus: EventBus | None = None
_bus_lock = Lock()


def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Returns:
        Global EventBus instance.
    """
    global _event_bus
    if _event_bus is None:
        with _bus_lock:
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    with _bus_lock:
        _event_bus = None


# Convenience alias
event_bus = get_event_bus
