"""Backpressure monitoring and event handling.

This module provides monitoring capabilities for backpressure,
including event emission, metrics collection, and alerting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from truthound.stores.backpressure.base import (
    BackpressureMetrics,
    BackpressureState,
    BackpressureStrategy,
    PressureLevel,
)


class BackpressureEventType(str, Enum):
    """Types of backpressure events."""

    PRESSURE_INCREASED = "pressure_increased"
    PRESSURE_DECREASED = "pressure_decreased"
    PAUSE_STARTED = "pause_started"
    PAUSE_ENDED = "pause_ended"
    RATE_ADJUSTED = "rate_adjusted"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    RECOVERY_STARTED = "recovery_started"
    CRITICAL_PRESSURE = "critical_pressure"


@dataclass
class BackpressureEvent:
    """Event emitted by backpressure monitor.

    Attributes:
        event_type: Type of the event.
        timestamp: When the event occurred.
        pressure_level: Current pressure level.
        previous_level: Previous pressure level (for changes).
        metrics: Current metrics snapshot.
        details: Additional event details.
    """

    event_type: BackpressureEventType
    timestamp: datetime = field(default_factory=datetime.now)
    pressure_level: PressureLevel = PressureLevel.NONE
    previous_level: PressureLevel | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "pressure_level": self.pressure_level.value,
            "previous_level": self.previous_level.value if self.previous_level else None,
            "metrics": self.metrics,
            "details": self.details,
        }


EventHandler = Callable[[BackpressureEvent], None]
AsyncEventHandler = Callable[[BackpressureEvent], "asyncio.Future[None]"]


@dataclass
class MonitorConfig:
    """Configuration for backpressure monitor.

    Attributes:
        poll_interval_ms: Interval for polling metrics.
        event_buffer_size: Maximum events to buffer.
        enable_async_handlers: Enable async event handlers.
        emit_rate_changes: Emit events for rate adjustments.
        rate_change_threshold: Min rate change % to emit event.
    """

    poll_interval_ms: float = 100.0
    event_buffer_size: int = 1000
    enable_async_handlers: bool = True
    emit_rate_changes: bool = True
    rate_change_threshold: float = 10.0  # 10% change


class BackpressureMonitor:
    """Monitor for tracking and reporting backpressure events.

    Monitors a backpressure strategy and emits events when pressure
    levels change, pauses occur, or rates are adjusted.

    Example:
        >>> bp = AdaptiveBackpressure()
        >>> monitor = BackpressureMonitor(bp)
        >>>
        >>> @monitor.on_event
        >>> def handle_event(event: BackpressureEvent):
        ...     print(f"Event: {event.event_type}")
        >>>
        >>> async with monitor:
        ...     while True:
        ...         await bp.acquire()
        ...         await process_item()
        ...         bp.release()
    """

    def __init__(
        self,
        strategy: BackpressureStrategy,
        config: MonitorConfig | None = None,
    ) -> None:
        """Initialize backpressure monitor.

        Args:
            strategy: Backpressure strategy to monitor.
            config: Monitor configuration.
        """
        self._strategy = strategy
        self._config = config or MonitorConfig()
        self._handlers: list[EventHandler] = []
        self._async_handlers: list[AsyncEventHandler] = []
        self._event_buffer: list[BackpressureEvent] = []
        self._last_pressure_level = PressureLevel.NONE
        self._last_rate = strategy.config.base_rate
        self._is_paused = False
        self._running = False
        self._poll_task: asyncio.Task | None = None

    @property
    def events(self) -> list[BackpressureEvent]:
        """Get buffered events."""
        return self._event_buffer.copy()

    @property
    def strategy(self) -> BackpressureStrategy:
        """Get monitored strategy."""
        return self._strategy

    def on_event(self, handler: EventHandler) -> EventHandler:
        """Decorator to register event handler.

        Args:
            handler: Function to handle events.

        Returns:
            The handler function.
        """
        self._handlers.append(handler)
        return handler

    def on_async_event(self, handler: AsyncEventHandler) -> AsyncEventHandler:
        """Decorator to register async event handler.

        Args:
            handler: Async function to handle events.

        Returns:
            The handler function.
        """
        self._async_handlers.append(handler)
        return handler

    def add_handler(self, handler: EventHandler) -> None:
        """Add event handler."""
        self._handlers.append(handler)

    def add_async_handler(self, handler: AsyncEventHandler) -> None:
        """Add async event handler."""
        self._async_handlers.append(handler)

    def remove_handler(self, handler: EventHandler) -> None:
        """Remove event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def _emit_event(self, event: BackpressureEvent) -> None:
        """Emit event to all handlers."""
        # Buffer event
        self._event_buffer.append(event)
        if len(self._event_buffer) > self._config.event_buffer_size:
            self._event_buffer.pop(0)

        # Call sync handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass  # Don't let handler errors affect monitoring

    async def _emit_event_async(self, event: BackpressureEvent) -> None:
        """Emit event to all handlers including async."""
        self._emit_event(event)

        # Call async handlers
        if self._config.enable_async_handlers:
            for handler in self._async_handlers:
                try:
                    await handler(event)
                except Exception:
                    pass

    def _check_pressure_change(self) -> BackpressureEvent | None:
        """Check for pressure level changes."""
        current_level = self._strategy.state.pressure_level

        if current_level != self._last_pressure_level:
            level_order = [
                PressureLevel.NONE,
                PressureLevel.LOW,
                PressureLevel.MEDIUM,
                PressureLevel.HIGH,
                PressureLevel.CRITICAL,
            ]

            if level_order.index(current_level) > level_order.index(
                self._last_pressure_level
            ):
                event_type = BackpressureEventType.PRESSURE_INCREASED
            else:
                event_type = BackpressureEventType.PRESSURE_DECREASED

            event = BackpressureEvent(
                event_type=event_type,
                pressure_level=current_level,
                previous_level=self._last_pressure_level,
                metrics=self._strategy.metrics.to_dict(),
            )

            self._last_pressure_level = current_level

            # Check for critical pressure
            if current_level == PressureLevel.CRITICAL:
                critical_event = BackpressureEvent(
                    event_type=BackpressureEventType.CRITICAL_PRESSURE,
                    pressure_level=current_level,
                    metrics=self._strategy.metrics.to_dict(),
                    details={"consecutive_high": self._strategy.state.consecutive_high_pressure},
                )
                self._emit_event(critical_event)

            return event

        return None

    def _check_pause_change(self) -> BackpressureEvent | None:
        """Check for pause state changes."""
        current_paused = self._strategy.state.is_paused

        if current_paused != self._is_paused:
            if current_paused:
                event_type = BackpressureEventType.PAUSE_STARTED
            else:
                event_type = BackpressureEventType.PAUSE_ENDED

            event = BackpressureEvent(
                event_type=event_type,
                pressure_level=self._strategy.state.pressure_level,
                metrics=self._strategy.metrics.to_dict(),
                details={
                    "pause_count": self._strategy.metrics.pause_count,
                    "total_pause_time_ms": self._strategy.metrics.total_pause_time_ms,
                },
            )

            self._is_paused = current_paused
            return event

        return None

    def _check_rate_change(self) -> BackpressureEvent | None:
        """Check for significant rate changes."""
        if not self._config.emit_rate_changes:
            return None

        current_rate = self._strategy.state.current_rate
        change_percent = abs(current_rate - self._last_rate) / self._last_rate * 100

        if change_percent >= self._config.rate_change_threshold:
            event = BackpressureEvent(
                event_type=BackpressureEventType.RATE_ADJUSTED,
                pressure_level=self._strategy.state.pressure_level,
                metrics=self._strategy.metrics.to_dict(),
                details={
                    "previous_rate": self._last_rate,
                    "current_rate": current_rate,
                    "change_percent": change_percent,
                },
            )

            self._last_rate = current_rate
            return event

        return None

    async def _poll_loop(self) -> None:
        """Main polling loop for monitoring."""
        while self._running:
            try:
                # Check for changes
                events = []

                pressure_event = self._check_pressure_change()
                if pressure_event:
                    events.append(pressure_event)

                pause_event = self._check_pause_change()
                if pause_event:
                    events.append(pause_event)

                rate_event = self._check_rate_change()
                if rate_event:
                    events.append(rate_event)

                # Emit events
                for event in events:
                    await self._emit_event_async(event)

                await asyncio.sleep(self._config.poll_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Continue monitoring despite errors

    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def __aenter__(self) -> "BackpressureMonitor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    def get_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        event_counts = {}
        for event in self._event_buffer:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self._event_buffer),
            "event_counts": event_counts,
            "current_pressure": self._strategy.state.pressure_level.value,
            "current_rate": self._strategy.state.current_rate,
            "is_paused": self._strategy.state.is_paused,
            "metrics": self._strategy.metrics.to_dict(),
        }

    def clear_events(self) -> None:
        """Clear event buffer."""
        self._event_buffer.clear()
