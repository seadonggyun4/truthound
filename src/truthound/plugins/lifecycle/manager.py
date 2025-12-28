"""Lifecycle manager for plugins.

This module provides lifecycle state management and event handling
for plugin loading, activation, and unloading.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.plugins.base import Plugin

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = auto()   # Found but not loaded
    LOADING = auto()      # Currently loading
    LOADED = auto()       # Loaded but not active
    ACTIVATING = auto()   # Being activated
    ACTIVE = auto()       # Fully active
    DEACTIVATING = auto() # Being deactivated
    INACTIVE = auto()     # Loaded but inactive
    UNLOADING = auto()    # Being unloaded
    UNLOADED = auto()     # Removed from system
    ERROR = auto()        # Error state


class LifecycleEvent(Enum):
    """Events in plugin lifecycle."""

    BEFORE_LOAD = auto()
    AFTER_LOAD = auto()
    BEFORE_ACTIVATE = auto()
    AFTER_ACTIVATE = auto()
    BEFORE_DEACTIVATE = auto()
    AFTER_DEACTIVATE = auto()
    BEFORE_UNLOAD = auto()
    AFTER_UNLOAD = auto()
    ON_ERROR = auto()
    ON_RELOAD = auto()


@dataclass
class LifecycleTransition:
    """Record of a lifecycle state transition."""

    plugin_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[["Plugin", LifecycleEvent, dict[str, Any]], None]
AsyncEventHandler = Callable[["Plugin", LifecycleEvent, dict[str, Any]], "asyncio.coroutines"]


class LifecycleManager:
    """Manages plugin lifecycle states and transitions.

    Provides:
    - State tracking for all plugins
    - Event hooks for lifecycle events
    - Transition validation
    - History tracking

    Example:
        >>> manager = LifecycleManager()
        >>> manager.on(LifecycleEvent.AFTER_LOAD, my_handler)
        >>> await manager.transition(plugin, LifecycleState.LOADING)
    """

    # Valid state transitions
    VALID_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
        LifecycleState.DISCOVERED: {LifecycleState.LOADING, LifecycleState.ERROR},
        LifecycleState.LOADING: {LifecycleState.LOADED, LifecycleState.ERROR},
        LifecycleState.LOADED: {
            LifecycleState.ACTIVATING,
            LifecycleState.UNLOADING,
            LifecycleState.ERROR,
        },
        LifecycleState.ACTIVATING: {LifecycleState.ACTIVE, LifecycleState.ERROR},
        LifecycleState.ACTIVE: {
            LifecycleState.DEACTIVATING,
            LifecycleState.ERROR,
        },
        LifecycleState.DEACTIVATING: {
            LifecycleState.INACTIVE,
            LifecycleState.ERROR,
        },
        LifecycleState.INACTIVE: {
            LifecycleState.ACTIVATING,
            LifecycleState.UNLOADING,
            LifecycleState.ERROR,
        },
        LifecycleState.UNLOADING: {LifecycleState.UNLOADED, LifecycleState.ERROR},
        LifecycleState.UNLOADED: {LifecycleState.LOADING},  # Can reload
        LifecycleState.ERROR: {
            LifecycleState.LOADING,  # Retry
            LifecycleState.UNLOADING,  # Cleanup
        },
    }

    def __init__(
        self,
        max_history: int = 1000,
    ) -> None:
        """Initialize lifecycle manager.

        Args:
            max_history: Maximum transition history to keep
        """
        self._states: dict[str, LifecycleState] = {}
        self._handlers: dict[LifecycleEvent, list[EventHandler | AsyncEventHandler]] = {
            event: [] for event in LifecycleEvent
        }
        self._history: list[LifecycleTransition] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    def get_state(self, plugin_id: str) -> LifecycleState:
        """Get current state of a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Current state (UNLOADED if unknown)
        """
        return self._states.get(plugin_id, LifecycleState.UNLOADED)

    def set_state(
        self,
        plugin_id: str,
        state: LifecycleState,
    ) -> None:
        """Set plugin state directly (bypasses validation).

        Args:
            plugin_id: Plugin identifier
            state: New state
        """
        self._states[plugin_id] = state

    async def transition(
        self,
        plugin: "Plugin",
        to_state: LifecycleState,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Transition a plugin to a new state.

        Args:
            plugin: Plugin instance
            to_state: Target state
            metadata: Optional metadata for handlers

        Returns:
            True if transition succeeded

        Raises:
            ValueError: If transition is invalid
        """
        async with self._lock:
            plugin_id = plugin.name
            from_state = self.get_state(plugin_id)

            # Validate transition
            valid_targets = self.VALID_TRANSITIONS.get(from_state, set())
            if to_state not in valid_targets:
                raise ValueError(
                    f"Invalid transition for '{plugin_id}': "
                    f"{from_state.name} -> {to_state.name}"
                )

            # Fire before event
            before_event = self._get_before_event(to_state)
            if before_event:
                await self._fire_event(plugin, before_event, metadata or {})

            # Update state
            self._states[plugin_id] = to_state

            # Record transition
            transition = LifecycleTransition(
                plugin_id=plugin_id,
                from_state=from_state,
                to_state=to_state,
                metadata=metadata or {},
            )
            self._add_history(transition)

            # Fire after event
            after_event = self._get_after_event(to_state)
            if after_event:
                await self._fire_event(plugin, after_event, metadata or {})

            logger.debug(
                f"Plugin '{plugin_id}' transitioned: {from_state.name} -> {to_state.name}"
            )
            return True

    async def transition_to_error(
        self,
        plugin: "Plugin",
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Transition a plugin to error state.

        Args:
            plugin: Plugin instance
            error: The error that occurred
            metadata: Optional metadata
        """
        plugin_id = plugin.name
        from_state = self.get_state(plugin_id)

        self._states[plugin_id] = LifecycleState.ERROR

        transition = LifecycleTransition(
            plugin_id=plugin_id,
            from_state=from_state,
            to_state=LifecycleState.ERROR,
            error=str(error),
            metadata=metadata or {},
        )
        self._add_history(transition)

        await self._fire_event(
            plugin,
            LifecycleEvent.ON_ERROR,
            {"error": error, **(metadata or {})},
        )

    def on(
        self,
        event: LifecycleEvent,
        handler: EventHandler | AsyncEventHandler,
    ) -> None:
        """Register an event handler.

        Args:
            event: Event to handle
            handler: Handler function
        """
        self._handlers[event].append(handler)

    def off(
        self,
        event: LifecycleEvent,
        handler: EventHandler | AsyncEventHandler,
    ) -> bool:
        """Unregister an event handler.

        Args:
            event: Event type
            handler: Handler to remove

        Returns:
            True if handler was removed
        """
        try:
            self._handlers[event].remove(handler)
            return True
        except ValueError:
            return False

    async def _fire_event(
        self,
        plugin: "Plugin",
        event: LifecycleEvent,
        data: dict[str, Any],
    ) -> None:
        """Fire an event to all registered handlers.

        Args:
            plugin: Plugin instance
            event: Event type
            data: Event data
        """
        for handler in self._handlers[event]:
            try:
                result = handler(plugin, event, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in lifecycle handler for {event.name}: {e}")

    def _get_before_event(self, state: LifecycleState) -> LifecycleEvent | None:
        """Get before event for a state transition."""
        mapping = {
            LifecycleState.LOADING: LifecycleEvent.BEFORE_LOAD,
            LifecycleState.ACTIVATING: LifecycleEvent.BEFORE_ACTIVATE,
            LifecycleState.DEACTIVATING: LifecycleEvent.BEFORE_DEACTIVATE,
            LifecycleState.UNLOADING: LifecycleEvent.BEFORE_UNLOAD,
        }
        return mapping.get(state)

    def _get_after_event(self, state: LifecycleState) -> LifecycleEvent | None:
        """Get after event for a state transition."""
        mapping = {
            LifecycleState.LOADED: LifecycleEvent.AFTER_LOAD,
            LifecycleState.ACTIVE: LifecycleEvent.AFTER_ACTIVATE,
            LifecycleState.INACTIVE: LifecycleEvent.AFTER_DEACTIVATE,
            LifecycleState.UNLOADED: LifecycleEvent.AFTER_UNLOAD,
        }
        return mapping.get(state)

    def _add_history(self, transition: LifecycleTransition) -> None:
        """Add a transition to history."""
        self._history.append(transition)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(
        self,
        plugin_id: str | None = None,
        limit: int = 100,
    ) -> list[LifecycleTransition]:
        """Get transition history.

        Args:
            plugin_id: Filter by plugin (None for all)
            limit: Maximum entries to return

        Returns:
            List of transitions (newest first)
        """
        history = self._history
        if plugin_id:
            history = [t for t in history if t.plugin_id == plugin_id]
        return list(reversed(history[-limit:]))

    def get_plugins_in_state(self, state: LifecycleState) -> list[str]:
        """Get all plugins in a specific state.

        Args:
            state: State to filter by

        Returns:
            List of plugin IDs
        """
        return [
            plugin_id
            for plugin_id, plugin_state in self._states.items()
            if plugin_state == state
        ]

    def clear(self) -> None:
        """Clear all state and history."""
        self._states.clear()
        self._history.clear()

    def remove_plugin(self, plugin_id: str) -> None:
        """Remove a plugin from tracking.

        Args:
            plugin_id: Plugin to remove
        """
        self._states.pop(plugin_id, None)
