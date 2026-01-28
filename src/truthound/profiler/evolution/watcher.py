"""Schema watcher for continuous monitoring.

This module provides real-time schema monitoring with file system
watching, periodic polling, and event-driven change detection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from truthound.profiler.evolution.changes import (
    ChangeSeverity,
    SchemaChange,
    SchemaChangeSummary,
)
from truthound.profiler.evolution.detector import SchemaEvolutionDetector
from truthound.profiler.evolution.history import SchemaHistory, SchemaVersion
from truthound.profiler.evolution.breaking_alerts import (
    BreakingChangeAlert,
    BreakingChangeAlertManager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Types
# =============================================================================


class WatcherState(str, Enum):
    """Watcher lifecycle states."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WatchEvent:
    """Event emitted when schema change is detected.

    Attributes:
        event_id: Unique event identifier.
        event_type: Type of event.
        source: Source of the schema (file path, table name, etc.).
        old_schema: Previous schema (if available).
        new_schema: New schema.
        changes: Detected changes.
        summary: Change summary.
        timestamp: When the event occurred.
        metadata: Additional metadata.
    """

    event_id: str
    event_type: str
    source: str
    old_schema: dict[str, Any] | None
    new_schema: dict[str, Any]
    changes: list[SchemaChange]
    summary: SchemaChangeSummary
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "old_schema": self.old_schema,
            "new_schema": {k: str(v) for k, v in self.new_schema.items()},
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def has_breaking_changes(self) -> bool:
        """Check if event contains breaking changes."""
        return self.summary.is_breaking()


@runtime_checkable
class SchemaSource(Protocol):
    """Protocol for schema sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get source name."""
        ...

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Get current schema.

        Returns:
            Schema as dict mapping column names to types.
        """
        ...

    @abstractmethod
    def get_checksum(self) -> str:
        """Get checksum of current schema.

        Returns:
            String checksum for change detection.
        """
        ...


@runtime_checkable
class WatchEventHandler(Protocol):
    """Protocol for watch event handlers."""

    @abstractmethod
    def handle(self, event: WatchEvent) -> None:
        """Handle a watch event.

        Args:
            event: The watch event to handle.
        """
        ...


# =============================================================================
# Schema Sources
# =============================================================================


class FileSchemaSource:
    """Schema source from JSON file.

    Watches a JSON file containing schema definition.
    """

    def __init__(self, path: str | Path):
        """Initialize with file path.

        Args:
            path: Path to JSON schema file.
        """
        self._path = Path(path)

    @property
    def name(self) -> str:
        return str(self._path)

    def get_schema(self) -> dict[str, Any]:
        """Read schema from file."""
        if not self._path.exists():
            return {}

        try:
            with open(self._path, "r") as f:
                data = json.load(f)

            # Handle different formats
            if isinstance(data, dict):
                # Direct column: type mapping
                if all(isinstance(v, str) for v in data.values()):
                    return data
                # Nested format with "columns" key
                if "columns" in data:
                    return data["columns"]
                # Schema wrapper
                if "schema" in data:
                    return data["schema"]

            return data
        except Exception as e:
            logger.error(f"Failed to read schema from {self._path}: {e}")
            return {}

    def get_checksum(self) -> str:
        """Get file checksum."""
        if not self._path.exists():
            return ""

        try:
            with open(self._path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class PolarsSchemaSource:
    """Schema source from Polars DataFrame/LazyFrame."""

    def __init__(
        self,
        data_factory: Callable[[], Any],
        name: str = "polars_source",
    ):
        """Initialize with data factory.

        Args:
            data_factory: Callable that returns DataFrame/LazyFrame.
            name: Source name.
        """
        self._factory = data_factory
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_schema(self) -> dict[str, Any]:
        """Get schema from data."""
        try:
            import polars as pl

            data = self._factory()

            if isinstance(data, pl.LazyFrame):
                schema = data.collect_schema()
            elif isinstance(data, pl.DataFrame):
                schema = data.schema
            else:
                return {}

            return {name: str(dtype) for name, dtype in schema.items()}
        except Exception as e:
            logger.error(f"Failed to get Polars schema: {e}")
            return {}

    def get_checksum(self) -> str:
        """Get schema checksum."""
        schema = self.get_schema()
        content = json.dumps(schema, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


class DictSchemaSource:
    """Simple dict-based schema source."""

    def __init__(
        self,
        schema: dict[str, Any],
        name: str = "dict_source",
    ):
        """Initialize with schema dict.

        Args:
            schema: Schema dictionary.
            name: Source name.
        """
        self._schema = schema
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_schema(self) -> dict[str, Any]:
        return self._schema.copy()

    def get_checksum(self) -> str:
        content = json.dumps(self._schema, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def update(self, schema: dict[str, Any]) -> None:
        """Update the schema.

        Args:
            schema: New schema dictionary.
        """
        self._schema = schema


# =============================================================================
# Event Handlers
# =============================================================================


class LoggingEventHandler:
    """Handler that logs events."""

    def __init__(
        self,
        logger_name: str = "truthound.schema_watcher",
        min_severity: ChangeSeverity = ChangeSeverity.INFO,
    ):
        """Initialize with logger.

        Args:
            logger_name: Name of logger to use.
            min_severity: Minimum severity to log.
        """
        self._logger = logging.getLogger(logger_name)
        self._min_severity = min_severity

    def handle(self, event: WatchEvent) -> None:
        """Log the event."""
        if event.summary.is_breaking():
            self._logger.warning(
                f"Breaking schema changes detected in {event.source}: "
                f"{event.summary.breaking_changes} breaking changes"
            )
            for change in event.changes:
                if change.breaking:
                    self._logger.warning(f"  - {change.description}")
        else:
            self._logger.info(
                f"Schema changes detected in {event.source}: "
                f"{event.summary.total_changes} changes"
            )


class CallbackEventHandler:
    """Handler that invokes a callback."""

    def __init__(
        self,
        callback: Callable[[WatchEvent], None],
        async_callback: bool = False,
    ):
        """Initialize with callback.

        Args:
            callback: Function to call with event.
            async_callback: Whether callback is async.
        """
        self._callback = callback
        self._async = async_callback

    def handle(self, event: WatchEvent) -> None:
        """Invoke the callback."""
        try:
            if self._async:
                asyncio.create_task(self._callback(event))
            else:
                self._callback(event)
        except Exception as e:
            logger.error(f"Event callback failed: {e}")


class AlertingEventHandler:
    """Handler that creates alerts for breaking changes."""

    def __init__(
        self,
        alert_manager: BreakingChangeAlertManager,
        only_breaking: bool = True,
    ):
        """Initialize with alert manager.

        Args:
            alert_manager: Manager for creating alerts.
            only_breaking: Only alert on breaking changes.
        """
        self._manager = alert_manager
        self._only_breaking = only_breaking

    def handle(self, event: WatchEvent) -> None:
        """Create alert if applicable."""
        if self._only_breaking and not event.has_breaking_changes():
            return

        self._manager.create_alert(
            changes=event.changes,
            source=event.source,
            metadata={"event_id": event.event_id},
        )


class HistoryEventHandler:
    """Handler that saves schema versions to history."""

    def __init__(self, history: SchemaHistory):
        """Initialize with history.

        Args:
            history: SchemaHistory to save versions to.
        """
        self._history = history

    def handle(self, event: WatchEvent) -> None:
        """Save schema version."""
        self._history.save(
            schema=event.new_schema,
            metadata={
                "source": event.source,
                "event_id": event.event_id,
                "breaking_changes": event.summary.breaking_changes,
            },
        )


# =============================================================================
# Schema Watcher
# =============================================================================


class SchemaWatcher:
    """Continuous schema monitoring with change detection.

    Monitors schema sources for changes and emits events when
    changes are detected.

    Example:
        # Create watcher
        watcher = SchemaWatcher()

        # Add schema source
        watcher.add_source(FileSchemaSource("schema.json"))

        # Add event handlers
        watcher.add_handler(LoggingEventHandler())
        watcher.add_handler(AlertingEventHandler(alert_manager))

        # Start watching
        watcher.start(poll_interval=60)  # Poll every 60 seconds

        # ... later ...
        watcher.stop()
    """

    def __init__(
        self,
        detector: SchemaEvolutionDetector | None = None,
        history: SchemaHistory | None = None,
    ):
        """Initialize the watcher.

        Args:
            detector: Schema change detector.
            history: Schema history for storing versions.
        """
        self._detector = detector or SchemaEvolutionDetector()
        self._history = history

        self._sources: dict[str, SchemaSource] = {}
        self._handlers: list[WatchEventHandler] = []
        self._last_checksums: dict[str, str] = {}
        self._last_schemas: dict[str, dict[str, Any]] = {}

        self._state = WatcherState.CREATED
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._event_counter = 0

    @property
    def state(self) -> WatcherState:
        """Get current watcher state."""
        return self._state

    def add_source(self, source: SchemaSource) -> None:
        """Add a schema source to watch.

        Args:
            source: Schema source to watch.
        """
        self._sources[source.name] = source
        # Initialize with current schema
        self._last_checksums[source.name] = source.get_checksum()
        self._last_schemas[source.name] = source.get_schema()

    def remove_source(self, name: str) -> bool:
        """Remove a schema source.

        Args:
            name: Name of source to remove.

        Returns:
            True if removed.
        """
        if name in self._sources:
            del self._sources[name]
            self._last_checksums.pop(name, None)
            self._last_schemas.pop(name, None)
            return True
        return False

    def add_handler(self, handler: WatchEventHandler) -> None:
        """Add an event handler.

        Args:
            handler: Handler to add.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: WatchEventHandler) -> bool:
        """Remove an event handler.

        Args:
            handler: Handler to remove.

        Returns:
            True if removed.
        """
        try:
            self._handlers.remove(handler)
            return True
        except ValueError:
            return False

    def check_now(self) -> list[WatchEvent]:
        """Check all sources for changes immediately.

        Returns:
            List of events for detected changes.
        """
        events: list[WatchEvent] = []

        for name, source in self._sources.items():
            try:
                event = self._check_source(source)
                if event:
                    events.append(event)
                    self._emit_event(event)
            except Exception as e:
                logger.error(f"Error checking source {name}: {e}")

        return events

    def _check_source(self, source: SchemaSource) -> WatchEvent | None:
        """Check a single source for changes.

        Args:
            source: Source to check.

        Returns:
            WatchEvent if changes detected, None otherwise.
        """
        name = source.name
        new_checksum = source.get_checksum()
        old_checksum = self._last_checksums.get(name, "")

        # No change if checksum matches
        if new_checksum == old_checksum:
            return None

        # Get schemas
        new_schema = source.get_schema()
        old_schema = self._last_schemas.get(name, {})

        # Detect changes
        changes = self._detector.detect_changes(new_schema, old_schema)

        if not changes:
            # Checksum changed but no schema changes (e.g., formatting)
            self._last_checksums[name] = new_checksum
            return None

        # Create event
        self._event_counter += 1
        event = WatchEvent(
            event_id=f"EVT-{self._event_counter:08d}",
            event_type="schema_changed",
            source=name,
            old_schema=old_schema,
            new_schema=new_schema,
            changes=changes,
            summary=self._detector.get_change_summary(changes),
            metadata={"checksum": new_checksum},
        )

        # Update state
        self._last_checksums[name] = new_checksum
        self._last_schemas[name] = new_schema

        return event

    def _emit_event(self, event: WatchEvent) -> None:
        """Emit event to all handlers.

        Args:
            event: Event to emit.
        """
        for handler in self._handlers:
            try:
                handler.handle(event)
            except Exception as e:
                logger.error(f"Handler {handler} failed: {e}")

    def start(
        self,
        poll_interval: float = 60.0,
        daemon: bool = True,
    ) -> None:
        """Start the watcher.

        Args:
            poll_interval: Seconds between checks.
            daemon: Whether to run as daemon thread.
        """
        with self._state_lock:
            if self._state == WatcherState.RUNNING:
                return

            self._state = WatcherState.STARTING
            self._stop_event.clear()

        def watch_loop() -> None:
            self._state = WatcherState.RUNNING
            logger.info(f"Schema watcher started (interval: {poll_interval}s)")

            while not self._stop_event.wait(poll_interval):
                if self._state == WatcherState.PAUSED:
                    continue

                try:
                    self.check_now()
                except Exception as e:
                    logger.error(f"Watch loop error: {e}")
                    with self._state_lock:
                        if self._state == WatcherState.RUNNING:
                            self._state = WatcherState.ERROR

            self._state = WatcherState.STOPPED
            logger.info("Schema watcher stopped")

        self._watch_thread = threading.Thread(target=watch_loop, daemon=daemon)
        self._watch_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the watcher.

        Args:
            timeout: Maximum time to wait for thread to stop.
        """
        with self._state_lock:
            if self._state not in (WatcherState.RUNNING, WatcherState.PAUSED):
                return
            self._state = WatcherState.STOPPING

        self._stop_event.set()

        if self._watch_thread:
            self._watch_thread.join(timeout)

    def pause(self) -> None:
        """Pause the watcher."""
        with self._state_lock:
            if self._state == WatcherState.RUNNING:
                self._state = WatcherState.PAUSED
                logger.info("Schema watcher paused")

    def resume(self) -> None:
        """Resume the watcher."""
        with self._state_lock:
            if self._state == WatcherState.PAUSED:
                self._state = WatcherState.RUNNING
                logger.info("Schema watcher resumed")

    def get_source_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all sources.

        Returns:
            Dict mapping source names to status info.
        """
        status = {}
        for name, source in self._sources.items():
            status[name] = {
                "name": name,
                "last_checksum": self._last_checksums.get(name),
                "column_count": len(self._last_schemas.get(name, {})),
            }
        return status


# =============================================================================
# Async Watcher
# =============================================================================


class AsyncSchemaWatcher:
    """Async version of SchemaWatcher.

    Uses asyncio for non-blocking operation.

    Example:
        watcher = AsyncSchemaWatcher()
        watcher.add_source(FileSchemaSource("schema.json"))

        async def main():
            await watcher.start(poll_interval=60)
            # ... later ...
            await watcher.stop()

        asyncio.run(main())
    """

    def __init__(
        self,
        detector: SchemaEvolutionDetector | None = None,
    ):
        """Initialize async watcher."""
        self._detector = detector or SchemaEvolutionDetector()
        self._sources: dict[str, SchemaSource] = {}
        self._handlers: list[Callable[[WatchEvent], Any]] = []
        self._last_checksums: dict[str, str] = {}
        self._last_schemas: dict[str, dict[str, Any]] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._event_counter = 0

    def add_source(self, source: SchemaSource) -> None:
        """Add a schema source."""
        self._sources[source.name] = source
        self._last_checksums[source.name] = source.get_checksum()
        self._last_schemas[source.name] = source.get_schema()

    def add_handler(self, handler: Callable[[WatchEvent], Any]) -> None:
        """Add an event handler (can be sync or async)."""
        self._handlers.append(handler)

    async def check_now(self) -> list[WatchEvent]:
        """Check all sources for changes."""
        events: list[WatchEvent] = []

        for name, source in self._sources.items():
            try:
                event = await self._check_source(source)
                if event:
                    events.append(event)
                    await self._emit_event(event)
            except Exception as e:
                logger.error(f"Error checking source {name}: {e}")

        return events

    async def _check_source(self, source: SchemaSource) -> WatchEvent | None:
        """Check a single source."""
        name = source.name
        new_checksum = source.get_checksum()
        old_checksum = self._last_checksums.get(name, "")

        if new_checksum == old_checksum:
            return None

        new_schema = source.get_schema()
        old_schema = self._last_schemas.get(name, {})

        changes = self._detector.detect_changes(new_schema, old_schema)

        if not changes:
            self._last_checksums[name] = new_checksum
            return None

        self._event_counter += 1
        event = WatchEvent(
            event_id=f"EVT-{self._event_counter:08d}",
            event_type="schema_changed",
            source=name,
            old_schema=old_schema,
            new_schema=new_schema,
            changes=changes,
            summary=self._detector.get_change_summary(changes),
        )

        self._last_checksums[name] = new_checksum
        self._last_schemas[name] = new_schema

        return event

    async def _emit_event(self, event: WatchEvent) -> None:
        """Emit event to handlers."""
        for handler in self._handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Handler failed: {e}")

    async def start(self, poll_interval: float = 60.0) -> None:
        """Start watching."""
        if self._running:
            return

        self._running = True

        async def watch_loop() -> None:
            logger.info(f"Async schema watcher started (interval: {poll_interval}s)")
            while self._running:
                await self.check_now()
                await asyncio.sleep(poll_interval)
            logger.info("Async schema watcher stopped")

        self._task = asyncio.create_task(watch_loop())

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_watcher(
    sources: list[SchemaSource] | None = None,
    poll_interval: float = 60.0,
    enable_logging: bool = True,
    enable_history: bool = True,
    history_path: str | Path | None = None,
    alert_manager: BreakingChangeAlertManager | None = None,
    on_change: Callable[[WatchEvent], None] | None = None,
    auto_start: bool = False,
) -> SchemaWatcher:
    """Factory function to create a configured SchemaWatcher.

    Args:
        sources: List of schema sources to watch.
        poll_interval: Seconds between checks.
        enable_logging: Enable logging handler.
        enable_history: Enable history tracking.
        history_path: Path for schema history storage.
        alert_manager: Alert manager for breaking changes.
        on_change: Callback for all changes.
        auto_start: Start watching immediately.

    Returns:
        Configured SchemaWatcher.
    """
    # Create history if enabled
    history = None
    if enable_history:
        if history_path:
            history = SchemaHistory.create(
                storage_type="file",
                path=history_path,
            )
        else:
            history = SchemaHistory.create(storage_type="memory")

    # Create watcher
    watcher = SchemaWatcher(history=history)

    # Add sources
    if sources:
        for source in sources:
            watcher.add_source(source)

    # Add handlers
    if enable_logging:
        watcher.add_handler(LoggingEventHandler())

    if enable_history and history:
        watcher.add_handler(HistoryEventHandler(history))

    if alert_manager:
        watcher.add_handler(AlertingEventHandler(alert_manager))

    if on_change:
        watcher.add_handler(CallbackEventHandler(on_change))

    # Auto-start if requested
    if auto_start:
        watcher.start(poll_interval=poll_interval)

    return watcher
