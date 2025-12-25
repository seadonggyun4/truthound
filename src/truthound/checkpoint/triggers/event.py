"""Event-based triggers.

These triggers execute checkpoints based on events like file changes,
webhooks, or custom events.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from truthound.checkpoint.triggers.base import (
    BaseTrigger,
    TriggerConfig,
    TriggerResult,
)

if TYPE_CHECKING:
    pass


@dataclass
class EventConfig(TriggerConfig):
    """Configuration for event trigger.

    Attributes:
        event_type: Type of event to listen for.
        event_filter: Filter for matching events.
        debounce_seconds: Minimum time between trigger fires.
        batch_events: Batch multiple events into single run.
        batch_window_seconds: Time window for batching events.
    """

    event_type: str = "custom"
    event_filter: dict[str, Any] = field(default_factory=dict)
    debounce_seconds: float = 1.0
    batch_events: bool = False
    batch_window_seconds: float = 5.0


class EventTrigger(BaseTrigger[EventConfig]):
    """Trigger based on custom events.

    Executes checkpoints when specific events occur. Events can be
    pushed programmatically or received from external sources.

    Example:
        >>> trigger = EventTrigger(
        ...     event_type="data_updated",
        ...     event_filter={"source": "production"},
        ... )
        >>>
        >>> # Fire the trigger
        >>> trigger.fire_event({"source": "production", "table": "users"})
    """

    trigger_type = "event"

    def __init__(self, config: EventConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._pending_events: list[dict[str, Any]] = []
        self._event_callback: Callable[[dict[str, Any]], None] | None = None
        self._last_event_time: datetime | None = None

    @classmethod
    def _default_config(cls) -> EventConfig:
        return EventConfig()

    def fire_event(self, event_data: dict[str, Any]) -> bool:
        """Fire an event to trigger the checkpoint.

        Args:
            event_data: Data associated with the event.

        Returns:
            True if event was accepted, False if filtered out.
        """
        # Check filter
        filter_config = self._config.event_filter
        if filter_config:
            for key, expected in filter_config.items():
                if event_data.get(key) != expected:
                    return False

        # Check debounce
        now = datetime.now()
        if self._last_event_time:
            elapsed = (now - self._last_event_time).total_seconds()
            if elapsed < self._config.debounce_seconds:
                if self._config.batch_events:
                    self._pending_events.append(event_data)
                return False

        self._last_event_time = now
        self._pending_events.append(event_data)

        if self._event_callback:
            self._event_callback(event_data)

        return True

    def should_trigger(self) -> TriggerResult:
        """Check if there are pending events."""
        if not self._pending_events:
            return TriggerResult(
                should_run=False,
                reason="No pending events",
            )

        # Check batch window
        if self._config.batch_events:
            if self._last_event_time:
                elapsed = (datetime.now() - self._last_event_time).total_seconds()
                if elapsed < self._config.batch_window_seconds:
                    return TriggerResult(
                        should_run=False,
                        reason=f"Batching events, {elapsed:.1f}s elapsed",
                        context={"pending_count": len(self._pending_events)},
                    )

        # Consume events
        events = self._pending_events.copy()
        self._pending_events.clear()

        return TriggerResult(
            should_run=True,
            reason=f"Received {len(events)} event(s)",
            context={"events": events},
        )

    def on_event(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register callback for event notifications.

        Args:
            callback: Function to call when events arrive.
        """
        self._event_callback = callback

    def clear_pending(self) -> int:
        """Clear pending events.

        Returns:
            Number of cleared events.
        """
        count = len(self._pending_events)
        self._pending_events.clear()
        return count


@dataclass
class FileWatchConfig(TriggerConfig):
    """Configuration for file watch trigger.

    Attributes:
        paths: Paths to watch (files or directories).
        patterns: Glob patterns to match (e.g., "*.csv").
        recursive: Watch directories recursively.
        events: File events to watch ("created", "modified", "deleted").
        ignore_patterns: Patterns to ignore.
        hash_check: Only trigger if file content changed (hash check).
        poll_interval_seconds: Polling interval for checking changes.
    """

    paths: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=lambda: ["*"])
    recursive: bool = True
    events: list[str] = field(default_factory=lambda: ["modified", "created"])
    ignore_patterns: list[str] = field(default_factory=lambda: [".*", "__pycache__", "*.pyc"])
    hash_check: bool = True
    poll_interval_seconds: float = 5.0


class FileWatchTrigger(BaseTrigger[FileWatchConfig]):
    """Trigger based on file system changes.

    Executes checkpoints when files matching specified patterns
    are created, modified, or deleted.

    Example:
        >>> trigger = FileWatchTrigger(
        ...     paths=["./data"],
        ...     patterns=["*.csv", "*.parquet"],
        ...     events=["modified", "created"],
        ... )
    """

    trigger_type = "file_watch"

    def __init__(self, config: FileWatchConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._file_hashes: dict[str, str] = {}
        self._file_mtimes: dict[str, float] = {}
        self._last_scan: datetime | None = None
        self._pending_changes: list[dict[str, Any]] = []

    @classmethod
    def _default_config(cls) -> FileWatchConfig:
        return FileWatchConfig()

    def _get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file content."""
        hasher = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (IOError, OSError):
            return ""

    def _matches_pattern(self, filepath: Path) -> bool:
        """Check if file matches any include pattern and no ignore pattern."""
        filename = filepath.name

        # Check ignore patterns
        for pattern in self._config.ignore_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return False
            if fnmatch.fnmatch(str(filepath), pattern):
                return False

        # Check include patterns
        for pattern in self._config.patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
            if fnmatch.fnmatch(str(filepath), pattern):
                return True

        return False

    def _scan_files(self) -> list[Path]:
        """Scan configured paths for matching files."""
        files: list[Path] = []

        for path_str in self._config.paths:
            path = Path(path_str).expanduser().resolve()

            if path.is_file():
                if self._matches_pattern(path):
                    files.append(path)
            elif path.is_dir():
                if self._config.recursive:
                    for filepath in path.rglob("*"):
                        if filepath.is_file() and self._matches_pattern(filepath):
                            files.append(filepath)
                else:
                    for filepath in path.glob("*"):
                        if filepath.is_file() and self._matches_pattern(filepath):
                            files.append(filepath)

        return files

    def _detect_changes(self) -> list[dict[str, Any]]:
        """Detect file changes since last scan."""
        changes: list[dict[str, Any]] = []
        current_files = self._scan_files()
        current_file_set = set(str(f) for f in current_files)
        previous_file_set = set(self._file_mtimes.keys())

        # Check for new and modified files
        for filepath in current_files:
            filepath_str = str(filepath)
            try:
                mtime = filepath.stat().st_mtime
            except (IOError, OSError):
                continue

            if filepath_str not in self._file_mtimes:
                # New file
                if "created" in self._config.events:
                    file_hash = self._get_file_hash(filepath) if self._config.hash_check else ""
                    self._file_hashes[filepath_str] = file_hash
                    self._file_mtimes[filepath_str] = mtime
                    changes.append({
                        "event": "created",
                        "path": filepath_str,
                        "mtime": mtime,
                    })
            elif mtime != self._file_mtimes[filepath_str]:
                # Potentially modified
                if "modified" in self._config.events:
                    if self._config.hash_check:
                        new_hash = self._get_file_hash(filepath)
                        old_hash = self._file_hashes.get(filepath_str, "")
                        if new_hash != old_hash:
                            self._file_hashes[filepath_str] = new_hash
                            self._file_mtimes[filepath_str] = mtime
                            changes.append({
                                "event": "modified",
                                "path": filepath_str,
                                "mtime": mtime,
                            })
                        else:
                            # mtime changed but content didn't
                            self._file_mtimes[filepath_str] = mtime
                    else:
                        self._file_mtimes[filepath_str] = mtime
                        changes.append({
                            "event": "modified",
                            "path": filepath_str,
                            "mtime": mtime,
                        })

        # Check for deleted files
        if "deleted" in self._config.events:
            for filepath_str in previous_file_set - current_file_set:
                del self._file_mtimes[filepath_str]
                if filepath_str in self._file_hashes:
                    del self._file_hashes[filepath_str]
                changes.append({
                    "event": "deleted",
                    "path": filepath_str,
                })

        return changes

    def should_trigger(self) -> TriggerResult:
        """Check for file system changes."""
        now = datetime.now()

        # Check poll interval
        if self._last_scan:
            elapsed = (now - self._last_scan).total_seconds()
            if elapsed < self._config.poll_interval_seconds:
                return TriggerResult(
                    should_run=False,
                    reason=f"Waiting for poll interval ({elapsed:.1f}s elapsed)",
                )

        self._last_scan = now

        # Detect changes
        changes = self._detect_changes()

        if not changes:
            return TriggerResult(
                should_run=False,
                reason="No file changes detected",
            )

        return TriggerResult(
            should_run=True,
            reason=f"Detected {len(changes)} file change(s)",
            context={
                "changes": changes,
                "changed_files": [c["path"] for c in changes],
            },
        )

    def _on_start(self) -> None:
        """Initialize file tracking on start."""
        # Initial scan to establish baseline
        for filepath in self._scan_files():
            filepath_str = str(filepath)
            try:
                self._file_mtimes[filepath_str] = filepath.stat().st_mtime
                if self._config.hash_check:
                    self._file_hashes[filepath_str] = self._get_file_hash(filepath)
            except (IOError, OSError):
                pass

        self._last_scan = datetime.now()

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.paths:
            errors.append("At least one path is required")

        for path_str in self._config.paths:
            path = Path(path_str).expanduser()
            if not path.exists():
                errors.append(f"Path does not exist: {path_str}")

        for event in self._config.events:
            if event not in ("created", "modified", "deleted"):
                errors.append(f"Invalid event type: {event}")

        return errors
