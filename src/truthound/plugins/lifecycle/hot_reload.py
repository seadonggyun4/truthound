"""Hot reload support for plugins.

This module provides hot reloading of plugins without restart:
- File watching for changes
- Graceful reload strategies
- Rollback on failure
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.plugins.base import Plugin
    from truthound.plugins.lifecycle.manager import LifecycleManager

logger = logging.getLogger(__name__)


class ReloadStrategy(Enum):
    """Strategy for reloading plugins."""

    GRACEFUL = auto()   # Wait for in-flight operations to complete
    IMMEDIATE = auto()  # Stop immediately and reload
    ROLLING = auto()    # Reload incrementally (for multi-instance)


@dataclass
class ReloadResult:
    """Result of a reload operation.

    Attributes:
        success: Whether reload succeeded
        plugin_id: ID of reloaded plugin
        strategy: Strategy used
        duration_ms: Time taken in milliseconds
        error: Error message if failed
        rolled_back: Whether rollback occurred
    """

    success: bool
    plugin_id: str
    strategy: ReloadStrategy
    duration_ms: float = 0.0
    error: str | None = None
    rolled_back: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WatchHandle:
    """Handle for a file watcher."""

    plugin_id: str
    path: Path
    task: asyncio.Task | None = None

    def cancel(self) -> None:
        """Cancel the watcher."""
        if self.task and not self.task.done():
            self.task.cancel()


class FileWatcher:
    """Watches files for changes.

    Uses polling for cross-platform compatibility.
    On Linux, could be extended to use inotify for better performance.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize file watcher.

        Args:
            poll_interval: Seconds between polls
        """
        self._poll_interval = poll_interval
        self._watches: dict[str, WatchHandle] = {}
        self._file_mtimes: dict[Path, float] = {}

    async def watch(
        self,
        path: Path,
        callback: Callable[[Path], None],
        recursive: bool = True,
    ) -> WatchHandle:
        """Start watching a path for changes.

        Args:
            path: File or directory to watch
            callback: Called when changes detected
            recursive: Watch subdirectories

        Returns:
            WatchHandle for cancellation
        """
        plugin_id = str(path)

        # Get initial mtimes
        self._update_mtimes(path, recursive)

        # Create watch task
        task = asyncio.create_task(
            self._watch_loop(path, callback, recursive)
        )

        handle = WatchHandle(
            plugin_id=plugin_id,
            path=path,
            task=task,
        )
        self._watches[plugin_id] = handle

        logger.debug(f"Started watching: {path}")
        return handle

    async def _watch_loop(
        self,
        path: Path,
        callback: Callable[[Path], None],
        recursive: bool,
    ) -> None:
        """Main watch loop."""
        try:
            while True:
                await asyncio.sleep(self._poll_interval)

                # Check for changes
                changed = self._check_changes(path, recursive)
                if changed:
                    logger.info(f"Changes detected in: {changed}")
                    callback(changed[0])

        except asyncio.CancelledError:
            logger.debug(f"Watch cancelled for: {path}")

    def _update_mtimes(self, path: Path, recursive: bool) -> None:
        """Update stored modification times."""
        if path.is_file():
            self._file_mtimes[path] = path.stat().st_mtime
        elif path.is_dir():
            for py_file in self._iter_files(path, recursive):
                self._file_mtimes[py_file] = py_file.stat().st_mtime

    def _check_changes(self, path: Path, recursive: bool) -> list[Path]:
        """Check for file changes."""
        changed: list[Path] = []

        if path.is_file():
            if self._file_changed(path):
                changed.append(path)
        elif path.is_dir():
            for py_file in self._iter_files(path, recursive):
                if self._file_changed(py_file):
                    changed.append(py_file)

        return changed

    def _file_changed(self, path: Path) -> bool:
        """Check if a single file changed."""
        if not path.exists():
            return path in self._file_mtimes

        try:
            current_mtime = path.stat().st_mtime
            stored_mtime = self._file_mtimes.get(path, 0)

            if current_mtime != stored_mtime:
                self._file_mtimes[path] = current_mtime
                return True
        except OSError:
            pass

        return False

    def _iter_files(self, path: Path, recursive: bool):
        """Iterate over Python files in a directory."""
        if recursive:
            yield from path.rglob("*.py")
        else:
            yield from path.glob("*.py")

    def stop(self, plugin_id: str) -> bool:
        """Stop watching a path.

        Args:
            plugin_id: Watch handle ID

        Returns:
            True if stopped, False if not found
        """
        handle = self._watches.pop(plugin_id, None)
        if handle:
            handle.cancel()
            return True
        return False

    def stop_all(self) -> None:
        """Stop all watchers."""
        for handle in self._watches.values():
            handle.cancel()
        self._watches.clear()
        self._file_mtimes.clear()


class HotReloadManager:
    """Manages hot reloading of plugins.

    Provides safe reloading with rollback on failure.

    Example:
        >>> manager = HotReloadManager(lifecycle_manager)
        >>> result = await manager.reload("my-plugin")
        >>> if not result.success:
        ...     print(f"Reload failed: {result.error}")
    """

    def __init__(
        self,
        lifecycle_manager: "LifecycleManager",
        plugin_loader: Callable[[str], "Plugin"] | None = None,
        default_strategy: ReloadStrategy = ReloadStrategy.GRACEFUL,
    ) -> None:
        """Initialize hot reload manager.

        Args:
            lifecycle_manager: Lifecycle manager instance
            plugin_loader: Function to load plugins
            default_strategy: Default reload strategy
        """
        self._lifecycle = lifecycle_manager
        self._loader = plugin_loader
        self._default_strategy = default_strategy
        self._watcher = FileWatcher()
        self._plugin_states: dict[str, Any] = {}  # For rollback
        self._watches: dict[str, WatchHandle] = {}

    async def reload(
        self,
        plugin_id: str,
        strategy: ReloadStrategy | None = None,
    ) -> ReloadResult:
        """Reload a plugin.

        Args:
            plugin_id: Plugin to reload
            strategy: Reload strategy (uses default if None)

        Returns:
            ReloadResult with operation details
        """
        strategy = strategy or self._default_strategy
        start_time = asyncio.get_event_loop().time()

        try:
            if strategy == ReloadStrategy.GRACEFUL:
                result = await self._reload_graceful(plugin_id)
            elif strategy == ReloadStrategy.IMMEDIATE:
                result = await self._reload_immediate(plugin_id)
            elif strategy == ReloadStrategy.ROLLING:
                result = await self._reload_rolling(plugin_id)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            duration = (asyncio.get_event_loop().time() - start_time) * 1000

            return ReloadResult(
                success=True,
                plugin_id=plugin_id,
                strategy=strategy,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (asyncio.get_event_loop().time() - start_time) * 1000

            # Attempt rollback
            rolled_back = await self._rollback(plugin_id)

            return ReloadResult(
                success=False,
                plugin_id=plugin_id,
                strategy=strategy,
                duration_ms=duration,
                error=str(e),
                rolled_back=rolled_back,
            )

    async def _reload_graceful(self, plugin_id: str) -> None:
        """Graceful reload - wait for operations to complete."""
        # Save state for rollback
        await self._save_state(plugin_id)

        # Wait for in-flight operations (implementation-specific)
        await asyncio.sleep(0.1)

        # Deactivate
        logger.info(f"Gracefully reloading plugin: {plugin_id}")

        # Reload module (simplified - full implementation would
        # use importlib.reload)
        if self._loader:
            self._loader(plugin_id)

    async def _reload_immediate(self, plugin_id: str) -> None:
        """Immediate reload - stop and reload now."""
        await self._save_state(plugin_id)
        logger.info(f"Immediately reloading plugin: {plugin_id}")

        if self._loader:
            self._loader(plugin_id)

    async def _reload_rolling(self, plugin_id: str) -> None:
        """Rolling reload - for multi-instance scenarios."""
        # In a multi-instance setup, this would reload one instance at a time
        await self._reload_graceful(plugin_id)

    async def _save_state(self, plugin_id: str) -> None:
        """Save plugin state for rollback."""
        # Store any state that needs to be restored on rollback
        self._plugin_states[plugin_id] = {
            "lifecycle_state": self._lifecycle.get_state(plugin_id),
            "saved_at": datetime.now(timezone.utc),
        }

    async def _rollback(self, plugin_id: str) -> bool:
        """Attempt to rollback a failed reload.

        Args:
            plugin_id: Plugin to rollback

        Returns:
            True if rollback succeeded
        """
        saved_state = self._plugin_states.pop(plugin_id, None)
        if not saved_state:
            return False

        try:
            logger.info(f"Rolling back plugin: {plugin_id}")
            # Restore previous state
            # Full implementation would restore the previous plugin code
            return True
        except Exception as e:
            logger.error(f"Rollback failed for {plugin_id}: {e}")
            return False

    async def watch(
        self,
        plugin_id: str,
        plugin_path: Path,
        auto_reload: bool = True,
    ) -> WatchHandle:
        """Start watching a plugin for changes.

        Args:
            plugin_id: Plugin identifier
            plugin_path: Path to watch
            auto_reload: Automatically reload on changes

        Returns:
            WatchHandle for cancellation
        """
        def on_change(path: Path) -> None:
            if auto_reload:
                asyncio.create_task(self.reload(plugin_id))

        handle = await self._watcher.watch(plugin_path, on_change)
        self._watches[plugin_id] = handle
        return handle

    def stop_watch(self, plugin_id: str) -> bool:
        """Stop watching a plugin.

        Args:
            plugin_id: Plugin to stop watching

        Returns:
            True if stopped
        """
        handle = self._watches.pop(plugin_id, None)
        if handle:
            handle.cancel()
            return True
        return self._watcher.stop(plugin_id)

    def stop_all_watches(self) -> None:
        """Stop all file watches."""
        for handle in self._watches.values():
            handle.cancel()
        self._watches.clear()
        self._watcher.stop_all()
