"""Hot reload support for plugins.

This module provides hot reloading of plugins without restart:
- File watching for changes
- Graceful reload strategies
- Rollback on failure
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
from collections.abc import Callable  # noqa: TC003 - public annotation introspection
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReloadStrategy(Enum):
    """Strategy for reloading plugins."""

    GRACEFUL = auto()  # Wait for in-flight operations to complete
    IMMEDIATE = auto()  # Stop immediately and reload
    ROLLING = auto()  # Reload incrementally (for multi-instance)


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


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
    """Owned asynchronous file polling with awaited callbacks and visible errors."""

    def __init__(self, poll_interval: float = 1.0) -> None:
        if (
            isinstance(poll_interval, bool)
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise ValueError("poll_interval must be finite and positive")
        self._poll_interval = poll_interval
        self._watches: dict[str, WatchHandle] = {}
        self._retired: list[asyncio.Task] = []

    def _snapshot(self, path: Path, recursive: bool) -> dict[Path, bytes]:
        if path.is_dir():
            paths = sorted(path.rglob("*.py") if recursive else path.glob("*.py"))
        else:
            paths = [path]
        snapshot = {}
        for selected in paths:
            try:
                with selected.open("rb") as stream:
                    snapshot[selected] = hashlib.file_digest(stream, "sha256").digest()
            except FileNotFoundError:
                continue
        return snapshot

    async def watch(
        self, path: Path, callback: Callable[[Path], Any], recursive: bool = True
    ) -> WatchHandle:
        path = Path(path).absolute()
        plugin_id = str(path)
        if plugin_id in self._watches:
            raise ValueError("Path is already watched")
        initial = self._snapshot(path, recursive)
        task = asyncio.create_task(self._watch_loop(path, callback, recursive, initial))
        handle = WatchHandle(plugin_id=plugin_id, path=path, task=task)
        self._watches[plugin_id] = handle
        return handle

    async def _watch_loop(self, path, callback, recursive, previous):
        while True:
            await asyncio.sleep(self._poll_interval)
            current = self._snapshot(path, recursive)
            changed = sorted(
                key
                for key in previous.keys() | current.keys()
                if previous.get(key) != current.get(key)
            )
            previous = current
            for selected in changed:
                result = callback(selected)
                if inspect.isawaitable(result):
                    await result

    def stop(self, plugin_id: str) -> bool:
        handle = self._watches.pop(plugin_id, None)
        if handle is None:
            return False
        handle.cancel()
        if handle.task is not None:
            self._retired.append(handle.task)
        return True

    def stop_all(self) -> None:
        for plugin_id in list(self._watches):
            self.stop(plugin_id)

    async def aclose(self) -> None:
        """Cancel and join every owned task, reporting callback/read failures."""
        self.stop_all()
        tasks, self._retired = self._retired, []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("File watcher failed", errors)


class HotReloadManager:
    """Reload through the shared PluginManager with observed cleanup and rollback.

    Operator factories return a fresh plugin class. Imports and callback effects
    remain trusted; restoration creates a fresh old-class instance, not a rollback
    of external side effects or arbitrary in-memory state.
    """

    def __init__(
        self,
        lifecycle_manager,
        plugin_loader=None,
        default_strategy=ReloadStrategy.GRACEFUL,
        *,
        timeout=30.0,
    ):
        from truthound.plugins.manager import PluginManager

        if not isinstance(lifecycle_manager, PluginManager):
            raise TypeError("Hot reload requires the shared PluginManager")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._manager = lifecycle_manager
        self._loader = plugin_loader
        self._default_strategy = default_strategy
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._active = {}
        self._reloading = set()
        self._watcher = FileWatcher()
        self._watches = {}
        self._closed = False
        self.last_results = {}

    @asynccontextmanager
    async def operation(self, plugin_id):
        """Track cooperative callers so graceful reload can drain their work."""
        async with self._condition:
            if self._closed or plugin_id in self._reloading:
                raise RuntimeError("Plugin reload is pending or owner is closed")
            self._active[plugin_id] = self._active.get(plugin_id, 0) + 1
        try:
            yield self._manager.get_plugin(plugin_id)
        finally:
            async with self._condition:
                self._active[plugin_id] -= 1
                self._condition.notify_all()

    def _guard(self):
        guard = getattr(self._manager, "_execution_guard", None)
        if guard is not None:
            guard()

    async def reload(self, plugin_id, strategy=None):
        from truthound.plugins.base import Plugin, PluginState

        strategy = strategy or self._default_strategy
        started = asyncio.get_running_loop().time()
        rolled_back = False
        error_type = None
        async with self._lock:
            if self._closed:
                raise RuntimeError("Hot reload owner is closed")
            self._reloading.add(plugin_id)
            try:
                if strategy not in (ReloadStrategy.GRACEFUL, ReloadStrategy.IMMEDIATE):
                    raise ValueError(
                        "A single manager does not provide rolling multi-instance reload"
                    )
                if self._loader is None:
                    raise ValueError("An operator plugin class loader is required")
                if strategy == ReloadStrategy.GRACEFUL:
                    async with self._condition:
                        await asyncio.wait_for(
                            self._condition.wait_for(lambda: self._active.get(plugin_id, 0) == 0),
                            self._timeout,
                        )
                self._guard()
                candidate = self._loader(plugin_id)
                if inspect.isawaitable(candidate):
                    candidate = await asyncio.wait_for(candidate, self._timeout)
                if not isinstance(candidate, type) or not issubclass(candidate, Plugin):
                    raise TypeError("Plugin loader must return a Plugin class")
                with self._manager._lock:
                    previous = self._manager.get_plugin(plugin_id)
                    old_class, config = type(previous), previous.config
                    active = previous.state == PluginState.ACTIVE
                    # Signed bindings require an explicit operator renewal workflow.
                    if plugin_id in getattr(self._manager, "_signed_plugins", {}):
                        raise ValueError("Unload and renew the signed class binding explicitly")
                    self._guard()
                    receipt = self._manager.unload_plugin(plugin_id, detailed=True)
                    if not receipt.success:
                        raise RuntimeError(
                            "Previous plugin cleanup failed; replacement not started"
                        )
                    self._manager._discovered_classes[plugin_id] = candidate
                    try:
                        self._guard()
                        loaded = self._manager.load_plugin(
                            plugin_id, config=config, activate=active
                        )
                        if loaded.name != plugin_id:
                            receipt = self._manager.unload_plugin(loaded.name, detailed=True)
                            if not receipt.success:
                                raise ExceptionGroup(
                                    "Candidate identity cleanup failed", [RuntimeError("cleanup")]
                                )
                            raise ValueError("Replacement plugin identity changed")
                    except Exception as error:
                        clean = not isinstance(error.__cause__, ExceptionGroup) and not isinstance(
                            error, ExceptionGroup
                        )
                        if self._manager.is_plugin_loaded(plugin_id):
                            clean = (
                                self._manager.unload_plugin(plugin_id, detailed=True).success
                                and clean
                            )
                        self._manager._discovered_classes[plugin_id] = old_class
                        if clean:
                            try:
                                self._guard()
                                self._manager.load_plugin(plugin_id, config=config, activate=active)
                                rolled_back = True
                            except Exception:
                                rolled_back = False
                        raise error
            except Exception as error:
                error_type = type(error).__name__
            finally:
                self._reloading.discard(plugin_id)
        result = ReloadResult(
            success=error_type is None,
            plugin_id=plugin_id,
            strategy=strategy,
            duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            error=error_type,
            rolled_back=rolled_back,
        )
        self.last_results[plugin_id] = result
        return result

    async def watch(self, plugin_id, plugin_path, auto_reload=True):
        if self._closed:
            raise RuntimeError("Hot reload owner is closed")
        if plugin_id in self._watches:
            raise ValueError("Plugin is already watched")

        async def on_change(path):
            if auto_reload:
                result = await self.reload(plugin_id)
                if not result.success:
                    raise RuntimeError("Automatic plugin reload failed")

        handle = await self._watcher.watch(plugin_path, on_change)
        self._watches[plugin_id] = handle
        return handle

    def stop_watch(self, plugin_id):
        handle = self._watches.pop(plugin_id, None)
        return self._watcher.stop(handle.plugin_id) if handle else False

    def stop_all_watches(self):
        self._watches.clear()
        self._watcher.stop_all()

    async def aclose(self):
        self._closed = True
        self.stop_all_watches()
        await self._watcher.aclose()
        async with self._lock:
            pass
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: not any(self._active.values())), self._timeout
            )
