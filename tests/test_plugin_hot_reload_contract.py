"""Owned file watchers observe deletions and await callbacks."""

import asyncio

import pytest

from truthound.plugins.lifecycle.hot_reload import FileWatcher


@pytest.mark.asyncio
async def test_watcher_awaits_callback_and_reports_deletion(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("before = 1\n")
    watcher = FileWatcher(poll_interval=0.01)
    changed = asyncio.Event()
    seen = []

    async def on_change(value):
        seen.append(value)
        changed.set()

    handle = await watcher.watch(path, on_change)
    try:
        path.unlink()
        await asyncio.wait_for(changed.wait(), timeout=1)
        assert seen == [path]
    finally:
        watcher.stop_all()
        await asyncio.gather(handle.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_duplicate_watch_does_not_abandon_owned_task(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    watcher = FileWatcher(poll_interval=0.01)
    first = await watcher.watch(path, lambda path: None)
    second = None
    try:
        with pytest.raises(ValueError):
            second = await watcher.watch(path, lambda path: None)
    finally:
        watcher.stop_all()
        first.cancel()
        await asyncio.gather(first.task, *([second.task] if second else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_reload_replaces_native_plugin_and_restores_after_failed_setup():
    from truthound.plugins.base import Plugin, PluginInfo, PluginType
    from truthound.plugins.lifecycle.hot_reload import HotReloadManager
    from truthound.plugins.manager import PluginManager

    calls = []

    class Original(Plugin):
        @property
        def info(self):
            return PluginInfo(name="reloadable", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            calls.append("original.setup")

        def teardown(self):
            calls.append("original.teardown")

    class Replacement(Original):
        def setup(self):
            calls.append("replacement.setup")

        def teardown(self):
            calls.append("replacement.teardown")

    class Broken(Replacement):
        def setup(self):
            calls.append("broken.setup")
            raise ValueError("setup failed")

        def teardown(self):
            calls.append("broken.teardown")

    manager = PluginManager()
    manager.load_from_class(Original, activate=False)
    selected = Replacement
    reloads = HotReloadManager(manager, plugin_loader=lambda name: selected)
    try:
        result = await reloads.reload("reloadable")
        assert result.success and type(manager.get_plugin("reloadable")) is Replacement
        assert calls == ["original.setup", "original.teardown", "replacement.setup"]
        selected = Broken
        result = await reloads.reload("reloadable")
        assert not result.success and result.rolled_back
        assert type(manager.get_plugin("reloadable")) is Replacement
        assert calls[-4:] == [
            "replacement.teardown",
            "broken.setup",
            "broken.teardown",
            "replacement.setup",
        ]
    finally:
        manager.shutdown()


@pytest.mark.asyncio
async def test_watcher_failure_is_observed_when_joined(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    watcher = FileWatcher(poll_interval=0.01)

    async def failing(path):
        raise OSError("callback failed")

    handle = await watcher.watch(path, failing)
    path.write_text("value = 2\n")
    with pytest.raises(OSError):
        await asyncio.wait_for(asyncio.shield(handle.task), 1)
    with pytest.raises(ExceptionGroup):
        await watcher.aclose()
    assert handle.task.done()


@pytest.mark.asyncio
async def test_graceful_reload_timeout_keeps_original_and_close_drains_leases():
    from truthound.plugins.base import Plugin, PluginInfo, PluginType
    from truthound.plugins.lifecycle.hot_reload import HotReloadManager
    from truthound.plugins.manager import PluginManager

    class Sample(Plugin):
        @property
        def info(self):
            return PluginInfo(name="sample", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            pass

        def teardown(self):
            pass

    manager = PluginManager()
    original = manager.load_from_class(Sample, activate=False)
    reloads = HotReloadManager(manager, plugin_loader=lambda name: Sample, timeout=0.05)
    try:
        async with reloads.operation("sample"):
            result = await reloads.reload("sample")
            assert not result.success and result.error == "TimeoutError"
            assert manager.get_plugin("sample") is original
        await reloads.aclose()
        with pytest.raises(RuntimeError):
            async with reloads.operation("sample"):
                pass
    finally:
        manager.shutdown()


@pytest.mark.asyncio
async def test_enterprise_native_reload_and_watcher_cleanup(tmp_path):
    from truthound.plugins.base import Plugin, PluginInfo, PluginType
    from truthound.plugins.enterprise_manager import (
        EnterprisePluginManager,
        EnterprisePluginManagerConfig,
    )

    class Sample(Plugin):
        @property
        def info(self):
            return PluginInfo(name="sample", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            pass

        def teardown(self):
            pass

    replaced = asyncio.Event()

    class Replacement(Sample):
        def setup(self):
            replaced.set()

    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    manager = EnterprisePluginManager(
        EnterprisePluginManagerConfig(
            enable_hot_reload=True, watch_for_changes=True, auto_activate=False
        )
    )
    manager.load_from_class(Sample)
    try:
        await manager.configure_hot_reload("sample", lambda name: Replacement, path=path)
        handle = manager._hot_reload._watches["sample"]
        with pytest.raises(RuntimeError):
            manager.shutdown()
        path.write_text("value = 2\n")
        await asyncio.wait_for(replaced.wait(), timeout=3)
        result = manager._hot_reload.last_results["sample"]
        assert result.success and type(manager.get_plugin("sample")) is Replacement
    finally:
        await manager.unload_all()
    assert handle.task.done()
    assert not manager.is_plugin_loaded("sample")
