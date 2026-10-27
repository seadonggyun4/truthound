"""Owned plugin teardown failures cannot become successful owner closure."""

import pytest

from truthound.mcp.platform_services import PluginRuntime
from truthound.mcp.platform_worker import PlatformOwner
from truthound.plugins.base import Plugin, PluginInfo, PluginType
from truthound.plugins.manager import PluginManager


@pytest.mark.asyncio
async def test_owner_close_reports_native_failed_receipts_and_attempts_all_plugins(tmp_path):
    calls = []

    class Broken(Plugin):
        @property
        def info(self):
            return PluginInfo(name="broken-close", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            calls.append("setup")

        def teardown(self):
            calls.append("broken")
            raise ValueError("synthetic private teardown detail")

    class Healthy(Plugin):
        @property
        def info(self):
            return PluginInfo(name="healthy-close", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            calls.append("setup")

        def teardown(self):
            calls.append("healthy")

    manager = PluginManager()
    manager.load_from_class(Broken)
    manager.load_from_class(Healthy)
    owner = PlatformOwner(
        PluginRuntime(manager),
        {
            "native_group": "plugin_services",
            "control": str(tmp_path),
            "digest": "unused",
            "plugins": {},
        },
    )
    owner.loaded = {"broken": "broken-close", "healthy": "healthy-close"}
    with pytest.raises(ExceptionGroup) as error:
        await owner.close()
    assert calls.count("setup") == 2
    assert calls[-2:] == ["broken", "healthy"]
    assert not owner.loaded
    assert "synthetic private teardown detail" not in str(error.value)
