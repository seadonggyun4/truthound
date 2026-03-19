import pytest

from truthound import __version__
from truthound.plugins.base import Plugin, PluginConfig, PluginInfo, PluginType
from truthound.plugins.enterprise_manager import EnterprisePluginManager, EnterprisePluginManagerConfig
from truthound.plugins.manager import PluginManager, PluginManagerConfig


def test_plugin_manager_host_version_defaults_to_package_version():
    config = PluginManagerConfig()

    assert config.host_version == __version__


def test_enterprise_manager_uses_unified_plugin_manager_runtime():
    manager = EnterprisePluginManager(EnterprisePluginManagerConfig())

    assert isinstance(manager, PluginManager)
    assert 'versioning' in manager.list_capabilities()


class AsyncSamplePlugin(Plugin[PluginConfig]):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name='async-sample',
            version='1.0.0',
            plugin_type=PluginType.CUSTOM,
        )

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass


@pytest.mark.asyncio
async def test_enterprise_manager_async_load_facade():
    manager = EnterprisePluginManager(EnterprisePluginManagerConfig(auto_activate=False))
    manager._discovered_classes['async-sample'] = AsyncSamplePlugin

    plugin = await manager.load('async-sample', activate=False)

    assert plugin.name == 'async-sample'
    assert manager.is_plugin_loaded('async-sample')
