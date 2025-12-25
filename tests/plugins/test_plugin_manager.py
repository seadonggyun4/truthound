"""Tests for PluginManager."""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

from truthound.plugins import (
    Plugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    PluginState,
    PluginManager,
    PluginRegistry,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
)
from truthound.plugins.manager import PluginManagerConfig, get_plugin_manager, reset_plugin_manager


# =============================================================================
# Test Fixtures
# =============================================================================


class SamplePlugin1(Plugin[PluginConfig]):
    """First test plugin."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="test-plugin-1",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            description="First test plugin",
        )

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass


class SamplePlugin2(Plugin[PluginConfig]):
    """Second test plugin."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="test-plugin-2",
            version="2.0.0",
            plugin_type=PluginType.VALIDATOR,
            dependencies=("test-plugin-1",),
        )

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass


class FailingSetupPlugin(Plugin[PluginConfig]):
    """Plugin that fails during setup."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="failing-setup",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )

    def setup(self) -> None:
        raise RuntimeError("Setup failed!")

    def teardown(self) -> None:
        pass


class TrackingPlugin(Plugin[PluginConfig]):
    """Plugin that tracks lifecycle calls."""

    calls: list[str] = []

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="tracking-plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )

    def setup(self) -> None:
        TrackingPlugin.calls.append("setup")

    def teardown(self) -> None:
        TrackingPlugin.calls.append("teardown")

    def register(self, manager) -> None:
        TrackingPlugin.calls.append("register")

    def unregister(self, manager) -> None:
        TrackingPlugin.calls.append("unregister")


@pytest.fixture
def manager():
    """Create a fresh plugin manager for each test."""
    reset_plugin_manager()
    mgr = PluginManager()
    yield mgr
    mgr.shutdown()


@pytest.fixture(autouse=True)
def reset_tracking():
    """Reset tracking plugin between tests."""
    TrackingPlugin.calls = []
    yield


# =============================================================================
# PluginManager Tests
# =============================================================================


class TestPluginManager:
    """Tests for PluginManager."""

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert isinstance(manager.registry, PluginRegistry)
        assert len(manager.registry) == 0

    def test_load_from_class(self, manager):
        """Test loading plugin from class."""
        plugin = manager.load_from_class(SamplePlugin1)

        assert plugin.name == "test-plugin-1"
        assert plugin.state == PluginState.ACTIVE  # Auto-activated
        assert manager.is_plugin_loaded("test-plugin-1")

    def test_load_plugin_without_activation(self, manager):
        """Test loading without auto-activation."""
        config = PluginManagerConfig(auto_activate=False)
        manager = PluginManager(config)

        plugin = manager.load_from_class(SamplePlugin1, activate=False)
        assert plugin.state == PluginState.LOADED
        assert not manager.is_plugin_active("test-plugin-1")

    def test_activate_deactivate_plugin(self, manager):
        """Test activating and deactivating plugins."""
        manager.load_from_class(TrackingPlugin, activate=False)

        # Activate
        manager.activate_plugin("tracking-plugin")
        assert manager.is_plugin_active("tracking-plugin")
        assert "register" in TrackingPlugin.calls

        # Deactivate
        manager.deactivate_plugin("tracking-plugin")
        assert not manager.is_plugin_active("tracking-plugin")
        assert "unregister" in TrackingPlugin.calls

    def test_unload_plugin(self, manager):
        """Test unloading a plugin."""
        manager.load_from_class(TrackingPlugin)
        assert manager.is_plugin_loaded("tracking-plugin")

        manager.unload_plugin("tracking-plugin")
        assert not manager.is_plugin_loaded("tracking-plugin")
        assert "teardown" in TrackingPlugin.calls

    def test_plugin_lifecycle_order(self, manager):
        """Test correct lifecycle method order."""
        manager.load_from_class(TrackingPlugin)
        manager.unload_plugin("tracking-plugin")

        # Should be: setup -> register -> unregister -> teardown
        assert TrackingPlugin.calls == ["setup", "register", "unregister", "teardown"]

    def test_load_plugin_not_found(self, manager):
        """Test loading non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            manager.load_plugin("non-existent")

    def test_load_plugin_setup_failure(self, manager):
        """Test handling of setup failure."""
        with pytest.raises(PluginLoadError, match="setup failed"):
            manager.load_from_class(FailingSetupPlugin)

    def test_get_plugin(self, manager):
        """Test getting a loaded plugin."""
        manager.load_from_class(SamplePlugin1)

        plugin = manager.get_plugin("test-plugin-1")
        assert plugin.name == "test-plugin-1"

    def test_get_plugin_not_found(self, manager):
        """Test getting non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            manager.get_plugin("non-existent")

    def test_get_plugins_by_type(self, manager):
        """Test filtering plugins by type."""
        manager.load_from_class(SamplePlugin1)  # CUSTOM
        manager.load_from_class(SamplePlugin2)  # VALIDATOR

        validators = manager.get_plugins_by_type(PluginType.VALIDATOR)
        assert len(validators) == 1
        assert validators[0].name == "test-plugin-2"

        custom = manager.get_plugins_by_type(PluginType.CUSTOM)
        assert len(custom) == 1

    def test_get_active_plugins(self, manager):
        """Test getting only active plugins."""
        manager.load_from_class(SamplePlugin1)
        manager.load_from_class(SamplePlugin2, activate=False)

        active = manager.get_active_plugins()
        # SamplePlugin1 is auto-activated, SamplePlugin2 is not
        assert len(active) == 1
        assert active[0].name == "test-plugin-1"

    def test_enable_disable_plugin(self, manager):
        """Test enabling and disabling plugins."""
        manager.load_from_class(SamplePlugin1)

        manager.disable_plugin("test-plugin-1")
        assert not manager.is_plugin_active("test-plugin-1")

        manager.enable_plugin("test-plugin-1")
        assert manager.is_plugin_active("test-plugin-1")

    def test_plugin_config_management(self, manager):
        """Test plugin configuration management."""
        config = PluginConfig(priority=50, settings={"key": "value"})
        manager.set_plugin_config("test-plugin-1", config)

        retrieved = manager.get_plugin_config("test-plugin-1")
        assert retrieved.priority == 50
        assert retrieved.settings == {"key": "value"}

    def test_load_all(self, manager):
        """Test loading all discovered plugins."""
        # Add plugins to discovered
        manager._discovered_classes["test-plugin-1"] = SamplePlugin1
        manager._discovered_classes["test-plugin-2"] = SamplePlugin2

        loaded = manager.load_all()
        assert len(loaded) == 2
        assert manager.is_plugin_loaded("test-plugin-1")
        assert manager.is_plugin_loaded("test-plugin-2")

    def test_unload_all(self, manager):
        """Test unloading all plugins."""
        manager.load_from_class(SamplePlugin1)
        manager.load_from_class(SamplePlugin2)

        manager.unload_all()
        assert len(manager.registry) == 0

    def test_shutdown(self, manager):
        """Test manager shutdown."""
        manager.load_from_class(SamplePlugin1)
        manager.shutdown()

        assert len(manager.registry) == 0
        assert len(manager._discovered_classes) == 0

    def test_context_manager(self):
        """Test manager as context manager."""
        with PluginManager() as manager:
            manager.load_from_class(SamplePlugin1)
            assert manager.is_plugin_loaded("test-plugin-1")
        # After exit, should be cleaned up


class TestPluginManagerDependencies:
    """Tests for plugin dependency management."""

    def test_dependency_check_success(self, manager):
        """Test loading plugin with satisfied dependencies."""
        # Load dependency first
        manager.load_from_class(SamplePlugin1)

        # Now load dependent plugin
        plugin = manager.load_from_class(SamplePlugin2)
        assert plugin.state == PluginState.ACTIVE

    def test_resolve_load_order(self, manager):
        """Test resolving plugin load order."""
        manager._discovered_classes["test-plugin-1"] = SamplePlugin1
        manager._discovered_classes["test-plugin-2"] = SamplePlugin2

        order = manager.resolve_load_order()

        # test-plugin-1 should come before test-plugin-2
        assert order.index("test-plugin-1") < order.index("test-plugin-2")


class TestPluginManagerConfig:
    """Tests for PluginManagerConfig."""

    def test_default_config(self):
        """Test default manager configuration."""
        config = PluginManagerConfig()
        assert config.scan_entrypoints is True
        assert config.auto_load is False
        assert config.auto_activate is True
        assert config.strict_dependencies is True
        assert config.check_compatibility is True

    def test_custom_config(self):
        """Test custom manager configuration."""
        config = PluginManagerConfig(
            scan_entrypoints=False,
            auto_load=True,
            auto_activate=False,
        )
        manager = PluginManager(config)
        assert manager._config.scan_entrypoints is False
        manager.shutdown()


class TestGlobalPluginManager:
    """Tests for global plugin manager functions."""

    def test_get_plugin_manager(self):
        """Test getting global manager."""
        reset_plugin_manager()
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is manager2
        reset_plugin_manager()

    def test_reset_plugin_manager(self):
        """Test resetting global manager."""
        manager1 = get_plugin_manager()
        manager1.load_from_class(SamplePlugin1)

        reset_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager2 is not manager1
        assert len(manager2.registry) == 0
        reset_plugin_manager()
