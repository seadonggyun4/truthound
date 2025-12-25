"""Tests for PluginRegistry."""

from __future__ import annotations

import pytest
import threading
from concurrent.futures import ThreadPoolExecutor

from truthound.plugins import (
    Plugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    PluginState,
    PluginRegistry,
    PluginError,
    PluginNotFoundError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockPluginA(Plugin[PluginConfig]):
    """Test plugin A."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="plugin-a",
            version="1.0.0",
            plugin_type=PluginType.VALIDATOR,
            tags=("testing", "validation"),
        )

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass


class MockPluginB(Plugin[PluginConfig]):
    """Test plugin B."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="plugin-b",
            version="2.0.0",
            plugin_type=PluginType.REPORTER,
            tags=("testing", "reporting"),
        )

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass


class MockPluginC(Plugin[PluginConfig]):
    """Test plugin C."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="plugin-c",
            version="1.0.0",
            plugin_type=PluginType.VALIDATOR,
            tags=("production",),
        )

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return PluginRegistry()


# =============================================================================
# Registry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_plugin(self, registry):
        """Test registering a plugin."""
        plugin = MockPluginA()
        registry.register(plugin)

        assert len(registry) == 1
        assert "plugin-a" in registry

    def test_register_duplicate_raises(self, registry):
        """Test that registering duplicate raises error."""
        plugin = MockPluginA()
        registry.register(plugin)

        with pytest.raises(PluginError, match="already registered"):
            registry.register(plugin)

    def test_unregister_plugin(self, registry):
        """Test unregistering a plugin."""
        plugin = MockPluginA()
        registry.register(plugin)

        unregistered = registry.unregister("plugin-a")
        assert unregistered is plugin
        assert "plugin-a" not in registry

    def test_unregister_not_found(self, registry):
        """Test unregistering non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            registry.unregister("non-existent")

    def test_get_plugin(self, registry):
        """Test getting a plugin by name."""
        plugin = MockPluginA()
        registry.register(plugin)

        retrieved = registry.get("plugin-a")
        assert retrieved is plugin

    def test_get_not_found(self, registry):
        """Test getting non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            registry.get("non-existent")

    def test_get_or_none(self, registry):
        """Test get_or_none method."""
        plugin = MockPluginA()
        registry.register(plugin)

        assert registry.get_or_none("plugin-a") is plugin
        assert registry.get_or_none("non-existent") is None

    def test_get_by_type(self, registry):
        """Test getting plugins by type."""
        registry.register(MockPluginA())  # VALIDATOR
        registry.register(MockPluginB())  # REPORTER
        registry.register(MockPluginC())  # VALIDATOR

        validators = registry.get_by_type(PluginType.VALIDATOR)
        assert len(validators) == 2

        reporters = registry.get_by_type(PluginType.REPORTER)
        assert len(reporters) == 1

    def test_get_by_state(self, registry):
        """Test getting plugins by state."""
        plugin_a = MockPluginA()
        plugin_b = MockPluginB()
        plugin_a._state = PluginState.ACTIVE
        plugin_b._state = PluginState.LOADED

        registry.register(plugin_a)
        registry.register(plugin_b)

        active = registry.get_by_state(PluginState.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "plugin-a"

    def test_get_by_tag(self, registry):
        """Test getting plugins by tag."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())
        registry.register(MockPluginC())

        testing_plugins = registry.get_by_tag("testing")
        assert len(testing_plugins) == 2

        production_plugins = registry.get_by_tag("production")
        assert len(production_plugins) == 1

    def test_get_active(self, registry):
        """Test get_active method."""
        plugin = MockPluginA()
        plugin._state = PluginState.ACTIVE
        registry.register(plugin)

        inactive = MockPluginB()
        inactive._state = PluginState.LOADED
        registry.register(inactive)

        active = registry.get_active()
        assert len(active) == 1
        assert active[0].name == "plugin-a"

    def test_filter(self, registry):
        """Test filtering plugins."""
        plugin_a = MockPluginA()
        plugin_a._state = PluginState.ACTIVE
        plugin_a._config = PluginConfig(enabled=True)

        plugin_b = MockPluginB()
        plugin_b._state = PluginState.LOADED
        plugin_b._config = PluginConfig(enabled=False)

        registry.register(plugin_a)
        registry.register(plugin_b)

        # Filter by type and state
        result = registry.filter(
            plugin_type=PluginType.VALIDATOR,
            state=PluginState.ACTIVE,
        )
        assert len(result) == 1
        assert result[0].name == "plugin-a"

        # Filter by enabled
        enabled = registry.filter(enabled=True)
        assert len(enabled) == 1

    def test_update_state(self, registry):
        """Test updating plugin state."""
        plugin = MockPluginA()
        registry.register(plugin)

        assert plugin.state == PluginState.DISCOVERED

        registry.update_state("plugin-a", PluginState.ACTIVE)
        assert plugin.state == PluginState.ACTIVE

    def test_list_all(self, registry):
        """Test listing all plugins."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        all_plugins = registry.list_all()
        assert len(all_plugins) == 2
        assert "plugin-a" in all_plugins
        assert "plugin-b" in all_plugins

    def test_list_names(self, registry):
        """Test listing plugin names."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        names = registry.list_names()
        assert set(names) == {"plugin-a", "plugin-b"}

    def test_list_types(self, registry):
        """Test listing plugin types."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        types = registry.list_types()
        assert PluginType.VALIDATOR in types
        assert PluginType.REPORTER in types

    def test_list_tags(self, registry):
        """Test listing all tags."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        tags = registry.list_tags()
        assert "testing" in tags
        assert "validation" in tags
        assert "reporting" in tags

    def test_contains(self, registry):
        """Test contains method."""
        registry.register(MockPluginA())

        assert registry.contains("plugin-a") is True
        assert registry.contains("non-existent") is False
        assert "plugin-a" in registry

    def test_get_info(self, registry):
        """Test getting plugin info."""
        registry.register(MockPluginA())

        info = registry.get_info("plugin-a")
        assert info.name == "plugin-a"
        assert info.version == "1.0.0"

    def test_get_all_info(self, registry):
        """Test getting all plugin info."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        infos = registry.get_all_info()
        assert len(infos) == 2

    def test_clear(self, registry):
        """Test clearing registry."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        registry.clear()
        assert len(registry) == 0

    def test_iteration(self, registry):
        """Test iterating over registry."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        names = [p.name for p in registry]
        assert set(names) == {"plugin-a", "plugin-b"}

    def test_repr(self, registry):
        """Test registry string representation."""
        registry.register(MockPluginA())
        registry.register(MockPluginB())

        repr_str = repr(registry)
        assert "PluginRegistry" in repr_str
        assert "total=2" in repr_str


class TestPluginRegistryThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self):
        """Test concurrent plugin registration."""
        registry = PluginRegistry()
        errors = []

        class DynamicPlugin(Plugin[PluginConfig]):
            def __init__(self, name: str):
                super().__init__()
                self._name = name

            @property
            def info(self) -> PluginInfo:
                return PluginInfo(
                    name=self._name,
                    version="1.0.0",
                    plugin_type=PluginType.CUSTOM,
                )

            def setup(self) -> None:
                pass

            def teardown(self) -> None:
                pass

        def register_plugin(i):
            try:
                plugin = DynamicPlugin(f"plugin-{i}")
                registry.register(plugin)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(register_plugin, range(100)))

        assert len(errors) == 0
        assert len(registry) == 100

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        registry = PluginRegistry()
        registry.register(MockPluginA())

        errors = []

        def read_plugin():
            try:
                for _ in range(100):
                    registry.get_or_none("plugin-a")
                    list(registry)
            except Exception as e:
                errors.append(e)

        def write_plugin():
            try:
                for i in range(10):
                    class TempPlugin(Plugin[PluginConfig]):
                        @property
                        def info(self):
                            return PluginInfo(
                                name=f"temp-{threading.current_thread().ident}-{i}",
                                version="1.0.0",
                                plugin_type=PluginType.CUSTOM,
                            )

                        def setup(self):
                            pass

                        def teardown(self):
                            pass

                    try:
                        registry.register(TempPlugin())
                    except PluginError:
                        pass  # Ignore duplicate errors
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=read_plugin) for _ in range(5)
        ] + [
            threading.Thread(target=write_plugin) for _ in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
