"""Tests for plugin base classes and types."""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from truthound.plugins import (
    Plugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    PluginState,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginDependencyError,
    PluginCompatibilityError,
    ValidatorPlugin,
    ReporterPlugin,
    HookPlugin,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleTestPlugin(Plugin[PluginConfig]):
    """Simple test plugin implementation."""

    def __init__(self, config: PluginConfig | None = None):
        super().__init__(config)
        self.setup_called = False
        self.teardown_called = False
        self.register_called = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="simple-test",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            description="A simple test plugin",
            author="Test Author",
        )

    def setup(self) -> None:
        self.setup_called = True

    def teardown(self) -> None:
        self.teardown_called = True

    def register(self, manager) -> None:
        self.register_called = True


class FailingPlugin(Plugin[PluginConfig]):
    """Plugin that fails during setup."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="failing-plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )

    def setup(self) -> None:
        raise RuntimeError("Intentional setup failure")

    def teardown(self) -> None:
        pass


# =============================================================================
# PluginInfo Tests
# =============================================================================


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_valid_plugin_info(self):
        """Test creating valid plugin info."""
        info = PluginInfo(
            name="my-plugin",
            version="1.0.0",
            plugin_type=PluginType.VALIDATOR,
            description="A test plugin",
        )
        assert info.name == "my-plugin"
        assert info.version == "1.0.0"
        assert info.plugin_type == PluginType.VALIDATOR

    def test_plugin_name_validation(self):
        """Test that invalid plugin names are rejected."""
        # Name cannot be empty
        with pytest.raises(ValueError, match="cannot be empty"):
            PluginInfo(name="", version="1.0.0", plugin_type=PluginType.CUSTOM)

        # Name must start with lowercase
        with pytest.raises(ValueError, match="Invalid plugin name"):
            PluginInfo(name="MyPlugin", version="1.0.0", plugin_type=PluginType.CUSTOM)

        # Name cannot contain spaces
        with pytest.raises(ValueError, match="Invalid plugin name"):
            PluginInfo(name="my plugin", version="1.0.0", plugin_type=PluginType.CUSTOM)

    def test_valid_plugin_names(self):
        """Test valid plugin name formats."""
        valid_names = [
            "my-plugin",
            "my_plugin",
            "myplugin",
            "plugin123",
            "a",
        ]
        for name in valid_names:
            info = PluginInfo(name=name, version="1.0.0", plugin_type=PluginType.CUSTOM)
            assert info.name == name

    def test_version_required(self):
        """Test that version is required."""
        with pytest.raises(ValueError, match="version cannot be empty"):
            PluginInfo(name="test", version="", plugin_type=PluginType.CUSTOM)

    def test_compatibility_check(self):
        """Test version compatibility checking."""
        info = PluginInfo(
            name="test",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            min_truthound_version="0.1.0",
            max_truthound_version="2.0.0",
        )

        # Compatible version
        assert info.is_compatible("1.0.0") is True
        assert info.is_compatible("0.5.0") is True
        assert info.is_compatible("1.9.9") is True

        # Incompatible versions
        assert info.is_compatible("0.0.1") is False
        assert info.is_compatible("2.0.1") is False
        assert info.is_compatible("3.0.0") is False

    def test_compatibility_no_constraints(self):
        """Test compatibility with no version constraints."""
        info = PluginInfo(
            name="test",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )
        assert info.is_compatible("999.0.0") is True

    def test_immutability(self):
        """Test that PluginInfo is immutable."""
        info = PluginInfo(name="test", version="1.0.0", plugin_type=PluginType.CUSTOM)
        with pytest.raises(AttributeError):
            info.name = "changed"  # type: ignore


# =============================================================================
# PluginConfig Tests
# =============================================================================


class TestPluginConfig:
    """Tests for PluginConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PluginConfig()
        assert config.enabled is True
        assert config.priority == 100
        assert config.settings == {}
        assert config.auto_load is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PluginConfig(
            enabled=False,
            priority=50,
            settings={"key": "value"},
            auto_load=False,
        )
        assert config.enabled is False
        assert config.priority == 50
        assert config.settings == {"key": "value"}
        assert config.auto_load is False

    def test_config_merge(self):
        """Test merging configurations."""
        base = PluginConfig(
            priority=100,
            settings={"a": 1, "b": 2},
        )
        override = PluginConfig(
            priority=50,
            settings={"b": 3, "c": 4},
        )

        merged = base.merge(override)

        assert merged.priority == 50  # Override takes precedence
        assert merged.settings == {"a": 1, "b": 3, "c": 4}  # Merged settings


# =============================================================================
# Plugin Base Class Tests
# =============================================================================


class TestPlugin:
    """Tests for Plugin base class."""

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        plugin = SimpleTestPlugin()
        assert plugin.name == "simple-test"
        assert plugin.version == "1.0.0"
        assert plugin.plugin_type == PluginType.CUSTOM
        assert plugin.state == PluginState.DISCOVERED

    def test_plugin_with_config(self):
        """Test plugin with custom config."""
        config = PluginConfig(priority=50)
        plugin = SimpleTestPlugin(config)
        assert plugin.config.priority == 50

    def test_plugin_setup_teardown(self):
        """Test setup and teardown methods."""
        plugin = SimpleTestPlugin()
        assert not plugin.setup_called
        assert not plugin.teardown_called

        plugin.setup()
        assert plugin.setup_called

        plugin.teardown()
        assert plugin.teardown_called

    def test_plugin_config_change(self):
        """Test configuration change notification."""
        plugin = SimpleTestPlugin()
        plugin._state = PluginState.ACTIVE

        old_config = plugin.config
        new_config = PluginConfig(priority=25)

        # Should trigger on_config_change
        plugin.config = new_config
        assert plugin.config.priority == 25

    def test_plugin_health_check(self):
        """Test health check method."""
        plugin = SimpleTestPlugin()
        assert plugin.health_check() is False  # Not active

        plugin._state = PluginState.ACTIVE
        assert plugin.health_check() is True

    def test_plugin_repr(self):
        """Test plugin string representation."""
        plugin = SimpleTestPlugin()
        repr_str = repr(plugin)
        assert "SimpleTestPlugin" in repr_str
        assert "simple-test" in repr_str


# =============================================================================
# PluginType Tests
# =============================================================================


class TestPluginType:
    """Tests for PluginType enumeration."""

    def test_all_plugin_types(self):
        """Test all plugin types are defined."""
        expected = [
            "validator",
            "reporter",
            "datasource",
            "profiler",
            "hook",
            "transformer",
            "exporter",
            "custom",
        ]
        actual = [t.value for t in PluginType]
        assert set(expected) == set(actual)


# =============================================================================
# PluginState Tests
# =============================================================================


class TestPluginState:
    """Tests for PluginState enumeration."""

    def test_all_plugin_states(self):
        """Test all plugin states are defined."""
        expected = [
            "discovered",
            "loading",
            "loaded",
            "active",
            "inactive",
            "error",
            "unloading",
        ]
        actual = [s.value for s in PluginState]
        assert set(expected) == set(actual)


# =============================================================================
# Exception Tests
# =============================================================================


class TestPluginExceptions:
    """Tests for plugin exceptions."""

    def test_plugin_error(self):
        """Test base PluginError."""
        error = PluginError("Test error", plugin_name="test")
        assert str(error) == "Test error"
        assert error.plugin_name == "test"

    def test_plugin_load_error(self):
        """Test PluginLoadError."""
        error = PluginLoadError("Load failed", plugin_name="test")
        assert isinstance(error, PluginError)

    def test_plugin_not_found_error(self):
        """Test PluginNotFoundError."""
        error = PluginNotFoundError("Not found", plugin_name="missing")
        assert isinstance(error, PluginError)

    def test_plugin_dependency_error(self):
        """Test PluginDependencyError."""
        error = PluginDependencyError(
            "Missing deps",
            plugin_name="test",
            missing_deps=["dep1", "dep2"],
        )
        assert error.missing_deps == ["dep1", "dep2"]

    def test_plugin_compatibility_error(self):
        """Test PluginCompatibilityError."""
        error = PluginCompatibilityError(
            "Incompatible",
            plugin_name="test",
            required_version="2.0.0",
            current_version="1.0.0",
        )
        assert error.required_version == "2.0.0"
        assert error.current_version == "1.0.0"
