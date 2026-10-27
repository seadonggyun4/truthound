"""Declared package versions and native plugin dependencies gate setup."""

import pytest

from truthound.plugins.base import (
    Plugin,
    PluginCompatibilityError,
    PluginDependencyError,
    PluginInfo,
    PluginType,
)
from truthound.plugins.manager import PluginManager, PluginManagerConfig


class Dependency(Plugin):
    @property
    def info(self):
        return PluginInfo(name="dependency", version="1.0.0", plugin_type=PluginType.CUSTOM)

    def setup(self):
        pass

    def teardown(self):
        pass


def test_discovered_dependency_is_not_a_loaded_dependency():
    class Consumer(Dependency):
        @property
        def info(self):
            return PluginInfo(
                name="consumer",
                version="1.0.0",
                plugin_type=PluginType.CUSTOM,
                dependencies=("dependency",),
            )

    manager = PluginManager()
    manager._discovered_classes = {"dependency": Dependency, "consumer": Consumer}
    with pytest.raises(PluginDependencyError):
        manager.load_plugin("consumer")
    manager.load_all()
    assert manager.is_plugin_loaded("consumer")
    manager.shutdown()


def test_python_requirement_uses_distribution_metadata_without_importing_it():
    class Consumer(Dependency):
        @property
        def info(self):
            return PluginInfo(
                name="consumer",
                version="1.0.0",
                plugin_type=PluginType.CUSTOM,
                python_dependencies=("polars>=1.0",),
            )

    manager = PluginManager()
    manager.load_from_class(Consumer)
    manager.shutdown()


def test_invalid_host_version_fails_strict_compatibility():
    manager = PluginManager(PluginManagerConfig(host_version="not-a-version"))
    with pytest.raises(PluginCompatibilityError):
        manager.load_from_class(Dependency)


def test_dependency_versions_and_optional_policy_are_enforced():
    class Consumer(Dependency):
        @property
        def info(self):
            return PluginInfo(
                name="consumer",
                version="1.0.0",
                plugin_type=PluginType.CUSTOM,
                dependencies=("dependency>=2",),
                optional_dependencies=("optional",),
            )

    manager = PluginManager()
    manager.load_from_class(Dependency)
    with pytest.raises(PluginDependencyError):
        manager.load_from_class(Consumer)
    manager.shutdown()
    relaxed = PluginManager(PluginManagerConfig(strict_version_check=False))
    relaxed.load_from_class(Dependency)
    relaxed.load_from_class(Consumer)
    relaxed.shutdown()
    required_optional = PluginManager(
        PluginManagerConfig(strict_version_check=False, allow_missing_optional=False)
    )
    required_optional.load_from_class(Dependency)
    with pytest.raises(PluginDependencyError):
        required_optional.load_from_class(Consumer)
    required_optional.shutdown()


def test_installed_python_version_mismatch_is_not_success():
    class Consumer(Dependency):
        @property
        def info(self):
            return PluginInfo(
                name="consumer",
                version="1.0.0",
                plugin_type=PluginType.CUSTOM,
                python_dependencies=("polars>=999",),
            )

    with pytest.raises(PluginDependencyError):
        PluginManager().load_from_class(Consumer)


def test_manager_captures_admission_configuration():
    config = PluginManagerConfig(host_version="not-a-version")
    manager = PluginManager(config)
    config.check_compatibility = False
    with pytest.raises(PluginCompatibilityError):
        manager.load_from_class(Dependency)


def test_discovery_name_mismatch_rejects_before_setup():
    from truthound.plugins.base import PluginLoadError

    calls = []

    class Candidate(Dependency):
        def setup(self):
            calls.append("setup")

    manager = PluginManager()
    manager._discovered_classes["different-name"] = Candidate
    try:
        with pytest.raises(PluginLoadError):
            manager.load_plugin("different-name")
        assert calls == []
        assert not manager.is_plugin_loaded("dependency")
        assert not manager.is_plugin_loaded("different-name")
    finally:
        manager.shutdown()
