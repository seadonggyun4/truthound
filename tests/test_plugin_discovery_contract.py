"""Real file discovery preserves imports and reports invalid sources explicitly."""

import inspect

import pytest

from truthound.plugins.discovery import PluginDiscovery

PLUGIN = """from truthound.plugins.base import Plugin, PluginInfo, PluginType
class LocalPlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(name="local-plugin", version="1.0.0", plugin_type=PluginType.CUSTOM)
    def setup(self):
        pass
    def teardown(self):
        pass
"""


def test_discovered_class_keeps_source_identity_for_signed_admission(tmp_path):
    source = tmp_path / "plugin.py"
    source.write_text(PLUGIN)
    discovered = PluginDiscovery(scan_entrypoints=False).discover_directory(tmp_path)
    assert inspect.getsourcefile(discovered["local-plugin"]) == str(source)


def test_import_failure_is_not_empty_success_and_partial_results_remain_visible(tmp_path):
    (tmp_path / "a_good.py").write_text(PLUGIN)
    (tmp_path / "z_bad.py").write_text('raise RuntimeError("synthetic-private-source-detail")\n')
    discovery = PluginDiscovery(scan_entrypoints=False)
    with pytest.raises(ExceptionGroup) as failure:
        discovery.discover_directory(tmp_path)
    assert "local-plugin" in discovery.discovered_plugins
    assert "synthetic-private-source-detail" not in str(failure.value)


def test_directory_relative_imports_execute_in_owned_namespace(tmp_path):
    (tmp_path / "_helper.py").write_text('VERSION = "1.0.0"\n')
    (tmp_path / "plugin.py").write_text(
        "from ._helper import VERSION\n" + PLUGIN.replace('version="1.0.0"', "version=VERSION")
    )
    discovered = PluginDiscovery(scan_entrypoints=False).discover_directory(tmp_path)
    assert discovered["local-plugin"]().info.version == "1.0.0"


def test_failed_package_initializer_is_not_reexecuted_for_each_child(tmp_path):
    marker = tmp_path / "attempts.txt"
    initializer = tmp_path / "__init__.py"
    initializer.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(marker)!r})\n"
        'p.write_text((p.read_text() if p.exists() else "") + "attempt\\n")\n'
        'raise RuntimeError("initializer failure")\n'
    )
    (tmp_path / "a.py").write_text(PLUGIN)
    (tmp_path / "b.py").write_text(PLUGIN)
    with PluginDiscovery(scan_entrypoints=False) as discovery, pytest.raises(ExceptionGroup):
        discovery.discover_directory(tmp_path)
    assert marker.read_text().splitlines() == ["attempt"]


def test_package_only_discovery_and_owner_close_release_modules(tmp_path):
    import sys

    (tmp_path / "__init__.py").write_text(PLUGIN)
    with PluginDiscovery(scan_entrypoints=False) as discovery:
        discovered = discovery.discover_directory(tmp_path)
        module = discovered["local-plugin"].__module__
        assert module in sys.modules
    assert module not in sys.modules


def test_explicit_rediscovery_does_not_reuse_stale_relative_import_bytecode(tmp_path):
    import os

    helper = tmp_path / "_helper.py"
    helper.write_text('VERSION = "1.0.0"\n')
    original_stat = helper.stat()
    (tmp_path / "plugin.py").write_text(
        "from ._helper import VERSION\n" + PLUGIN.replace('version="1.0.0"', "version=VERSION")
    )
    with PluginDiscovery(scan_entrypoints=False) as discovery:
        original = discovery.discover_directory(tmp_path)["local-plugin"]
        helper.write_text('VERSION = "2.0.0"\n')
        os.utime(helper, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        replacement = discovery.discover_directory(tmp_path)["local-plugin"]
        assert original().info.version == "1.0.0"
        assert replacement().info.version == "2.0.0"
