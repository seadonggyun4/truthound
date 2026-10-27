"""Actual same-manager stable ports feed the canonical kernel and reporters."""

import asyncio

import polars as pl
import pytest

from truthound.core import CheckSpec, build_validation_asset
from truthound.plugins import PluginManager
from truthound.plugins.hooks import HookType
from truthound.reporters.json_reporter import JSONReporter
from truthound.validators import get_validator

pytestmark = pytest.mark.contract


def native_manager(events):
    manager = PluginManager()

    class Factory:
        name = "local-checks"

        def build_check_specs(self):
            events.append("factory")
            return [
                CheckSpec(
                    id="configured-check",
                    name="selected-positive",
                    category="range",
                    factory=lambda: get_validator("between")(columns=["x"], min_value=0),
                )
            ]

    class Provider:
        name = "local-data"

        def get_asset(self):
            events.append("provider")
            return build_validation_asset(pl.DataFrame({"x": [-1, 1]}))

    manager.register_check_factory("checks", Factory())
    manager.register_data_asset_provider("data", Provider())
    manager.register_reporter("json", JSONReporter())

    async def before(**kwargs):
        events.append("before")
        assert kwargs["checks"][0].id == "configured-check"

    async def found(issue, validator, **kwargs):
        events.append("issue")
        assert validator.id == "configured-check"

    manager.register_hook(HookType.BEFORE_VALIDATION, before)
    manager.register_hook(HookType.ON_ISSUE_FOUND, found)
    return manager


def test_native_same_manager_provider_checks_reporter_and_hook_outcomes():
    events = []
    manager = native_manager(events)
    result = manager.validate(["checks"], asset_provider="data", result_format="complete")
    assert result.run.row_count == 2 and result.run.issues[0].count == 1
    assert not result.hooks_failed
    assert events == ["provider", "factory", "before", "issue"]
    rendered = manager.render("json", result.run)
    import json
    from datetime import datetime

    actual = json.loads(rendered.content)
    direct = json.loads(manager.get_reporter("json").render(result.run))
    datetime.strptime(
        actual.pop("generated_at"), manager.get_reporter("json").config.timestamp_format
    )
    direct.pop("generated_at")
    assert actual == direct
    assert not rendered.hooks_failed


def test_native_workflow_hook_failure_stays_visible_without_relabeling_validation(caplog):
    manager = native_manager([])

    def failing(**kwargs):
        raise RuntimeError("synthetic-private-workflow-error")

    manager.register_hook(HookType.AFTER_VALIDATION, failing)
    result = manager.validate(["checks"], data=pl.DataFrame({"x": [1]}))
    assert not result.run.issues and result.hooks_failed
    assert result.hooks["after_validation"][0].error_type == "RuntimeError"
    assert "synthetic-private-workflow-error" not in caplog.text


def test_native_sync_workflow_rejects_running_loop_before_provider_or_factory_effects():
    async def run():
        events = []
        manager = native_manager(events)
        with pytest.raises(RuntimeError, match="async"):
            manager.validate(["checks"], asset_provider="data")
        assert not events

    asyncio.run(run())


def test_native_structural_v3_reporter_port_is_usable_without_framework_inheritance():
    manager = native_manager([])

    class LocalReporter:
        contract_version = 3

        def render(self, run):
            return run.suite_name

    manager.register_reporter("local-structural", LocalReporter)
    bundle = manager.validate(["checks"], data=pl.DataFrame({"x": [1]}))
    assert manager.get_reporter("local-structural")().render(bundle.run) == "plugin-suite"
    assert manager.render("local-structural", bundle.run).content == "plugin-suite"


def test_native_configured_reporter_port_does_not_poison_global_constructor_registry():
    from truthound.reporters import factory

    key = "local-configured-instance"
    reporter = JSONReporter()
    manager = PluginManager()
    try:
        manager.register_reporter(key, reporter)
        assert manager.get_reporter(key) is reporter
        with pytest.raises(factory.ReporterError):
            factory.get_reporter(key)
    finally:
        factory._reporter_registry.pop(key, None)


def test_revoked_native_unload_hook_cannot_prevent_owned_plugin_cleanup():
    from truthound.plugins import Plugin, PluginInfo, PluginType

    events = []
    manager = PluginManager()

    class ResourcePlugin(Plugin):
        @property
        def info(self):
            return PluginInfo(name="local-resource", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            events.append("setup")

        def register(self, manager):
            events.append("register")

        def unregister(self, manager):
            events.append("unregister")

        def teardown(self):
            events.append("closed")

    plugin = manager.load_from_class(ResourcePlugin, activate=True)

    def guard():
        raise PermissionError("revoked")

    manager.register_hook(HookType.ON_PLUGIN_UNLOAD, lambda **kwargs: events.append("unload-hook"))
    manager.set_execution_guard(guard)
    with pytest.raises(PermissionError, match="revoked"):
        manager.unload_plugin(plugin.name)
    assert events == ["setup", "register", "unregister", "closed"]
    assert not manager.is_plugin_loaded(plugin.name)


def test_native_cleanup_keeps_failures_visible_and_omits_private_exception_logs(caplog):
    from truthound.plugins import Plugin, PluginInfo, PluginType

    class FailedCleanup(Plugin):
        @property
        def info(self):
            return PluginInfo(name="failed-cleanup", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            pass

        def register(self, manager):
            pass

        def unregister(self, manager):
            raise ValueError("synthetic-private-unregister-detail")

        def teardown(self):
            raise RuntimeError("synthetic-private-teardown-detail")

    manager = PluginManager()
    plugin = manager.load_from_class(FailedCleanup, activate=True)
    manager.unload_plugin(plugin.name)
    assert "synthetic-private-unregister-detail" not in caplog.text
    assert "synthetic-private-teardown-detail" not in caplog.text
    plugin = manager.load_from_class(FailedCleanup, activate=True)
    receipt = manager.unload_plugin(plugin.name, detailed=True)
    assert not receipt.success and receipt.removed
    assert not receipt.deactivated and not receipt.teardown_completed
    assert receipt.error_types == ("ValueError", "RuntimeError")


def test_native_port_removal_preserves_another_managers_replacement_reporter():
    from truthound.reporters.factory import get_reporter, unregister_reporter

    class First:
        contract_version = 3

        def render(self, run):
            return "first:" + run.suite_name

    class Second(First):
        def render(self, run):
            return "second:" + run.suite_name

    class Invalid(First):
        contract_version = "invalid"

    name = "native-manager-owned"
    first, second = PluginManager(), PluginManager()
    try:
        first.register_reporter(name, First)
        with pytest.raises(ValueError):
            first.register_reporter(name, Invalid)
        assert first.get_reporter(name) is First
        second.register_reporter(name, Second)
        assert first.unregister_reporter(name)
        run = native_manager([]).validate(["checks"], data=pl.DataFrame({"x": [1]})).run
        assert get_reporter(name).render(run) == "second:plugin-suite"
        assert second.unregister_reporter(name)
        assert not second.unregister_reporter(name)
    finally:
        unregister_reporter(name)
    manager = native_manager([])
    assert manager.unregister_check_factory("checks")
    assert manager.unregister_data_asset_provider("data")
    manager.register_capability("local", object())
    assert manager.unregister_capability("local")
    assert manager.get_check_factory("checks") is None
    assert manager.get_data_asset_provider("data") is None
    assert manager.get_capability("local") is None
