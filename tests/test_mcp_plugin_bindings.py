"""Real Core plugin entrypoint/lifecycle from an installed wheel in a private worker.

The synthetic distribution lives only in a private test venv. No production
environment is installed into or changed; no plugin is loaded before approval.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import venv
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from truthound.mcp.approval import ApprovalStore
from truthound.mcp.plugin_bindings import approved_capabilities, fingerprint
from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


# Wrongly typed operator objects can survive native construction. These are
# separate from natural failures admitted by the ordinary JSON contract.
OBJECT_FAILURES = [
    ("column_count", "min_count", "wrong", TypeError),
    ("table_row_count_range", "min_rows", "wrong", TypeError),
    ("column_exists", "columns", [["x"]], TypeError),
    ("table_column_types", "column_types", {"x": None}, TypeError),
    ("table_schema_compare", "reference_table", object(), AttributeError),
]


@pytest.mark.parametrize(
    "name,parameter,value,error", OBJECT_FAILURES, ids=[r[0] for r in OBJECT_FAILURES]
)
def test_native_runtime_type_error_is_rejected_before_ordinary_mcp_execution(
    runtime, name, parameter, value, error
):
    import polars as pl

    from truthound.mcp.validator_wire import settings
    from truthound.validators import get_validator

    validator = get_validator(name)(**{parameter: value})
    with pytest.raises(error):
        validator.validate(pl.DataFrame({"x": [-1, 1]}).lazy())
    # object() itself cannot be JSON. A JSON object is also not an asset handle.
    wire_value = {} if parameter == "reference_table" else value
    with pytest.raises(MCPError, match="invalid_arguments"):
        settings(name, {parameter: wire_value})
    adapter, _, marker, _, asset = runtime
    failed = invoke(
        adapter,
        "assets_validate",
        asset_id=asset,
        checks=[{"validator": name, "settings": {parameter: wire_value}}],
    )
    assert failed["execution_status"] == "failed"
    assert failed["issues"] == [{"code": "invalid_arguments"}]
    assert not failed.get("data") and not marker.exists()


@pytest.mark.parametrize(
    "name,parameter,value,error", OBJECT_FAILURES, ids=[r[0] for r in OBJECT_FAILURES]
)
def test_approved_object_runtime_failure_matches_core_without_reexecution(
    runtime, tmp_path, name, parameter, value, error
):
    import base64

    import polars as pl

    import truthound as th
    from truthound.mcp.operations import _context
    from truthound.mcp.tables import decode
    from truthound.validators import get_validator

    native = get_validator(name)(**{parameter: value})
    with pytest.raises(error):
        native.validate(pl.DataFrame({"x": [-1, 1]}).lazy())
    direct = th.check(
        pl.DataFrame({"x": [-1, 1]}),
        validators=[native],
        result_format="complete",
        context=_context(str(tmp_path / "direct/input.ipc")),
    ).to_dict()
    assert direct["execution_issues"]
    adapter, control, marker, _, asset = runtime
    path = control / "bindings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    capability = "bad-" + name
    config["plugins"]["fixture"]["capabilities"][capability] = capability
    config["validators"]["invalid-object"] = {
        "validator": name,
        "object_parameters": {parameter: capability},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    request = dict(
        asset_id=asset,
        checks=[{"validator": name, "binding_id": "invalid-object"}],
        result_format="complete",
        idempotency_key="object-failure-" + name,
    )
    pending = invoke(adapter, "assets_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    assert "installed_extension_execute" in pending["data"]["effects"]
    assert not marker.exists()
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    failed = invoke(adapter, "assets_validate", **request)
    assert failed["execution_status"] == "failed", failed
    assert failed["issues"] == [{"code": "core_execution_failed"}]
    identity = failed["data"]["artifact"]["resource_id"]
    chunks, offset = [], 0
    while offset is not None:
        part = invoke(adapter, "artifacts_read", resource_id=identity, offset=offset)
        assert part["execution_status"] == "succeeded"
        chunks.append(base64.b64decode(part["data"]["base64"]))
        offset = part["data"]["next_offset"]
    actual = decode(json.loads(b"".join(chunks)))
    for key in (
        "checks",
        "issues",
        "execution_issues",
        "row_count",
        "column_count",
        "result_format",
        "execution_mode",
        "planned_execution_mode",
        "suite_name",
        "source",
    ):
        assert actual[key] == direct[key], key
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]
    replay = invoke(adapter, "assets_validate", **request)
    assert replay["execution_status"] == "failed"
    assert replay["data"]["replayed"]
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]


@pytest.mark.parametrize(
    ("polarity", "values"),
    [("positive", [1, 2]), ("negative", [-1, 2]), ("caught_error", [None, 2])],
    ids=["positive-row_level", "negative-row_level", "caught_error-row_level"],
)
def test_row_level_algorithm_matches_core_with_full_canonical_evidence(
    runtime, tmp_path, polarity, values
):
    import base64

    import polars as pl

    import truthound as th
    from truthound.mcp.operations import _context
    from truthound.mcp.tables import decode
    from truthound.validators import get_validator

    def callback(row):
        return row["x"] > 0

    if polarity == "caught_error":
        with pytest.raises(TypeError):
            callback({"x": None})
    direct = th.check(
        pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)}),
        validators=[get_validator("row_level")(row_validator=callback, columns=["x"])],
        result_format="complete",
        context=_context(str(tmp_path / "direct/input.ipc")),
    ).to_dict()
    assert not direct["execution_issues"]  # Core catches callback exceptions as failed rows.
    assert bool(direct["issues"]) == (polarity != "positive")
    adapter, control, marker, _, _ = runtime
    asset = invoke(
        adapter,
        "assets_inline",
        columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": values}],
    )
    assert asset["execution_status"] == "succeeded"
    arguments = {
        "asset_id": asset["data"]["resource_id"],
        "checks": [{"validator": "row_level", "binding_id": "positive-rows"}],
        "idempotency_key": "row-parity-" + polarity,
        "result_format": "complete",
    }
    pending = invoke(adapter, "assets_validate", **arguments)
    assert pending["issues"] == [{"code": "approval_required"}]
    assert not marker.exists()
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    response = invoke(adapter, "assets_validate", **arguments)
    assert response["execution_status"] == "succeeded", response
    identity = response["data"]["artifact"]["resource_id"]
    chunks, offset = [], 0
    while offset is not None:
        chunk = invoke(adapter, "artifacts_read", resource_id=identity, offset=offset)
        assert chunk["execution_status"] == "succeeded"
        chunks.append(base64.b64decode(chunk["data"]["base64"]))
        offset = chunk["data"]["next_offset"]
    actual = decode(json.loads(b"".join(chunks)))
    for key in (
        "checks",
        "issues",
        "execution_issues",
        "row_count",
        "column_count",
        "result_format",
        "execution_mode",
        "planned_execution_mode",
        "suite_name",
        "source",
    ):
        assert actual[key] == direct[key], key
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]


def invoke(adapter, name, **arguments):
    return asyncio.run(adapter.call_operation_async(name, arguments))


def prepare_source(runtime, monkeypatch, capability="source-factory"):
    from types import SimpleNamespace

    from truthound.mcp import model_process

    adapter, control, marker, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    path = control / "bindings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["plugins"]["fixture"]["capabilities"]["source-factory"] = capability
    config["sources"] = {"approved-source": "source-factory"}
    path.write_text(json.dumps(config), encoding="utf-8")
    request = {"binding_id": "approved-source", "idempotency_key": "approved-source-open"}
    pending = invoke(adapter, "sources_open_binding", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert not marker.exists()
    assert set(pending["data"]["effects"]) == {
        "installed_extension_execute",
        "native_datasource_connect",
        "native_datasource_read",
        "native_datasource_close",
    }
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    return request


def source_artifact(adapter, identity):
    import base64

    from truthound.mcp.tables import decode

    chunks, offset = [], 0
    while offset is not None:
        response = invoke(adapter, "artifacts_read", resource_id=identity, offset=offset)
        assert response["execution_status"] == "succeeded", response
        chunks.append(base64.b64decode(response["data"]["base64"]))
        offset = response["data"]["next_offset"]
    return decode(json.loads(b"".join(chunks)))


def test_approved_native_source_preserves_engine_results_sampling_and_owned_cleanup(
    runtime, monkeypatch, tmp_path
):
    import truthound as th
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig
    from truthound.mcp.operations import _context
    from truthound.mcp.tables import decode
    from truthound.validators import get_validator

    adapter, _, marker, _, _ = runtime
    request = prepare_source(runtime, monkeypatch)
    opened = invoke(adapter, "sources_open_binding", **request)
    assert opened["execution_status"] == "succeeded", opened
    source = opened["data"]["resource_id"]
    assert invoke(adapter, "sources_open_binding", **request) == opened
    worker = adapter._session._sources.workers[source]
    with SQLiteDataSource(
        database=str(tmp_path / "operator.db"),
        table="items",
        config=SQLiteDataSourceConfig(name="approved-source"),
    ) as native:
        direct = th.check(
            source=native,
            validators=[get_validator("null")(columns=["x"])],
            result_format="complete",
            context=_context(str(tmp_path / "direct/input.ipc")),
        ).to_dict()
        result = invoke(
            adapter,
            "sources_validate",
            source_id=source,
            checks=[{"validator": "null", "settings": {"columns": ["x"]}}],
            result_format="complete",
        )
        assert result["execution_status"] == "succeeded", result
        actual = source_artifact(adapter, result["data"]["artifact"]["resource_id"])
        assert result["provenance"].get("input_sha256") is None
        assert result["provenance"].get("input_identity") is None
        for key in (
            "checks",
            "issues",
            "execution_issues",
            "source",
            "row_count",
            "column_count",
            "execution_mode",
            "planned_execution_mode",
            "result_format",
        ):
            assert actual[key] == direct[key], key
        assert direct["issues"]
        inspected = invoke(adapter, "sources_inspect", source_id=source)
        info = source_artifact(adapter, inspected["data"]["resource_id"])
        assert info["config"] is None and info["name"] == native.name
        assert info["schema"] == {key: value.value for key, value in native.schema.items()}
        assert info["row_count"] == native.row_count
        assert str(tmp_path / "operator.db") not in json.dumps(info)
        for op, args, expected in (
            ("sources_validate_connection", {}, {"connected": native.validate_connection()}),
            (
                "sources_validate_columns",
                {"required_columns": ["missing", "x"]},
                {"missing_columns": native.validate_columns(["missing", "x"])},
            ),
            ("sources_check_size_limits", {}, {"within_limits": True}),
        ):
            response = invoke(adapter, op, source_id=source, **args)
            assert response["execution_status"] == "succeeded", response
            assert source_artifact(adapter, response["data"]["resource_id"]) == expected
        sample = invoke(adapter, "sources_sample", source_id=source, n=2, seed=42)
        assert sample["execution_status"] == "succeeded", sample
        page = invoke(adapter, "assets_page", asset_id=sample["data"]["resource_id"])
        sampled = native.sample(2, seed=42)
        try:
            assert (
                decode(page["data"]["rows"]) == sampled.to_polars_lazyframe().collect().to_dicts()
            )
        finally:
            sampled.__exit__(None, None, None)
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "import",
        "setup",
        "source-create",
        "source-enter",
    ]
    assert (
        invoke(adapter, "resources_release", resource_id=source)["execution_status"] == "succeeded"
    )
    assert worker.closed and worker.process.poll() is not None
    assert marker.read_text(encoding="utf-8").splitlines()[-2:] == ["source-close", "teardown"]
    assert (
        invoke(adapter, "artifacts_read", resource_id=inspected["data"]["resource_id"])[
            "execution_status"
        ]
        == "failed"
    )
    assert invoke(adapter, "sources_open_binding", **request)["execution_status"] == "failed"
    assert marker.read_text(encoding="utf-8").count("source-create") == 1


@pytest.mark.parametrize("change", ["binding", "package", "policy", "close", "expiry"])
def test_approved_source_revocation_and_shutdown_reap_native_lifecycle(
    runtime, monkeypatch, change
):
    from dataclasses import replace

    adapter, control, marker, code, _ = runtime
    request = prepare_source(runtime, monkeypatch)
    result = invoke(adapter, "sources_open_binding", **request)
    assert result["execution_status"] == "succeeded", result
    identity = result["data"]["resource_id"]
    worker = adapter._session._sources.workers[identity]
    artifact = invoke(adapter, "sources_inspect", source_id=identity)["data"]["resource_id"]
    if change == "binding":
        (control / "bindings.json").write_text("{}", encoding="utf-8")
    elif change == "package":
        code.write_text(code.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    elif change == "policy":
        adapter.policy = replace(adapter.policy, max_items=99)
    elif change == "expiry":
        expires = adapter._session.resources.get(identity).expires
        adapter._session.resources.clock = lambda: expires + 1
    else:
        adapter.close()
    assert invoke(adapter, "artifacts_read", resource_id=artifact)["execution_status"] == "failed"
    assert worker.closed and worker.process.poll() is not None
    assert marker.read_text(encoding="utf-8").splitlines()[-2:] == ["source-close", "teardown"]


def test_approved_source_factory_failure_is_not_retried_or_exposed(runtime, monkeypatch):
    adapter, _, marker, _, _ = runtime
    request = prepare_source(runtime, monkeypatch, "failed-source")
    for _ in range(2):
        result = invoke(adapter, "sources_open_binding", **request)
        assert result["issues"] == [{"code": "outcome_unknown"}], result
        assert "synthetic-connection-secret" not in json.dumps(result)
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "import",
        "setup",
        "factory-failed",
        "teardown",
    ]
    assert not adapter._session._sources.workers


def test_revoked_source_plan_and_untyped_authority_never_import_plugin(runtime, monkeypatch):
    adapter, control, marker, _, _ = runtime
    request = prepare_source(runtime, monkeypatch)
    store = ApprovalStore(control)
    try:
        plan_id = store.connection.execute(
            "SELECT identity FROM plans WHERE state='approved'"
        ).fetchone()[0]
        store.revoke(plan_id)
    finally:
        store.close()
    result = invoke(adapter, "sources_open_binding", **request)
    assert result["issues"] == [{"code": "approval_required"}]
    for extra in (
        {"connection_string": "secret://value"},
        {"factory": "os.system"},
        {"settings": {"database": "private"}},
    ):
        result = invoke(adapter, "sources_open_binding", **request, **extra)
        assert result["issues"] == [{"code": "invalid_arguments"}]
    assert not marker.exists()


def test_approved_source_timeout_reaps_owned_worker_and_blocks_retry(runtime, monkeypatch):
    from dataclasses import replace

    from truthound.mcp import sources

    adapter, _, marker, _, _ = runtime
    adapter.policy = replace(adapter.policy, timeout_seconds=1)
    request = prepare_source(runtime, monkeypatch, "slow-source")
    created = []
    original = sources.SourceProcess

    def tracked(ttl):
        worker = original(ttl)
        created.append(worker)
        return worker

    monkeypatch.setattr(sources, "SourceProcess", tracked)
    for _ in range(2):
        result = invoke(adapter, "sources_open_binding", **request)
        assert result["issues"] == [{"code": "outcome_unknown"}], result
    assert len(created) == 1 and created[0].closed
    assert created[0].process.poll() is not None
    assert not adapter._session._sources.workers
    if marker.exists():
        assert "source-enter" not in marker.read_text(encoding="utf-8")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import sqlite3

    database = tmp_path / "operator.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (x INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(-1,), (1,), (None,)])
    root = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(root)
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    scheme = "nt" if os.name == "nt" else "posix_prefix"
    site = Path(
        sysconfig.get_path(
            "purelib", scheme=scheme, vars={"base": str(root), "platbase": str(root)}
        )
    )
    # Add the exact test interpreter's already installed Core/SDK dependencies.
    # Only package/code directories, never admitted customer-data directories.
    paths = [str(Path(item).resolve()) for item in sys.path if item and Path(item).is_dir()]
    (site / "test-runtime.pth").write_text("\n".join(paths) + "\n", encoding="utf-8")
    build = tmp_path / "fixture-build"
    build.mkdir()
    module = build / "mcp_operator_fixture"
    module.mkdir()
    marker = tmp_path / "plugin-effects.txt"
    source = module / "__init__.py"
    source.write_text(
        "from pathlib import Path\n"
        "from truthound.plugins import Plugin, PluginInfo, PluginType\n"
        "import polars as pl\n"
        f"MARKER = Path({str(marker)!r})\n"
        "def note(text):\n"
        "    with MARKER.open('a', encoding='utf-8') as stream: stream.write(text+'\\n')\n"
        "note('import')\n"
        "from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig\n"
        "class NativeSource(SQLiteDataSource):\n"
        "    def __enter__(self): note('source-enter'); return super().__enter__()\n"
        "    def __exit__(self, *args):\n"
        "        try: return super().__exit__(*args)\n"
        "        finally: note('source-close')\n"
        "def source_factory():\n"
        "    note('source-create')\n"
        f"    return NativeSource(database={str(database)!r}, table='items', config=SQLiteDataSourceConfig(name='approved-source'))\n"
        "def failed_source_factory():\n"
        "    note('factory-failed'); raise RuntimeError('synthetic-connection-secret')\n"
        "def slow_source_factory():\n"
        "    import time\n"
        "    note('factory-blocked'); time.sleep(10); return source_factory()\n"
        "from truthound.datasources.async_base import AsyncBaseDataSource, AsyncDataSourceConfig\n"
        "from truthound.datasources._protocols import ColumnType\n"
        "class AsyncNativeSource(AsyncBaseDataSource):\n"
        "    def __init__(self, values=None, failed=False):\n"
        "        super().__init__(AsyncDataSourceConfig(name='approved-source'))\n"
        "        self.values = [-1, 1, None] if values is None else values\n"
        "        self.failed = failed\n"
        "    async def _create_connection_factory(self): raise NotImplementedError\n"
        "    async def connect_async(self):\n"
        "        import asyncio\n"
        "        self.loop = asyncio.get_running_loop(); self._is_connected=True; note('async-connect')\n"
        "    async def disconnect_async(self):\n"
        "        import asyncio\n"
        "        assert self.loop is asyncio.get_running_loop()\n"
        "        if MARKER.with_name('slow-async-child-close').exists() and len(self.values) == 1:\n"
        "            note('async-child-close-blocked')\n"
        "            await asyncio.sleep(30)\n"
        "        self._is_connected=False; note('async-disconnect')\n"
        "    async def get_schema_async(self):\n"
        "        if self.failed: raise ValueError('synthetic-connection-secret')\n"
        "        self._cached_schema={'x': ColumnType.INTEGER}; return self._cached_schema\n"
        "    async def get_row_count_async(self): return len(self.values)\n"
        "    async def validate_connection_async(self):\n"
        "        import asyncio\n"
        "        return self._is_connected and self.loop is asyncio.get_running_loop()\n"
        "    async def to_polars_lazyframe_async(self):\n"
        "        assert await self.validate_connection_async()\n"
        "        note('async-read'); return pl.DataFrame({'x': pl.Series(self.values,dtype=pl.Int64)}).lazy()\n"
        "    async def sample_async(self,n=1000,seed=None): return AsyncNativeSource(self.values[:n])\n"
        "def async_source_factory(): return AsyncNativeSource()\n"
        "def failed_async_source_factory(): return AsyncNativeSource(failed=True)\n"
        "from truthound.profiler.evolution.watcher import FileSchemaSource\n"
        "from truthound.profiler.evolution.breaking_alerts import CallbackNotifier\n"
        "def schema_factory():\n"
        "    note('schema-create')\n"
        f"    return FileSchemaSource({str(tmp_path / 'schema.json')!r})\n"
        "def alert_factory():\n"
        "    import json, urllib.request\n"
        "    def send(alert):\n"
        "        note('send')\n"
        f"        target = Path({str(tmp_path / 'endpoint.txt')!r}).read_text(encoding='utf-8')\n"
        "        request = urllib.request.Request(target, data=json.dumps(alert.to_dict()).encode(), headers={'Content-Type':'application/json'})\n"
        "        with urllib.request.urlopen(request, timeout=2) as response:\n"
        "            assert response.status == 200\n"
        "    return CallbackNotifier(send)\n"
        "class FalseNotifier:\n"
        "    def send(self, alert): note('send-false'); return False\n"
        "from contextlib import contextmanager\n"
        "from truthound.checkpoint import Checkpoint\n"
        "from truthound.checkpoint.actions.base import BaseAction, ActionConfig, ActionResult, ActionStatus\n"
        "class RecordAction(BaseAction):\n"
        "    action_type = 'record'\n"
        "    @classmethod\n"
        "    def _default_config(cls): return ActionConfig(name='record')\n"
        "    def _execute(self, result):\n"
        "        note('checkpoint-action')\n"
        "        if self.config.metadata.get('fail'): raise RuntimeError('synthetic-action-error')\n"
        "        return ActionResult(self.name, self.action_type, ActionStatus.SUCCESS, details={'status': result.status.value})\n"
        "@contextmanager\n"
        "def checkpoint_factory():\n"
        "    note('checkpoint-enter')\n"
        "    try:\n"
        "        import json, time\n"
        f"        options_path = Path({str(tmp_path / 'checkpoint-settings.json')!r})\n"
        "        options = json.loads(options_path.read_text(encoding='utf-8')) if options_path.exists() else {}\n"
        "        if options.get('block'): time.sleep(10)\n"
        "        if options.get('wrong'): yield object(); return\n"
        "        from truthound.checkpoint.routing import ActionRouter, Route, AlwaysRule\n"
        "        action = RecordAction(config=ActionConfig(name='record', retry_count=options.get('retries', 0), retry_delay_seconds=0, notify_on=options.get('notify_on', 'always'), fail_checkpoint_on_error=True, metadata={'fail': options.get('fail', False)}))\n"
        "        router = ActionRouter().add_route(Route(name='approved-route', rule=AlwaysRule(), actions=[action])) if options.get('router') else None\n"
        "        with source_factory() as source:\n"
        "            yield Checkpoint(name='approved-checkpoint', data_source=source, validators=['null'], actions=[] if router else [action], router=router)\n"
        "    finally: note('checkpoint-close')\n"
        "from contextlib import asynccontextmanager\n"
        "@asynccontextmanager\n"
        "async def async_checkpoint_factory():\n"
        "    from truthound.checkpoint.async_checkpoint import AsyncCheckpoint\n"
        "    with checkpoint_factory() as checkpoint:\n"
        "        if not isinstance(checkpoint, Checkpoint): yield checkpoint; return\n"
        "        async with AsyncCheckpoint(name=checkpoint.name, data_source=checkpoint.config.data_source, validators=['null'], actions=checkpoint.actions) as native:\n"
        "            try: yield native\n"
        "            finally: note('async-checkpoint-close')\n"
        "@contextmanager\n"
        "def checkpoint_runner_factory():\n"
        "    from truthound.checkpoint.runner import CheckpointRunner\n"
        "    from truthound.checkpoint.triggers.schedule import ScheduleTrigger\n"
        "    with checkpoint_factory() as checkpoint:\n"
        "        checkpoint.add_trigger(ScheduleTrigger(name='approved-schedule', interval_seconds=60, max_runs=1))\n"
        "        import json\n"
        f"        options_path = Path({str(tmp_path / 'checkpoint-settings.json')!r})\n"
        "        options = json.loads(options_path.read_text(encoding='utf-8')) if options_path.exists() else {}\n"
        f"        runner = CheckpointRunner(poll_interval_seconds=0.05, journal_path={str(tmp_path / 'runner.sqlite')!r} if options.get('durable') else None)\n"
        "        runner.add_checkpoint(checkpoint)\n"
        "        try: yield runner\n"
        "        finally: runner.stop(); note('runner-close')\n"
        "@asynccontextmanager\n"
        "async def escalation_factory():\n"
        "    from truthound.checkpoint.escalation.engine import EscalationEngine, EscalationEngineConfig\n"
        "    from truthound.checkpoint.escalation.protocols import EscalationPolicy, EscalationLevel, EscalationTarget\n"
        "    from truthound.checkpoint.escalation.scheduler import SchedulerConfig, SchedulerType\n"
        "    import json, os\n"
        f"    options_path = Path({str(tmp_path / 'escalation-settings.json')!r})\n"
        "    options = json.loads(options_path.read_text(encoding='utf-8')) if options_path.exists() else {}\n"
        "    async def notify(record, level, targets):\n"
        "        note('escalation-notify')\n"
        "        if options.get('crash'): os._exit(0)\n"
        "        return True\n"
        f"    database = {str(tmp_path / 'escalation.sqlite')!r}\n"
        "    durable = options.get('durable', False)\n"
        "    engine = EscalationEngine(EscalationEngineConfig(store_type='sqlite' if durable else 'memory', store_config={'database': database} if durable else {}, scheduler_config=SchedulerConfig(scheduler_type=SchedulerType(options.get('scheduler', 'asyncio')), journal_path=database if durable else None), default_notification_handler=notify))\n"
        "    engine.register_policy(EscalationPolicy(name='approved-policy', levels=[EscalationLevel(level=1, targets=[EscalationTarget.user('local')]), EscalationLevel(level=2, delay_minutes=options.get('delay_minutes', 60), targets=[EscalationTarget.user('local')])]))\n"
        "    try: yield engine\n"
        "    finally: await engine.stop(); note('escalation-close')\n"
        "@asynccontextmanager\n"
        "async def broker_factory():\n"
        "    from truthound.realtime.adapters.mock import MockAdapter, MockAdapterConfig\n"
        "    from truthound.realtime.processing.windows import TumblingWindowProcessor, WindowConfig, CountAggregator\n"
        "    from truthound.realtime.processing.exactly_once import ExactlyOnceProcessor\n"
        "    from truthound.mcp.brokers import BrokerRuntime\n"
        "    class ObservedAdapter(MockAdapter):\n"
        "        async def _do_connect(self): note('broker-connect'); await super()._do_connect()\n"
        "        async def _do_disconnect(self): note('broker-disconnect'); await super()._do_disconnect()\n"
        "        async def _do_commit(self, message): note('broker-ack'); await super()._do_commit(message)\n"
        "        async def _do_produce(self, message): note('broker-produce'); await super()._do_produce(message)\n"
        "    adapter = ObservedAdapter(MockAdapterConfig(num_messages=3, seed=42, error_rate=0, enable_health_check=False))\n"
        "    window = TumblingWindowProcessor(WindowConfig(size_ms=60000), CountAggregator)\n"
        "    processor = ExactlyOnceProcessor(adapter, process_fn=lambda message: message)\n"
        "    try: yield BrokerRuntime(adapter, window, processor, output_topic='operator-output')\n"
        "    finally: await adapter.disconnect(); note('broker-close')\n"
        "@asynccontextmanager\n"
        "async def ai_factory():\n"
        "    import json\n"
        "    import truthound as th\n"
        "    from truthound.context import TruthoundContext\n"
        "    from truthound.ai import StructuredProviderResponse\n"
        "    from truthound.mcp.ai_services import AIRuntime\n"
        "    class LocalProvider:\n"
        "        provider_name = 'local-test'\n"
        "        api_key_env = None\n"
        "        supports_structured_outputs = True\n"
        "        default_model_name = 'deterministic-local'\n"
        "        def generate_structured(self, request):\n"
        "            note('ai-provider')\n"
        "            if request.response_format_name == 'suite_proposal':\n"
        "                payload = {'summary': 'Local uniqueness proposal', 'rationale': 'Operator review', 'proposed_checks': [{'intent': 'unique', 'columns': ['x'], 'params': {}, 'rationale': 'Stable identifiers'}]}\n"
        "            else:\n"
        "                payload = {'summary': 'Local run analysis', 'recommended_next_actions': ['Review results'], 'evidence_refs': [reference.ref for reference in request.input_refs[:1]]}\n"
        "            return StructuredProviderResponse(provider_name=self.provider_name, model_name=request.model_name, output_text=json.dumps(payload), parsed_output=payload, usage={}, finish_reason='stop')\n"
        f"    context = TruthoundContext(Path({str(tmp_path / 'ai-workspace')!r}))\n"
        "    data = pl.DataFrame({'x': [-1, 1]})\n"
        "    run = th.check(data, context=context, validators=['null'])\n"
        "    try: yield AIRuntime(context, LocalProvider(), data=data, run=run)\n"
        "    finally: note('ai-close')\n"
        "@asynccontextmanager\n"
        "async def audit_factory():\n"
        "    from truthound.audit.storage import SQLiteAuditStorage\n"
        "    from truthound.mcp.administration import AuditRuntime\n"
        f"    storage = SQLiteAuditStorage({str(tmp_path / 'audit.sqlite')!r})\n"
        "    try: yield AuditRuntime(storage)\n"
        "    finally: storage.close(); note('audit-close')\n"
        '@asynccontextmanager\n'
        'async def documentation_factory():\n'
        '    from truthound.docs import DocConfig, DocumentationGenerator\n'
        '    from truthound.mcp.documentation_services import DocumentationJob, DocumentationRuntime\n'
        "    source = MARKER.with_name('synthetic-api.py')\n"
        "    source.write_text('def add(a: int, b: int) -> int:\\n    return a + b\\n')\n"
        "    generator = DocumentationGenerator(DocConfig(format='json'))\n"
        "    from truthound.cli_modules.scaffolding import ReporterScaffold, ScaffoldConfig, ValidatorScaffold\n"
        "    try: yield DocumentationRuntime({'local': DocumentationJob(generator, source), 'failure': DocumentationJob(generator, 'synthetic_nonexistent_documentation_module'), 'scaffold': DocumentationJob(ReporterScaffold(), ScaffoldConfig(name='synthetic_component', template_variant='full', description='Synthetic \"quoted\" description', author='Synthetic \"Author\"'), 'scaffold'), 'validate': DocumentationJob(generator, source, 'validate', strict=True), 'invalid_scaffold': DocumentationJob(ValidatorScaffold(), ScaffoldConfig(name='synthetic_invalid', template_variant='range', extra={'min_value': float('nan')}), 'scaffold')})\n"
        "    finally: note('documentation-close')\n"
        '@asynccontextmanager\n'
        'async def benchmark_factory():\n'
        '    from truthound.benchmark import Benchmark, BenchmarkConfig, BenchmarkRunner, BenchmarkSuite, RunnerConfig, benchmark_registry\n'
        '    from truthound.benchmark.comparison import BenchmarkComparator, PerformanceThreshold, RegressionDetector\n'
        '    from truthound.mcp.benchmark_services import BenchmarkRuntime, BenchmarkWorkload\n'
        '    class FailedBenchmark(Benchmark):\n'
        "        name = 'synthetic-failed-benchmark'\n"
        "        def run_iteration(self): raise RuntimeError('synthetic-private-benchmark-secret')\n"
        '    benchmark_registry.register(FailedBenchmark)\n'
        '    config = BenchmarkConfig(warmup_iterations=0,measure_iterations=1,cooldown_seconds=0)\n'
        '    runner = BenchmarkRunner(RunnerConfig(save_results=False,verbose=False), config)\n'
        "    suite = BenchmarkSuite('synthetic-small').add('profile',row_count=8)\n"
        "    runtime = BenchmarkRuntime({'profile':BenchmarkWorkload(runner,'single','profile',{'row_count':8}), 'suite':BenchmarkWorkload(runner,'suite',suite), 'failure':BenchmarkWorkload(runner,'single',FailedBenchmark.name)}, {'native':BenchmarkComparator()}, {'strict':PerformanceThreshold('profile',max_duration_seconds=0)})\n"
        "    runtime.workloads['duplicate'] = BenchmarkWorkload(runner, 'suite', BenchmarkSuite('synthetic-duplicate').add('profile', row_count=8).add('profile', row_count=8))\n"
        "    runtime.histories['local'] = RegressionDetector(MARKER.with_name('benchmark-history'))\n"
        "    from truthound.benchmark.generators import TabularDataGenerator, GeneratorConfig\n"
        "    runtime.workloads['generated'] = BenchmarkWorkload(TabularDataGenerator(), 'generator_lazy', GeneratorConfig(row_count=8, seed=0, null_ratio=0.25, duplicate_ratio=0.25))\n"
        '    try: yield runtime\n'
        '    finally:\n'
        '        benchmark_registry.unregister(FailedBenchmark.name)\n'
        "        note('benchmark-close')\n"
        '@asynccontextmanager\n'
        'async def lineage_factory():\n'
        '    from truthound.lineage import LineageTracker, LineageConfig\n'
        '    from truthound.mcp.lineage_services import LineageRuntime, LineageJob\n'
        "    state = MARKER.with_name('lineage-tracker-state.json')\n"
        '    tracker = LineageTracker(LineageConfig(persist_path=state))\n'
        "    if not tracker.graph.has_node('input'): tracker.track_source('input')\n"
        '    def success(native, context):\n'
        '        assert LineageTracker.get_current() is native\n'
        "        native.track_transformation('output', ['input'])\n"
        "        context.add_source('input'); context.add_target('output')\n"
        "        context.metadata['phase'] = 'complete'\n"
        '    def failure(native, context):\n'
        "        native.track_source('partial')\n"
        "        raise RuntimeError('synthetic-private-lineage-secret')\n"
        "    runtime = LineageRuntime(tracker, {'success':LineageJob(success,'synthetic-context'), 'failure':LineageJob(failure,'synthetic-failure')}, {'primary':state,'backup':MARKER.with_name('lineage-tracker-backup.json')})\n"
        '    try: yield runtime\n'
        "    finally: note('lineage-close')\n"
        '@asynccontextmanager\n'
        'async def openlineage_factory():\n'
        '    from truthound.lineage.integrations.openlineage import OpenLineageConfig, OpenLineageEmitter\n'
        '    from truthound.mcp.openlineage_services import OpenLineageRuntime\n'
        "    endpoint = MARKER.with_name('lineage-endpoint').read_text()\n"
        "    emitter = OpenLineageEmitter(OpenLineageConfig(endpoint=endpoint,api_key='synthetic-lineage-secret'))\n"
        "    note('lineage-open')\n"
        "    try: yield OpenLineageRuntime(emitter, {'input':emitter.build_dataset('synthetic-input'), 'output':emitter.build_output_dataset('synthetic-output',row_count=0)})\n"
        '    finally:\n'
        '        await emitter.aclose()\n'
        "        note('lineage-close')\n"
        "@asynccontextmanager\n"
        "async def telemetry_factory():\n"
        "    from truthound.observability.metrics import MetricsRegistry, Counter, InMemoryExporter\n"
        "    from truthound.observability.tracing.provider import TracerProvider\n"
        "    from truthound.observability.tracing.exporter import InMemorySpanExporter\n"
        "    from truthound.observability.tracing.processor import SimpleSpanProcessor\n"
        "    from truthound.mcp.administration import TelemetryRuntime\n"
        "    class MetricExporter(InMemoryExporter):\n"
        "        def export(self, registry): note('metric-export'); return super().export(registry)\n"
        "    class TraceExporter(InMemorySpanExporter):\n"
        "        def export(self, spans): note('span-export'); return super().export(spans)\n"
        "    class FailingFlushProcessor(SimpleSpanProcessor):\n"
        "        def force_flush(self, timeout_millis=30000): return False\n"
        "    registry = MetricsRegistry()\n"
        "    counter = Counter('requests', labels=['kind'])\n"
        "    registry.register(counter)\n"
        "    provider = TracerProvider(processors=[FailingFlushProcessor(TraceExporter())])\n"
        "    try: yield TelemetryRuntime(registry, provider, exporters={'local': MetricExporter()}, metrics={'requests': counter})\n"
        "    finally: provider.shutdown(); note('telemetry-close')\n"
        "@asynccontextmanager\n"
        "async def dashboard_factory():\n"
        "    import json\n"
        "    from truthound.profiler.dashboard.config import DashboardConfig\n"
        "    from truthound.profiler.dashboard.server import DashboardServer\n"
        f"    options = json.loads(Path({str(tmp_path / 'dashboard-settings.json')!r}).read_text(encoding='utf-8'))\n"
        "    server = DashboardServer(DashboardConfig(host=options.get('host', '127.0.0.1'), port=options['port'], title=options.get('title', 'Local dashboard')), python_executable=options.get('python'))\n"
        "    try: yield server\n"
        "    finally: server.close(); note('dashboard-close')\n"
        "@asynccontextmanager\n"
        "async def reflex_dashboard_factory():\n"
        "    import json\n"
        "    from truthound.datadocs.dashboard import DashboardConfig, ReflexDashboardServer\n"
        f"    options = json.loads(Path({str(tmp_path / 'dashboard-settings.json')!r}).read_text(encoding='utf-8'))\n"
        "    root = Path(options['project'])\n"
        "    config = DashboardConfig(**json.loads((root/'configuration.json').read_text()))\n"
        "    server = ReflexDashboardServer(root, config)\n"
        "    try: yield server\n"
        "    finally: server.close(); note('reflex-dashboard-close')\n"
        "@asynccontextmanager\n"
        "async def security_factory():\n"
        "    from truthound.rbac import RBACManager\n"
        "    from truthound.mcp.platform_services import RBACRuntime\n"
        "    from truthound.rbac.storage import SQLiteRoleStore, SQLitePrincipalStore\n"
        f"    storage = Path({str(tmp_path / 'security-settings.json')!r})\n"
        "    stores = []\n"
        "    if storage.exists():\n"
        f"        stores = [SQLiteRoleStore({str(tmp_path / 'security-roles.db')!r}), SQLitePrincipalStore({str(tmp_path / 'security-principals.db')!r})]\n"
        "    manager = RBACManager(*stores)\n"
        "    if manager.get_role('other-role') is None: manager.create_role('Outside', role_id='other-role', tenant_id='other', permissions={'dataset:read'})\n"
        "    try: yield RBACRuntime(manager, tenant_id='local', attribute_keys=('department',))\n"
        "    finally:\n"
        "        for store in stores: store.close()\n"
        "        note('security-close')\n"
        "@asynccontextmanager\n"
        "async def rate_limit_factory():\n"
        "    from truthound.ratelimit import RateLimiter\n"
        "    from truthound.ratelimit.core import RateLimitConfig, QuotaConfig\n"
        "    from truthound.ratelimit.storage import MemoryStorage\n"
        "    from truthound.ratelimit.policy import PolicyRegistry, QuotaManager, DefaultPolicy\n"
        "    from truthound.mcp.governance import RateLimitRuntime\n"
        "    registry, quotas = PolicyRegistry(), QuotaManager()\n"
        "    config = RateLimitConfig(requests_per_second=0.0001,burst_size=2)\n"
        "    limiter = RateLimiter(config,storage=MemoryStorage(),policy_registry=registry,quota_manager=quotas)\n"
        "    try: yield RateLimitRuntime(limiter,subjects={'reviewer':'internal-private-subject'},contexts={'local':{}},policies={'default':DefaultPolicy(config)},quotas={'daily':QuotaConfig('internal-daily',2,3600)},policy_registry=registry,quota_manager=quotas)\n"
        "    finally: note('rate-limit-close')\n"
        "@asynccontextmanager\n"
        "async def tenant_factory():\n"
        "    from truthound.multitenancy.manager import TenantManager\n"
        "    from truthound.multitenancy.storage import SQLiteTenantStore\n"
        "    from truthound.multitenancy.core import TenantSettings\n"
        "    from truthound.multitenancy.isolation import RowLevelPolicy\n"
        "    from truthound.mcp.tenancy import TenantRuntime\n"
        f"    store = SQLiteTenantStore({str(tmp_path / 'native-tenants.db')!r})\n"
        "    manager = TenantManager(store=store)\n"
        "    if manager.get('outside') is None: manager.create('Outside',tenant_id='outside')\n"
        '    from contextlib import ExitStack\n'
        '    sources = {}\n'
        "    tenants = {'local':'internal-private-tenant'}\n"
        '    stack = ExitStack()\n'
        "    if MARKER.with_name('tenant-source-mode').exists():\n"
        '        import sqlite3\n'
        '        from truthound.multitenancy.core import IsolationLevel\n'
        '        from truthound.multitenancy.sources import TenantSQLSource\n'
        '        from truthound.datasources.sql.sqlite import SQLiteDataSource\n'
        "        path = MARKER.with_name('private-tenant-data.db')\n"
        '        with sqlite3.connect(path) as connection:\n'
        "            connection.execute('CREATE TABLE IF NOT EXISTS records (x INTEGER)')\n"
        "            if connection.execute('SELECT COUNT(*) FROM records').fetchone()[0] == 0:\n"
        "                connection.executemany('INSERT INTO records VALUES (?)', [(1,), (2,)])\n"
        "        tenant = manager.get('private-structured')\n"
        '        if tenant is None:\n'
        "            tenant = manager.create('Structured',tenant_id='private-structured',isolation_level=IsolationLevel.DATABASE)\n"
        '        tenant.database_name = str(path)\n'
        '        store.save(tenant)\n'
        "        tenants['structured'] = tenant.id\n"
        "        source = stack.enter_context(SQLiteDataSource(table='records',database=str(path)))\n"
        "        sources['records'] = TenantSQLSource(source,tenant_id=tenant.id)\n"
        "    policies = {'positive':RowLevelPolicy('private-positive-policy',lambda row,tenant: row['x'] > 0), 'untyped':RowLevelPolicy('private-untyped-policy',lambda row,tenant: 'False')}\n"
        "    try: yield TenantRuntime(manager,tenants=tenants,sources=sources,settings={'local':TenantSettings(timezone='Asia/Seoul')},policies=policies,contexts={'selected':{'headers':{'X-Tenant-ID':'internal-private-tenant','Authorization':'synthetic-private-context-token'}},'outside':{'headers':{'X-Tenant-ID':'outside'}},'missing':{}})\n"
        "    finally: stack.close(); store.close(); note('tenant-close')\n"
        "@asynccontextmanager\n"
        "async def secret_factory():\n"
        "    from truthound.secrets.base import BaseSecretProvider, SecretReference, SecretNotFoundError\n"
        "    from truthound.secrets.manager import SecretManager\n"
        "    from truthound.mcp.governance import SecretRuntime, SecretProviderSlot\n"
        "    class LocalProvider(BaseSecretProvider):\n"
        "        @property\n"
        "        def name(self): return 'internal-private-provider'\n"
        "        def _fetch(self,key,version=None,field=None):\n"
        "            note('secret-fetch')\n"
        "            if key == 'failing-private': raise RuntimeError('synthetic-sensitive-provider-message')\n"
        "            if key != 'internal-private-secret-key': raise SecretNotFoundError(key,self.name)\n"
        "            return 'synthetic-not-a-real-credential'\n"
        "    provider = LocalProvider()\n"
        "    manager = SecretManager(audit_callback=lambda *args: note('secret-read'))\n"
        "    manager.add_provider(provider,name='internal-provider-alias')\n"
        "    try: yield SecretRuntime(manager,references={'primary':SecretReference('internal-private-secret-key',provider='internal-provider-alias'),'missing':SecretReference('absent',provider='internal-provider-alias'),'failure':SecretReference('failing-private',provider='internal-provider-alias')},providers={'local':SecretProviderSlot(provider,'internal-provider-alias')})\n"
        "    finally: manager.clear_cache(); note('secret-close')\n"
        "@asynccontextmanager\n"
        "async def plugin_factory():\n"
        "    from truthound.plugins import PluginManager\n"
        "    from truthound.mcp.platform_services import PluginRuntime\n"
        "    class TargetPlugin(Plugin):\n"
        "        @property\n"
        "        def info(self): return PluginInfo(name='bound-target', version='1.0.0', plugin_type=PluginType.CUSTOM)\n"
        "        def setup(self): note('target-setup')\n"
        "        def teardown(self):\n"
        "            note('target-teardown')\n"
        "            if MARKER.with_name('failed-target-cleanup').exists(): raise RuntimeError('synthetic-private-target-teardown')\n"
        "        def register(self, manager): manager.register_capability('bound-target-capability', object()); note('target-register')\n"
        "        def unregister(self, manager):\n"
        "            manager.unregister_capability('bound-target-capability')\n"
        "            note('target-unregister')\n"
        "            if MARKER.with_name('failed-target-cleanup').exists(): raise ValueError('synthetic-private-target-unregister')\n"
        "    class ReplacementPlugin(TargetPlugin):\n"
        "        def setup(self):\n"
        "            note('replacement-setup')\n"
        "            if MARKER.with_name('failed-replacement-setup').exists(): raise RuntimeError('synthetic-private-reload')\n"
        "        def teardown(self): note('replacement-teardown')\n"
        "    async def custom_hook(payload): note('custom-hook'); return payload.get('scope')\n"
        "    from truthound.core import CheckSpec, build_validation_asset\n"
        "    from truthound.validators import get_validator\n"
        "    from truthound.reporters.json_reporter import JSONReporter\n"
        "    class NativeChecks:\n"
        "        name = 'native-checks'\n"
        "        def build_check_specs(self):\n"
        "            note('native-check-build')\n"
        "            return [CheckSpec(id='native-positive',name='selected-positive',category='range',factory=lambda:get_validator('between')(columns=['x'],min_value=0))]\n"
        "    class NativeProvider:\n"
        "        name = 'native-data'\n"
        "        def get_asset(self):\n"
        "            note('native-provider-read')\n"
        "            return build_validation_asset(pl.DataFrame({'x':[-1,1]}))\n"
        "    async def native_before(**kwargs):\n"
        "        assert kwargs['checks'][0].id == 'native-positive'\n"
        "        note('native-before')\n"
        "    async def native_after(**kwargs):\n"
        "        assert kwargs['result'].row_count == 2\n"
        "        note('native-after')\n"
        "    async def native_issue(issue,validator,**kwargs):\n"
        "        assert validator.id == 'native-positive'\n"
        "        note('native-issue')\n"
        "    async def native_report(**kwargs):\n"
        "        assert kwargs['report'].row_count == 2\n"
        "        note('native-report')\n"
        "    async def failed_hook(**kwargs):\n"
        "        note('native-hook-failed')\n"
        "        raise RuntimeError('synthetic-private-hook-detail')\n"
        "    manager = PluginManager()\n"
        "    try: yield PluginRuntime(manager, classes={'target': TargetPlugin}, reload_factories={'target': lambda name: ReplacementPlugin}, hooks={'local-hook': custom_hook,'before':native_before,'after':native_after,'issue':native_issue,'report':native_report,'failure':failed_hook}, providers={'native-data':NativeProvider()},reporters={'native-json':JSONReporter()},check_factories={'native-checks':NativeChecks()}, capabilities={'approved-capability': object()})\n"
        "    finally: manager.unload_all(); note('platform-plugin-close')\n"
        "@asynccontextmanager\n"
        "async def execution_factory():\n"
        "    from truthound.mcp.execution_services import ExecutionRuntime\n"
        "    from truthound.execution.distributed.aggregations import AggregationPlan\n"
        "    from truthound.execution.distributed.protocols import AggregationScope,ExecutionMode,PartitionStrategy\n"
        "    from truthound.validators.completeness.null import NullValidator\n"
        "    from truthound.execution.polars_engine import PolarsExecutionEngine\n"
        "    from truthound.execution.sql_engine import SQLExecutionEngine\n"
        "    source = None\n"
        "    mode_path = MARKER.with_name('execution-mode')\n"
        "    mode = mode_path.read_text() if mode_path.exists() else 'polars'\n"
        "    if mode in {'ray','spark'}:\n"
        "        import os,json,time\n"
        "        started=time.monotonic()\n"
        "        from .distributed_fixture import engine as owned_engine, runtime_directory\n"
        "        java = MARKER.with_name('execution-java-home')\n"
        "        if java.exists(): os.environ['JAVA_HOME'] = java.read_text()\n"
        "        hadoop = MARKER.with_name('execution-hadoop-home')\n"
        "        if hadoop.exists(): os.environ['HADOOP_HOME'] = hadoop.read_text()\n"
        "        os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'\n"
        "        os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'\n"
        "        try:\n"
        "            with runtime_directory('th-fixture-') as directory:\n"
        "                def phase(label): note('native-fixture-diagnostic:'+json.dumps({'factory':'execution_factory','phase':label,'elapsed_seconds':round(time.monotonic()-started,3)}))\n"
        "                with owned_engine(mode,Path(directory),progress=phase,direct_spark_launcher=MARKER.with_name('execution-direct-launcher').exists()) as native:\n"
        "                    if mode == 'spark': native.spark_dataframe.createOrReplaceTempView('items')\n"
        "                    try:\n"
        "                        yield ExecutionRuntime(native,plans={'stats':AggregationPlan().add_stats('x'),'partitions':AggregationPlan(scope=AggregationScope.PARTITION).add_count().add_sum('x')},validators={'nulls':NullValidator()},predicates={'positive':'x > 0'},queries={'count':'SELECT COUNT(*) AS total FROM items'},row_filters={'positive':lambda row: row['x'] is not None and row['x'] > 0})\n"
        "                    finally:\n"
        "                        if mode == 'spark': note('native-sdk-exit-callbacks:'+str(native._verification_sdk_launcher_state['cancelled_callbacks']))\n"
        "        finally: note('execution-close')\n"
        "        return\n"
        "    if mode == 'sql':\n"
        "        source = source_factory()\n"
        "        source.__enter__()\n"
        "        engine = SQLExecutionEngine(source)\n"
        "    elif mode in {'dask-checkpoint','dask-checkpoint-reopen'}:\n"
        "        import dask.dataframe as dd\n"
        "        import pandas as pd\n"
        "        from truthound.execution.distributed.dask_engine import DaskExecutionEngine,DaskEngineConfig\n"
        "        config = DaskEngineConfig(scheduler='synchronous',checkpoint_enabled=True,checkpoint_dir=str(MARKER.with_name('native-engine-checkpoints')),cache_results=False)\n"
        "        pointer = MARKER.with_name('native-engine-checkpoint-location')\n"
        "        if mode == 'dask-checkpoint-reopen':\n"
        "            engine = DaskExecutionEngine.from_checkpoint(pointer.read_text(),config)\n"
        "        else:\n"
        "            data = dd.from_pandas(pd.DataFrame({'x':[-1.,1.,None]}),npartitions=2)\n"
        "            def observed(frame):\n"
        "                note('checkpoint-source-partition')\n"
        "                return frame\n"
        "            engine = DaskExecutionEngine(data.map_partitions(observed,meta=data._meta),config)\n"
        "            pointer.write_text(engine.checkpoint_path)\n"
        "    elif mode in {'dask','dask-synchronous','dask-eager','dask-hash','dask-streaming'}:\n"
        "        import dask.dataframe as dd\n"
        "        import pandas as pd\n"
        "        from truthound.execution.distributed.dask_engine import DaskExecutionEngine,DaskEngineConfig\n"
        "        import threading\n"
        "        data = dd.from_pandas(pd.DataFrame({'x':[-1.,1.,None]}),npartitions=2)\n"
        "        if mode in {'dask-synchronous','dask-eager','dask-hash','dask-streaming'}:\n"
        "            expected_thread = threading.get_ident()\n"
        "            def require_selected_scheduler(frame):\n"
        "                if threading.get_ident() != expected_thread: raise RuntimeError('synthetic-selected-scheduler-not-used')\n"
        "                if mode == 'dask-eager': note('eager-native-partition')\n"
        "                if mode == 'dask-streaming': note('streaming-native-partition')\n"
        "                return frame\n"
        "            data = data.map_partitions(require_selected_scheduler,meta=data._meta)\n"
        "        engine = DaskExecutionEngine(data,DaskEngineConfig(scheduler='synchronous' if mode in {'dask-synchronous','dask-eager','dask-hash','dask-streaming'} else 'threads',execution_mode=ExecutionMode.EAGER if mode == 'dask-eager' else ExecutionMode.STREAMING if mode == 'dask-streaming' else ExecutionMode.LAZY,num_partitions=3 if mode == 'dask-hash' else 0,partition_strategy=PartitionStrategy.COLUMN if mode == 'dask-hash' else PartitionStrategy.ROW_HASH,partition_columns=('x',) if mode == 'dask-hash' else ()))\n"
        "    else:\n"
        "        engine = PolarsExecutionEngine(pl.DataFrame({'x': [-1, 1, None]}).lazy())\n"
        "    try:\n"
        "        yield ExecutionRuntime(engine,plans={'stats':AggregationPlan().add_stats('x'),'partitions':AggregationPlan(scope=AggregationScope.PARTITION).add_count().add_sum('x')},validators={'nulls':NullValidator()},predicates={'positive':'x > 0'},queries={'count':'SELECT COUNT(*) AS total FROM items'},expressions={'positive':pl.col('x') > 0,'double':(pl.col('x')*2).alias('double')})\n"
        "    finally:\n"
        "        if source is not None: source.__exit__(None,None,None)\n"
        "        note('execution-close')\n"
        "        if mode == 'polars-close-failure': raise RuntimeError('synthetic-private-native-close')\n"
        "@asynccontextmanager\n"
        "async def store_factory():\n"
        "    from datetime import datetime\n"
        "    from truthound.mcp.store_services import StoreRuntime\n"
        "    from truthound.stores.results import ValidationResult, ResultStatus\n"
        "    from truthound.stores.backends.filesystem import FileSystemStore, FileSystemExpectationStore\n"
        "    from truthound.stores.caching.store import CachedStore\n"
        "    from truthound.stores.caching.backends import LRUCache\n"
        "    from truthound.stores.batching.store import BatchedStore\n"
        "    from truthound.stores.batching.base import BatchConfig\n"
        "    from truthound.stores.versioning.store import VersionedStore\n"
        "    from truthound.stores.retention.store import RetentionStore\n"
        "    from truthound.stores.retention.base import RetentionConfig, RetentionSchedule\n"
        "    from truthound.stores.retention.policies import CountBasedPolicy\n"
        "    from truthound.stores.tiering.store import TieredStore, InMemoryTierMetadataStore\n"
        "    from truthound.stores.tiering.base import StorageTier, TieringConfig, TierType\n"
        "    from truthound.stores.tiering.policies import AgeBasedTierPolicy\n"
        "    from truthound.stores.replication.store import ReplicatedStore\n"
        "    from truthound.stores.replication.base import ReplicaTarget, ReplicationConfig, ReplicationMode\n"
        "    from truthound.stores.base import StoreWriteError\n"
        "    mode_path = MARKER.with_name('store-mode')\n"
        "    mode = mode_path.read_text() if mode_path.exists() else 'filesystem'\n"
        "    file_options = {'serialization_format':'yaml','file_extension':'.yaml','compression':True} if mode == 'filesystem-yaml' else {'compression':'zstd'} if mode == 'filesystem-zstd' else {}\n"
        "    base = (FileSystemExpectationStore if mode == 'expectation' else FileSystemStore)(base_path=str(MARKER.with_name('native-store')),**file_options)\n"
        "    if mode.startswith('ordinary-azure'):\n"
        "        from truthound.stores.backends import azure_blob\n"
        "        from .native_cloud_transports import AzureObjectHTTP\n"
        "        transport = AzureObjectHTTP(MARKER.with_name('native-azure-objects.json'))\n"
        "        native_client = azure_blob.BlobServiceClient\n"
        "        azure_blob.BlobServiceClient = lambda **kwargs: native_client(transport=transport,retry_total=0,**kwargs)\n"
        "        options = {'serialization_format':'yaml'} if mode == 'ordinary-azure-yaml' else {}\n"
        "        base = azure_blob.AzureBlobStore('synthetic',account_url='https://synthetic.blob.core.windows.net',account_key='c3ludGhldGljLWF6dXJlLWtleS1vbmx5',compression='zstd' if mode == 'ordinary-azure-zstd' else True,access_tier='Cool',metadata={'owner':'synthetic-private-azure-header'},record_metadata={'configured':'synthetic-private-azure-default'},**options)\n"
        "    if mode.startswith('ordinary-gcs'):\n"
        "        from truthound.stores.backends import gcs\n"
        "        from .native_cloud_transports import GoogleObjectHTTP,google_client\n"
        "        transport = GoogleObjectHTTP(MARKER.with_name('native-google-objects.json'))\n"
        "        client = google_client(transport)\n"
        "        gcs.storage.Client = lambda **kwargs: client\n"
        "        options = {'serialization_format':'yaml'} if mode == 'ordinary-gcs-yaml' else {}\n"
        "        base = gcs.GCSStore('synthetic',compression='zstd' if mode == 'ordinary-gcs-zstd' else True,metadata={'configured':'synthetic-private-gcs-default'},**options)\n"
        "    if mode.startswith('ordinary-s3'):\n"
        "        from truthound.stores.backends import s3\n"
        "        from .synthetic_s3 import OrdinaryMemoryS3\n"
        "        client = OrdinaryMemoryS3(MARKER.with_name('ordinary-s3-objects.json'))\n"
        "        s3.boto3.client = lambda *args,**kwargs: client\n"
        "        options = {'serialization_format':'yaml'} if mode == 'ordinary-s3-yaml' else {}\n"
        "        base = s3.S3Store('synthetic',compression='zstd' if mode == 'ordinary-s3-zstd' else False,metadata={'configured':'synthetic-private-object-default'},**options)\n"
        "    if mode in {'memory','memory-expectation'}:\n"
        "        from truthound.stores.backends.memory import MemoryStore, MemoryExpectationStore\n"
        "        base = (MemoryExpectationStore if mode == 'memory-expectation' else MemoryStore)(max_items=2,metadata={'configured':'synthetic-private-memory-default'})\n"
        "    if mode in {'database','database-legacy','database-failure','database-prefixed'}:\n"
        "        from truthound.stores.backends.database import DatabaseStore, PoolingConfig\n"
        "        base = DatabaseStore('sqlite:///'+str(MARKER.with_name('native-store.db')),table_prefix='isolated_' if mode == 'database-prefixed' else '',use_pool_manager=mode != 'database-legacy',pooling=PoolingConfig(enable_health_checks=mode != 'database-failure'))\n"
        "        if mode == 'database-failure':\n"
        "            from sqlalchemy import event\n"
        "            from sqlalchemy.exc import OperationalError\n"
        "            base.initialize()\n"
        "            def fail_database(*args): raise OperationalError('synthetic',{},RuntimeError('synthetic-private-db-connection'))\n"
        "            event.listen(base._engine,'before_cursor_execute',fail_database)\n"
        "    if mode == 'concurrent-cleanup-crash':\n"
        "        import shutil,os\n"
        "        native_cleanup = shutil.rmtree\n"
        "        def interrupted_cleanup(path,*args,**kwargs):\n"
        "            path=Path(path)\n"
        "            if path.name.startswith('.batch-resolved-'):\n"
        "                for child in path.iterdir():\n"
        "                    if child.is_file(): child.unlink(); break\n"
        "                os._exit(75)\n"
        "            return native_cleanup(path,*args,**kwargs)\n"
        "        shutil.rmtree=interrupted_cleanup\n"
        "    if mode.startswith('concurrent'):\n"
        "        from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore, ConcurrencyConfig\n"
        "        class Concurrent(ConcurrentFileSystemStore):\n"
        "            def _do_save(self,item,txn=None):\n"
        "                if mode == 'concurrent-failure' and item.id == 'private-fail': raise StoreWriteError('synthetic-private-batch-failure')\n"
        "                answer = super()._do_save(item,txn)\n"
        "                if mode == 'concurrent-crash' and item.id == 'private-native-run':\n"
        "                    import os; os._exit(73)\n"
        "                return answer\n"
        "        selected = dict(serialization_format='yaml',file_extension='.yaml',compression=True) if mode == 'concurrent-yaml' else {}\n"
        "        base = Concurrent(base_path=str(MARKER.with_name('native-store')),concurrency=ConcurrencyConfig(lock_strategy='filelock') if mode == 'concurrent-filelock' else None, **selected)\n"
        "    if mode == 'concurrent-expectation':\n"
        "        from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemExpectationStore\n"
        "        base = ConcurrentFileSystemExpectationStore(base_path=str(MARKER.with_name('native-store')),serialization_format='yaml',file_extension='.yaml',use_compression=True)\n"
        "    if mode == 'cached': store = CachedStore(base, cache=LRUCache())\n"
        "    elif mode == 'cached-behind':\n"
        "        from truthound.stores.caching.store import CacheMode\n"
        "        store = CachedStore(base, cache=LRUCache(), mode=CacheMode.WRITE_BEHIND)\n"
        "    elif mode == 'versioned': store = VersionedStore(base)\n"
        "    elif mode == 'batched': store = BatchedStore(base, BatchConfig(batch_size=100,flush_interval_seconds=3600))\n"
        "    elif mode == 'retention': store = RetentionStore(base,RetentionConfig(policies=[CountBasedPolicy(1)],schedule=RetentionSchedule(enabled=False),preserve_latest=False))\n"
        "    elif mode == 'observable':\n"
        "        from truthound.stores.observability.store import ObservableStore\n"
        "        from truthound.stores.observability.config import ObservabilityConfig, AuditConfig, TracingConfig\n"
        "        from truthound.stores.observability.protocols import ObservabilityContext\n"
        "        store = ObservableStore(base,ObservabilityConfig(audit=AuditConfig(backend='memory'),tracing=TracingConfig(enabled=False)),ObservabilityContext('private-correlation',user_id='private-user',tenant_id='private-tenant',extra={'secret':'synthetic-private-observability'}))\n"
        "    elif mode.startswith('migration'):\n"
        "        from truthound.stores.migration import MigratableStore, MigrationConfig, MigrationRegistry\n"
        "        registry = MigrationRegistry()\n"
        "        @registry.register('1.0.0','2.0.0')\n"
        "        def upgrade(data):\n"
        "            note('migration-callback')\n"
        "            if mode == 'migration-failure': raise RuntimeError('synthetic-private-migration')\n"
        "            data['metadata']['applied'] = data['metadata'].get('applied',0)+1\n"
        "            return data\n"
        "        store = MigratableStore(base,MigrationConfig(current_version='2.0.0'),registry)\n"
        "    elif mode == 'tiered':\n"
        "        secondary = FileSystemStore(base_path=str(MARKER.with_name('native-secondary')))\n"
        "        metadata = InMemoryTierMetadataStore()\n"
        "        store = TieredStore([StorageTier('private-hot',base),StorageTier('private-cold',secondary,TierType.COLD,priority=2)],TieringConfig(default_tier='private-hot',policies=[AgeBasedTierPolicy('private-hot','private-cold',after_hours=1)]),metadata_store=metadata)\n"
        "    elif mode.startswith('replicated'):\n"
        "        class Replica(FileSystemStore):\n"
        "            def save(self,item):\n"
        "                note('replica-save')\n"
        "                if mode == 'replicated-failure': raise StoreWriteError('synthetic-private-replica-failure')\n"
        "                return super().save(item)\n"
        "        secondary = Replica(base_path=str(MARKER.with_name('native-secondary')))\n"
        "        target = ReplicaTarget('private-replica',secondary,'local',max_retry_attempts=1)\n"
        "        target.mark_healthy()\n"
        "        store = ReplicatedStore(base,ReplicationConfig(mode=ReplicationMode.SYNC,targets=[target],enable_health_checks=False))\n"
        "    else: store = base\n"
        "    record = ValidationResult('private-native-run',datetime(2026,1,1),'private-native-asset',ResultStatus.FAILURE,metadata={'private-record':'synthetic-private-record'},runtime_environment={'private-path':'/operator/private'})\n"
        "    if mode.startswith('migration') and not base.exists(record.id): base.save(record)\n"
        "    records = {'seed':record} if mode not in {'expectation','memory-expectation'} else {}\n"
        "    if mode.startswith('concurrent'): records['fail'] = ValidationResult('private-fail',datetime(2026,1,2),'private-native-asset',ResultStatus.SUCCESS)\n"
        "    if mode == 'retention': records['second'] = ValidationResult('private-second',datetime(2026,1,2),'private-native-asset',ResultStatus.SUCCESS)\n"
        "    if mode == 'tiered':\n"
        "        store.save(record)\n"
        "        info = metadata.get_info(record.id)\n"
        "        info.created_at = datetime(2026,1,1)\n"
        "        metadata.save_info(info)\n"
        "    metrics = {}\n"
        "    if mode == 'observable':\n"
        "        from truthound.mcp.store_services import StoreMetric\n"
        "        native = store.observability.metrics\n"
        "        metrics['saves'] = StoreMetric('operations_total', labels={'store_type':native.store_type,'store_id':native.store_id,'operation':'save'})\n"
        "    try: yield StoreRuntime(store, metrics=metrics, records=records,data_assets={'local':'private-native-asset'},tiers={'hot':'private-hot','cold':'private-cold'} if mode == 'tiered' else {},replicas={'local':'private-replica'} if mode.startswith('replicated') else {})\n"
        "    finally:\n"
        "        if mode == 'batched': await store.stop()\n"
        "        if mode.startswith('replicated'): await store.stop()\n"
        "        if mode == 'tiered' or mode.startswith('replicated'): secondary.close()\n"
        "        if mode == 'observable': store.close()\n"
        "        else: base.close()\n"
        "        note('store-close')\n"
        '@asynccontextmanager\n'
        'async def streaming_store_factory():\n'
        '    from truthound.mcp.streaming_store_services import StreamingRun, StreamingStoreRuntime\n'
        '    from truthound.stores.streaming.filesystem import StreamingFileSystemStore\n'
        '    from truthound.stores.streaming.database import StreamingDatabaseStore\n'
        '    from truthound.stores.streaming.base import CompressionType, StreamingFormat\n'
        "    mode = MARKER.with_name('streaming-store-mode').read_text() if MARKER.with_name('streaming-store-mode').exists() else 'filesystem'\n"
        "    settings = dict(buffer_size=2,checkpoint_interval=1,flush_interval_seconds=0,format=StreamingFormat.CSV,compression=CompressionType.GZIP)\n"
        "    if mode.endswith('uncheckpointed'): settings.update(buffer_size=1,checkpoint_interval=100)\n"
        "    store = StreamingDatabaseStore('sqlite:///'+str(MARKER.with_name('native-stream.db')),**settings) if mode.startswith('sqlite') else StreamingFileSystemStore(str(MARKER.with_name('native-stream')),**settings)\n"
        "    if mode.startswith('s3'):\n"
        "        from truthound.stores.streaming.s3 import StreamingS3Store\n"
        "        from .synthetic_s3 import PersistentMemoryS3\n"
        "        store = StreamingS3Store('synthetic',**settings)\n"
        "        store._s3_client = PersistentMemoryS3(MARKER.with_name('synthetic-objects.json'))\n"
        "    multipart_receipts = {}\n"
        "    if mode.startswith('s3-multipart-'):\n"
        "        from .synthetic_s3 import PersistentMultipartS3\n"
        "        once = MARKER.with_name('multipart-crash-once')\n"
        "        phase = mode.removeprefix('s3-multipart-') if not once.exists() else None\n"
        "        once.touch()\n"
        "        settings.update(buffer_size=1,multipart_threshold=1,max_retries=0)\n"
        "        store = StreamingS3Store('synthetic',**settings)\n"
        "        client = PersistentMultipartS3(MARKER.with_name('synthetic-objects.json'),crash_at=phase)\n"
        "        store._s3_client = client\n"
        "        if client.pending: multipart_receipts['recovered'] = {key: identity for identity,(key,_) in client.pending.items()}\n"
        "    note('streaming-store-open')\n"
        "    try: yield StreamingStoreRuntime(store,{'local':StreamingRun('private-native-run','private-native-asset')},multipart_receipts=multipart_receipts)\n"
        "    finally:\n"
        "        try: store.close()\n"
        "        finally: note('streaming-store-close')\n"
        '@asynccontextmanager\n'
        'async def profiler_factory():\n'
        "    assert Path.home().resolve() == Path.cwd().resolve(), 'Native home must be the owned workspace'\n"
        "    print('\\u2713-synthetic-private-native-progress')\n"
        "    if MARKER.with_name('pdf-native-mode').exists():\n"
        "        import os, sys\n"
        "        if sys.platform == 'darwin' and Path('/opt/homebrew/lib/libgobject-2.0.dylib').exists():\n"
        "            os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/opt/homebrew/lib'\n"
        '    from truthound.mcp.profiler_services import ProfilerRuntime\n'
        '    from truthound.profiler import DataProfiler, ColumnProfiler, ProfilerConfig\n'
        '    from truthound.profiler.base import DataType, TypeInferrer\n'
        '    from truthound.profiler.column_profiler import BasicStatsAnalyzer, ColumnAnalyzer, PatternAnalyzer, PatternDefinition\n'
        '    from truthound.profiler.generators.base import GeneratedRule, RuleCategory\n'
        '    from truthound.profiler.generators.suite_generator import ValidationSuiteGenerator\n'
        '    from truthound.profiler.generators.schema_rules import SchemaRuleGenerator\n'
        '    from truthound.profiler.integration.executor import SuiteExecutor\n'
        '    class CustomAnalyzer(ColumnAnalyzer):\n'
        "        name = 'custom-analyzer'\n"
        '        def analyze(self, column, frame, config):\n'
        "            note('custom-analyze')\n"
        "            if MARKER.with_name('profiler-failed').exists(): raise RuntimeError('synthetic-private-profile-query')\n"
        '            return {}\n'
        '    class CustomInferrer(TypeInferrer):\n'
        "        name, priority = 'custom-inferrer', 1000\n"
        '        def infer(self, column, frame, dtype):\n'
        "            note('custom-infer')\n"
        '            return DataType.STRING\n'
        '    class MissingGenerator(SchemaRuleGenerator):\n'
        '        def generate(self, profile, strictness):\n'
        "            return [GeneratedRule('missing', 'UninstalledSyntheticValidator', RuleCategory.SCHEMA)]\n"
        '    config = ProfilerConfig(sample_size=2)\n'
        "    columns = ColumnProfiler(config=config, analyzers=[BasicStatsAnalyzer(), CustomAnalyzer(), PatternAnalyzer([PatternDefinition('custom-code', r'^A[0-9]+$', DataType.STRING)])], type_inferrers=[CustomInferrer()])\n"
        "    runtime = ProfilerRuntime(profilers={'custom':DataProfiler(config=config,column_profiler=columns)},generators={'schema':ValidationSuiteGenerator([SchemaRuleGenerator()]),'missing':ValidationSuiteGenerator([MissingGenerator()])},executors={'native':SuiteExecutor()},profile_slots={'local':MARKER.with_name('operator-profile.json')},suite_slots={'local':MARKER.with_name('operator-suite.json')})\n"
        '    from truthound.profiler.distributed import DistributedProfiler\n'
        "    distributed = DistributedProfiler.create(backend='local',num_partitions=2)\n"
        "    runtime.profilers['distributed'] = distributed\n"
        '    from truthound.profiler.quality import RuleQualityScorer, ValidationRule, RuleType\n'
        "    runtime.quality_scorers['sampling'] = RuleQualityScorer(estimator_options={'random_seed':7})\n"
        "    runtime.quality_rules['positive'] = ValidationRule('private-quality-rule',RuleType.RANGE,column='x',min_value=0,nullable=False)\n"
        '    from truthound.profiler.quality import QualityTrendAnalyzer\n'
        "    runtime.quality_trends['local'] = QualityTrendAnalyzer(MARKER.with_name('private-quality-history.json'))\n"
        '    from truthound.profiler.sampling import Sampler, SamplingConfig, SamplingMethod\n'
        '    from truthound.mcp.profiler_services import NativeSamplingBinding\n'
        '    from truthound.profiler.parallel_sampling import ParallelBlockSampler, ParallelSamplingConfig, SketchBasedSampler\n'
        "    runtime.samplers['sketch'] = NativeSamplingBinding(SketchBasedSampler(ParallelSamplingConfig(target_rows=1,seed=0)), SamplingConfig(max_rows=1,min_sample_size=1))\n"
        "    runtime.samplers['parallel'] = NativeSamplingBinding(ParallelBlockSampler(ParallelSamplingConfig(target_rows=1,block_size=1,max_workers=2,seed=0)), SamplingConfig(max_rows=1,min_sample_size=1))\n"
        "    runtime.samplers['head'] = Sampler(SamplingConfig(strategy=SamplingMethod.HEAD,max_rows=1,min_sample_size=1,seed=7))\n"
        '    from truthound.profiler.ml_inference import MLTypeInferrer, InferrerConfig\n'
        "    runtime.inferrers['local'] = MLTypeInferrer(config=InferrerConfig(model='naive_bayes',feedback_batch_size=2,model_path=str(MARKER.with_name('private-inference-model.json'))))\n"
        '    from truthound.mcp.profiler_classifier import NativeClassifierBinding\n'
        '    from truthound.profiler.ml.pretrained import PretrainedModelManager\n'
        "    runtime.inferrers['native-rules'] = NativeClassifierBinding(PretrainedModelManager.load_default())\n"
        "    runtime.inference_contexts['selected'] = {'table_name':'private-native-table'}\n"
        '    from truthound.profiler.visualization import HTMLReportGenerator, ReportConfig\n'
        "    runtime.visualizers['svg'] = HTMLReportGenerator(ReportConfig(renderer='svg'))\n"
        "    runtime.visualization_slots['local'] = MARKER.with_name('private-native-profile.html')\n"
        "    from truthound.profiler.evolution.compatibility import CompatibilityRule, SchemaCompatibilityAnalyzer\n"
        "    from truthound.profiler.evolution.changes import SchemaChange, ChangeType\n"
        "    class WarningSchemaRule(CompatibilityRule):\n"
        "        name = 'bound-warning-rule'\n"
        "        def check(self,old,new):\n"
        "            note('schema-rule-check')\n"
        "            return True,[SchemaChange(ChangeType.CONSTRAINT_ADDED,'x',breaking=False)]\n"
        "    class FailedSchemaRule(CompatibilityRule):\n"
        "        name = 'bound-failed-rule'\n"
        "        def check(self,old,new):\n"
        "            note('schema-rule-failed')\n"
        "            raise RuntimeError('synthetic-private-schema-rule')\n"
        "    runtime.schema_analyzers.update({'warning':SchemaCompatibilityAnalyzer([WarningSchemaRule()]),'strict':SchemaCompatibilityAnalyzer([WarningSchemaRule()],strict=True),'failure':SchemaCompatibilityAnalyzer([FailedSchemaRule()])})\n"
        "    if MARKER.with_name('schema-only-mode').exists(): runtime.profilers.clear()\n"
        '    from truthound.profiler.incremental import IncrementalProfiler\n'
        '    from truthound.profiler.streaming import StreamingProfiler\n'
        '    import truthound.profiler.streaming as native_streaming\n'
        '    native_stats = native_streaming.IncrementalStats\n'
        '    class FailedOnce(native_stats):\n'
        '        failed = False\n'
        '        def update_from_chunk(self, *args, **kwargs):\n'
        "            if MARKER.with_name('profiler-stream-failed').exists() and not FailedOnce.failed:\n"
        '                FailedOnce.failed = True\n'
        "                raise RuntimeError('synthetic-private-chunk-query')\n"
        '            return super().update_from_chunk(*args, **kwargs)\n'
        '    native_streaming.IncrementalStats = FailedOnce\n'
        '    @asynccontextmanager\n'
        '    async def chunks_source():\n'
        '        class Chunks:\n'
        '            total_chunks = 2\n'
        '            def __iter__(self):\n'
        '                try:\n'
        "                    yield pl.DataFrame({'x':[1]})\n"
        "                    yield pl.DataFrame({'changed':[2]}) if MARKER.with_name('profiler-stream-schema').exists() else pl.DataFrame({'x':[2]})\n"
        "                finally: note('chunk-iterator-close')\n"
        "        note('chunk-source-open')\n"
        '        try: yield Chunks()\n'
        "        finally: note('chunk-source-close')\n"
        "    runtime.incrementals = {'native':IncrementalProfiler()}\n"
        "    runtime.streaming = {'native':StreamingProfiler(chunk_size=1)}\n"
        "    runtime.chunk_factories = {'local':chunks_source}\n"
        "    runtime.stream_progress = {'local':lambda progress: note('chunk-progress')}\n"
        '    from truthound.profiler.caching import ProfileCache, FileCacheBackend, CacheKey\n'
        "    runtime.caches = {'local':ProfileCache(FileCacheBackend(MARKER.with_name('private-profile-cache')))}\n"
        "    runtime.cache_keys = {'local':CacheKey('internal-private-cache-key')}\n"
        "    runtime.cache_patterns = {'selected':'default:1:internal-private-*'}\n"
        "    if 'distributed' in runtime.profilers: distributed.__enter__()\n"
        "    note('profiler-open')\n"
        '    try: yield runtime\n'
        "    finally:\n"
        "        if 'distributed' in runtime.profilers: distributed.__exit__(None,None,None)\n"
        "        note('profiler-close')\n"
        '@asynccontextmanager\n'
        'async def diagnosed_native(factory):\n'
        '    import json,time\n'
        '    from .distributed_fixture import failure_diagnostic,startup_watchdog\n'
        '    started=time.monotonic()\n'
        "    def diagnostic(payload): note('native-fixture-diagnostic:'+json.dumps({'factory':factory.__name__,'elapsed_seconds':round(time.monotonic()-started,3),**payload}))\n"
        '    try:\n'
        '        with startup_watchdog(diagnostic) as cancel_startup:\n'
        "            diagnostic({'phase':'starting'})\n"
        '            async with factory() as runtime:\n'
        '                cancel_startup()\n'
        "                diagnostic({'phase':'entered'})\n"
        '                yield runtime\n'
        '    except Exception as error:\n'
        "        note('native-fixture-diagnostic:'+json.dumps({'factory':factory.__name__,**failure_diagnostic(error)}))\n"
        '        raise\n'
        "class FixturePlugin(Plugin):\n"
        "    @property\n"
        "    def info(self): return PluginInfo(name='mcp-operator-fixture', version='1.0.0', plugin_type=PluginType.CUSTOM)\n"
        "    def setup(self): note('setup')\n"
        "    def teardown(self): note('teardown')\n"
        "    def register(self, manager):\n"
        "        manager.register_capability('row-positive', lambda row: row['x'] > 0)\n"
        "        manager.register_capability('expr-positive', pl.col('x') > 0)\n"
        "        manager.register_capability('expected-types', {'x': pl.Int64})\n"
        "        manager.register_capability('reference-frame', pl.DataFrame({'x': [-1, 1]}))\n"
        "        manager.register_capability('bad-column_count', 'wrong')\n"
        "        manager.register_capability('bad-table_row_count_range', 'wrong')\n"
        "        manager.register_capability('bad-column_exists', [['x']])\n"
        "        manager.register_capability('bad-table_column_types', {'x': None})\n"
        "        manager.register_capability('bad-table_schema_compare', object())\n"
        "        manager.register_capability('source-factory', source_factory)\n"
        "        manager.register_capability('async-source-factory', async_source_factory)\n"
        "        manager.register_capability('failed-async-source', failed_async_source_factory)\n"
        "        manager.register_capability('failed-source', failed_source_factory)\n"
        "        manager.register_capability('slow-source', slow_source_factory)\n"
        "        manager.register_capability('schema-factory', schema_factory)\n"
        "        manager.register_capability('alert-factory', alert_factory)\n"
        "        manager.register_capability('false-notifier', FalseNotifier)\n"
        "        manager.register_capability('checkpoint-factory', checkpoint_factory)\n"
        "        manager.register_capability('async-checkpoint-factory', async_checkpoint_factory)\n"
        "        manager.register_capability('checkpoint-runner-factory', checkpoint_runner_factory)\n"
        "        manager.register_capability('escalation-factory', escalation_factory)\n"
        "        manager.register_capability('broker-factory', broker_factory)\n"
        "        manager.register_capability('ai-factory', lambda: diagnosed_native(ai_factory))\n"
        "        manager.register_capability('audit-factory', audit_factory)\n"
        "        manager.register_capability('documentation-factory', documentation_factory)\n"
        "        manager.register_capability('benchmark-factory', benchmark_factory)\n"
        "        manager.register_capability('lineage-factory', lineage_factory)\n"
        "        manager.register_capability('openlineage-factory', openlineage_factory)\n"
        "        manager.register_capability('telemetry-factory', telemetry_factory)\n"
        "        manager.register_capability('dashboard-factory', dashboard_factory)\n"
        "        manager.register_capability('reflex-dashboard-factory', reflex_dashboard_factory)\n"
        "        manager.register_capability('security-factory', security_factory)\n"
        "        manager.register_capability('rate-limit-factory', rate_limit_factory)\n"
        "        manager.register_capability('tenant-factory', tenant_factory)\n"
        "        manager.register_capability('execution-factory', lambda: diagnosed_native(execution_factory))\n"
        "        manager.register_capability('store-factory', lambda: diagnosed_native(store_factory))\n"
        "        manager.register_capability('profiler-factory', lambda: diagnosed_native(profiler_factory))\n"
        "        from .ml_monitor_fixture import factory as ml_factory\n"
        "        manager.register_capability('ml-factory', ml_factory)\n"
        "        from .store_utility_fixture import factory as store_utility_factory\n"
        "        manager.register_capability('store-utility-factory', store_utility_factory)\n"
        "        manager.register_capability('streaming-store-factory', streaming_store_factory)\n"
        "        manager.register_capability('secret-factory', secret_factory)\n"
        "        manager.register_capability('plugin-factory', plugin_factory)\n",
        # Keep the fixture an installed distribution, including the new factories.
        encoding="utf-8",
    )
    # Ship the synthetic local SDK lifetime inside the installed fixture wheel;
    # the private worker cannot import arbitrary checkout helpers under -I.
    (module / "distributed_fixture.py").write_text(
        (Path(__file__).resolve().parents[1] / "verification/native_distributed_probe.py")
        .read_text(encoding="utf-8"), encoding="utf-8",
    )
    (module / "native_cloud_transports.py").write_text(
        Path(__file__).with_name("native_cloud_transports.py").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (module / "synthetic_s3.py").write_text(
        Path(__file__).with_name("synthetic_s3.py").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (module / "ml_monitor_fixture.py").write_text(
        Path(__file__).with_name("ml_monitor_fixture.py").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (module / "store_utility_fixture.py").write_text(
        Path(__file__).with_name("store_utility_fixture.py").read_text(encoding="utf-8"), encoding="utf-8",
    )
    metadata = build / "truthound_mcp_operator_fixture-1.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: truthound-mcp-operator-fixture\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (metadata / "entry_points.txt").write_text(
        "[truthound.plugins]\nfixture = mcp_operator_fixture:FixturePlugin\n", encoding="utf-8"
    )
    (metadata / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: truthound-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    records = [
        source,
        module / "distributed_fixture.py",
        module / "synthetic_s3.py",
        module / "native_cloud_transports.py",
        module / "ml_monitor_fixture.py",
        module / "store_utility_fixture.py",
        metadata / "METADATA",
        metadata / "entry_points.txt",
        metadata / "WHEEL",
        metadata / "RECORD",
    ]
    (metadata / "RECORD").write_text(
        "\n".join(path.relative_to(build).as_posix() + ",," for path in records) + "\n",
        encoding="utf-8",
    )
    wheel = tmp_path / "truthound_mcp_operator_fixture-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as bundle:
        for path in records:
            bundle.write(path, path.relative_to(build).as_posix())
    uv = shutil.which("uv")
    assert uv, "The mandatory plugin test requires the development uv installer"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-index", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    source = site / "mcp_operator_fixture/__init__.py"
    assert source.read_bytes() == (module / "__init__.py").read_bytes()
    monkeypatch.syspath_prepend(str(site))
    _, digest, _ = fingerprint("truthound-mcp-operator-fixture")
    assert not marker.exists()  # fingerprint is metadata-only
    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    configuration = {
        "plugins": {
            "fixture": {
                "distribution": "truthound-mcp-operator-fixture",
                "version": "1.0.0",
                "entry_point": "fixture",
                "sha256": digest,
                "capabilities": {"positive": "row-positive"},
            }
        },
        "validators": {
            "positive-rows": {
                "validator": "row_level",
                "settings": {
                    "row_validator": {"operator_capability": "positive"},
                    "columns": ["x"],
                },
            }
        },
    }
    bindings = control / "bindings.json"
    configuration["plugins"]["fixture"]["capabilities"].update(
        {
            "expr-positive": "expr-positive",
            "expected-types": "expected-types",
            "reference-frame": "reference-frame",
        }
    )
    configuration["validators"].update(
        {
            "native-expression": {
                "validator": "custom_expression",
                "object_parameters": {"filter_expr": "expr-positive"},
            },
            "native-dtypes": {
                "validator": "column_type",
                "object_parameters": {"expected_types": "expected-types"},
            },
            "native-frame": {
                "validator": "cross_table_row_count",
                "object_parameters": {"reference_data": "reference-frame"},
            },
        }
    )
    bindings.write_text(json.dumps(configuration), encoding="utf-8")
    bindings.chmod(0o600)
    adapter = MCPService(
        MCPPolicy((data,), profile="core", control_directory=control, allow_sensitive_results=True)
    )
    adapter._worker_argv = lambda: [str(python), "-I", "-m", "truthound.mcp.worker"]
    created = invoke(
        adapter,
        "assets_inline",
        columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [-1, 1]}],
    )
    assert created["execution_status"] == "succeeded", json.dumps(created)
    yield adapter, control, marker, source, created["data"]["resource_id"]
    adapter.close()


def request_and_approve(runtime, key):
    adapter, control, marker, _, asset = runtime
    request = {
        "asset_id": asset,
        "checks": [{"validator": "row_level", "binding_id": "positive-rows"}],
        "idempotency_key": key,
    }
    pending = invoke(adapter, "assets_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert "installed_extension_execute" in pending["data"]["effects"]
    assert not marker.exists()
    store = ApprovalStore(control)
    try:
        assert "constructor_settings" not in store.review(pending["data"]["plan_id"])["request"]
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    return request


def test_row_level_constructor_failure_is_effect_unknown(runtime):
    from truthound.validators import get_validator

    def callback(row):
        return row["x"] > 0

    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        get_validator("row_level")(row_validator=callback, columns=["x"], timeout_seconds=0.0)
    adapter, control, marker, _, _ = runtime
    path = control / "bindings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["validators"]["positive-rows"]["settings"]["timeout_seconds"] = 0.0
    path.write_text(json.dumps(config), encoding="utf-8")
    request = request_and_approve(runtime, "row-constructor-error")
    failed = invoke(adapter, "assets_validate", **request)
    assert failed["execution_status"] == "outcome_unknown"
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]
    replay = invoke(adapter, "assets_validate", **request)
    assert replay["execution_status"] == "outcome_unknown"


def test_actual_installed_entrypoint_executes_only_after_approval_through_core_lifecycle(runtime):
    adapter, _, marker, _, _ = runtime
    request = request_and_approve(runtime, "plugin-validation-1")
    result = invoke(adapter, "assets_validate", **request)
    assert result["execution_status"] == "succeeded", result
    assert result["data"]["summary"]["issue_count"] == 1
    assert result["data"]["summary"]["success"] is False
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]
    assert invoke(adapter, "assets_validate", **request)["data"]["replayed"]
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]


def test_changed_plugin_code_cannot_execute_under_old_package_digest(runtime):
    adapter, _, marker, source, _ = runtime
    request = request_and_approve(runtime, "plugin-validation-2")
    source.write_text(source.read_text(encoding="utf-8") + "\n# changed code\n", encoding="utf-8")
    result = invoke(adapter, "assets_validate", **request)
    assert result["execution_status"] == "outcome_unknown", result
    assert not marker.exists()
    assert invoke(adapter, "assets_validate", **request)["issues"] == [{"code": "outcome_unknown"}]


@pytest.mark.fault
def test_distribution_metadata_cannot_authorize_shadowed_module_execution(
    runtime, tmp_path, monkeypatch
):
    _, control, marker, _, _ = runtime
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "mcp_operator_fixture.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unapproved import')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(shadow))
    plugins = json.loads((control / "bindings.json").read_text(encoding="utf-8"))["plugins"]
    with (
        pytest.raises(MCPError, match="plugin_entrypoint_not_owned"),
        approved_capabilities(plugins),
    ):
        raise AssertionError("unapproved code must not run")
    assert not marker.exists()


@pytest.mark.parametrize(
    ("binding", "validator", "success"),
    [
        ("native-expression", "custom_expression", False),
        ("native-dtypes", "column_type", True),
        ("native-frame", "cross_table_row_count", True),
    ],
)
def test_reviewed_native_expression_dtype_and_frame_objects_are_not_lossy_wire_reconstructions(
    runtime, binding, validator, success
):
    adapter, control, marker, _, asset = runtime
    request = {
        "asset_id": asset,
        "checks": [{"validator": validator, "binding_id": binding}],
        "idempotency_key": "native-object-" + binding,
    }
    pending = invoke(adapter, "assets_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert not marker.exists()
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    result = invoke(adapter, "assets_validate", **request)
    assert result["execution_status"] == "succeeded", result
    assert result["data"]["summary"]["success"] is success
    assert result["provenance"]["authorization_plan_id"] == pending["data"]["plan_id"]
    assert marker.read_text(encoding="utf-8").splitlines() == ["import", "setup", "teardown"]
