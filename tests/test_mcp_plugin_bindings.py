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
    from truthound.validators import get_validator
    from truthound.mcp.validator_wire import settings

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
    from truthound.validators import get_validator
    from truthound.mcp.operations import _context
    from truthound.mcp.tables import decode

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


@pytest.fixture
def runtime(tmp_path, monkeypatch):
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
        "        manager.register_capability('bad-table_schema_compare', object())\n",
        encoding="utf-8",
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
