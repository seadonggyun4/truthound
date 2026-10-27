"""Real Core re-entry and strict constructor/secondary-asset boundaries."""

import asyncio
import base64
import inspect
import json

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.approval import ApprovalStore
from truthound.mcp.encoding import encode
from truthound.mcp.operations import _context
from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.validator_wire import constructor, contracts, settings

pytestmark = pytest.mark.contract


def test_float_contract_preserves_finite_integers_without_coercion():
    from truthound.mcp.validator_wire import convert

    for value in (10**400, -(10**400), 1, 0.25):
        actual = convert({"kind": "float"}, value, build=True)
        assert actual == value and type(actual) is type(value)
    for value in (float("inf"), float("-inf"), float("nan"), True, "1"):
        with pytest.raises(MCPError, match="invalid_arguments"):
            convert({"kind": "float"}, value, build=True)


def test_polygon_files_need_operator_binding_before_native_or_filesystem_access():
    with pytest.raises(MCPError, match="operator_configuration_required"):
        settings(
            "polygon",
            {"lat_column": "lat", "lon_column": "lon", "geojson_file": "/ungranted/boundary.json"},
        )


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


@pytest.fixture
def service(tmp_path):
    roots = [tmp_path / key for key in ("data", "workspace", "control")]
    for root in roots:
        root.mkdir(mode=0o700)
    adapter = MCPService(
        MCPPolicy(
            (roots[0],),
            profile="core",
            workspace_directories=(roots[1],),
            control_directory=roots[2],
            allow_sensitive_results=True,
        )
    )
    yield adapter, roots
    adapter.close()


def asset(service):
    result = invoke(
        service,
        "assets_inline",
        columns=[
            {"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 3, None, 3]},
            {"name": "s", "dtype": {"kind": "String"}, "values": ["a", "bb", "bad", None]},
        ],
    )
    assert result["execution_status"] == "succeeded", result
    return result["data"]["resource_id"]


def test_native_nested_configuration_rejection_is_not_generic_worker_failure(service):
    from truthound.validators.base import ValidatorConfig

    with pytest.raises(ValueError, match="sample_size"):
        ValidatorConfig(sample_size=-1)
    adapter, _ = service
    response = invoke(
        adapter,
        "assets_validate",
        asset_id=asset(adapter),
        checks=[{"validator": "null", "settings": {"config": {"sample_size": -1}}}],
    )
    assert response["execution_status"] == "failed"
    assert response["issues"] == [{"code": "validator_configuration_error"}]
    assert response["data"] is None


def canonical(service, result):
    assert result["execution_status"] == "succeeded", result
    handle = result["data"]["artifact"]["resource_id"]
    pieces, offset = [], 0
    while offset is not None:
        page = invoke(service, "artifacts_read", resource_id=handle, offset=offset, limit=32768)
        assert page["execution_status"] == "succeeded", page
        pieces.append(base64.b64decode(page["data"]["base64"]))
        offset = page["data"]["next_offset"]
    return json.loads(b"".join(pieces))


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("null", {"columns": ["x"], "mostly": 0.5, "sample_size": 2}),
        (
            "not_null",
            {
                "config": {
                    "columns": ["s"],
                    "result_format": {"format": "complete", "partial_unexpected_count": 2},
                }
            },
        ),
        ("unique", {"columns": ["x"], "severity_override": "critical"}),
        ("duplicate", {"subset": ["x"]}),
        ("between", {"min_value": 2, "max_value": 4, "columns": ["x"]}),
        ("length", {"min_length": 2, "columns": ["s"]}),
        ("column_count", {"exact_count": 3}),
        ("row_count", {"min_count": 5}),
        ("column_exists", {"columns": ["absent"]}),
        ("column_type", {"expected_types": {"x": "String"}}),
        ("in_set", {"allowed_values": [1, 2], "columns": ["x"]}),
        ("feature_scale", {"max_magnitude": 2.0, "columns": ["x"]}),
    ],
)
def test_registry_call_preserves_canonical_core_result(service, tmp_path, name, config):
    import truthound as th

    adapter, _ = service
    handle = asset(adapter)
    frame = pl.DataFrame({"x": [1, 3, None, 3], "s": ["a", "bb", "bad", None]})
    # Construct the expected Core instance independently of wire conversion.
    # Enum/config objects are materialized deliberately.
    direct_config = dict(config)
    if name == "not_null":
        from truthound.types import ResultFormat, ResultFormatConfig
        from truthound.validators.base import ValidatorConfig

        direct_config = {
            "config": ValidatorConfig(
                columns=("s",),
                result_format=ResultFormatConfig(
                    format=ResultFormat.COMPLETE, partial_unexpected_count=2
                ),
            )
        }
    if "severity_override" in direct_config:
        from truthound.types import Severity

        direct_config["severity_override"] = Severity(direct_config["severity_override"])
    from truthound.validators import get_validator

    direct = th.check(
        frame,
        validators=[get_validator(name)(**direct_config)],
        context=_context(str(tmp_path / "direct/input.ipc")),
    )
    result = invoke(
        adapter,
        "assets_validate",
        asset_id=handle,
        checks=[{"validator": name, "settings": config}],
    )
    actual = canonical(adapter, result)
    expected = encode(direct.to_dict())
    # Identity/time/workspace are intentionally execution-local, not algorithm
    # outcomes. Every issue, detail, evidence field and check verdict is exact.
    for field in (
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
        assert actual["entries"][field] == expected["entries"][field], field
    assert result["quality_status"] == "unknown"  # no invented global accounting


def test_secondary_frames_expression_and_learned_schema_reentry(service, tmp_path):
    import truthound as th
    from truthound.validators import get_validator

    adapter, _ = service
    handle = asset(adapter)
    reference = asset(adapter)
    configured = {"reference_data": {"asset_id": reference}, "tolerance": 0}
    result = invoke(
        adapter,
        "assets_validate",
        asset_id=handle,
        checks=[{"validator": "cross_table_row_count", "settings": configured}],
    )
    assert canonical(adapter, result)["entries"]["checks"][0]["entries"]["success"] is True
    expression = {
        "op": "gt",
        "left": {"op": "column", "name": "x"},
        "right": {"op": "literal", "value": 2},
    }
    result = invoke(
        adapter,
        "assets_validate",
        asset_id=handle,
        checks=[{"validator": "custom_expression", "settings": {"filter_expr": expression}}],
    )
    frame = pl.DataFrame({"x": [1, 3, None, 3], "s": ["a", "bb", "bad", None]})
    direct = th.check(
        frame,
        validators=[get_validator("custom_expression")(filter_expr=pl.col("x") > 2)],
        context=_context(str(tmp_path / "direct/input.ipc")),
    )
    assert (
        canonical(adapter, result)["entries"]["issues"]
        == encode(direct.to_dict())["entries"]["issues"]
    )
    learned = invoke(adapter, "schemas_learn", asset_id=handle)["data"]["resource_id"]
    result = invoke(
        adapter,
        "assets_validate",
        asset_id=handle,
        schema_id=learned,
        checks=[{"validator": "null"}],
    )
    assert canonical(adapter, result)["entries"]["checks"][0]["entries"]["name"] == "schema"
    invoke(adapter, "resources_release", resource_id=reference)
    revoked = invoke(
        adapter,
        "assets_validate",
        asset_id=handle,
        checks=[{"validator": "cross_table_row_count", "settings": configured}],
    )
    assert revoked["issues"] == [{"code": "invalid_handle"}]


def test_persistent_validation_is_approved_once_and_replays_without_core_write(service):
    adapter, (_, workspace, control) = service
    handle = asset(adapter)
    context = invoke(adapter, "contexts_open", path=str(workspace))["data"]["context_id"]
    request = {
        "asset_id": handle,
        "context_id": context,
        "checks": [{"validator": "null"}],
        "idempotency_key": "persist-validation-1",
    }
    pending = invoke(adapter, "assets_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    assert not (workspace / ".truthound").exists()
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    completed = invoke(adapter, "assets_validate", **request)
    canonical(adapter, completed)
    runs = list((workspace / ".truthound/runs").glob("*.json"))
    assert len(runs) == 1
    stamp = runs[0].stat().st_mtime_ns
    replay = invoke(adapter, "assets_validate", **request)
    assert replay["data"]["replayed"]
    assert len(list((workspace / ".truthound/runs").glob("*.json"))) == 1
    assert runs[0].stat().st_mtime_ns == stamp


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("null", {"typo_columns": ["x"]}),
        ("null", {"sample_size": True}),
        ("column_type", {"expected_types": {"x": {"kind": "String", "time_zone": "UTC"}}}),
        ("json_schema", {"schema": {"$ref": "file:///etc/passwd"}}),
        ("query_returns_rows", {"query": "SELECT * FROM read_csv('/etc/passwd')"}),
        ("custom_expression", {"filter_expr": {"op": "eval", "value": "import os"}}),
    ],
)
def test_closed_constructor_rejects_silent_drops_and_effectful_strings(name, config):
    with pytest.raises(MCPError):
        settings(name, config)


@pytest.mark.parametrize(
    "nested",
    [
        {"properties": {"id": {"$ref": "https://example.test/schema"}}},
        {"$defs": {"value": {"$dynamicRef": "file:///secret"}}},
        {"items": [{"$recursiveRef": "https://example.test/schema"}]},
        {"dependencies": {"x": {"properties": {"y": {"pattern": "a+"}}}}},
        {"if": {"required": ["x"]}, "then": {"$id": "https://example.test/schema"}},
        {"prefixItems": [{"patternProperties": {"a+": {"type": "string"}}}]},
    ],
)
def test_json_schema_nested_execution_keywords_still_require_operator(nested):
    with pytest.raises(MCPError, match="operator_configuration_required"):
        settings("json_schema", {"schema": nested})


def test_json_schema_property_names_and_literal_data_are_not_executable_keywords():
    document = {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "pattern": {"type": "string"}},
        "$defs": {"id": {"type": "integer"}},
        "additionalProperties": False,
        "const": {"$ref": "https://example.test/literal", "pattern": "literal"},
        "examples": [{"$id": "https://example.test/literal"}],
    }
    assert settings("json_schema", {"schema": document})["schema"] == document


def test_internally_supplied_parent_arguments_are_not_required_or_caller_overrides():
    config = {
        "antecedent_column": "x",
        "antecedent_value": 1,
        "consequent_column": "y",
        "consequent_value": 2,
    }
    assert settings("column_implication", config) == config
    with pytest.raises(MCPError):
        settings("column_implication", {**config, "columns": ["other", "y"]})


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#/const", "const": {"$ref": "https://example.invalid/schema"}},
        {"$ref": "#%2Fdefault", "default": {"pattern": "a+"}},
        {"$ref": "#/examples/0", "examples": [{"$ref": "file:///secret"}]},
        {"$ref": "#/default/a~1b", "default": {"a/b": {"$ref": "https://example.invalid/schema"}}},
        {"$ref": "#hidden", "const": {"$id": "#hidden", "$ref": "https://example.invalid/schema"}},
    ],
)
def test_local_schema_refs_cannot_turn_literal_data_into_external_execution(schema):
    with pytest.raises(MCPError, match="operator_configuration_required"):
        settings("json_schema", {"schema": schema})


def test_local_schema_refs_preserve_safe_pointer_anchor_and_literal_names():
    for document in (
        {"$ref": "#/const", "const": {"type": "integer"}},
        {"$ref": "#/definitions/id", "definitions": {"id": {"type": "integer"}}},
        {"$ref": "#value", "definitions": {"x": {"$id": "#value", "type": "integer"}}},
        {"$ref": "#", "properties": {"id": {"type": "integer"}}},
    ):
        assert settings("json_schema", {"schema": document})["schema"] == document


def test_schema_reference_walk_has_a_separate_bounded_depth_budget():
    definitions = {f"n{i}": {"$ref": f"#/definitions/n{i + 1}"} for i in range(40)}
    definitions["n40"] = {"type": "integer"}
    with pytest.raises(MCPError, match="resource_exhausted"):
        settings(
            "json_schema", {"schema": {"$ref": "#/definitions/n0", "definitions": definitions}}
        )


def test_every_registered_concrete_source_has_an_exact_reviewable_contract():
    from truthound.validators.registry import registry

    registered = registry.list_all()
    concrete = {name: cls for name, cls in registered.items() if not inspect.isabstract(cls)}
    assert set(concrete) == set(contracts()["validators"])
    assert len(concrete) == 260  # includes all six native streaming validators
    from truthound.validators.streaming import completeness
    from truthound.validators.streaming import range as streaming_range

    for module in (completeness, streaming_range):
        for cls in vars(module).values():
            if inspect.isclass(cls) and cls.__module__ == module.__name__:
                assert registry.get(cls.name) is cls
    assert registry.list_all() == registered
    for name, cls in concrete.items():
        assert constructor(name) is cls


def test_mcp_complete_evidence_stays_on_failed_column_after_passing_column(service):
    adapter, _ = service
    captured = invoke(
        adapter, 'assets_inline',
        columns=[
            {'name': 'first', 'dtype': {'kind': 'Int64'}, 'values': [1, 2, 3]},
            {'name': 'second', 'dtype': {'kind': 'Int64'}, 'values': [2, -7, -7]},
        ],
    )
    assert captured['execution_status'] == 'succeeded', captured
    result = invoke(
        adapter, 'assets_validate', asset_id=captured['data']['resource_id'], result_format='complete',
        checks=[{'validator': 'between', 'settings': {
            'min_value': 0, 'max_value': 5,
            'config': {'columns': ['first', 'second'], 'result_format': {'format': 'complete'}},
        }}],
    )
    report = canonical(adapter, result)['entries']
    assert len(report['issues']) == 1
    issue = report['issues'][0]['entries']
    assert issue['column'] == 'second' and issue['count'] == 2
    assert issue['sample_values'] == [-7, -7]
    evidence = issue['result']['entries']
    assert evidence['partial_unexpected_counts'] == [encode({'value': -7, 'count': 2})]
    assert evidence['unexpected_index_list'] == [1, 2]
