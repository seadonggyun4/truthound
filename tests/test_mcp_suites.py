"""Kernel suites retain intent identity and execute through the real planner/runtime."""

import asyncio
import base64
import json

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.approval import ApprovalStore
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


def call(service, name, **arguments):
    response = invoke(service, name, **arguments)
    assert response["execution_status"] == "succeeded", response
    return response["data"]


def contents(service, identity):
    chunks, offset = [], 0
    while offset is not None:
        chunk = call(service, "artifacts_read", resource_id=identity, offset=offset)
        chunks.append(base64.b64decode(chunk["base64"]))
        offset = chunk["next_offset"]
    return json.loads(b"".join(chunks))


@pytest.fixture
def runtime(tmp_path):
    roots = [tmp_path / name for name in ("data", "workspace", "control")]
    for root in roots:
        root.mkdir(mode=0o700)
    service = MCPService(
        MCPPolicy(
            (roots[0],),
            profile="core",
            workspace_directories=(roots[1],),
            control_directory=roots[2],
            allow_sensitive_results=True,
        )
    )
    yield service, roots
    service.close()


def test_schema_only_suite_uses_native_schema_check_and_revocation(runtime):
    from truthound.core import ScanPlanner, ValidationRuntime, build_validation_asset
    from truthound.core.suite import EvidencePolicy, SchemaSpec, ValidationSuite
    from truthound.schema import Schema

    service, _ = runtime
    specification = {"columns": {"x": {"name": "x", "dtype": "Int64", "max_value": 10.0}}}
    schema = call(service, "schemas_create", schema_spec=specification)["resource_id"]
    suite = call(service, "suites_create", spec={"schema_id": schema})["resource_id"]
    asset = call(
        service,
        "assets_inline",
        columns=[
            {
                "name": "x",
                "dtype": {"kind": "Int64"},
                "values": [1, 50],
            }
        ],
    )["resource_id"]
    result = call(service, "suites_validate", suite_id=suite, asset_id=asset)
    actual = decode(contents(service, result["artifact"]["resource_id"]))
    policy = EvidencePolicy()
    native_schema = SchemaSpec(schema=Schema.from_dict(specification))
    native = ValidationSuite(
        name="mcp-suite",
        evidence_policy=policy,
        schema_spec=native_schema,
        checks=(native_schema.to_check_spec(evidence_policy=policy),),
    )
    frame = build_validation_asset(pl.DataFrame({"x": [1, 50]}))
    expected = (
        ValidationRuntime()
        .execute(
            asset=frame,
            plan=ScanPlanner().plan(suite=native, asset=frame),
        )
        .to_dict()
    )
    assert expected["issues"]
    for key in ("checks", "issues", "execution_issues", "metadata", "result_format"):
        assert actual[key] == expected[key], key
    conflict = invoke(
        service,
        "suites_create",
        spec={
            "schema_id": schema,
            "checks": [{"check_id": "schema", "validator": "null"}],
        },
    )
    assert conflict["execution_status"] == "failed"
    call(service, "resources_release", resource_id=schema)
    revoked = invoke(service, "suites_validate", suite_id=suite, asset_id=asset)
    assert revoked["execution_status"] == "failed"


def test_kernel_suite_matches_native_policies_metadata_parallel_and_persistence(runtime):
    from truthound.core import ScanPlanner, ValidationRuntime, build_validation_asset
    from truthound.core.suite import CheckSpec, EvidencePolicy, SeverityPolicy, ValidationSuite
    from truthound.types import ResultFormat, ResultFormatConfig, Severity
    from truthound.validators import get_validator

    service, (_, workspace, control) = runtime
    # Distinct failure frequencies make Core's hash-based top-k ordering
    # unambiguous across worker processes; every evidence field stays compared.
    frame = pl.DataFrame({"x": [-1, -1, 4, 150]})
    asset = call(
        service,
        "assets_inline",
        columns=[
            {
                "name": "x",
                "dtype": {"kind": "Int64"},
                "values": frame["x"].to_list(),
            }
        ],
    )["resource_id"]
    spec = {
        "name": "production-intent",
        "checks": [
            {
                "check_id": "bounded-age",
                "validator": "between",
                "check_name": "between",
                "settings": {"columns": ["x"], "min_value": 0, "max_value": 120},
                "tags": ["required"],
                "metadata": {"owner": "quality"},
            },
            {"check_id": "non-null", "validator": "null", "settings": {"columns": ["x"]}},
        ],
        "evidence_policy": {"result_format": {"format": "complete", "partial_unexpected_count": 1}},
        "severity_policy": {"min_severity": "high"},
        "metadata": {"revision": 7},
    }
    suite = call(service, "suites_create", spec=spec)["resource_id"]
    inspected = decode(call(service, "suites_inspect", resource_id=suite)["result"])
    assert inspected["checks"][0]["check_id"] == "bounded-age"
    assert contents(service, suite)["metadata"] == {"revision": 7}
    native = ValidationSuite(
        name=spec["name"],
        checks=(
            CheckSpec(
                id="bounded-age",
                name="between",
                category="distribution",
                tags=("required",),
                metadata={"owner": "quality"},
                factory=lambda: get_validator("between")(columns=["x"], min_value=0, max_value=120),
            ),
            CheckSpec(
                id="non-null",
                name="null",
                category="completeness",
                factory=lambda: get_validator("null")(columns=["x"]),
            ),
        ),
        evidence_policy=EvidencePolicy(
            ResultFormatConfig(format=ResultFormat.COMPLETE, partial_unexpected_count=1)
        ),
        severity_policy=SeverityPolicy(Severity.HIGH),
        metadata={"revision": 7},
    )
    data_asset = build_validation_asset(frame)
    direct = (
        ValidationRuntime()
        .execute(
            asset=data_asset,
            plan=ScanPlanner().plan(suite=native, asset=data_asset, parallel=True, max_workers=2),
        )
        .filter_by_severity(Severity.HIGH)
        .to_dict()
    )
    result = call(
        service, "suites_validate", asset_id=asset, suite_id=suite, parallel=True, max_workers=2
    )
    actual = decode(contents(service, result["artifact"]["resource_id"]))
    for field in (
        "checks",
        "issues",
        "execution_issues",
        "suite_name",
        "metadata",
        "result_format",
        "execution_mode",
    ):
        assert actual[field] == direct[field], field
    assert not (workspace / ".truthound").exists()
    context = call(service, "contexts_open", path=str(workspace))["context_id"]
    request = {
        "asset_id": asset,
        "suite_id": suite,
        "context_id": context,
        "idempotency_key": "kernel-persist-01",
    }
    pending = invoke(service, "suites_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    completed = call(service, "suites_validate", **request)
    assert not completed["replayed"]
    assert call(service, "suites_validate", **request)["replayed"]
    paths = list((workspace / ".truthound/runs").glob("*.json"))
    assert len(paths) == 1 and json.loads(paths[0].read_text())["suite_name"] == spec["name"]


@pytest.mark.fault
def test_suite_rejects_duplicate_ids_overrides_forgery_and_releases_descendants(runtime):
    service, _ = runtime
    check = {"check_id": "required", "validator": "null"}
    assert invoke(service, "suites_create", spec={"checks": [check, check]})["issues"] == [
        {"code": "invalid_arguments"}
    ]
    suite = call(service, "suites_create", spec={"checks": [check]})["resource_id"]
    asset = call(
        service, "assets_inline", columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1]}]
    )["resource_id"]
    for field, value in (("checks", []), ("result_format", "boolean_only"), ("approved", True)):
        denied = invoke(
            service, "suites_validate", asset_id=asset, suite_id=suite, **{field: value}
        )
        assert denied["issues"] == [{"code": "invalid_arguments"}]
    result = call(service, "suites_validate", asset_id=asset, suite_id=suite)
    call(service, "resources_release", resource_id=suite)
    assert (
        invoke(service, "artifacts_read", resource_id=result["artifact"]["resource_id"])[
            "execution_status"
        ]
        == "failed"
    )
