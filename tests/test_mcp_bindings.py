"""Protected exact operator templates are not model consent or executable strings."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from truthound.mcp.approval import ApprovalStore
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


def invoke(adapter, name, **arguments):
    return asyncio.run(adapter.call_operation_async(name, arguments))


@pytest.fixture
def operator(tmp_path):
    data, control = tmp_path / "data", tmp_path / "control"
    for directory in (data, control):
        directory.mkdir(mode=0o700)
    file = control / "bindings.json"
    file.write_text(
        json.dumps(
            {
                "validators": {
                    "capital-a": {
                        "validator": "regex",
                        "settings": {"pattern": "^A+$", "columns": ["s"]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    file.chmod(0o600)
    adapter = MCPService(
        MCPPolicy((data,), profile="core", control_directory=control, allow_sensitive_results=True)
    )
    created = invoke(
        adapter,
        "assets_inline",
        columns=[{"name": "s", "dtype": {"kind": "String"}, "values": ["AA", "b"]}],
    )
    assert created["execution_status"] == "succeeded", created
    yield adapter, control, file, created["data"]["resource_id"]
    adapter.close()


def approve(control, plan_id):
    store = ApprovalStore(control)
    try:
        reviewed = store.review(plan_id)
        assert len(reviewed["request"]["constructor_digest"]) == 64
        assert reviewed["request"]["checks"][0]["binding_id"] == "capital-a"
        store.approve(plan_id)
    finally:
        store.close()


def test_optional_dependency_probes_preserve_late_bound_pydantic_field_identity(operator, tmp_path):
    import importlib.util
    from pathlib import Path

    import pydantic

    original = pydantic
    descriptor = importlib.util.spec_from_file_location(
        "truthound_ai_optional_probe_cases", Path(__file__).with_name("test_ai_import_contract.py")
    )
    probes = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(probes)
    for function in (
        probes.test_truthound_ai_namespace_requires_optional_dependency,
        probes.test_truthound_root_ai_support_probe_is_safe_without_optional_dependency,
    ):
        with pytest.MonkeyPatch.context() as patches:
            function(patches)
        assert __import__("pydantic") is original
    with pytest.MonkeyPatch.context() as patches:
        probes.test_truthound_ai_cli_command_surfaces_install_hint_when_optional_dependency_missing(
            patches, tmp_path
        )
    assert __import__("pydantic") is original
    from truthound.mcp.bindings import Bindings

    assert Bindings().plugins == {}
    adapter, _, _, asset = operator
    response = invoke(
        adapter,
        "assets_validate",
        asset_id=asset,
        checks=[{"validator": "regex", "binding_id": "capital-a"}],
        idempotency_key="after-optional-dependency-probes",
    )
    assert response["issues"] == [{"code": "approval_required"}]


def test_regex_template_requires_exact_human_approval_then_real_core_run_and_revocation(operator):
    adapter, control, file, asset = operator
    request = {
        "asset_id": asset,
        "checks": [{"validator": "regex", "binding_id": "capital-a"}],
        "idempotency_key": "trusted-regex-run-1",
    }
    pending = invoke(adapter, "assets_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert pending["data"]["effects"] == ["execute_trusted_binding"]
    approve(control, pending["data"]["plan_id"])
    completed = invoke(adapter, "assets_validate", **request)
    assert completed["execution_status"] == "succeeded", completed
    assert completed["data"]["summary"]["success"] is False
    assert completed["data"]["summary"]["issue_count"] == 1
    assert not completed["data"]["replayed"]
    replay = invoke(adapter, "assets_validate", **request)
    assert replay["data"]["replayed"]
    for key in ("authorization_plan_id", "configuration_identity"):
        assert replay["provenance"][key] == completed["provenance"][key]
    artifact = completed["data"]["artifact"]["resource_id"]
    assert (
        invoke(adapter, "artifacts_read", resource_id=artifact)["execution_status"] == "succeeded"
    )
    file.write_text('{"validators":{}}', encoding="utf-8")
    assert invoke(adapter, "artifacts_read", resource_id=artifact)["issues"] == [
        {"code": "access_denied"}
    ]


def test_template_changes_or_overrides_cannot_reuse_approval(operator):
    adapter, control, file, asset = operator
    request = {
        "asset_id": asset,
        "checks": [{"validator": "regex", "binding_id": "capital-a"}],
        "idempotency_key": "trusted-regex-run-2",
    }
    pending = invoke(adapter, "assets_validate", **request)
    approve(control, pending["data"]["plan_id"])
    raw = json.loads(file.read_text(encoding="utf-8"))
    raw["validators"]["capital-a"]["settings"]["pattern"] = ".*"
    file.write_text(json.dumps(raw), encoding="utf-8")
    changed = invoke(adapter, "assets_validate", **request)
    assert changed["issues"] == [{"code": "idempotency_conflict"}]
    override = invoke(
        adapter,
        "assets_validate",
        asset_id=asset,
        checks=[
            {"validator": "regex", "binding_id": "capital-a", "settings": {"columns": ["other"]}}
        ],
        idempotency_key="trusted-regex-run-3",
    )
    assert override["issues"] == [{"code": "binding_override_denied"}]
    forged = invoke(adapter, "assets_validate", **request, approved=True)
    assert forged["issues"] == [{"code": "invalid_arguments"}]


def test_unbound_regex_is_still_denied_and_binding_does_not_enable_imports_or_callbacks(operator):
    adapter, _, _, asset = operator
    result = invoke(
        adapter,
        "assets_validate",
        asset_id=asset,
        checks=[{"validator": "regex", "settings": {"pattern": ".*"}}],
    )
    assert result["issues"] == [{"code": "operator_configuration_required"}]
    unknown = invoke(
        adapter,
        "assets_validate",
        asset_id=asset,
        checks=[{"validator": "os.system", "binding_id": "capital-a"}],
        idempotency_key="unknown-validator-1",
    )
    assert unknown["issues"] == [{"code": "unknown_validator"}]


def test_exact_sql_template_matches_core_without_durable_query_text(operator):
    import base64

    import polars as pl

    import truthound as th
    from truthound.context import TruthoundContext, TruthoundContextConfig
    from truthound.mcp.tables import decode
    from truthound.validators import get_validator

    adapter, control, file, asset = operator
    query = "SELECT * FROM data WHERE s = 'AA'"
    file.write_text(
        json.dumps(
            {
                "validators": {
                    "exact-query": {
                        "validator": "query_row_count",
                        "settings": {"query": query, "expected_count": 2},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    request = {
        "asset_id": asset,
        "checks": [
            {
                "validator": "query_row_count",
                "binding_id": "exact-query",
            }
        ],
        "idempotency_key": "trusted-query-run-1",
    }
    pending = invoke(adapter, "assets_validate", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(control)
    try:
        review = store.review(pending["data"]["plan_id"])
        assert query not in json.dumps(review)
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    completed = invoke(adapter, "assets_validate", **request)
    assert completed["execution_status"] == "succeeded", completed
    assert completed["data"]["summary"]["success"] is False
    chunk = invoke(
        adapter, "artifacts_read", resource_id=completed["data"]["artifact"]["resource_id"]
    )
    actual = decode(json.loads(base64.b64decode(chunk["data"]["base64"])))
    context = TruthoundContext(
        control / "direct-only",
        config=TruthoundContextConfig(
            persist_runs=False,
            persist_docs=False,
        ),
    )
    direct = th.check(
        pl.DataFrame({"s": ["AA", "b"]}),
        validators=[
            get_validator("query_row_count")(query=query, expected_count=2),
        ],
        context=context,
    ).to_dict()
    assert actual["issues"] == direct["issues"]
    assert actual["checks"] == direct["checks"]
    for path in control.glob("*.sqlite*"):
        assert query.encode() not in path.read_bytes()
