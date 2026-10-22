"""Execute native metadata audits through owned MCP artifacts, including failures."""

import asyncio
import base64
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode
from truthound.schema_audit import (
    VERSION,
    CatalogSnapshot,
    Coverage,
    Identifier,
    NameOrigin,
    Reason,
    audit_implementation,
    audit_standards,
    dump_document,
    semantic_digest,
)

pytestmark = pytest.mark.contract


def documents():
    path = Path(__file__).parent / "schema_audit/test_standards.py"
    spec = importlib.util.spec_from_file_location("mcp_native_standard_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    standard, design = module.fixture()
    table = design.tables[0]

    def ident(value):
        return replace(
            value,
            origin=NameOrigin.CATALOG,
            text=value.text.lower() if value.origin is NameOrigin.UNQUOTED else value.text,
        )

    table = replace(
        table,
        namespace=ident(table.namespace),
        physical_name=ident(table.physical_name),
        logical_name=replace(table.logical_name, source="COMMENT"),
        columns=tuple(
            replace(
                c,
                physical_name=ident(c.physical_name),
                logical_name=replace(c.logical_name, source="COMMENT"),
            )
            for c in table.columns
        ),
        constraints=tuple(replace(c, name=ident(c.name)) for c in table.constraints),
    )
    catalog = CatalogSnapshot(
        contract_version=VERSION,
        kind="CATALOG",
        id="snapshot",
        revision=1,
        dialect=design.dialect,
        binding_ref="synthetic",
        adapter_version="test",
        captured_at="2026-10-22T00:00:00Z",
        scope=(Identifier(text="public", origin=NameOrigin.CATALOG),),
        coverage=tuple(
            Coverage(facet=f, state="COMPLETE", reason=Reason.NONE)
            for f in ("TABLES", "COLUMNS", "CONSTRAINTS", "LOGICAL_NAMES", "TYPES", "DEFAULTS")
        ),
        tables=(table,),
        sources=(),
    )
    return standard, design, catalog


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


def call(service, name, **arguments):
    result = invoke(service, name, **arguments)
    assert result["execution_status"] == "succeeded", result
    return result


def read(service, identity):
    chunks, offset = [], 0
    while offset is not None:
        part = call(service, "artifacts_read", resource_id=identity, offset=offset)["data"]
        chunks.append(base64.b64decode(part["base64"]))
        offset = part["next_offset"]
    return decode(json.loads(b"".join(chunks)))


def retain(service, document):
    return call(
        service, "schema_audit_create", document_json=dump_document(document).decode("utf-8")
    )["data"]["resource_id"]


@pytest.fixture
def service(tmp_path):
    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    instance = MCPService(
        MCPPolicy((data,), profile="core", allow_sensitive_results=True, control_directory=control)
    )
    yield instance
    instance.close()


@pytest.mark.parametrize("mode", ["pass", "fail", "unknown"])
def test_standard_and_implementation_reports_preserve_native_evidence(service, mode):
    standard, design, catalog = documents()
    if mode == "fail":
        table = design.tables[0]
        design = replace(
            design,
            tables=(
                replace(table, physical_name=Identifier(text="wrong", origin=NameOrigin.UNQUOTED)),
            ),
        )
    elif mode == "unknown":
        design = replace(design, tables=())
        catalog = replace(catalog, tables=())
    standard_id, design_id, catalog_id = (retain(service, d) for d in (standard, design, catalog))
    for name, parameters, expected in (
        (
            "schema_audit_standards",
            {"standard_id": standard_id, "design_id": design_id},
            audit_standards(standard, design),
        ),
        (
            "schema_audit_implementation",
            {"design_id": design_id, "catalog_id": catalog_id},
            audit_implementation(design, catalog),
        ),
    ):
        result = call(service, name, **parameters)
        assert result["quality_status"] == mode
        assert read(service, result["data"]["resource_id"]) == expected.as_dict()
        assert result["coverage"]["target_count"] == expected.target_count
    inspected = call(service, "schema_audit_inspect", resource_id=design_id)
    value = read(service, inspected["data"]["resource_id"])
    assert value["semantic_digest"] == semantic_digest(design)
    assert value["canonical_json"].encode() == dump_document(design)
    call(service, "resources_release", resource_id=design_id)
    denied = invoke(service, "artifacts_read", resource_id=inspected["data"]["resource_id"])
    assert denied["issues"] == [{"code": "invalid_handle"}]


@pytest.mark.parametrize(
    "raw",
    [
        '{"kind":"DESIGN","kind":"CATALOG"}',
        '{"kind":"CODE","module":"os"}',
        '{"kind":"DESIGN","secret":1.5}',
        "{",
    ],
)
def test_schema_audit_invalid_document_cannot_create_a_successful_result(service, raw):
    result = invoke(service, "schema_audit_create", document_json=raw)
    assert result["execution_status"] == "failed" and result["quality_status"] == "unknown"
    assert result["data"] is None


def test_schema_audit_operator_mapping_requires_approval_and_revokes_artifact(service):
    from truthound.mcp.approval import ApprovalStore
    from truthound.schema_audit import TermBinding

    standard, design, _ = documents()
    binding = TermBinding(
        object_id="column",
        term_id="customer-number",
        standard_digest=semantic_digest(standard),
        design_digest=semantic_digest(design),
        approval_digest="a" * 64,
    )
    from dataclasses import asdict

    path = service.policy.control_directory / "bindings.json"
    path.write_text(json.dumps({"schema_audit": {"approved": [asdict(binding)]}}), encoding="utf-8")
    path.chmod(0o600)
    arguments = {
        "standard_id": retain(service, standard),
        "design_id": retain(service, design),
        "binding_id": "approved",
        "idempotency_key": "audit-1",
    }
    pending = invoke(service, "schema_audit_standards", **arguments)
    assert pending["issues"] == [{"code": "approval_required"}]
    approval = ApprovalStore(service.policy.control_directory)
    try:
        approval.approve(pending["data"]["plan_id"])
    finally:
        approval.close()
    actual = call(service, "schema_audit_standards", **arguments)
    assert (
        read(service, actual["data"]["resource_id"])
        == audit_standards(standard, design, bindings=(binding,)).as_dict()
    )
    replay = call(service, "schema_audit_standards", **arguments)
    assert replay["data"]["resource_id"] == actual["data"]["resource_id"]
    path.write_text('{"schema_audit":{}}', encoding="utf-8")
    denied = invoke(service, "artifacts_read", resource_id=actual["data"]["resource_id"])
    assert denied["issues"] == [{"code": "access_denied"}]


def test_schema_audit_wrong_document_kind_and_foreign_handle_fail(service, tmp_path):
    standard, design, _ = documents()
    standard_id = retain(service, standard)
    result = invoke(
        service, "schema_audit_implementation", design_id=standard_id, catalog_id=standard_id
    )
    assert result["execution_status"] == "failed"
    other = MCPService(service.policy)
    try:
        assert invoke(other, "schema_audit_inspect", resource_id=standard_id)["issues"] == [
            {"code": "invalid_handle"}
        ]
    finally:
        other.close()


def test_schema_audit_native_contract_and_conditional_authority(service):
    from truthound.schema_audit import json_schema

    result = call(service, "schema_audit_contract")
    assert "input_identity" not in result["provenance"]
    assert read(service, result["data"]["resource_id"]) == json_schema()
    descriptor = call(service, "capability_describe", operation_id="schema_audit_standards")
    assert descriptor["data"]["conditional_grants"]["binding_id supplied"] == [
        "control_directory",
        "operator_plan_approval",
    ]


def test_schema_audit_quota_failure_keeps_native_error_and_discards_partial_file(tmp_path):
    instance = MCPService(MCPPolicy((tmp_path,), profile="core", max_session_bytes=1024))
    try:
        result = invoke(instance, "schema_audit_contract")
        assert result["execution_status"] == "failed"
        assert result["issues"] == [{"code": "resource_exhausted"}]
        assert instance._session.resources.available_bytes == 1024
        assert not instance._session.resources._pending
        assert not list(instance._session.resources.root.iterdir())
    finally:
        instance.close()
