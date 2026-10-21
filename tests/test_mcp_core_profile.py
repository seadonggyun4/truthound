"""Core-profile contracts: actual workers/SDK and failure-first semantic parity."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from truthound.mcp.catalog import CATALOG
from truthound.mcp.encoding import encode
from truthound.mcp.operations import _context, execute
from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text("email,age\na@example.com,10\na@example.com,10\n,150\n", encoding="utf-8")
    return path


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


def service_for(path, *, consent=False, **options):
    return MCPService(
        MCPPolicy((path.parent,), profile="core", allow_sensitive_results=consent, **options)
    )


@pytest.mark.parametrize(
    "check",
    [
        {"validator": "null"},
        {"validator": "not_null", "columns": ["email"], "mostly": 0.9},
        {"validator": "unique", "columns": ["email"]},
        {"validator": "duplicate", "subset": ["age"]},
        {"validator": "between", "columns": ["age"], "min_value": 0, "max_value": 100},
        {"validator": "length", "columns": ["email"], "exact_length": 5},
    ],
)
def test_typed_rules_execute_same_core_semantics(dataset, check):
    import truthound as th
    from truthound.validators import get_validator

    result = invoke(service_for(dataset), "validation_check", path=str(dataset), checks=[check])
    assert not (dataset.parent / ".truthound").exists()
    reference = dataset.parent / "reference"
    reference.mkdir()
    validator = get_validator(check["validator"])(
        **{k: v for k, v in check.items() if k != "validator"}
    )
    direct = th.check(
        str(dataset), context=_context(str(reference / "input.csv")), validators=[validator]
    )
    assert result["execution_status"] == "succeeded", result
    assert result["data"]["success"] == direct.success
    assert result["data"]["issue_count"] == len(direct.issues)
    assert result["data"]["checks"] == [
        {"name": c.name, "success": c.success, "issue_count": c.issue_count} for c in direct.checks
    ]
    assert result["quality_status"] == ("pass" if direct.success else "fail")
    assert "input_sha256" not in result["provenance"]
    assert len(result["provenance"]["input_identity"]) == 64
    assert "a@example.com" not in json.dumps(result)
    assert not (dataset.parent / ".truthound").exists()


@pytest.mark.parametrize(
    "name,extra",
    [
        ("data_read", {}),
        ("data_mask", {"columns": ["email"]}),
        ("schema_learn", {}),
        ("validation_check", {"evidence": "full"}),
        ("data_profile", {"evidence": "full"}),
        ("privacy_scan", {"evidence": "full"}),
    ],
)
def test_sensitive_routes_require_operator_grant_before_file_access(dataset, name, extra):
    result = invoke(service_for(dataset), name, path="not-present.csv", **extra)
    assert result["execution_status"] == "failed"
    assert result["issues"] == [{"code": "approval_required"}]
    assert result["quality_status"] == "unknown" and result["data"] is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"checks": []},
        {"checks": [{"validator": "os.system"}]},
        {"checks": [{"validator": "between", "minimum": 1}]},
        {"checks": [{"validator": "between", "min_value": 10, "max_value": 1}]},
        {"checks": [{"validator": "length"}]},
        {"checks": [{"validator": "null", "callable": "payload"}]},
        {"parallel": "true"},
        {"approved": True},
        {
            "schema_spec": {
                "columns": {"email": {"name": "email", "dtype": "String", "pattern": "(a+)+$"}}
            }
        },
        {"schema_spec": {"columns": {"email": {"name": "different", "dtype": "String"}}}},
        {"schema_spec": {"columns": {}}},
        {"checks": [{"validator": "null", "mostly": float("nan")}]},
    ],
)
def test_unbound_parameters_and_authority_cannot_be_smuggled(dataset, arguments):
    result = invoke(service_for(dataset), "validation_check", path="not-present.csv", **arguments)
    assert result["issues"] == [{"code": "invalid_arguments"}], result
    assert result["data"] is None


def test_discovery_is_paged_and_does_not_authorize_or_mutate_catalog(dataset, monkeypatch):
    service = service_for(dataset)
    monkeypatch.setattr(
        service.policy.__class__, "resolve", lambda *a: pytest.fail("Discovery touched input")
    )
    identities = []
    offset = 0
    while True:
        result = invoke(service, "capabilities_list", offset=offset, limit=2)
        identities.extend(op["operation_id"] for op in result["data"]["operations"])
        offset = result["data"]["next_offset"]
        if offset is None:
            break
    assert identities == list(CATALOG)
    assert not result["data"]["full_core_parity"]
    schema = invoke(service, "capability_describe", operation_id="validation_check")
    assert schema["data"]["input_schema"]["additionalProperties"] is False
    denied = invoke(service, "capability_describe", operation_id="unregistered.operation")
    assert denied["issues"] == [{"code": "unknown_operation"}]
    assert list(CATALOG) == identities


def test_authorized_lifecycle_facades_match_core_and_do_not_overwrite(dataset):
    import truthound as th

    service = service_for(dataset, consent=True)
    original = dataset.read_bytes()
    learned = invoke(service, "schema_learn", path=str(dataset), infer_constraints=False)
    assert learned["data"]["result"] == encode(
        th.learn(str(dataset), infer_constraints=False).to_dict()
    )
    read = invoke(service, "data_read", path=str(dataset), offset=1, limit=1)
    assert read["provenance"]["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert read["data"]["rows"] == encode(th.read(str(dataset)).slice(1, 1).to_dicts())
    assert read["data"]["next_offset"] == 2 and read["coverage"]["truncated"]
    masked = invoke(
        service,
        "data_mask",
        path=str(dataset),
        columns=["email"],
        strategy="hash",
        limit=3,
        strict=True,
    )
    assert masked["data"]["rows"] == encode(
        th.mask(str(dataset), columns=["email"], strategy="hash", strict=True).to_dicts()
    )
    assert "a@example.com" not in json.dumps(masked)
    assert dataset.read_bytes() == original
    failed = invoke(service, "data_mask", path=str(dataset), columns=["missing"], strict=True)
    assert failed["execution_status"] == "failed" and failed["quality_status"] == "unknown"
    assert "email" not in json.dumps(failed)  # no raw exception / available-column echo


def test_execution_failure_never_becomes_quality_success(dataset):
    result = invoke(
        service_for(dataset),
        "validation_check",
        path=str(dataset),
        checks=[{"validator": "duplicate", "subset": ["missing"]}],
    )
    assert result["execution_status"] == "failed", result
    assert result["quality_status"] == "unknown"
    assert result["coverage"]["execution_issue_count"] > 0
    assert "missing" not in json.dumps(result)


def test_incomplete_core_accounting_is_explicit_not_a_pass(dataset, monkeypatch):
    import truthound as th

    direct = th.check(str(dataset), context=_context(str(dataset)), validators=["null"])
    request = {
        "operation": "validation_check",
        "arguments": {"path": str(dataset)},
        "path": str(dataset),
        "max_items": 100,
        "provenance": {},
    }
    for uncertain in (
        replace(direct, checks=(), issues=(), execution_issues=()),
        replace(direct, execution_mode="parallel"),
    ):
        monkeypatch.setattr(th, "check", lambda *args, _result=uncertain, **kwargs: _result)
        result = execute(request)
        assert result["quality_status"] == "unknown"
        assert not result["coverage"]["accounting_complete"]


@pytest.mark.parametrize(
    "check",
    [
        {"validator": "between", "columns": ["email"], "max_value": 200},
        {"validator": "length", "columns": ["age"], "max_length": 10},
        {"validator": "null", "columns": ["not_present"]},
        {"validator": "unique", "columns": ["email"], "exclude_columns": ["email"]},
    ],
)
def test_unapplied_rules_are_not_passes_even_when_core_has_no_issues(dataset, check):
    result = invoke(service_for(dataset), "validation_check", path=str(dataset), checks=[check])
    assert result["execution_status"] == "succeeded", result
    assert result["quality_status"] == "unknown"
    assert not result["coverage"]["accounting_complete"]


def test_applied_positive_rule_can_pass_but_auto_suite_coverage_is_not_invented(dataset):
    service = service_for(dataset)
    result = invoke(
        service,
        "validation_check",
        path=str(dataset),
        checks=[{"validator": "between", "columns": ["age"], "min_value": 0, "max_value": 200}],
    )
    assert result["quality_status"] == "pass", result
    automatic = invoke(service, "validation_check", path=str(dataset))
    assert automatic["execution_status"] == "succeeded", automatic
    assert automatic["quality_status"] == "unknown"


def test_tagged_encoding_preserves_rich_values_and_cannot_spoof_tags():
    rich = {
        "$type": "decimal",
        "value": "secret",
        "amount": Decimal("1.2300"),
        "at": datetime(2026, 1, 1, tzinfo=UTC),
        "n": float("nan"),
        "binary": b"\x00\xff",
    }
    encoded = encode(rich)
    assert encoded["$type"] == "object"
    assert encoded["entries"]["amount"] == {"$type": "decimal", "value": "1.2300"}
    assert encoded["entries"]["at"]["value"].endswith("+00:00")
    assert encoded["entries"]["n"]["value"] == "nan"
    json.dumps(encoded, allow_nan=False)

    class Hostile:
        def __repr__(self):
            pytest.fail("Unknown objects must not be stringified")

    for item in (Hostile(), {1: "key"}):
        with pytest.raises(MCPError, match="unsupported_value"):
            encode(item)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(MCPError, match="unsupported_value"):
        encode(cycle)


def test_new_tool_limits_and_timeout_preserve_unknown_status(dataset, monkeypatch):
    import anyio

    service = service_for(dataset, consent=True, max_items=1)
    large = invoke(service, "data_read", path=str(dataset), limit=2)
    assert large["issues"] == [{"code": "resource_exhausted"}]

    async def timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(anyio, "run_process", timeout)
    result = invoke(service, "validation_check", path=str(dataset))
    assert result["issues"] == [{"code": "timeout"}]
    assert result["quality_status"] == "unknown"


def test_nanosecond_data_is_not_silently_truncated_or_downgraded(tmp_path):
    import polars as pl

    path = tmp_path / "precise.parquet"
    pl.DataFrame({"at": pl.Series([1_000_000_001], dtype=pl.Datetime("ns"))}).write_parquet(path)
    result = invoke(service_for(path, consent=True), "data_read", path=str(path), limit=1)
    assert result["execution_status"] == "failed"
    assert result["issues"] == [{"code": "unsupported_value"}]
    assert result["data"] is None


def test_full_result_and_discovery_limits_fail_without_successful_partial_claim(tmp_path):
    path = tmp_path / "large.csv"
    path.write_text("value\n" + "x" * 100_000 + "\n", encoding="utf-8")
    result = invoke(service_for(path, consent=True), "data_read", path=str(path), limit=1)
    assert result["issues"] == [{"code": "response_too_large"}]
    assert result["quality_status"] == "unknown" and result["data"] is None
    result = invoke(
        service_for(path, max_response_bytes=1024),
        "capability_describe",
        operation_id="validation_check",
    )
    assert result["issues"] == [{"code": "response_too_large"}]


def test_summary_identity_is_stable_within_session_but_not_a_public_content_digest(dataset):
    service = service_for(dataset)
    first = invoke(service, "data_profile", path=str(dataset))
    second = invoke(service, "privacy_scan", path=str(dataset))
    other = invoke(service_for(dataset), "data_profile", path=str(dataset))
    assert first["provenance"]["input_identity"] == second["provenance"]["input_identity"]
    assert first["provenance"]["input_identity"] != other["provenance"]["input_identity"]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() not in json.dumps(
        [first, second, other]
    )


def test_random_outputs_do_not_advertise_inconsistent_page_continuations(dataset):
    service = service_for(dataset, consent=True)
    result = invoke(service, "data_read", path=str(dataset), sample_size=2, limit=1)
    assert result["execution_status"] == "succeeded"
    assert result["coverage"]["truncated"] and result["data"]["next_offset"] is None
    for name, extra in (("data_read", {"sample_size": 2}), ("data_mask", {"strategy": "fake"})):
        denied = invoke(service, name, path=str(dataset), offset=1, limit=1, **extra)
        assert denied["issues"] == [{"code": "invalid_arguments"}]


def test_unknown_output_objects_and_malformed_envelopes_cannot_be_success(dataset, monkeypatch):
    from types import SimpleNamespace

    import anyio

    async def malformed(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b'{"execution_status":"succeeded"}')

    monkeypatch.setattr(anyio, "run_process", malformed)
    result = invoke(service_for(dataset), "validation_check", path=str(dataset))
    assert result["execution_status"] == "failed"
    assert result["issues"] == [{"code": "worker_error"}]


def test_real_sdk_catalog_schema_authorization_and_three_core_facades(dataset):
    async def scenario():
        for consent in (False, True):
            args = [
                "-m",
                "truthound.mcp",
                "serve",
                "--allow-dir",
                str(dataset.parent),
                "--profile",
                "core",
            ]
            if consent:
                args.append("--allow-sensitive-results")
            async with (
                stdio_client(StdioServerParameters(command=sys.executable, args=args)) as (
                    read,
                    write,
                ),
                ClientSession(read, write) as client,
            ):
                await client.initialize()
                tools = await client.list_tools()
                assert {tool.name for tool in tools.tools} == set(CATALOG) | {
                    "check_file",
                    "profile_file",
                    "scan_pii",
                }
                for name, extra in (
                    (
                        "validation_check",
                        {
                            "checks": [
                                {"validator": "between", "columns": ["age"], "max_value": 100}
                            ]
                        },
                    ),
                    ("data_profile", {}),
                    ("privacy_scan", {}),
                ):
                    result = await client.call_tool(name, {"path": str(dataset), **extra})
                    assert not result.isError, result
                    assert "a@example.com" not in result.model_dump_json()
                result = await client.call_tool("data_read", {"path": str(dataset), "limit": 1})
                assert result.isError is (not consent), result
                if consent:
                    assert "a@example.com" in result.model_dump_json()
                else:
                    assert result.structuredContent["issues"] == [{"code": "approval_required"}]
                assert {tool.name for tool in (await client.list_tools()).tools} == {
                    tool.name for tool in tools.tools
                }

    asyncio.run(scenario())
