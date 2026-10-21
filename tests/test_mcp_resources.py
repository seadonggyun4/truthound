"""Owned snapshots: Core parity and adversarial lifetime/disclosure boundaries."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.resources import SessionResources
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


@pytest.mark.fault
def test_tiny_nested_array_declarations_cannot_expand_before_budget_validation(monkeypatch):
    from truthound.mcp.session_contracts import InlineColumn
    from truthound.mcp.tables import inline_table

    def forbidden(*args, **kwargs):
        raise AssertionError("native allocation started before admission")

    monkeypatch.setattr(pl, "Series", forbidden)
    dtype = {"kind": "Float64"}
    for _ in range(3):
        dtype = {"kind": "Array", "inner": dtype, "width": 1000}
    column = InlineColumn(name="x", dtype=dtype, values=[None])
    with pytest.raises(MCPError, match="resource_exhausted"):
        inline_table([column])
    nested = InlineColumn(
        name="x",
        dtype={
            "kind": "List",
            "inner": {
                "kind": "Array",
                "width": 500,
                "inner": {
                    "kind": "Array",
                    "width": 500,
                    "inner": {"kind": "Float64"},
                },
            },
        },
        values=[[None] * 1000],
    )
    with pytest.raises(MCPError, match="resource_exhausted"):
        inline_table([nested])
    small = InlineColumn(name="x", dtype={"kind": "Int64"}, values=[1, 2])
    with pytest.raises(MCPError, match="resource_exhausted"):
        inline_table([small], maximum_bytes=31)


@pytest.fixture
def service(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    yield service
    service.close()


def call(service, name, **arguments):
    result = asyncio.run(service.call_operation_async(name, arguments))
    assert result["execution_status"] == "succeeded", result
    return result["data"]


def test_retained_core_sample_mask_schema_and_artifact_roundtrip(tmp_path, service):
    import truthound as th

    path = tmp_path / "input.csv"
    path.write_text(
        "email,x\na@example.com,1\nb@example.com,2\nc@example.com,3\n", encoding="utf-8"
    )
    asset = call(service, "assets_capture", path=str(path))["resource_id"]
    sample = call(service, "assets_sample", asset_id=asset, n=2, seed=42)["resource_id"]
    first = call(service, "assets_page", asset_id=sample, offset=0, limit=1)
    second = call(service, "assets_page", asset_id=sample, offset=1, limit=1)
    repeated = call(service, "assets_page", asset_id=sample, offset=0, limit=1)
    assert first == repeated and first["next_offset"] == 1 and second["next_offset"] is None
    masked = call(
        service, "assets_mask", asset_id=asset, columns=["email"], strategy="hash", strict=True
    )["resource_id"]
    artifact = call(service, "artifacts_export", resource_id=masked, format="ipc")["resource_id"]
    chunk = call(service, "artifacts_read", resource_id=artifact, limit=32768)
    import io

    frame = pl.read_ipc(io.BytesIO(base64.b64decode(chunk["base64"])))
    assert frame.equals(th.mask(str(path), columns=["email"], strategy="hash", strict=True))
    schema = call(service, "schemas_learn", asset_id=asset)["resource_id"]
    from truthound.mcp.encoding import encode

    assert call(service, "schemas_inspect", resource_id=schema)["result"] == encode(
        th.learn(str(path)).to_dict()
    )
    path.unlink()  # retained snapshots do not reopen a missing original file
    assert call(service, "assets_page", asset_id=asset, limit=1)["total_rows"] == 3


def test_capture_is_immutable_and_every_read_rechecks_grants(tmp_path, service):
    path = tmp_path / "input.csv"
    path.write_text("x\noriginal\n", encoding="utf-8")
    asset = call(service, "assets_capture", path=str(path))["resource_id"]
    path.write_text("x\nchanged\n", encoding="utf-8")
    page = call(service, "assets_page", asset_id=asset)
    assert "original" in json.dumps(page) and "changed" not in json.dumps(page)
    empty = tmp_path / "other"
    empty.mkdir()
    service.policy = replace(service.policy, allowed_directories=(empty,))
    result = asyncio.run(service.call_operation_async("assets_page", {"asset_id": asset}))
    assert result["issues"] == [{"code": "access_denied"}]
    assert call(service, "resources_release", resource_id=asset)["released"]


def test_handles_are_session_bound_and_parent_release_invalidates_derivatives(tmp_path, service):
    asset = call(
        service,
        "assets_inline",
        columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2]}],
    )["resource_id"]
    sample = call(service, "assets_sample", asset_id=asset, n=1, seed=1)["resource_id"]
    other = MCPService(service.policy)
    try:
        result = asyncio.run(other.call_operation_async("assets_page", {"asset_id": asset}))
        assert result["issues"] == [{"code": "invalid_handle"}]
    finally:
        other.close()
    call(service, "resources_release", resource_id=asset)
    result = asyncio.run(service.call_operation_async("assets_page", {"asset_id": sample}))
    assert result["issues"] == [{"code": "invalid_handle"}]


def test_nanosecond_nested_and_large_integer_values_roundtrip_without_python_truncation(service):
    temporal = {
        "$type": "temporal",
        "kind": "Datetime",
        "unit": "ns",
        "time_zone": "UTC",
        "value": "1000000001",
    }
    data = [
        {
            "name": "at",
            "dtype": {"kind": "Datetime", "unit": "ns", "time_zone": "UTC"},
            "values": [temporal, None],
        },
        {
            "name": "large",
            "dtype": {"kind": "Int64"},
            "values": [{"$type": "integer", "value": "9007199254740993"}, None],
        },
        {
            "name": "nested",
            "dtype": {
                "kind": "List",
                "inner": {"kind": "Datetime", "unit": "ns", "time_zone": "UTC"},
            },
            "values": [[temporal], None],
        },
    ]
    asset = call(service, "assets_inline", columns=data)["resource_id"]
    rows = call(service, "assets_page", asset_id=asset)["rows"]
    assert rows[0]["entries"]["at"] == temporal
    assert rows[0]["entries"]["large"]["value"] == "9007199254740993"
    assert rows[0]["entries"]["nested"][0]["value"] == "1000000001"
    assert rows[0]["entries"]["nested"][0]["time_zone"] == "UTC"
    assert rows[1]["entries"] == {"at": None, "large": None, "nested": None}


@pytest.mark.parametrize(
    ("key", "value"), [("unit", "us"), ("time_zone", None), ("kind", "Duration")]
)
def test_temporal_wire_metadata_cannot_silently_reinterpret_physical_values(service, key, value):
    tagged = {
        "$type": "temporal",
        "kind": "Datetime",
        "unit": "ns",
        "time_zone": "UTC",
        "value": "1000000001",
        key: value,
    }
    result = asyncio.run(
        service.call_operation_async(
            "assets_inline",
            {
                "columns": [
                    {
                        "name": "time",
                        "dtype": {"kind": "Datetime", "unit": "ns", "time_zone": "UTC"},
                        "values": [tagged],
                    }
                ]
            },
        )
    )
    assert result["issues"] == [{"code": "invalid_arguments"}]


def test_resource_disclosure_is_separate_and_close_removes_owned_files(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core"))
    asset = call(
        service,
        "assets_inline",
        columns=[{"name": "x", "dtype": {"kind": "String"}, "values": ["private"]}],
    )
    assert "sha256" not in asset and "private" not in json.dumps(asset)
    result = asyncio.run(
        service.call_operation_async("assets_page", {"asset_id": asset["resource_id"]})
    )
    assert result["issues"] == [{"code": "approval_required"}]
    root = service._session.resources.root
    service.close()
    assert not root.exists()
    result = asyncio.run(
        service.call_operation_async("resources_describe", {"resource_id": asset["resource_id"]})
    )
    assert result["issues"] == [{"code": "session_closed"}]


def test_expiry_quota_and_tampered_snapshot_are_not_successes(tmp_path):
    policy = MCPPolicy((tmp_path,), profile="core", max_handles=1, max_session_bytes=1024)
    now = [1.0]
    resources = SessionResources(lambda: policy, clock=lambda: now[0])
    try:
        identity, path = resources.reserve("asset")
        path.write_bytes(b"one")
        resources.commit(identity, kind="asset", media_type="application/test")
        with pytest.raises(MCPError, match="resource_exhausted"):
            resources.reserve("asset")
        path.write_bytes(b"two")
        with pytest.raises(MCPError, match="resource_changed"):
            resources.get(identity)
        now[0] += policy.handle_ttl_seconds
        with pytest.raises(MCPError, match="handle_expired"):
            resources.get(identity)
        assert not path.exists()
        large, path = resources.reserve("asset")
        path.write_bytes(b"x" * 1025)
        with pytest.raises(MCPError, match="resource_exhausted"):
            resources.commit(large, kind="asset", media_type="application/test")
        assert not path.exists()
    finally:
        resources.close()
