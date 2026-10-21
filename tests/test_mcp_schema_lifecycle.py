"""Native schema construction, re-entry, output and revocation boundaries."""

import asyncio
import base64
import json

import pytest

pytest.importorskip("mcp")

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode
from truthound.schema import Schema

pytestmark = pytest.mark.contract


def call(adapter, name, **arguments):
    response = asyncio.run(adapter.call_operation_async(name, arguments))
    assert response["execution_status"] == "succeeded", response
    return response["data"]


def read(adapter, identity):
    parts, offset = [], 0
    while offset is not None:
        chunk = call(adapter, "artifacts_read", resource_id=identity, offset=offset)
        parts.append(base64.b64decode(chunk["base64"]))
        offset = chunk["next_offset"]
    return b"".join(parts)


def test_explicit_schema_native_yaml_reentry_and_validation_preserve_constraints(tmp_path):
    specification = {
        "version": "custom-2",
        "row_count": 2,
        "columns": {
            "age": {
                "name": "age",
                "dtype": "Int64",
                "nullable": False,
                "unique": True,
                "min_value": 1.0,
                "max_value": 100.0,
                "allowed_values": [1, 2],
                "mean": 1.5,
                "std": 0.5,
                "quantiles": {"25%": 1.0},
            },
        },
    }
    native = Schema.from_dict(specification)
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        schema = call(adapter, "schemas_create", schema_spec=specification)["resource_id"]
        inspected = decode(call(adapter, "schemas_inspect", resource_id=schema)["result"])
        assert inspected == native.to_dict()
        yaml_artifact = call(adapter, "schemas_export", resource_id=schema)["resource_id"]
        path = tmp_path / "native-schema.yaml"
        path.write_bytes(read(adapter, yaml_artifact))
        assert Schema.load(path).to_dict() == native.to_dict()
        json_artifact = call(adapter, "schemas_export", resource_id=schema, format="json")
        assert decode(json.loads(read(adapter, json_artifact["resource_id"]))) == native.to_dict()
        asset = call(
            adapter,
            "assets_inline",
            columns=[
                {
                    "name": "age",
                    "dtype": {"kind": "Int64"},
                    "values": [1, 150],
                }
            ],
        )["resource_id"]
        result = call(
            adapter,
            "assets_validate",
            asset_id=asset,
            schema_id=schema,
            checks=[{"validator": "null", "settings": {"columns": ["age"]}}],
        )
        assert not result["summary"]["success"]
        assert result["summary"]["issue_count"] >= 1
        call(adapter, "resources_release", resource_id=schema)
        revoked = asyncio.run(
            adapter.call_operation_async("artifacts_read", {"resource_id": yaml_artifact})
        )
        assert revoked["execution_status"] == "failed"
    finally:
        adapter.close()


@pytest.mark.fault
def test_schema_constructor_does_not_silently_drop_unknown_or_regex_fields(tmp_path):
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core"))
    try:
        for field in ("pattern", "unrecognized"):
            response = asyncio.run(
                adapter.call_operation_async(
                    "schemas_create",
                    {
                        "schema_spec": {
                            "columns": {
                                "x": {
                                    "name": "x",
                                    "dtype": "String",
                                    field: ".*",
                                }
                            }
                        }
                    },
                )
            )
            assert response["issues"] == [{"code": "invalid_arguments"}]
    finally:
        adapter.close()
