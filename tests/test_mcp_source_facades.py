"""Resident datasource facades consume native sources and retain re-enterable results."""

import asyncio

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_sources import call, invoke, read
from truthound.datasources.polars_source import PolarsDataSource, PolarsDataSourceConfig
from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def source(service):
    frame = pl.DataFrame({"email": ["one@example.com", "two@example.com", None], "x": [1, 2, 3]})
    asset = call(
        service,
        "assets_inline",
        columns=[
            {"name": name, "dtype": {"kind": str(dtype)}, "values": frame[name].to_list()}
            for name, dtype in frame.schema.items()
        ],
    )["resource_id"]
    identity = call(
        service, "sources_open_polars", asset_id=asset, config={"name": "facade-source"}
    )["resource_id"]
    return frame, identity


def retained_frame(service, identity):
    return pl.read_ipc(service._session.resources.get(identity, "asset").path, memory_map=False)


def test_native_source_facades_results_reentry_and_immutable_mask(tmp_path):
    import truthound as th

    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        frame, identity = source(service)
        outputs = []
        with PolarsDataSource(
            frame.lazy(), config=PolarsDataSourceConfig(name="facade-source")
        ) as native:
            for operation, function in (
                ("sources_profile", th.profile),
                ("sources_scan", th.scan),
                ("sources_learn", th.learn),
            ):
                result = call(service, operation, source_id=identity)
                outputs.append(result["resource_id"])
                assert read(service, result["resource_id"]) == function(source=native).to_dict()
            learned = outputs[-1]
            assert (
                decode(call(service, "schemas_inspect", resource_id=learned)["result"])
                == th.learn(source=native).to_dict()
            )
            validation = call(service, "sources_validate", source_id=identity, schema_id=learned)
            assert read(service, validation["artifact"]["resource_id"])["row_count"] == 3
            result = call(service, "sources_read", source_id=identity, sample_size=2)["resource_id"]
            outputs.append(result)
            assert_frame_equal(retained_frame(service, result), th.read(native, sample_size=2))
            result = call(
                service,
                "sources_mask",
                source_id=identity,
                columns=["email"],
                strategy="hash",
                strict=True,
            )["resource_id"]
            outputs.append(result)
            assert_frame_equal(
                retained_frame(service, result),
                th.mask(source=native, columns=["email"], strategy="hash", strict=True),
            )
        fake = call(
            service, "sources_mask", source_id=identity, columns=["email"], strategy="fake"
        )["resource_id"]
        first = call(service, "assets_page", asset_id=fake)
        assert first == call(service, "assets_page", asset_id=fake)
        assert_frame_equal(
            retained_frame(
                service, call(service, "sources_read", source_id=identity)["resource_id"]
            ),
            frame,
        )
        call(service, "resources_release", resource_id=identity)
        for result in outputs:
            assert invoke(service, "resources_describe", resource_id=result)["issues"] == [
                {"code": "invalid_handle"}
            ]
    finally:
        service.close()


@pytest.mark.fault
def test_native_mask_failure_and_output_quota_leave_source_usable(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        _, identity = source(service)
        failed = invoke(
            service, "sources_mask", source_id=identity, columns=["private-missing"], strict=True
        )
        assert failed["issues"] == [{"code": "native_source_error"}]
        assert "private-missing" not in str(failed)
        worker = service._session._sources.workers[identity]
        assert not worker.closed
        output = tmp_path / "bounded.ipc"
        with pytest.raises(MCPError, match="resource_exhausted"):
            asyncio.run(
                worker.call(
                    {
                        "operation": "sources_read",
                        "arguments": {"source_id": identity, "sample_size": None},
                        "output_path": str(output),
                        "output_budget": 1,
                    },
                    10,
                )
            )
        assert read(
            service, call(service, "sources_validate_connection", source_id=identity)["resource_id"]
        )["connected"]
        assert not service._session.resources._pending
    finally:
        service.close()


def test_actual_async_source_facades_share_resident_connection(runtime, monkeypatch):
    service, _, marker, _, _ = runtime
    request = plugins.prepare_source(runtime, monkeypatch, "async-source-factory")
    identity = call(service, "sources_open_binding", **request)["resource_id"]
    assert retained_frame(
        service, call(service, "sources_read", source_id=identity)["resource_id"]
    )["x"].to_list() == [-1, 1, None]
    for operation in ("sources_profile", "sources_scan", "sources_learn"):
        result = read(service, call(service, operation, source_id=identity)["resource_id"])
        assert result["row_count"] == 3
    import truthound as th

    # Native redaction uses string expressions; integer columns fail in Core too.
    with pytest.raises(pl.exceptions.InvalidOperationError):
        th.mask(pl.DataFrame({"x": [-1, 1, None]}), columns=["x"], strategy="redact")
    assert invoke(service, "sources_mask", source_id=identity, columns=["x"], strategy="redact")[
        "issues"
    ] == [{"code": "native_source_error"}]
    masked = call(service, "sources_mask", source_id=identity, columns=["x"], strategy="hash")[
        "resource_id"
    ]
    assert retained_frame(service, masked).height == 3
    assert marker.read_text().splitlines().count("async-connect") == 1
    assert "async-disconnect" not in marker.read_text()
    call(service, "resources_release", resource_id=identity)
    assert marker.read_text().splitlines().count("async-disconnect") == 1
