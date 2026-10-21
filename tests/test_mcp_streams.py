"""Native file-stream cursors, complete batches and safe retry boundaries."""

import asyncio

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


def call(service, operation, **arguments):
    result = asyncio.run(service.call_operation_async(operation, arguments))
    assert result["execution_status"] == "succeeded", result
    return result["data"]


@pytest.mark.parametrize("format", ["csv", "jsonl", "parquet", "arrow"])
def test_native_batches_exact_cursor_replay_projection_and_cleanup(tmp_path, format):
    from truthound.validators.streaming.sources import (
        ArrowIPCStreamingSource,
        CSVStreamingSource,
        JSONLStreamingSource,
        ParquetStreamingSource,
    )

    frame = pl.DataFrame({"x": list(range(11)), "text": ["synthetic"] * 11})
    path = tmp_path / ("input." + format)
    if format == "csv":
        # Native CSV skip_rows skips physical lines before the header.
        path.write_text("preamble one\npreamble two\n" + frame.write_csv(), encoding="utf-8")
    elif format == "jsonl":
        frame.write_ndjson(path)
    elif format == "parquet":
        frame.write_parquet(path)
    else:
        frame.write_ipc(path)
    factory = {
        "csv": CSVStreamingSource,
        "jsonl": JSONLStreamingSource,
        "parquet": ParquetStreamingSource,
        "arrow": ArrowIPCStreamingSource,
    }[format]
    config = {"chunk_size": 3, "columns": ["x"], "skip_rows": 2, "max_rows": 6}
    with factory(path, **config) as direct:
        expected = list(direct)
    assert expected and sum(len(batch) for batch in expected) > 0
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        stream = call(
            service, "streams_open", path=str(path), source={"format": format, "config": config}
        )["resource_id"]
        worker = service._session._streams.workers[stream]
        entry = service._session.resources.get(stream, "source")
        snapshot = entry.metadata["owned_input"]
        for index, batch in enumerate(expected):
            arguments = {"stream_id": stream, "idempotency_key": f"chunk-{index}"}
            actual = call(service, "streams_next", **arguments)
            assert not actual["exhausted"]
            asset = actual["artifact"]["resource_id"]
            result = service._session.resources.get(asset, "asset")
            assert pl.read_ipc(result.path).equals(batch)
            replay = call(service, "streams_next", **arguments)
            assert replay["artifact"]["resource_id"] == asset
        assert call(service, "streams_next", stream_id=stream, idempotency_key="end") == {
            "artifact": None,
            "exhausted": True,
        }
        call(service, "resources_release", resource_id=stream)
        assert worker.closed and worker.process.poll() is not None
        assert not service._session.resources.root.joinpath(snapshot).exists()
    finally:
        service.close()


@pytest.mark.fault
def test_stream_uncertain_advance_is_not_retried_and_open_schema_is_closed(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    pl.DataFrame({"x": [1, 2, 3]}).write_csv(path)
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        bad = asyncio.run(
            service.call_operation_async(
                "streams_open",
                {
                    "path": str(path),
                    "source": {"format": "csv", "config": {"arbitrary_code": "not executable"}},
                },
            )
        )
        assert bad["issues"] == [{"code": "invalid_arguments"}]
        identity = call(service, "streams_open", path=str(path), source={"format": "csv"})[
            "resource_id"
        ]
        worker = service._session._streams.workers[identity]

        def interrupted(_request):
            raise TimeoutError

        monkeypatch.setattr(worker, "_exchange", interrupted)
        failed = asyncio.run(
            service.call_operation_async(
                "streams_next", {"stream_id": identity, "idempotency_key": "chunk-1"}
            )
        )
        assert failed["execution_status"] == "failed"
        assert worker.closed and worker.process.poll() is not None
        replay = asyncio.run(
            service.call_operation_async(
                "streams_next", {"stream_id": identity, "idempotency_key": "chunk-1"}
            )
        )
        assert replay["execution_status"] == "failed"
    finally:
        service.close()


@pytest.mark.fault
def test_revoked_stream_grant_reaps_worker_and_owned_snapshot_without_new_read(tmp_path):
    from dataclasses import replace

    data, other = tmp_path / "data", tmp_path / "other"
    data.mkdir()
    other.mkdir()
    path = data / "input.csv"
    pl.DataFrame({"x": [1, 2]}).write_csv(path)
    service = MCPService(MCPPolicy((data,), profile="core"))
    try:
        stream = call(service, "streams_open", path=str(path), source={"format": "csv"})[
            "resource_id"
        ]
        worker = service._session._streams.workers[stream]
        snapshot = service._session.resources.get(stream).metadata["owned_input"]
        service.policy = replace(service.policy, allowed_directories=(other,))
        result = asyncio.run(service.call_operation_async("streams_inspect", {"stream_id": stream}))
        assert result["execution_status"] == "failed"
        assert worker.closed and worker.process.poll() is not None
        assert not service._session.resources.root.joinpath(snapshot).exists()
        assert path.exists()
    finally:
        service.close()
