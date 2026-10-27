"""Resident native tracking contexts, persistence fences and cancellation ownership."""

import asyncio
import json
import threading

import pytest

from truthound.lineage import LineageConfig, LineageTracker
from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.lineage_service_worker import LineageOwner
from truthound.mcp.lineage_services import LineageJob, LineageRuntime


@pytest.mark.asyncio
async def test_native_context_history_partial_failure_and_graph_reopen(tmp_path):
    primary, backup = tmp_path / "private-state.json", tmp_path / "private-backup.json"
    config = LineageConfig(persist_path=primary)
    tracker = LineageTracker(config)
    tracker.track_source("input")

    def success(native, context):
        assert LineageTracker.get_current() is native
        native.track_transformation("output", ["input"])
        context.add_source("input")
        context.add_target("output")
        context.metadata["phase"] = "complete"

    def failure(native, context):
        native.track_source("partial")
        raise RuntimeError("synthetic-private-lineage-secret")

    owner = LineageOwner(
        LineageRuntime(
            tracker,
            {
                "success": LineageJob(success, "synthetic-context"),
                "failure": LineageJob(failure, "synthetic-failure"),
            },
            {"primary": primary, "backup": backup},
        )
    )

    async def call(command, **args):
        return await owner.execute({"command": command, **args}, 1024 * 1024)

    try:
        result = await call("run", job_id="success")
        assert result["native_completed"], result
        history = await call("history", limit=1)
        assert history["history"][0]["completed"]
        assert history["history"][0]["operation_id"] == result["operation_id"]
        assert LineageTracker.get_current() is None
        assert (await call("path", source_id="input", target_id="output"))["path"]
        assert (await call("save", file_id="backup"))["saved"]
        failed = await call("run", job_id="failure")
        assert isinstance(failed, NativeServiceResult)
        assert tracker.graph.has_node("partial")
        history = await call("history")
        assert not history["history"][-1]["completed"]
        assert history["history"][-1]["error"] == "native_tracking_failed"
        assert "synthetic-private-lineage-secret" not in json.dumps(history, default=str)
        assert str(tmp_path) not in json.dumps(history, default=str)
        assert (await call("history", limit=0))["history"] == []
        assert (await call("load", file_id="backup"))["loaded"]
        assert not tracker.graph.has_node("partial")
        assert (await call("save", file_id="primary"))["saved"]
        assert LineageTracker(config).graph.has_node("output")
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_native_tracker_cancelled_waiter_joins_context_before_close():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    tracker = LineageTracker()

    def work(native, context):
        entered.set()
        assert release.wait(5)
        assert LineageTracker.get_current() is native
        finished.set()

    owner = LineageOwner(LineageRuntime(tracker, {"wait": LineageJob(work, "synthetic-wait")}))
    task = asyncio.create_task(owner.execute({"command": "run", "job_id": "wait"}, 1024 * 1024))
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        asyncio.get_running_loop().call_later(0.05, release.set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set() and tracker.history[-1].completed
        assert not tracker._active_contexts
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_render_failure_is_retained_without_losing_native_tracker(monkeypatch):
    import truthound.mcp.lineage_service_worker as module

    tracker = LineageTracker()
    tracker.track_source("input")
    owner = LineageOwner(LineageRuntime(tracker))

    def failed(*args):
        raise OSError("synthetic-private-render-secret")

    monkeypatch.setattr(module, "render_artifact", failed)
    try:
        reply = await owner.execute(
            {"command": "render", "rendering": {"renderer": "graphviz", "output_format": "png"}},
            1024 * 1024,
        )
        assert isinstance(reply, NativeServiceResult)
        assert "synthetic-private-render-secret" not in str(reply)
        assert (await owner.execute({"command": "snapshot"}, 1024 * 1024))["graph"]["nodes"]
    finally:
        await owner.close()
