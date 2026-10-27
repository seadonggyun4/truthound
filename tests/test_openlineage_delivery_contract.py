"""Actual loopback delivery must acknowledge or retain an uncertain transition."""

import pytest

from tests.verification_support import receiver
from truthound.lineage.integrations.openlineage import (
    EventType,
    OpenLineageConfig,
    OpenLineageEmitter,
)


def test_failed_start_is_not_acknowledged_as_an_active_run():
    with receiver() as remote:
        remote.status = 503
        emitter = OpenLineageEmitter(OpenLineageConfig(endpoint=remote.endpoint))
        with pytest.raises(RuntimeError):
            emitter.start_run("synthetic-job")
        assert len(remote.events) == 1


def test_failed_terminal_event_retains_prior_state_and_cannot_be_replayed():
    with receiver() as remote:
        emitter = OpenLineageEmitter(OpenLineageConfig(endpoint=remote.endpoint))
        run = emitter.start_run("synthetic-job")
        remote.status = 503
        with pytest.raises(RuntimeError):
            emitter.emit_complete(run)
        assert run.event_type == EventType.START
        assert run.run_id in emitter.unknown_runs
        with pytest.raises(RuntimeError):
            emitter.emit_complete(run)
        assert len(remote.events) == 2
        emitter.resolve_unknown(run.run_id, delivered=True)
        assert run.event_type == EventType.COMPLETE
        assert not emitter.active_runs and not emitter.unknown_runs
        emitter.close()


@pytest.mark.asyncio
async def test_async_cancellation_joins_delivery_and_close_leaves_no_background_send():
    import asyncio

    with receiver() as remote:
        remote.delay = 0.15
        emitter = OpenLineageEmitter(OpenLineageConfig(endpoint=remote.endpoint))
        with pytest.raises(RuntimeError, match="async"):
            emitter.start_run("wrong-api")
        assert not remote.events
        task = asyncio.create_task(emitter.start_run_async("cancelled-wait"))
        async with asyncio.timeout(5):
            while not remote.entered.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(remote.events) == 1
        assert not emitter.unknown_runs
        run = emitter.get_run(next(iter(emitter.active_runs)))
        remote.delay = 0
        await emitter.emit_complete_async(run)
        await emitter.aclose()
        assert not emitter._pending
        with pytest.raises(RuntimeError, match="closed"):
            await emitter.start_run_async("closed")
        assert len(remote.events) == 2


def test_owned_run_snapshot_and_facet_schema_are_not_mutated_by_callers():
    with receiver() as remote:
        config = OpenLineageConfig(
            endpoint=remote.endpoint, producer="https://example.invalid/test"
        )
        emitter = OpenLineageEmitter(config)
        config.namespace = "changed-after-construction"
        facets = {"progress": {"completed": 1}}
        run = emitter.start_run("owned", facets=facets)
        facets["progress"]["completed"] = 999
        run.job_name = "caller-mutated"
        output = emitter.build_output_dataset("output", row_count=0)
        assert output["outputFacets"]["outputStatistics"]["rowCount"] == 0
        emitter.emit_complete(run, outputs=[output])
        event = remote.events[-1]
        assert event["job"] == {"namespace": "truthound", "name": "owned"}
        assert event["producer"] == "https://example.invalid/test"
        assert event["run"]["facets"]["progress"]["completed"] == 1
        assert event["run"]["facets"]["progress"]["_producer"] == "https://example.invalid/test"
        assert not emitter.active_runs
        emitter.close()


def test_graph_preflight_rejects_later_oversized_event_before_any_delivery():
    from truthound.lineage import LineageTracker

    tracker = LineageTracker()
    tracker.track_source("input")
    tracker.track_transformation("small", ["input"])
    tracker.track_transformation("x" * 5000, ["small"])
    with receiver() as remote, OpenLineageEmitter(
        OpenLineageConfig(endpoint=remote.endpoint, max_event_bytes=2048)
    ) as emitter:
        with pytest.raises(ValueError, match="byte budget"):
            emitter.emit_from_graph(tracker.graph)
        assert not remote.events
        assert not emitter.active_runs
