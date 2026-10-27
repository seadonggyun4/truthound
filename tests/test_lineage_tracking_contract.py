"""Tracker configuration changes actual provenance and durable graph behavior."""

import pytest

from truthound.lineage import LineageConfig, LineageTracker, NodeNotFoundError


def test_column_tracking_disabled_omits_mappings():
    tracker = LineageTracker(LineageConfig(track_column_level=False))
    tracker.track_source("input")
    tracker.track_transformation("output", ["input"], column_mapping={"y": [("input", "x")]})
    assert not tracker.graph.get_node("output").column_lineage


def test_tracker_reopens_configured_persistence_and_rolls_back_failed_mutation(tmp_path):
    path = tmp_path / "graph.json"
    config = LineageConfig(persist_path=path)
    tracker = LineageTracker(config)
    tracker.track_source("input")
    before = path.read_bytes()
    reopened = LineageTracker(config)
    assert reopened.graph.has_node("input")
    with pytest.raises(ValueError):
        reopened.track_source("bad", invalid=float("nan"))
    assert not reopened.graph.has_node("bad")
    assert path.read_bytes() == before
    reopened.track_transformation("output", ["input"])
    assert LineageTracker(config).get_path("input", "output") is not None


def test_missing_source_does_not_publish_partial_transformation():
    tracker = LineageTracker()
    with pytest.raises(NodeNotFoundError):
        tracker.track_transformation("output", ["missing"])
    assert not tracker.graph.has_node("output")


def test_tracking_context_links_declared_inputs_and_outputs():
    tracker = LineageTracker()
    tracker.track_source("input")
    tracker.track_source("output")
    with tracker.track("clean") as ctx:
        ctx.add_source("input")
        ctx.add_target("output")
    assert tracker.get_path("input", "output") is not None
    assert tracker.history[-1].completed
    disabled = LineageTracker(LineageConfig(auto_track=False))
    with disabled.track("disabled") as ctx:
        ctx.add_source("absent")
        ctx.add_target("also_absent")
    assert disabled.graph.node_count == 0


def test_stale_tracker_cannot_overwrite_a_newer_snapshot(tmp_path):
    config = LineageConfig(persist_path=tmp_path / "graph.json")
    first = LineageTracker(config)
    second = LineageTracker(config)
    first.track_source("newer")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        second.track_source("stale")
    assert not second.graph.has_node("stale")
    assert LineageTracker(config).graph.has_node("newer")


def test_nested_contexts_restore_current_tracker_and_failed_context_has_no_edges():
    outer, inner = LineageTracker(), LineageTracker()
    previous = LineageTracker.get_current()
    with outer.track("outer"):
        assert LineageTracker.get_current() is outer
        with pytest.raises(ValueError), inner.track("inner") as ctx:
            assert LineageTracker.get_current() is inner
            ctx.add_source("missing")
            raise ValueError("synthetic body failure")
        assert LineageTracker.get_current() is outer
    assert LineageTracker.get_current() is previous
    assert inner.graph.node_count == 0
    assert not inner.history[-1].completed


def test_unknown_publication_fences_tracker_until_reopened(tmp_path, monkeypatch):
    from truthound.stores import _filesystem_io

    config = LineageConfig(persist_path=tmp_path / "graph.json")
    tracker = LineageTracker(config)
    tracker.track_source("old")
    original = _filesystem_io.publish

    def publish_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("synthetic acknowledgement failure after publication")

    with monkeypatch.context() as patch:
        patch.setattr(_filesystem_io, "publish", publish_then_fail)
        with pytest.raises(OSError):
            tracker.track_source("possibly_committed")
    assert not tracker.graph.has_node("possibly_committed")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        tracker.track_source("blocked")
    reopened = LineageTracker(config)
    assert reopened.graph.has_node("possibly_committed")
    reopened.track_source("next")


def test_row_tracking_records_positions_and_rejects_undeclared_sources(tmp_path):
    config = LineageConfig(track_row_level=True, persist_path=tmp_path / "rows.json")
    tracker = LineageTracker(config)
    tracker.track_source("input")
    tracker.track_transformation("output", ["input"], row_mapping={0: [("input", 4)]})
    row_lineage = tracker.graph.get_node("output").metadata.properties["row_lineage"]
    assert row_lineage == [{"row": 0, "sources": [{"node": "input", "row": 4}]}]
    assert (
        LineageTracker(config).graph.get_node("output").metadata.properties["row_lineage"]
        == row_lineage
    )
    with pytest.raises(ValueError):
        tracker.track_transformation("invalid", ["input"], row_mapping={0: [("missing", 1)]})
    assert not tracker.graph.has_node("invalid")
    disabled = LineageTracker()
    disabled.track_source("input")
    disabled.track_transformation("output", ["input"], row_mapping={0: [("input", 4)]})
    assert "row_lineage" not in disabled.graph.get_node("output").metadata.properties


def test_sample_opt_in_captures_bounded_values_without_caller_aliases(tmp_path):
    rows = [{"x": 1}]
    disabled = LineageTracker()
    disabled.track_source("input", samples=rows)
    assert "samples" not in disabled.graph.get_node("input").metadata.properties
    config = LineageConfig(store_samples=True, persist_path=tmp_path / "samples.json")
    enabled = LineageTracker(config)
    enabled.track_source("input", samples=rows)
    rows[0]["x"] = 999
    assert enabled.graph.get_node("input").metadata.properties["samples"] == [{"x": 1}]
    assert LineageTracker(config).graph.get_node("input").metadata.properties["samples"] == [
        {"x": 1}
    ]
    before = (tmp_path / "samples.json").read_bytes()
    with pytest.raises(ValueError):
        enabled.track_transformation("oversized", ["input"], samples=[{"x": 1}] * 101)
    assert not enabled.graph.has_node("oversized")
    assert (tmp_path / "samples.json").read_bytes() == before


def test_history_is_a_snapshot_of_completed_contexts():
    tracker = LineageTracker()
    with tracker.track("synthetic") as context:
        context.metadata["nested"] = [1]
    context.completed = False
    context.metadata["nested"].append(2)
    history = tracker.history
    history[0].sources.append("not-an-actual-source")
    assert tracker.history[0].completed
    assert tracker.history[0].metadata == {"nested": [1]}
    assert tracker.history[0].sources == []


def test_explicit_primary_save_cannot_overwrite_a_newer_tracker(tmp_path):
    path = tmp_path / "primary.json"
    config = LineageConfig(persist_path=path)
    first, second = LineageTracker(config), LineageTracker(config)
    second.track_source("newer")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="snapshot changed"):
        first.save(str(path))
    assert path.read_bytes() == before


def test_primary_load_reopens_a_stale_tracker_before_the_next_edit(tmp_path):
    path = tmp_path / "primary.json"
    config = LineageConfig(persist_path=path)
    first, second = LineageTracker(config), LineageTracker(config)
    second.track_source("newer")
    first.load(str(path))
    first.track_source("after-reopen")
    reopened = LineageTracker(config)
    assert reopened.graph.has_node("newer") and reopened.graph.has_node("after-reopen")


def test_missing_publication_witness_fences_tracker_without_changing_graph(tmp_path, monkeypatch):
    from truthound.stores import _filesystem_io
    tracker = LineageTracker(LineageConfig(persist_path=tmp_path / "state.json"))
    monkeypatch.setattr(_filesystem_io, "publish", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="confirmed"):
        tracker.track_source("not-published")
    assert not tracker.graph.has_node("not-published")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        tracker.save(str(tmp_path / "state.json"))


def test_native_graph_export_does_not_emit_nonstandard_nonfinite_json():
    from truthound.lineage import LineageMetadata, LineageNode, NodeType
    tracker = LineageTracker()
    tracker.graph.add_node(LineageNode(
        id="synthetic", name="synthetic", node_type=NodeType.TABLE,
        metadata=LineageMetadata(properties={"mean": float("nan")}),
    ))
    with pytest.raises(ValueError):
        tracker.export_to_json()
