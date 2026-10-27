"""Graph traversal and persistence preserve native lineage evidence."""

import pytest

from truthound.lineage.base import (
    CyclicDependencyError,
    EdgeType,
    LineageConfig,
    LineageEdge,
    LineageGraph,
    LineageNode,
    NodeType,
)
from truthound.lineage.tracker import LineageTracker


def node(name):
    return LineageNode(id=name, name=name, node_type=NodeType.TABLE)


def edge(source, target):
    return LineageEdge(source=source, target=target, edge_type=EdgeType.DERIVED_FROM)


def test_deep_graph_traversal_and_cycle_detection_do_not_use_python_recursion():
    graph = LineageGraph()
    for i in range(1200):
        graph.add_node(node(str(i)))
        if i:
            graph.add_edge(edge(str(i - 1), str(i)))
    assert len(graph.get_downstream("0")) == 1199
    tracker = LineageTracker()
    tracker._graph = graph
    assert len(tracker.get_path("0", "1199")) == 1200
    with pytest.raises(CyclicDependencyError):
        graph.add_edge(edge("1199", "0"))


def test_depth_limit_uses_shortest_reachable_distance():
    graph = LineageGraph()
    for name in "abcde":
        graph.add_node(node(name))
    for source, target in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("d", "e")]:
        graph.add_edge(edge(source, target))
    assert {n.id for n in graph.get_downstream("a", depth=3)} == set("bcde")


def test_failed_graph_serialization_preserves_previous_file(tmp_path):
    graph = LineageGraph()
    graph.add_node(node("a"))
    path = tmp_path / "graph.json"
    graph.save(path)
    before = path.read_bytes()
    graph.get_node("a").metadata.properties["not_json"] = object()
    with pytest.raises(TypeError):
        graph.save(path)
    assert path.read_bytes() == before


def test_zero_tracking_history_retains_no_entries():
    tracker = LineageTracker(LineageConfig(max_history=0))
    with tracker.track("operation"):
        pass
    assert tracker.history == []
