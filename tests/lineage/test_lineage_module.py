from __future__ import annotations

import pytest

from truthound.lineage import ImpactAnalyzer, LineageGraph
from truthound.lineage.base import (
    EdgeType,
    LineageEdge,
    LineageError,
    LineageNode,
    NodeNotFoundError,
    NodeType,
)


pytestmark = pytest.mark.contract


def _build_graph() -> LineageGraph:
    graph = LineageGraph()
    graph.add_node(LineageNode(id="orders_raw", name="orders_raw", node_type=NodeType.SOURCE))
    graph.add_node(LineageNode(id="orders_clean", name="orders_clean", node_type=NodeType.TABLE))
    graph.add_node(LineageNode(id="fraud_model", name="fraud_model", node_type=NodeType.MODEL))
    graph.add_edge(
        LineageEdge(
            source="orders_raw",
            target="orders_clean",
            edge_type=EdgeType.TRANSFORMED_TO,
        )
    )
    graph.add_edge(
        LineageEdge(
            source="orders_clean",
            target="fraud_model",
            edge_type=EdgeType.DERIVED_FROM,
        )
    )
    return graph


def test_impact_analysis_finds_downstream_critical_assets():
    analysis = ImpactAnalyzer(_build_graph()).analyze_impact("orders_raw")

    assert analysis.total_affected == 2
    assert analysis.max_depth == 2
    assert [node.node.id for node in analysis.get_critical_nodes()] == ["fraud_model"]


@pytest.mark.fault
def test_lineage_node_from_dict_reports_invalid_node_type():
    with pytest.raises(LineageError) as exc_info:
        LineageNode.from_dict({"id": "orders", "name": "orders", "node_type": "unknown"})

    assert "Invalid node_type 'unknown'" in str(exc_info.value)


@pytest.mark.fault
def test_graph_rejects_edges_to_missing_nodes():
    graph = LineageGraph()
    graph.add_node(LineageNode(id="orders_raw", name="orders_raw", node_type=NodeType.SOURCE))

    with pytest.raises(NodeNotFoundError):
        graph.add_edge(
            LineageEdge(
                source="orders_raw",
                target="missing_target",
                edge_type=EdgeType.DERIVED_FROM,
            )
        )


@pytest.mark.fault
def test_impact_analysis_rejects_unknown_source_nodes():
    analyzer = ImpactAnalyzer(_build_graph())

    with pytest.raises(NodeNotFoundError):
        analyzer.analyze_impact("missing_source")
