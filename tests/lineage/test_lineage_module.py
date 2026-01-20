"""Integration tests for the Lineage module."""

from __future__ import annotations

import pytest

from truthound.lineage import (
    LineageGraph,
    LineageTracker,
    ImpactAnalyzer,
)
from truthound.lineage.base import (
    LineageNode,
    LineageEdge,
    LineageError,
    NodeType,
    EdgeType,
    OperationType,
)


# =============================================================================
# Test LineageGraph
# =============================================================================


class TestLineageGraph:
    """Tests for LineageGraph."""

    def test_creation(self):
        """Test graph creation."""
        graph = LineageGraph()
        assert graph is not None

    def test_add_node(self):
        """Test adding nodes."""
        graph = LineageGraph()
        node = LineageNode(
            id="test",
            name="Test Node",
            node_type=NodeType.SOURCE,
        )
        graph.add_node(node)

        assert graph.get_node("test") is not None

    def test_add_edge(self):
        """Test adding edges."""
        graph = LineageGraph()

        # Add nodes first
        graph.add_node(LineageNode(
            id="source",
            name="Source",
            node_type=NodeType.SOURCE,
        ))
        graph.add_node(LineageNode(
            id="target",
            name="Target",
            node_type=NodeType.TABLE,
        ))

        # Add edge
        graph.add_edge(LineageEdge(
            source="source",
            target="target",
            edge_type=EdgeType.TRANSFORMED_TO,
        ))

        # Verify nodes exist
        assert graph.get_node("source") is not None
        assert graph.get_node("target") is not None


# =============================================================================
# Test LineageNode
# =============================================================================


class TestLineageNode:
    """Tests for LineageNode."""

    def test_creation(self):
        """Test node creation."""
        node = LineageNode(
            id="test",
            name="Test",
            node_type=NodeType.SOURCE,
        )

        assert node.id == "test"
        assert node.name == "Test"
        assert node.node_type == NodeType.SOURCE

    def test_metadata(self):
        """Test node with schema."""
        node = LineageNode(
            id="test",
            name="Test",
            node_type=NodeType.SOURCE,
            schema={"col1": "int", "col2": "str"},
        )

        assert "col1" in node.schema

    def test_from_dict_missing_node_type(self):
        """Test from_dict with missing node_type gives helpful error."""
        data = {"id": "test", "name": "Test"}

        with pytest.raises(LineageError) as exc_info:
            LineageNode.from_dict(data)

        error_msg = str(exc_info.value)
        assert "missing required field(s): node_type" in error_msg
        assert "Valid node_type values:" in error_msg
        assert "source" in error_msg
        assert "Example:" in error_msg

    def test_from_dict_missing_multiple_fields(self):
        """Test from_dict with multiple missing fields gives helpful error."""
        data = {"name": "Test"}

        with pytest.raises(LineageError) as exc_info:
            LineageNode.from_dict(data)

        error_msg = str(exc_info.value)
        assert "id" in error_msg
        assert "node_type" in error_msg

    def test_from_dict_invalid_node_type(self):
        """Test from_dict with invalid node_type gives helpful error."""
        data = {"id": "test", "name": "Test", "node_type": "invalid_type"}

        with pytest.raises(LineageError) as exc_info:
            LineageNode.from_dict(data)

        error_msg = str(exc_info.value)
        assert "Invalid node_type 'invalid_type'" in error_msg
        assert "Valid values:" in error_msg

    def test_from_dict_valid(self):
        """Test from_dict with valid data works correctly."""
        data = {"id": "test", "name": "Test", "node_type": "source"}

        node = LineageNode.from_dict(data)

        assert node.id == "test"
        assert node.name == "Test"
        assert node.node_type == NodeType.SOURCE


# =============================================================================
# Test LineageEdge
# =============================================================================


class TestLineageEdge:
    """Tests for LineageEdge."""

    def test_creation(self):
        """Test edge creation."""
        edge = LineageEdge(
            source="source",
            target="target",
            edge_type=EdgeType.TRANSFORMED_TO,
        )

        assert edge.source == "source"
        assert edge.target == "target"
        assert edge.edge_type == EdgeType.TRANSFORMED_TO

    def test_from_dict_missing_edge_type(self):
        """Test from_dict with missing edge_type gives helpful error."""
        data = {"source": "a", "target": "b"}

        with pytest.raises(LineageError) as exc_info:
            LineageEdge.from_dict(data)

        error_msg = str(exc_info.value)
        assert "missing required field(s): edge_type" in error_msg
        assert "Valid edge_type values:" in error_msg
        assert "derived_from" in error_msg
        assert "Example:" in error_msg

    def test_from_dict_missing_multiple_fields(self):
        """Test from_dict with multiple missing fields gives helpful error."""
        data = {"source": "a"}

        with pytest.raises(LineageError) as exc_info:
            LineageEdge.from_dict(data)

        error_msg = str(exc_info.value)
        assert "target" in error_msg
        assert "edge_type" in error_msg

    def test_from_dict_invalid_edge_type(self):
        """Test from_dict with invalid edge_type gives helpful error."""
        data = {"source": "a", "target": "b", "edge_type": "invalid_type"}

        with pytest.raises(LineageError) as exc_info:
            LineageEdge.from_dict(data)

        error_msg = str(exc_info.value)
        assert "Invalid edge_type 'invalid_type'" in error_msg
        assert "Valid values:" in error_msg

    def test_from_dict_valid(self):
        """Test from_dict with valid data works correctly."""
        data = {"source": "a", "target": "b", "edge_type": "derived_from"}

        edge = LineageEdge.from_dict(data)

        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.edge_type == EdgeType.DERIVED_FROM


# =============================================================================
# Test LineageTracker
# =============================================================================


class TestLineageTracker:
    """Tests for LineageTracker."""

    def test_creation(self):
        """Test tracker creation."""
        tracker = LineageTracker()
        assert tracker is not None
        assert tracker.graph is not None

    def test_track_source(self):
        """Test tracking a source."""
        tracker = LineageTracker()
        source_id = tracker.track_source(
            name="my_table",
            source_type="database",
        )

        assert source_id is not None


# =============================================================================
# Test ImpactAnalyzer
# =============================================================================


class TestImpactAnalyzer:
    """Tests for ImpactAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        graph = LineageGraph()
        analyzer = ImpactAnalyzer(graph)
        assert analyzer is not None


# =============================================================================
# Test NodeType and EdgeType
# =============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_node_types(self):
        """Test NodeType values."""
        assert NodeType.SOURCE.value == "source"
        assert NodeType.TRANSFORMATION.value == "transformation"
        assert NodeType.TABLE.value == "table"

    def test_edge_types(self):
        """Test EdgeType values."""
        assert EdgeType.DERIVED_FROM.value == "derived_from"
        assert EdgeType.TRANSFORMED_TO.value == "transformed_to"

    def test_operation_types(self):
        """Test OperationType values."""
        assert OperationType.FILTER.value == "filter"
        assert OperationType.TRANSFORM.value == "transform"
        assert OperationType.JOIN.value == "join"
