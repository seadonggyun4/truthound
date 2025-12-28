"""Integration tests for lineage tracking and visualization.

Tests the complete lineage pipeline including tracking,
visualization rendering, and OpenLineage integration.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestLineageTrackingIntegration:
    """Integration tests for lineage tracking."""

    @pytest.fixture
    def tracker(self):
        """Create lineage tracker."""
        from truthound.lineage import LineageTracker, LineageConfig

        config = LineageConfig(
            auto_track=True,
            track_column_level=True,
        )
        return LineageTracker(config=config)

    def test_full_pipeline_tracking(self, tracker):
        """Test tracking a complete data pipeline."""
        # Track source data
        tracker.track_source(
            "raw_customers",
            source_type="csv",
            path="/data/customers.csv",
        )

        tracker.track_source(
            "raw_orders",
            source_type="database",
        )

        # Track transformations
        tracker.track_transformation(
            "cleaned_customers",
            sources=["raw_customers"],
            operation="filter",
        )

        tracker.track_transformation(
            "enriched_orders",
            sources=["raw_orders", "cleaned_customers"],
            operation="join",
        )

        # Track validation
        tracker.track_validation(
            "validated_orders",
            sources=["enriched_orders"],
            validator="null_check",
        )

        # Track output
        tracker.track_output(
            "output_warehouse",
            sources=["validated_orders"],
        )

        # Get lineage graph
        graph = tracker.graph

        # Verify nodes (nodes is a list, use node_count and has_node)
        assert graph.node_count == 6
        assert graph.has_node("raw_customers")
        assert graph.has_node("output_warehouse")

    def test_impact_analysis(self, tracker):
        """Test impact analysis on lineage graph."""
        from truthound.lineage import ImpactAnalyzer

        # Build lineage
        tracker.track_source("source_a", source_type="file")
        tracker.track_source("source_b", source_type="file")
        tracker.track_transformation("transform_1", sources=["source_a"])
        tracker.track_transformation("transform_2", sources=["source_a", "source_b"])
        tracker.track_transformation("transform_3", sources=["transform_1", "transform_2"])
        tracker.track_output("output", sources=["transform_3"])

        graph = tracker.graph
        analyzer = ImpactAnalyzer(graph)

        # Analyze impact of changing source_a (method is analyze_impact, not analyze)
        result = analyzer.analyze_impact("source_a")

        # source_a affects: transform_1, transform_2, transform_3, output
        assert len(result.affected_nodes) >= 4

    def test_column_level_lineage(self, tracker):
        """Test column-level lineage tracking."""
        tracker.track_source(
            "users",
            source_type="database",
        )

        tracker.track_transformation(
            "users_with_full_name",
            sources=["users"],
            operation="derive",
        )

        graph = tracker.graph
        assert graph.has_node("users_with_full_name")
        node = graph.get_node("users_with_full_name")

        assert node is not None


class TestLineageVisualizationIntegration:
    """Integration tests for lineage visualization."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample lineage graph for visualization."""
        from truthound.lineage import LineageTracker

        tracker = LineageTracker()

        # Build sample pipeline
        tracker.track_source("source_1", source_type="csv")
        tracker.track_source("source_2", source_type="database")
        tracker.track_transformation("join_1", sources=["source_1", "source_2"])
        tracker.track_transformation("filter_1", sources=["join_1"])
        tracker.track_validation("validate_1", sources=["filter_1"])
        tracker.track_output("output_1", sources=["validate_1"])

        return tracker.graph

    def test_d3_renderer(self, sample_graph):
        """Test D3.js graph rendering."""
        import json
        from truthound.lineage.visualization.renderers.d3 import D3Renderer
        from truthound.lineage.visualization.protocols import RenderConfig

        renderer = D3Renderer()
        config = RenderConfig(
            width=1200,
            height=800,
        )

        # Render to JSON (D3 data format) - returns a JSON string
        json_str = renderer.render(sample_graph, config)
        json_output = json.loads(json_str)

        assert "nodes" in json_output
        assert "links" in json_output
        assert len(json_output["nodes"]) == 6
        assert len(json_output["links"]) >= 5

        # Render to HTML
        html_output = renderer.render_html(sample_graph, config)

        assert "<html>" in html_output
        assert "d3" in html_output.lower()

    def test_cytoscape_renderer(self, sample_graph):
        """Test Cytoscape.js graph rendering."""
        import json
        from truthound.lineage.visualization.renderers.cytoscape import CytoscapeRenderer
        from truthound.lineage.visualization.protocols import RenderConfig

        renderer = CytoscapeRenderer()
        config = RenderConfig(layout="dagre")

        # Render to Cytoscape format - returns a JSON string
        json_str = renderer.render(sample_graph, config)
        output = json.loads(json_str)

        assert "elements" in output
        # Cytoscape elements is a flat list containing both nodes and edges
        nodes_count = sum(1 for e in output["elements"] if e.get("group") == "nodes")
        edges_count = sum(1 for e in output["elements"] if e.get("group") == "edges")
        assert nodes_count == 6
        assert edges_count >= 5

        # Render to HTML
        html_output = renderer.render_html(sample_graph, config)

        assert "cytoscape" in html_output.lower()

    def test_graphviz_renderer(self, sample_graph):
        """Test Graphviz DOT rendering."""
        from truthound.lineage.visualization.renderers.graphviz import GraphvizRenderer
        from truthound.lineage.visualization.protocols import RenderConfig

        renderer = GraphvizRenderer()
        config = RenderConfig(
            orientation="LR",
        )

        # Render to DOT format
        dot_output = renderer.render(sample_graph, config)

        assert "digraph" in dot_output
        assert "source_1" in dot_output
        assert "->" in dot_output

    def test_mermaid_renderer(self, sample_graph):
        """Test Mermaid diagram rendering."""
        from truthound.lineage.visualization.renderers.mermaid import MermaidRenderer
        from truthound.lineage.visualization.protocols import RenderConfig

        renderer = MermaidRenderer()
        config = RenderConfig(orientation="LR")

        # Render to Mermaid format
        mermaid_output = renderer.render(sample_graph, config)

        assert "graph LR" in mermaid_output or "flowchart LR" in mermaid_output
        assert "source_1" in mermaid_output
        assert "-->" in mermaid_output

    def test_all_renderers_consistency(self, sample_graph):
        """Test that all renderers produce consistent node/edge counts."""
        import json
        from truthound.lineage.visualization.renderers.d3 import D3Renderer
        from truthound.lineage.visualization.renderers.cytoscape import CytoscapeRenderer
        from truthound.lineage.visualization.protocols import RenderConfig

        config = RenderConfig()

        d3_output = json.loads(D3Renderer().render(sample_graph, config))
        cyto_output = json.loads(CytoscapeRenderer().render(sample_graph, config))

        # Both should have same number of nodes
        d3_nodes = len(d3_output["nodes"])
        cyto_nodes = sum(1 for e in cyto_output["elements"] if e.get("group") == "nodes")

        assert d3_nodes == cyto_nodes == 6


class TestOpenLineageIntegration:
    """Integration tests for OpenLineage integration."""

    @pytest.fixture
    def emitter(self):
        """Create OpenLineage emitter."""
        from truthound.lineage.integrations.openlineage import (
            OpenLineageEmitter,
            OpenLineageConfig,
        )

        config = OpenLineageConfig(
            endpoint="http://localhost:5000/api/v1/lineage",
            namespace="truthound-test",
            producer="truthound-integration-test",
        )
        return OpenLineageEmitter(config)

    def test_run_lifecycle(self, emitter):
        """Test OpenLineage run lifecycle events."""
        # Start run
        run = emitter.start_run(
            job_name="test-job",
            inputs=[
                emitter.build_input_dataset(
                    "input-dataset",
                    schema=[
                        {"name": "id", "type": "integer"},
                        {"name": "value", "type": "double"},
                    ],
                )
            ],
        )

        assert run.run_id is not None
        assert run.job_name == "test-job"

        # Emit running
        emitter.emit_running(
            run,
            facets={"progress": {"completed": 50, "total": 100}},
        )

        # Complete run
        emitter.emit_complete(
            run,
            outputs=[
                emitter.build_output_dataset(
                    "output-dataset",
                    schema=[
                        {"name": "id", "type": "integer"},
                        {"name": "result", "type": "string"},
                    ],
                    row_count=1000,
                )
            ],
        )

    def test_run_failure(self, emitter):
        """Test OpenLineage failure event."""
        run = emitter.start_run("failing-job")

        # Emit failure
        emitter.emit_fail(
            run,
            error=ValueError("Data validation failed"),
            facets={"errorDetails": {"code": "VALIDATION_ERROR"}},
        )

    def test_run_abort(self, emitter):
        """Test OpenLineage abort event."""
        run = emitter.start_run("aborted-job")

        # Emit abort
        emitter.emit_abort(
            run,
            reason="User requested cancellation",
        )

    def test_nested_runs(self, emitter):
        """Test parent-child run relationships."""
        # Start parent run
        parent_run = emitter.start_run("parent-job")

        # Start child runs
        child_run_1 = emitter.start_run(
            "child-job-1",
            parent_run_id=parent_run.run_id,
        )
        child_run_2 = emitter.start_run(
            "child-job-2",
            parent_run_id=parent_run.run_id,
        )

        # Complete children
        emitter.emit_complete(child_run_1)
        emitter.emit_complete(child_run_2)

        # Complete parent
        emitter.emit_complete(parent_run)

    def test_emit_from_truthound_graph(self, emitter):
        """Test converting Truthound lineage to OpenLineage."""
        from truthound.lineage import LineageTracker

        tracker = LineageTracker()

        # Build lineage
        tracker.track_source("input_a", source_type="file")
        tracker.track_source("input_b", source_type="database")
        tracker.track_transformation("transform", sources=["input_a", "input_b"])
        tracker.track_output("output", sources=["transform"])

        graph = tracker.graph

        # Emit OpenLineage events
        runs = emitter.emit_from_graph(
            graph,
            job_name="truthound-pipeline",
        )

        # Should have created runs for transformations
        assert len(runs) >= 2

    def test_dataset_facets(self, emitter):
        """Test dataset facets creation."""
        from truthound.lineage.integrations.openlineage import DatasetFacets

        facets = DatasetFacets(
            schema_fields=[
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
                {"name": "amount", "type": "decimal"},
            ],
            data_source={"name": "production-db", "uri": "postgresql://host/db"},
            lifecycle_state="CREATE",
            ownership={"owners": [{"name": "data-team", "type": "GROUP"}]},
            quality_metrics={"rowCount": 1000000, "nullCount": {"id": 0, "name": 50}},
        )

        facets_dict = facets.to_dict()

        assert "schema" in facets_dict
        assert len(facets_dict["schema"]["fields"]) == 3
        assert "dataSource" in facets_dict
        assert "lifecycleStateChange" in facets_dict
        assert "ownership" in facets_dict
        assert "dataQualityMetrics" in facets_dict


class TestLineageWithStreamingIntegration:
    """Integration tests for lineage with streaming data."""

    @pytest.mark.asyncio
    async def test_streaming_lineage_tracking(self):
        """Test lineage tracking for streaming pipelines."""
        from truthound.lineage import LineageTracker

        tracker = LineageTracker()

        # Track streaming source
        tracker.track_source(
            "kafka_events",
            source_type="kafka",
            metadata={
                "topic": "user-events",
                "bootstrap_servers": "localhost:9092",
            },
        )

        # Track stream processing
        tracker.track_transformation(
            "filtered_events",
            sources=["kafka_events"],
            operation="filter",
        )

        tracker.track_transformation(
            "windowed_aggregation",
            sources=["filtered_events"],
            operation="aggregate",
        )

        # Track streaming output
        tracker.track_output(
            "metrics_topic",
            sources=["windowed_aggregation"],
        )

        graph = tracker.graph

        # Verify streaming pipeline lineage (node_count instead of len(nodes))
        assert graph.node_count == 4

    @pytest.mark.asyncio
    async def test_lineage_with_ml_monitoring(self):
        """Test lineage tracking for ML pipelines."""
        from truthound.lineage import LineageTracker

        tracker = LineageTracker()

        # Track training data
        tracker.track_source(
            "training_data",
            source_type="parquet",
        )

        # Track feature engineering
        tracker.track_transformation(
            "features",
            sources=["training_data"],
            operation="feature_engineering",
        )

        # Track model training
        tracker.track_transformation(
            "model_v1",
            sources=["features"],
            operation="train",
        )

        # Track prediction output
        tracker.track_output(
            "predictions",
            sources=["model_v1"],
        )

        graph = tracker.graph

        # Verify ML pipeline lineage (use get_node instead of nodes[])
        assert graph.has_node("model_v1")
        model_node = graph.get_node("model_v1")
        assert model_node is not None
