"""Data lineage tracking module for Truthound.

This module provides data lineage capabilities:
- Lineage graph representation
- Data flow tracking
- Impact analysis
- Lineage visualization (D3, Cytoscape, Graphviz, Mermaid)
- OpenLineage integration for cross-platform compatibility

Example:
    >>> from truthound import lineage
    >>> tracker = lineage.LineageTracker()
    >>> tracker.track_source("raw_data", source_type="csv", path="/data/input.csv")
    >>> tracker.track_transformation("cleaned_data", sources=["raw_data"], operation="clean")
    >>> tracker.track_validation("validated_data", sources=["cleaned_data"])
    >>>
    >>> # Analyze impact
    >>> impact = tracker.analyze_impact("raw_data")
    >>> print(f"Affected: {impact.affected_nodes}")

Example with visualization:
    >>> from truthound.lineage.visualization import D3Renderer, RenderConfig
    >>> renderer = D3Renderer()
    >>> html = renderer.render_html(tracker.get_graph(), RenderConfig())

Example with OpenLineage:
    >>> from truthound.lineage.integrations.openlineage import OpenLineageEmitter
    >>> emitter = OpenLineageEmitter()
    >>> run = emitter.start_run("my-job")
    >>> emitter.emit_complete(run, outputs=[...])
"""

from truthound.lineage.base import (
    # Enums
    NodeType,
    EdgeType,
    OperationType,
    # Data structures
    LineageNode,
    LineageEdge,
    LineageMetadata,
    # Configuration
    LineageConfig,
    # Graph
    LineageGraph,
    # Exceptions
    LineageError,
    NodeNotFoundError,
    CyclicDependencyError,
)

from truthound.lineage.tracker import (
    LineageTracker,
    TrackingContext,
)

from truthound.lineage.impact_analysis import (
    ImpactAnalyzer,
    ImpactResult,
    AffectedNode,
    ImpactLevel,
)

__all__ = [
    # Enums
    "NodeType",
    "EdgeType",
    "OperationType",
    # Data structures
    "LineageNode",
    "LineageEdge",
    "LineageMetadata",
    # Configuration
    "LineageConfig",
    # Graph
    "LineageGraph",
    # Exceptions
    "LineageError",
    "NodeNotFoundError",
    "CyclicDependencyError",
    # Tracker
    "LineageTracker",
    "TrackingContext",
    # Impact analysis
    "ImpactAnalyzer",
    "ImpactResult",
    "AffectedNode",
    "ImpactLevel",
]
