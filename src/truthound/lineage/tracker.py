"""Data lineage tracking.

Provides automatic and manual tracking of data lineage
during validation and transformation operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator
import uuid
import threading

import polars as pl

from truthound.lineage.base import (
    LineageConfig,
    LineageGraph,
    LineageNode,
    LineageEdge,
    LineageMetadata,
    ColumnLineage,
    NodeType,
    EdgeType,
    OperationType,
    NodeNotFoundError,
)


@dataclass
class TrackingContext:
    """Context for lineage tracking operations.

    Holds temporary state during a tracked operation.
    """

    operation_id: str
    operation_type: OperationType
    started_at: datetime
    sources: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    error: Exception | None = None


class LineageTracker:
    """Track data lineage for validation and transformation operations.

    Provides methods to:
    - Track data sources
    - Track transformations
    - Track validations
    - Query lineage history
    - Export lineage information

    Can be used as a context manager for automatic tracking.

    Example:
        >>> tracker = LineageTracker()
        >>>
        >>> # Manual tracking
        >>> tracker.track_source("raw_data", source_type="csv", path="/data/input.csv")
        >>> tracker.track_transformation(
        ...     "cleaned_data",
        ...     sources=["raw_data"],
        ...     operation="clean",
        ... )
        >>>
        >>> # Context manager for automatic tracking
        >>> with tracker.track("my_operation") as ctx:
        ...     # Operations here are automatically tracked
        ...     ctx.add_source("input_data")
        ...     ctx.add_target("output_data")
    """

    _current_tracker: "LineageTracker | None" = None
    _lock = threading.Lock()

    def __init__(self, config: LineageConfig | None = None):
        """Initialize the lineage tracker.

        Args:
            config: Optional configuration
        """
        self._config = config or LineageConfig()
        self._graph = LineageGraph(config)
        self._history: list[TrackingContext] = []
        self._active_contexts: dict[str, TrackingContext] = {}
        self._local = threading.local()

    @classmethod
    def get_current(cls) -> "LineageTracker | None":
        """Get the current active tracker."""
        return cls._current_tracker

    @classmethod
    def set_current(cls, tracker: "LineageTracker | None") -> None:
        """Set the current active tracker."""
        with cls._lock:
            cls._current_tracker = tracker

    @property
    def graph(self) -> LineageGraph:
        """Get the lineage graph."""
        return self._graph

    @property
    def history(self) -> list[TrackingContext]:
        """Get tracking history."""
        return list(self._history)

    def track_source(
        self,
        name: str,
        source_type: str = "file",
        location: str = "",
        schema: dict[str, str] | None = None,
        **metadata: Any,
    ) -> str:
        """Track a data source.

        Args:
            name: Source name/ID
            source_type: Type of source (file, table, stream, etc.)
            location: Physical location
            schema: Column schema
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        node_type_map = {
            "file": NodeType.FILE,
            "table": NodeType.TABLE,
            "stream": NodeType.STREAM,
            "external": NodeType.EXTERNAL,
        }
        node_type = node_type_map.get(source_type, NodeType.SOURCE)

        node = LineageNode(
            id=name,
            name=name,
            node_type=node_type,
            location=location,
            schema=schema or {},
            metadata=LineageMetadata(
                description=metadata.get("description", f"Source: {name}"),
                owner=metadata.get("owner", ""),
                tags=tuple(metadata.get("tags", [])),
                properties={k: v for k, v in metadata.items() if k not in ("description", "owner", "tags")},
            ),
        )

        self._graph.add_node(node)
        return name

    def track_transformation(
        self,
        name: str,
        sources: list[str],
        operation: str = "transform",
        location: str = "",
        schema: dict[str, str] | None = None,
        column_mapping: dict[str, list[tuple[str, str]]] | None = None,
        **metadata: Any,
    ) -> str:
        """Track a data transformation.

        Args:
            name: Transformation output name/ID
            sources: Source node IDs
            operation: Operation type/name
            location: Output location
            schema: Output schema
            column_mapping: Mapping of output columns to source columns
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        # Create column lineage
        column_lineage = []
        if column_mapping:
            for col, source_cols in column_mapping.items():
                column_lineage.append(ColumnLineage(
                    column=col,
                    source_columns=tuple(source_cols),
                    transformation=operation,
                    dtype=schema.get(col, "") if schema else "",
                ))

        node = LineageNode(
            id=name,
            name=name,
            node_type=NodeType.TRANSFORMATION,
            location=location,
            schema=schema or {},
            metadata=LineageMetadata(
                description=metadata.get("description", f"Transformation: {operation}"),
                owner=metadata.get("owner", ""),
                tags=tuple(metadata.get("tags", [])),
                properties={
                    "operation": operation,
                    **{k: v for k, v in metadata.items() if k not in ("description", "owner", "tags")},
                },
            ),
            column_lineage=tuple(column_lineage),
        )

        self._graph.add_node(node)

        # Create edges from sources
        operation_type_map = {
            "transform": OperationType.TRANSFORM,
            "filter": OperationType.FILTER,
            "join": OperationType.JOIN,
            "aggregate": OperationType.AGGREGATE,
            "mask": OperationType.MASK,
        }
        op_type = operation_type_map.get(operation, OperationType.TRANSFORM)

        edge_type_map = {
            "transform": EdgeType.TRANSFORMED_TO,
            "filter": EdgeType.FILTERED_TO,
            "join": EdgeType.JOINED_WITH,
            "aggregate": EdgeType.AGGREGATED_TO,
        }
        edge_type = edge_type_map.get(operation, EdgeType.DERIVED_FROM)

        for source_id in sources:
            if self._graph.has_node(source_id):
                edge = LineageEdge(
                    source=source_id,
                    target=name,
                    edge_type=edge_type,
                    operation=op_type,
                    metadata=LineageMetadata(
                        description=f"{source_id} -> {name} via {operation}",
                    ),
                )
                self._graph.add_edge(edge)

        return name

    def track_validation(
        self,
        name: str,
        sources: list[str],
        validators: list[str] | None = None,
        result_summary: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> str:
        """Track a validation operation.

        Args:
            name: Validation checkpoint name/ID
            sources: Source node IDs
            validators: List of validators used
            result_summary: Summary of validation results
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        node = LineageNode(
            id=name,
            name=name,
            node_type=NodeType.VALIDATION,
            metadata=LineageMetadata(
                description=metadata.get("description", f"Validation: {name}"),
                owner=metadata.get("owner", ""),
                tags=tuple(metadata.get("tags", [])),
                properties={
                    "validators": validators or [],
                    "result_summary": result_summary or {},
                    **{k: v for k, v in metadata.items() if k not in ("description", "owner", "tags")},
                },
            ),
        )

        self._graph.add_node(node)

        for source_id in sources:
            if self._graph.has_node(source_id):
                edge = LineageEdge(
                    source=source_id,
                    target=name,
                    edge_type=EdgeType.VALIDATED_BY,
                    operation=OperationType.VALIDATE,
                    metadata=LineageMetadata(
                        description=f"Validated {source_id}",
                    ),
                )
                self._graph.add_edge(edge)

        return name

    def track_output(
        self,
        name: str,
        sources: list[str],
        output_type: str = "report",
        location: str = "",
        **metadata: Any,
    ) -> str:
        """Track an output (report, export, etc.).

        Args:
            name: Output name/ID
            sources: Source node IDs
            output_type: Type of output
            location: Output location
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        node_type = NodeType.REPORT if output_type == "report" else NodeType.EXTERNAL

        node = LineageNode(
            id=name,
            name=name,
            node_type=node_type,
            location=location,
            metadata=LineageMetadata(
                description=metadata.get("description", f"Output: {name}"),
                owner=metadata.get("owner", ""),
                tags=tuple(metadata.get("tags", [])),
                properties={
                    "output_type": output_type,
                    **{k: v for k, v in metadata.items() if k not in ("description", "owner", "tags")},
                },
            ),
        )

        self._graph.add_node(node)

        for source_id in sources:
            if self._graph.has_node(source_id):
                edge = LineageEdge(
                    source=source_id,
                    target=name,
                    edge_type=EdgeType.USED_BY,
                    operation=OperationType.EXPORT,
                )
                self._graph.add_edge(edge)

        return name

    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEPENDS_ON,
    ) -> None:
        """Add a dependency between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of dependency
        """
        edge = LineageEdge(
            source=source_id,
            target=target_id,
            edge_type=edge_type,
        )
        self._graph.add_edge(edge)

    @contextmanager
    def track(
        self,
        operation_name: str,
        operation_type: OperationType = OperationType.TRANSFORM,
    ) -> Generator[TrackingContext, None, None]:
        """Context manager for tracking operations.

        Args:
            operation_name: Name of the operation
            operation_type: Type of operation

        Yields:
            TrackingContext for adding sources/targets
        """
        ctx = TrackingContext(
            operation_id=str(uuid.uuid4()),
            operation_type=operation_type,
            started_at=datetime.now(),
        )

        self._active_contexts[ctx.operation_id] = ctx

        try:
            yield ctx
            ctx.completed = True
        except Exception as e:
            ctx.error = e
            raise
        finally:
            del self._active_contexts[ctx.operation_id]
            self._history.append(ctx)

            # Limit history size
            if len(self._history) > self._config.max_history:
                self._history = self._history[-self._config.max_history:]

    def get_lineage(self, node_id: str, direction: str = "both") -> dict[str, Any]:
        """Get lineage information for a node.

        Args:
            node_id: Node ID
            direction: 'upstream', 'downstream', or 'both'

        Returns:
            Lineage information dictionary
        """
        result = {
            "node": self._graph.get_node(node_id).to_dict(),
            "edges": [e.to_dict() for e in self._graph.get_edges_for_node(node_id, direction)],
        }

        if direction in ("upstream", "both"):
            result["upstream"] = [n.to_dict() for n in self._graph.get_upstream(node_id)]

        if direction in ("downstream", "both"):
            result["downstream"] = [n.to_dict() for n in self._graph.get_downstream(node_id)]

        return result

    def get_path(self, source_id: str, target_id: str) -> list[LineageNode] | None:
        """Find path between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            List of nodes in path, or None if no path exists
        """
        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return None

        visited: set[str] = set()
        path: list[str] = []

        if self._find_path_dfs(source_id, target_id, visited, path):
            return [self._graph.get_node(nid) for nid in path]
        return None

    def _find_path_dfs(
        self,
        current: str,
        target: str,
        visited: set[str],
        path: list[str],
    ) -> bool:
        """DFS to find path between nodes."""
        visited.add(current)
        path.append(current)

        if current == target:
            return True

        for child_id in self._graph._adjacency.get(current, []):
            if child_id not in visited:
                if self._find_path_dfs(child_id, target, visited, path):
                    return True

        path.pop()
        return False

    def export_to_json(self) -> str:
        """Export lineage graph to JSON string."""
        import json
        return json.dumps(self._graph.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Save lineage graph to file."""
        self._graph.save(path)

    def load(self, path: str) -> None:
        """Load lineage graph from file."""
        self._graph = LineageGraph.load(path, self._config)

    def clear(self) -> None:
        """Clear all lineage data."""
        self._graph.clear()
        self._history.clear()

    def __repr__(self) -> str:
        return f"<LineageTracker nodes={self._graph.node_count} edges={self._graph.edge_count}>"


# Global tracker instance
_global_tracker: LineageTracker | None = None


def get_tracker() -> LineageTracker:
    """Get or create the global lineage tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LineageTracker()
    return _global_tracker


def set_tracker(tracker: LineageTracker) -> None:
    """Set the global lineage tracker."""
    global _global_tracker
    _global_tracker = tracker
