"""Data lineage tracking.

Provides automatic and manual tracking of data lineage
during validation and transformation operations.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Generator  # noqa: TC003 - public annotation introspection
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from truthound.lineage._tracking import fingerprint, graph_edit, publish_tracker_graph, writer_lock
from truthound.lineage.base import (
    ColumnLineage,
    EdgeType,
    LineageConfig,
    LineageEdge,
    LineageGraph,
    LineageMetadata,
    LineageNode,
    NodeType,
    OperationType,
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

    def add_source(self, node_id: str) -> None:
        if node_id not in self.sources:
            self.sources.append(node_id)

    def add_target(self, node_id: str) -> None:
        if node_id not in self.targets:
            self.targets.append(node_id)


def _snapshot_context(context: TrackingContext) -> TrackingContext:
    try:
        metadata = deepcopy(context.metadata)
    except Exception:
        # Native opaque metadata remains caller-owned; never lose the original failure.
        metadata = (
            dict.copy(context.metadata) if isinstance(context.metadata, dict) else context.metadata
        )
    return replace(
        context, sources=list(context.sources), targets=list(context.targets), metadata=metadata
    )


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

    _current_tracker: ContextVar[LineageTracker | None] = ContextVar(
        "truthound_lineage_tracker", default=None
    )
    _lock = threading.Lock()

    def __init__(self, config: LineageConfig | None = None):
        """Initialize the lineage tracker.

        Args:
            config: Optional configuration
        """
        self._config = deepcopy(config) if config is not None else LineageConfig()
        if type(self._config.max_history) is not int or self._config.max_history < 0:
            raise ValueError("max_history must be a nonnegative integer")
        for name in ("track_column_level", "track_row_level", "store_samples", "auto_track"):
            if type(getattr(self._config, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        self._mutation_lock = threading.RLock()
        self._persistence_failed = False
        path = self._config.persist_path
        self._persistence_version = None
        if path is not None:
            with writer_lock(path):
                self._persistence_version = fingerprint(path)
                self._graph = (
                    LineageGraph.load(path, self._config)
                    if self._persistence_version is not None
                    else LineageGraph(self._config)
                )
        else:
            self._graph = LineageGraph(self._config)
        self._history: list[TrackingContext] = []
        self._active_contexts: dict[str, TrackingContext] = {}
        self._local = threading.local()

    @classmethod
    def get_current(cls) -> LineageTracker | None:
        """Get the current active tracker."""
        return cls._current_tracker.get()

    @classmethod
    def set_current(cls, tracker: LineageTracker | None) -> None:
        """Set the current active tracker."""
        with cls._lock:
            cls._current_tracker.set(tracker)

    @property
    def graph(self) -> LineageGraph:
        """Get the lineage graph."""
        return self._graph

    @property
    def history(self) -> list[TrackingContext]:
        """Get tracking history."""
        with self._mutation_lock:
            return [_snapshot_context(context) for context in self._history]

    @graph_edit
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
                properties={
                    k: v for k, v in metadata.items() if k not in ("description", "owner", "tags")
                },
            ),
        )

        self._graph.add_node(node)
        return name

    @graph_edit
    def track_transformation(
        self,
        name: str,
        sources: list[str],
        operation: str = "transform",
        location: str = "",
        schema: dict[str, str] | None = None,
        column_mapping: dict[str, list[tuple[str, str]]] | None = None,
        row_mapping: dict[int, list[tuple[str, int]]] | None = None,
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
            row_mapping: Explicit output row positions to source node/row positions
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        if "row_lineage" in metadata:
            raise ValueError("Use row_mapping for row provenance")
        rows = []
        if row_mapping is not None and self._config.track_row_level:
            if len(row_mapping) > 10000:
                raise ValueError("Row provenance exceeds 10000 output rows")
            for row, origins in sorted(row_mapping.items()):
                if type(row) is not int or row < 0 or len(origins) > 10000:
                    raise ValueError("Invalid row provenance position or source count")
                references = []
                for node, position in origins:
                    if node not in sources or type(position) is not int or position < 0:
                        raise ValueError(
                            "Row provenance requires declared sources and nonnegative positions"
                        )
                    references.append({"node": node, "row": position})
                rows.append({"row": row, "sources": references})
        # Create column lineage
        column_lineage = []
        if column_mapping and self._config.track_column_level:
            for col, source_cols in column_mapping.items():
                column_lineage.append(
                    ColumnLineage(
                        column=col,
                        source_columns=tuple(source_cols),
                        transformation=operation,
                        dtype=schema.get(col, "") if schema else "",
                    )
                )

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
                    **(
                        {"row_lineage": rows}
                        if self._config.track_row_level and row_mapping is not None
                        else {}
                    ),
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k not in ("description", "owner", "tags")
                    },
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

    @graph_edit
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
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k not in ("description", "owner", "tags")
                    },
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

    @graph_edit
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
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k not in ("description", "owner", "tags")
                    },
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

    @graph_edit
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

        token = self._current_tracker.set(self)
        try:
            yield ctx
            if self._config.auto_track and (ctx.sources or ctx.targets):
                self._record_context(operation_name, ctx)
            ctx.completed = True
        except Exception as e:
            ctx.error = e
            raise
        finally:
            self._current_tracker.reset(token)
            del self._active_contexts[ctx.operation_id]
            with self._mutation_lock:
                self._history.append(_snapshot_context(ctx))
                if self._config.max_history == 0:
                    self._history.clear()
                elif len(self._history) > self._config.max_history:
                    self._history = self._history[-self._config.max_history :]

    @graph_edit
    def _record_context(self, name: str, context: TrackingContext) -> None:
        for node_id in [*context.sources, *context.targets]:
            self._graph.get_node(node_id)
        self._graph.add_node(
            LineageNode(
                id=context.operation_id,
                name=name,
                node_type=NodeType.TRANSFORMATION,
                metadata=LineageMetadata(properties={"operation": context.operation_type.value}),
            )
        )
        for source in context.sources:
            self._graph.add_edge(
                LineageEdge(
                    source=source,
                    target=context.operation_id,
                    edge_type=EdgeType.USED_BY,
                    operation=context.operation_type,
                )
            )
        for target in context.targets:
            self._graph.add_edge(
                LineageEdge(
                    source=context.operation_id,
                    target=target,
                    edge_type=EdgeType.DERIVED_FROM,
                    operation=context.operation_type,
                )
            )

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

        with self._graph._lock:
            path = self._graph._path_ids(source_id, target_id)
            return [self._graph.get_node(nid) for nid in path] if path is not None else None

    def export_to_json(self) -> str:
        """Export lineage graph to JSON string."""
        import json

        return json.dumps(self._graph.to_dict(), indent=2, allow_nan=False)

    def save(self, path: str) -> None:
        """Save lineage graph to file."""
        with self._mutation_lock:
            if self._primary_path(path):
                publish_tracker_graph(self, self._graph, path)
            else:
                self._graph.save(path)

    def _primary_path(self, path) -> bool:
        return (
            self._config.persist_path is not None
            and Path(path).resolve() == Path(self._config.persist_path).resolve()
        )

    def load(self, path: str) -> None:
        """Load a graph; reloading the primary file confirms its current version."""
        with self._mutation_lock:
            if not self._primary_path(path):
                self._graph = LineageGraph.load(path, self._config)
                return
            with writer_lock(path):
                before = fingerprint(path)
                loaded = LineageGraph.load(path, self._config)
                if fingerprint(path) != before:
                    raise RuntimeError("Lineage snapshot changed while loading")
                self._graph = loaded
                self._persistence_version = before
                self._persistence_failed = False

    @graph_edit
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
