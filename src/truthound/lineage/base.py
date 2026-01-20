"""Base classes and data structures for data lineage.

This module provides the core abstractions for lineage tracking:
- LineageNode: Represents a data asset in the lineage graph
- LineageEdge: Represents a relationship between nodes
- LineageGraph: The complete lineage DAG
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)
import threading
import json
from pathlib import Path
import uuid


# =============================================================================
# Enums
# =============================================================================


class NodeType(str, Enum):
    """Types of nodes in the lineage graph."""

    SOURCE = "source"  # Raw data source
    TABLE = "table"  # Database table
    FILE = "file"  # File-based data
    STREAM = "stream"  # Streaming source
    TRANSFORMATION = "transformation"  # Data transformation
    VALIDATION = "validation"  # Validation checkpoint
    MODEL = "model"  # ML model
    REPORT = "report"  # Output report
    EXTERNAL = "external"  # External system
    VIRTUAL = "virtual"  # Virtual/computed dataset


class EdgeType(str, Enum):
    """Types of edges in the lineage graph."""

    DERIVED_FROM = "derived_from"  # Data derivation
    VALIDATED_BY = "validated_by"  # Validation relationship
    USED_BY = "used_by"  # Usage relationship
    TRANSFORMED_TO = "transformed_to"  # Transformation
    JOINED_WITH = "joined_with"  # Join operation
    AGGREGATED_TO = "aggregated_to"  # Aggregation
    FILTERED_TO = "filtered_to"  # Filter operation
    DEPENDS_ON = "depends_on"  # Generic dependency


class OperationType(str, Enum):
    """Types of data operations."""

    READ = "read"
    WRITE = "write"
    TRANSFORM = "transform"
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    VALIDATE = "validate"
    PROFILE = "profile"
    MASK = "mask"
    EXPORT = "export"


# =============================================================================
# Exceptions
# =============================================================================


class LineageError(Exception):
    """Base exception for lineage-related errors."""

    pass


class NodeNotFoundError(LineageError):
    """Raised when a node is not found in the graph."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


class CyclicDependencyError(LineageError):
    """Raised when a cyclic dependency is detected."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        super().__init__(f"Cyclic dependency detected: {' -> '.join(cycle)}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LineageConfig:
    """Configuration for lineage tracking.

    Attributes:
        track_column_level: Track column-level lineage
        track_row_level: Track row-level lineage (expensive)
        store_samples: Store sample values at each node
        max_history: Maximum history entries per node
        auto_track: Automatically track operations
        persist_path: Path to persist lineage data
    """

    track_column_level: bool = True
    track_row_level: bool = False
    store_samples: bool = False
    max_history: int = 100
    auto_track: bool = True
    persist_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class LineageMetadata:
    """Metadata for a lineage node or edge.

    Stores additional context about the data or operation.
    """

    description: str = ""
    owner: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "owner": self.owner,
            "tags": list(self.tags),
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageMetadata":
        return cls(
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            tags=tuple(data.get("tags", [])),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class ColumnLineage:
    """Column-level lineage information.

    Tracks which columns derive from which source columns.
    """

    column: str
    source_columns: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (node_id, column_name)
    transformation: str = ""  # Description of transformation
    dtype: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "source_columns": [
                {"node": node, "column": col}
                for node, col in self.source_columns
            ],
            "transformation": self.transformation,
            "dtype": self.dtype,
        }


@dataclass
class LineageNode:
    """A node in the lineage graph.

    Represents a data asset (table, file, transformation result, etc.)
    in the data lineage.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        node_type: Type of node
        location: Physical location (path, URI, etc.)
        schema: Column schema if applicable
        metadata: Additional metadata
        column_lineage: Column-level lineage information
    """

    id: str
    name: str
    node_type: NodeType
    location: str = ""
    schema: dict[str, str] = field(default_factory=dict)  # column -> dtype
    metadata: LineageMetadata = field(default_factory=LineageMetadata)
    column_lineage: tuple[ColumnLineage, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineageNode):
            return False
        return self.id == other.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "location": self.location,
            "schema": self.schema,
            "metadata": self.metadata.to_dict(),
            "column_lineage": [cl.to_dict() for cl in self.column_lineage],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageNode":
        """Deserialize a LineageNode from a dictionary.

        Args:
            data: Dictionary containing node data

        Returns:
            LineageNode instance

        Raises:
            LineageError: If required fields are missing or invalid
        """
        # Validate required fields with helpful error messages
        required_fields = ["id", "name", "node_type"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            node_id = data.get("id", data.get("name", "<unknown>"))
            valid_types = ", ".join(t.value for t in NodeType)
            raise LineageError(
                f"Invalid lineage node '{node_id}': missing required field(s): {', '.join(missing)}. "
                f"Each node must have 'id', 'name', and 'node_type' fields. "
                f"Valid node_type values: {valid_types}. "
                f"Example: {{'id': 'my_table', 'name': 'My Table', 'node_type': 'table'}}"
            )

        # Validate node_type value
        node_type_value = data["node_type"]
        valid_types = [t.value for t in NodeType]
        if node_type_value not in valid_types:
            raise LineageError(
                f"Invalid node_type '{node_type_value}' for node '{data['id']}'. "
                f"Valid values: {', '.join(valid_types)}"
            )

        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType(node_type_value),
            location=data.get("location", ""),
            schema=data.get("schema", {}),
            metadata=LineageMetadata.from_dict(data.get("metadata", {})),
            column_lineage=tuple(
                ColumnLineage(
                    column=cl["column"],
                    source_columns=tuple(
                        (sc["node"], sc["column"])
                        for sc in cl.get("source_columns", [])
                    ),
                    transformation=cl.get("transformation", ""),
                    dtype=cl.get("dtype", ""),
                )
                for cl in data.get("column_lineage", [])
            ),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class LineageEdge:
    """An edge in the lineage graph.

    Represents a relationship between two nodes (e.g., derivation,
    transformation, validation).

    Attributes:
        source: Source node ID
        target: Target node ID
        edge_type: Type of relationship
        operation: Operation that created this relationship
        metadata: Additional metadata about the relationship
    """

    source: str
    target: str
    edge_type: EdgeType
    operation: OperationType = OperationType.TRANSFORM
    metadata: LineageMetadata = field(default_factory=LineageMetadata)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineageEdge):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.edge_type == other.edge_type
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "operation": self.operation.value,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageEdge":
        """Deserialize a LineageEdge from a dictionary.

        Args:
            data: Dictionary containing edge data

        Returns:
            LineageEdge instance

        Raises:
            LineageError: If required fields are missing or invalid
        """
        # Validate required fields with helpful error messages
        required_fields = ["source", "target", "edge_type"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            source = data.get("source", "<unknown>")
            target = data.get("target", "<unknown>")
            valid_types = ", ".join(t.value for t in EdgeType)
            raise LineageError(
                f"Invalid lineage edge '{source}' -> '{target}': missing required field(s): {', '.join(missing)}. "
                f"Each edge must have 'source', 'target', and 'edge_type' fields. "
                f"Valid edge_type values: {valid_types}. "
                f"Example: {{'source': 'raw_data', 'target': 'processed', 'edge_type': 'derived_from'}}"
            )

        # Validate edge_type value
        edge_type_value = data["edge_type"]
        valid_types = [t.value for t in EdgeType]
        if edge_type_value not in valid_types:
            raise LineageError(
                f"Invalid edge_type '{edge_type_value}' for edge '{data['source']}' -> '{data['target']}'. "
                f"Valid values: {', '.join(valid_types)}"
            )

        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=EdgeType(edge_type_value),
            operation=OperationType(data.get("operation", "transform")),
            metadata=LineageMetadata.from_dict(data.get("metadata", {})),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


# =============================================================================
# Lineage Graph
# =============================================================================


class LineageGraph:
    """A directed acyclic graph representing data lineage.

    Provides methods for:
    - Adding/removing nodes and edges
    - Querying lineage (upstream/downstream)
    - Detecting cycles
    - Serialization/deserialization
    - Graph traversal

    Example:
        >>> graph = LineageGraph()
        >>> graph.add_node(LineageNode(id="raw", name="Raw Data", node_type=NodeType.SOURCE))
        >>> graph.add_node(LineageNode(id="clean", name="Clean Data", node_type=NodeType.TRANSFORMATION))
        >>> graph.add_edge(LineageEdge(source="raw", target="clean", edge_type=EdgeType.TRANSFORMED_TO))
        >>> print(graph.get_downstream("raw"))
    """

    def __init__(self, config: LineageConfig | None = None):
        """Initialize the lineage graph.

        Args:
            config: Optional configuration
        """
        self._config = config or LineageConfig()
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        self._adjacency: dict[str, list[str]] = {}  # Forward edges
        self._reverse_adjacency: dict[str, list[str]] = {}  # Backward edges
        self._lock = threading.RLock()

    @property
    def config(self) -> LineageConfig:
        return self._config

    def add_node(self, node: LineageNode) -> None:
        """Add a node to the graph.

        Args:
            node: Node to add

        Raises:
            ValueError: If node with same ID already exists
        """
        with self._lock:
            if node.id in self._nodes:
                # Update existing node
                self._nodes[node.id] = node
            else:
                self._nodes[node.id] = node
                self._adjacency[node.id] = []
                self._reverse_adjacency[node.id] = []

    def remove_node(self, node_id: str) -> LineageNode:
        """Remove a node and all its edges.

        Args:
            node_id: ID of node to remove

        Returns:
            Removed node

        Raises:
            NodeNotFoundError: If node not found
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)

            node = self._nodes.pop(node_id)

            # Remove edges
            self._edges = [
                e for e in self._edges
                if e.source != node_id and e.target != node_id
            ]

            # Update adjacency
            if node_id in self._adjacency:
                del self._adjacency[node_id]
            if node_id in self._reverse_adjacency:
                del self._reverse_adjacency[node_id]

            for adj_list in self._adjacency.values():
                if node_id in adj_list:
                    adj_list.remove(node_id)
            for adj_list in self._reverse_adjacency.values():
                if node_id in adj_list:
                    adj_list.remove(node_id)

            return node

    def get_node(self, node_id: str) -> LineageNode:
        """Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            The node

        Raises:
            NodeNotFoundError: If node not found
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)
            return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        with self._lock:
            return node_id in self._nodes

    def add_edge(self, edge: LineageEdge) -> None:
        """Add an edge to the graph.

        Args:
            edge: Edge to add

        Raises:
            NodeNotFoundError: If source or target node not found
            CyclicDependencyError: If edge would create a cycle
        """
        with self._lock:
            if edge.source not in self._nodes:
                raise NodeNotFoundError(edge.source)
            if edge.target not in self._nodes:
                raise NodeNotFoundError(edge.target)

            # Check for cycles
            if self._would_create_cycle(edge.source, edge.target):
                cycle = self._find_cycle(edge.source, edge.target)
                raise CyclicDependencyError(cycle)

            # Check if edge already exists
            for existing in self._edges:
                if existing == edge:
                    return  # Edge already exists

            self._edges.append(edge)
            self._adjacency[edge.source].append(edge.target)
            self._reverse_adjacency[edge.target].append(edge.source)

    def remove_edge(self, source: str, target: str) -> LineageEdge | None:
        """Remove an edge from the graph.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Removed edge or None if not found
        """
        with self._lock:
            for i, edge in enumerate(self._edges):
                if edge.source == source and edge.target == target:
                    removed = self._edges.pop(i)
                    if target in self._adjacency.get(source, []):
                        self._adjacency[source].remove(target)
                    if source in self._reverse_adjacency.get(target, []):
                        self._reverse_adjacency[target].remove(source)
                    return removed
            return None

    def get_upstream(self, node_id: str, depth: int = -1) -> list[LineageNode]:
        """Get all upstream (parent) nodes.

        Args:
            node_id: Starting node ID
            depth: Maximum depth (-1 for unlimited)

        Returns:
            List of upstream nodes
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)

            visited: set[str] = set()
            result: list[LineageNode] = []
            self._traverse_upstream(node_id, visited, result, depth, 0)
            return result

    def _traverse_upstream(
        self,
        node_id: str,
        visited: set[str],
        result: list[LineageNode],
        max_depth: int,
        current_depth: int,
    ) -> None:
        """Recursive upstream traversal."""
        if max_depth != -1 and current_depth >= max_depth:
            return

        for parent_id in self._reverse_adjacency.get(node_id, []):
            if parent_id not in visited:
                visited.add(parent_id)
                result.append(self._nodes[parent_id])
                self._traverse_upstream(
                    parent_id, visited, result, max_depth, current_depth + 1
                )

    def get_downstream(self, node_id: str, depth: int = -1) -> list[LineageNode]:
        """Get all downstream (child) nodes.

        Args:
            node_id: Starting node ID
            depth: Maximum depth (-1 for unlimited)

        Returns:
            List of downstream nodes
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)

            visited: set[str] = set()
            result: list[LineageNode] = []
            self._traverse_downstream(node_id, visited, result, depth, 0)
            return result

    def _traverse_downstream(
        self,
        node_id: str,
        visited: set[str],
        result: list[LineageNode],
        max_depth: int,
        current_depth: int,
    ) -> None:
        """Recursive downstream traversal."""
        if max_depth != -1 and current_depth >= max_depth:
            return

        for child_id in self._adjacency.get(node_id, []):
            if child_id not in visited:
                visited.add(child_id)
                result.append(self._nodes[child_id])
                self._traverse_downstream(
                    child_id, visited, result, max_depth, current_depth + 1
                )

    def get_edges_for_node(
        self, node_id: str, direction: str = "both"
    ) -> list[LineageEdge]:
        """Get all edges connected to a node.

        Args:
            node_id: Node ID
            direction: 'incoming', 'outgoing', or 'both'

        Returns:
            List of edges
        """
        with self._lock:
            edges = []
            for edge in self._edges:
                if direction in ("outgoing", "both") and edge.source == node_id:
                    edges.append(edge)
                elif direction in ("incoming", "both") and edge.target == node_id:
                    edges.append(edge)
            return edges

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding an edge would create a cycle."""
        # If target can reach source, adding source->target creates a cycle
        visited: set[str] = set()
        return self._can_reach(target, source, visited)

    def _can_reach(
        self, start: str, end: str, visited: set[str]
    ) -> bool:
        """Check if end is reachable from start."""
        if start == end:
            return True
        if start in visited:
            return False
        visited.add(start)
        for neighbor in self._adjacency.get(start, []):
            if self._can_reach(neighbor, end, visited):
                return True
        return False

    def _find_cycle(self, source: str, target: str) -> list[str]:
        """Find the cycle path if adding source->target creates one."""
        path = [target]
        self._find_path(target, source, path)
        path.append(source)
        path.append(target)
        return path

    def _find_path(
        self, start: str, end: str, path: list[str]
    ) -> bool:
        """Find path from start to end."""
        for neighbor in self._adjacency.get(start, []):
            if neighbor == end:
                return True
            if neighbor not in path:
                path.append(neighbor)
                if self._find_path(neighbor, end, path):
                    return True
                path.pop()
        return False

    def get_roots(self) -> list[LineageNode]:
        """Get all root nodes (no incoming edges)."""
        with self._lock:
            roots = []
            for node_id, node in self._nodes.items():
                if not self._reverse_adjacency.get(node_id):
                    roots.append(node)
            return roots

    def get_leaves(self) -> list[LineageNode]:
        """Get all leaf nodes (no outgoing edges)."""
        with self._lock:
            leaves = []
            for node_id, node in self._nodes.items():
                if not self._adjacency.get(node_id):
                    leaves.append(node)
            return leaves

    def topological_sort(self) -> list[LineageNode]:
        """Return nodes in topological order."""
        with self._lock:
            in_degree: dict[str, int] = {
                node_id: len(self._reverse_adjacency.get(node_id, []))
                for node_id in self._nodes
            }

            queue = [
                node_id for node_id, degree in in_degree.items()
                if degree == 0
            ]
            result = []

            while queue:
                node_id = queue.pop(0)
                result.append(self._nodes[node_id])

                for child_id in self._adjacency.get(node_id, []):
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        queue.append(child_id)

            return result

    @property
    def nodes(self) -> list[LineageNode]:
        """Get all nodes."""
        with self._lock:
            return list(self._nodes.values())

    @property
    def edges(self) -> list[LineageEdge]:
        """Get all edges."""
        with self._lock:
            return list(self._edges)

    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self._edges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        with self._lock:
            return {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "edges": [edge.to_dict() for edge in self._edges],
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: LineageConfig | None = None) -> "LineageGraph":
        """Deserialize graph from dictionary."""
        graph = cls(config)
        for node_data in data.get("nodes", []):
            graph.add_node(LineageNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(LineageEdge.from_dict(edge_data))
        return graph

    def save(self, path: str | Path) -> None:
        """Save graph to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, config: LineageConfig | None = None) -> "LineageGraph":
        """Load graph from file.

        Args:
            path: Path to the lineage JSON file
            config: Optional lineage configuration

        Returns:
            Loaded LineageGraph

        Raises:
            FileNotFoundError: If the file does not exist
            LineageError: If the file is empty or contains invalid JSON
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Lineage file not found: {path}")

        # Check for empty file
        if path.stat().st_size == 0:
            raise LineageError(
                f"Lineage file is empty: {path}. "
                "Create a lineage graph first using LineageGraph.save() or "
                "the lineage tracking API."
            )

        try:
            with open(path) as f:
                content = f.read().strip()
                if not content:
                    raise LineageError(
                        f"Lineage file is empty: {path}. "
                        "Create a lineage graph first using LineageGraph.save() or "
                        "the lineage tracking API."
                    )
                data = json.loads(content)
        except json.JSONDecodeError as e:
            raise LineageError(
                f"Invalid JSON in lineage file: {path}. "
                f"Error: {e}. "
                "Ensure the file contains valid lineage data in JSON format."
            ) from e

        return cls.from_dict(data, config)

    def clear(self) -> None:
        """Clear all nodes and edges."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._adjacency.clear()
            self._reverse_adjacency.clear()

    def __repr__(self) -> str:
        return f"<LineageGraph nodes={self.node_count} edges={self.edge_count}>"
