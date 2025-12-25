"""Impact analysis for data lineage.

Provides tools to analyze the impact of changes to data assets
on downstream dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from truthound.lineage.base import (
    LineageGraph,
    LineageNode,
    LineageEdge,
    NodeType,
    EdgeType,
    NodeNotFoundError,
)


class ImpactLevel(str, Enum):
    """Severity level of impact."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AffectedNode:
    """A node affected by a change.

    Attributes:
        node: The affected lineage node
        distance: Distance from the changed node
        path: Path from changed node to this node
        impact_level: Severity of impact
        impact_reason: Reason for the impact level
    """

    node: LineageNode
    distance: int
    path: tuple[str, ...] = field(default_factory=tuple)
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    impact_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node.id,
            "node_name": self.node.name,
            "node_type": self.node.node_type.value,
            "distance": self.distance,
            "path": list(self.path),
            "impact_level": self.impact_level.value,
            "impact_reason": self.impact_reason,
        }


@dataclass
class ImpactResult:
    """Result of impact analysis.

    Attributes:
        source_node: The node that was changed
        affected_nodes: List of affected downstream nodes
        total_affected: Total number of affected nodes
        critical_paths: Paths to critical nodes
        recommendations: Suggested actions
    """

    source_node: LineageNode
    affected_nodes: tuple[AffectedNode, ...] = field(default_factory=tuple)
    total_affected: int = 0
    max_depth: int = 0
    analysis_time_ms: float = 0.0
    analyzed_at: datetime = field(default_factory=datetime.now)

    def get_by_level(self, level: ImpactLevel) -> list[AffectedNode]:
        """Get affected nodes by impact level."""
        return [n for n in self.affected_nodes if n.impact_level == level]

    def get_critical_nodes(self) -> list[AffectedNode]:
        """Get critically affected nodes."""
        return self.get_by_level(ImpactLevel.CRITICAL)

    def get_by_type(self, node_type: NodeType) -> list[AffectedNode]:
        """Get affected nodes by type."""
        return [n for n in self.affected_nodes if n.node.node_type == node_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node": {
                "id": self.source_node.id,
                "name": self.source_node.name,
                "type": self.source_node.node_type.value,
            },
            "affected_nodes": [n.to_dict() for n in self.affected_nodes],
            "total_affected": self.total_affected,
            "max_depth": self.max_depth,
            "by_level": {
                level.value: len(self.get_by_level(level))
                for level in ImpactLevel
                if len(self.get_by_level(level)) > 0
            },
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "analyzed_at": self.analyzed_at.isoformat(),
        }

    def summary(self) -> str:
        """Get a summary of the impact analysis."""
        lines = [
            f"Impact Analysis for: {self.source_node.name}",
            f"Total affected nodes: {self.total_affected}",
            f"Maximum depth: {self.max_depth}",
        ]

        for level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH, ImpactLevel.MEDIUM, ImpactLevel.LOW]:
            count = len(self.get_by_level(level))
            if count > 0:
                lines.append(f"  {level.value}: {count}")

        return "\n".join(lines)


class ImpactAnalyzer:
    """Analyze impact of changes to data assets.

    Provides methods to:
    - Analyze downstream impact of changes
    - Identify critical paths
    - Estimate impact severity
    - Generate recommendations

    Example:
        >>> analyzer = ImpactAnalyzer(graph)
        >>> result = analyzer.analyze_impact("raw_data")
        >>> print(result.summary())
        >>> for node in result.get_critical_nodes():
        ...     print(f"Critical: {node.node.name}")
    """

    def __init__(
        self,
        graph: LineageGraph,
        impact_rules: dict[NodeType, ImpactLevel] | None = None,
    ):
        """Initialize the impact analyzer.

        Args:
            graph: Lineage graph to analyze
            impact_rules: Custom impact level rules by node type
        """
        self._graph = graph
        self._impact_rules = impact_rules or self._default_impact_rules()

    def _default_impact_rules(self) -> dict[NodeType, ImpactLevel]:
        """Default impact rules based on node type."""
        return {
            NodeType.SOURCE: ImpactLevel.HIGH,
            NodeType.TABLE: ImpactLevel.HIGH,
            NodeType.TRANSFORMATION: ImpactLevel.MEDIUM,
            NodeType.VALIDATION: ImpactLevel.LOW,
            NodeType.REPORT: ImpactLevel.MEDIUM,
            NodeType.MODEL: ImpactLevel.CRITICAL,
            NodeType.EXTERNAL: ImpactLevel.HIGH,
        }

    def analyze_impact(
        self,
        node_id: str,
        max_depth: int = -1,
        include_validations: bool = True,
    ) -> ImpactResult:
        """Analyze the downstream impact of changes to a node.

        Args:
            node_id: ID of the changed node
            max_depth: Maximum depth to analyze (-1 for unlimited)
            include_validations: Include validation nodes in analysis

        Returns:
            ImpactResult with affected nodes and analysis
        """
        import time
        start = time.perf_counter()

        source_node = self._graph.get_node(node_id)
        affected: list[AffectedNode] = []
        visited: set[str] = set()
        max_found_depth = 0

        self._analyze_downstream(
            node_id,
            visited,
            affected,
            max_depth,
            current_depth=0,
            path=[node_id],
            include_validations=include_validations,
        )

        if affected:
            max_found_depth = max(a.distance for a in affected)

        elapsed = (time.perf_counter() - start) * 1000

        return ImpactResult(
            source_node=source_node,
            affected_nodes=tuple(affected),
            total_affected=len(affected),
            max_depth=max_found_depth,
            analysis_time_ms=elapsed,
        )

    def _analyze_downstream(
        self,
        node_id: str,
        visited: set[str],
        affected: list[AffectedNode],
        max_depth: int,
        current_depth: int,
        path: list[str],
        include_validations: bool,
    ) -> None:
        """Recursively analyze downstream nodes."""
        if max_depth != -1 and current_depth >= max_depth:
            return

        downstream = self._graph.get_downstream(node_id, depth=1)

        for node in downstream:
            if node.id in visited:
                continue

            if not include_validations and node.node_type == NodeType.VALIDATION:
                continue

            visited.add(node.id)
            new_path = path + [node.id]

            impact_level = self._calculate_impact_level(node, current_depth + 1)
            impact_reason = self._get_impact_reason(node, current_depth + 1)

            affected.append(AffectedNode(
                node=node,
                distance=current_depth + 1,
                path=tuple(new_path),
                impact_level=impact_level,
                impact_reason=impact_reason,
            ))

            self._analyze_downstream(
                node.id,
                visited,
                affected,
                max_depth,
                current_depth + 1,
                new_path,
                include_validations,
            )

    def _calculate_impact_level(
        self,
        node: LineageNode,
        distance: int,
    ) -> ImpactLevel:
        """Calculate impact level for a node."""
        # Base level from node type
        base_level = self._impact_rules.get(node.node_type, ImpactLevel.MEDIUM)

        # Adjust based on distance
        level_order = [
            ImpactLevel.NONE,
            ImpactLevel.LOW,
            ImpactLevel.MEDIUM,
            ImpactLevel.HIGH,
            ImpactLevel.CRITICAL,
        ]
        base_index = level_order.index(base_level)

        # Reduce severity for distant nodes
        if distance > 3:
            adjusted_index = max(0, base_index - 1)
        elif distance > 5:
            adjusted_index = max(0, base_index - 2)
        else:
            adjusted_index = base_index

        return level_order[adjusted_index]

    def _get_impact_reason(self, node: LineageNode, distance: int) -> str:
        """Get reason for impact level."""
        reasons = []

        if node.node_type == NodeType.MODEL:
            reasons.append("ML model depends on this data")
        elif node.node_type == NodeType.REPORT:
            reasons.append("Report output affected")
        elif node.node_type == NodeType.EXTERNAL:
            reasons.append("External system integration affected")

        if distance == 1:
            reasons.append("Direct dependency")
        elif distance <= 3:
            reasons.append(f"Indirect dependency (depth {distance})")
        else:
            reasons.append(f"Distant dependency (depth {distance})")

        return "; ".join(reasons) if reasons else "Downstream dependency"

    def find_critical_paths(self, node_id: str) -> list[list[LineageNode]]:
        """Find all paths to critical nodes.

        Args:
            node_id: Starting node ID

        Returns:
            List of paths (each path is a list of nodes)
        """
        result = self.analyze_impact(node_id)
        critical_nodes = result.get_critical_nodes()

        paths = []
        for affected in critical_nodes:
            path = [self._graph.get_node(nid) for nid in affected.path]
            paths.append(path)

        return paths

    def what_if_delete(self, node_id: str) -> dict[str, Any]:
        """Analyze what would happen if a node is deleted.

        Args:
            node_id: Node to potentially delete

        Returns:
            Analysis of deletion impact
        """
        result = self.analyze_impact(node_id)

        # Find nodes that would become orphaned
        orphaned = []
        for affected in result.affected_nodes:
            node = affected.node
            upstream = self._graph.get_upstream(node.id, depth=1)

            # Check if this is the only upstream
            upstream_ids = {n.id for n in upstream}
            if upstream_ids == {node_id}:
                orphaned.append(node)

        return {
            "node_to_delete": node_id,
            "total_affected": result.total_affected,
            "would_be_orphaned": [n.id for n in orphaned],
            "critical_impacts": [n.node.id for n in result.get_critical_nodes()],
            "recommendation": self._get_deletion_recommendation(result, orphaned),
        }

    def _get_deletion_recommendation(
        self,
        result: ImpactResult,
        orphaned: list[LineageNode],
    ) -> str:
        """Get recommendation for deletion."""
        if result.get_critical_nodes():
            return "NOT RECOMMENDED: Critical systems depend on this data"
        if orphaned:
            return f"CAUTION: {len(orphaned)} nodes would become orphaned"
        if result.total_affected > 10:
            return "CAUTION: Many downstream dependencies"
        if result.total_affected > 0:
            return "REVIEW: Some downstream dependencies exist"
        return "OK: No downstream dependencies"

    def compare_schemas(
        self,
        node_id: str,
        new_schema: dict[str, str],
    ) -> dict[str, Any]:
        """Analyze impact of schema changes.

        Args:
            node_id: Node with schema change
            new_schema: Proposed new schema

        Returns:
            Analysis of schema change impact
        """
        node = self._graph.get_node(node_id)
        old_schema = node.schema

        # Find changes
        added_columns = set(new_schema.keys()) - set(old_schema.keys())
        removed_columns = set(old_schema.keys()) - set(new_schema.keys())
        type_changes = {
            col: (old_schema[col], new_schema[col])
            for col in set(old_schema.keys()) & set(new_schema.keys())
            if old_schema[col] != new_schema[col]
        }

        # Analyze impact of removed/changed columns
        impact_result = self.analyze_impact(node_id)
        affected_by_removal = []

        for affected in impact_result.affected_nodes:
            for col_lineage in affected.node.column_lineage:
                for source_node, source_col in col_lineage.source_columns:
                    if source_node == node_id and source_col in removed_columns:
                        affected_by_removal.append({
                            "node": affected.node.id,
                            "column": col_lineage.column,
                            "depends_on": source_col,
                        })

        return {
            "node_id": node_id,
            "added_columns": list(added_columns),
            "removed_columns": list(removed_columns),
            "type_changes": type_changes,
            "affected_by_removal": affected_by_removal,
            "safe": len(removed_columns) == 0 and len(type_changes) == 0,
        }

    def get_dependency_chain(
        self,
        source_id: str,
        target_id: str,
    ) -> list[LineageNode] | None:
        """Get the dependency chain between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Chain of nodes, or None if not connected
        """
        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return None

        visited: set[str] = set()
        path: list[str] = []

        if self._find_path(source_id, target_id, visited, path):
            return [self._graph.get_node(nid) for nid in path]
        return None

    def _find_path(
        self,
        current: str,
        target: str,
        visited: set[str],
        path: list[str],
    ) -> bool:
        """DFS to find path."""
        visited.add(current)
        path.append(current)

        if current == target:
            return True

        for child in self._graph.get_downstream(current, depth=1):
            if child.id not in visited:
                if self._find_path(child.id, target, visited, path):
                    return True

        path.pop()
        return False
