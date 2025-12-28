"""Dependency graph for plugins.

This module provides a graph representation of plugin dependencies
with support for cycle detection and topological sorting.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Type of dependency between plugins."""

    REQUIRED = "required"    # Plugin won't work without it
    OPTIONAL = "optional"    # Enhanced functionality if present
    DEV = "dev"              # Only needed for development


@dataclass
class DependencyNode:
    """Node in the dependency graph.

    Represents a plugin and its dependencies.

    Attributes:
        plugin_id: Unique plugin identifier
        version: Plugin version
        dependencies: Map of dependency ID to type
        reverse_dependencies: Plugins that depend on this one
        metadata: Additional metadata
    """

    plugin_id: str
    version: str = "0.0.0"
    dependencies: dict[str, DependencyType] = field(default_factory=dict)
    reverse_dependencies: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_dependency(
        self,
        plugin_id: str,
        dep_type: DependencyType = DependencyType.REQUIRED,
    ) -> None:
        """Add a dependency.

        Args:
            plugin_id: Dependency plugin ID
            dep_type: Type of dependency
        """
        self.dependencies[plugin_id] = dep_type

    def add_reverse_dependency(self, plugin_id: str) -> None:
        """Add a reverse dependency (plugin that depends on this).

        Args:
            plugin_id: Dependent plugin ID
        """
        self.reverse_dependencies.add(plugin_id)

    @property
    def required_dependencies(self) -> list[str]:
        """Get list of required dependencies."""
        return [
            dep_id for dep_id, dep_type in self.dependencies.items()
            if dep_type == DependencyType.REQUIRED
        ]

    @property
    def optional_dependencies(self) -> list[str]:
        """Get list of optional dependencies."""
        return [
            dep_id for dep_id, dep_type in self.dependencies.items()
            if dep_type == DependencyType.OPTIONAL
        ]


class DependencyGraph:
    """Graph representation of plugin dependencies.

    Provides operations for:
    - Adding/removing nodes
    - Cycle detection
    - Topological sorting (load order)
    - Dependency queries

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_node("plugin-a", "1.0.0")
        >>> graph.add_node("plugin-b", "1.0.0", dependencies={"plugin-a": DependencyType.REQUIRED})
        >>> order = graph.get_load_order()
        >>> print(order)  # ['plugin-a', 'plugin-b']
    """

    def __init__(self) -> None:
        """Initialize empty dependency graph."""
        self._nodes: dict[str, DependencyNode] = {}

    def add_node(
        self,
        plugin_id: str,
        version: str = "0.0.0",
        dependencies: dict[str, DependencyType] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DependencyNode:
        """Add a node to the graph.

        Args:
            plugin_id: Plugin identifier
            version: Plugin version
            dependencies: Map of dependency IDs to types
            metadata: Additional metadata

        Returns:
            Created DependencyNode
        """
        node = DependencyNode(
            plugin_id=plugin_id,
            version=version,
            dependencies=dependencies or {},
            metadata=metadata or {},
        )
        self._nodes[plugin_id] = node

        # Update reverse dependencies
        for dep_id in node.dependencies:
            if dep_id in self._nodes:
                self._nodes[dep_id].add_reverse_dependency(plugin_id)

        return node

    def remove_node(self, plugin_id: str) -> bool:
        """Remove a node from the graph.

        Args:
            plugin_id: Plugin to remove

        Returns:
            True if removed, False if not found
        """
        if plugin_id not in self._nodes:
            return False

        node = self._nodes.pop(plugin_id)

        # Remove from reverse dependencies
        for dep_id in node.dependencies:
            if dep_id in self._nodes:
                self._nodes[dep_id].reverse_dependencies.discard(plugin_id)

        # Remove from other nodes' dependencies
        for other in self._nodes.values():
            other.dependencies.pop(plugin_id, None)
            other.reverse_dependencies.discard(plugin_id)

        return True

    def get_node(self, plugin_id: str) -> DependencyNode | None:
        """Get a node by ID.

        Args:
            plugin_id: Plugin ID

        Returns:
            DependencyNode or None
        """
        return self._nodes.get(plugin_id)

    def has_node(self, plugin_id: str) -> bool:
        """Check if node exists.

        Args:
            plugin_id: Plugin ID

        Returns:
            True if exists
        """
        return plugin_id in self._nodes

    def get_dependencies(
        self,
        plugin_id: str,
        recursive: bool = False,
        include_optional: bool = False,
    ) -> set[str]:
        """Get dependencies of a plugin.

        Args:
            plugin_id: Plugin ID
            recursive: Include transitive dependencies
            include_optional: Include optional dependencies

        Returns:
            Set of dependency IDs
        """
        node = self._nodes.get(plugin_id)
        if not node:
            return set()

        deps = set()
        if include_optional:
            deps.update(node.dependencies.keys())
        else:
            deps.update(node.required_dependencies)

        if recursive:
            to_visit = list(deps)
            while to_visit:
                dep_id = to_visit.pop()
                dep_node = self._nodes.get(dep_id)
                if dep_node:
                    if include_optional:
                        new_deps = set(dep_node.dependencies.keys()) - deps
                    else:
                        new_deps = set(dep_node.required_dependencies) - deps
                    deps.update(new_deps)
                    to_visit.extend(new_deps)

        return deps

    def get_dependents(
        self,
        plugin_id: str,
        recursive: bool = False,
    ) -> set[str]:
        """Get plugins that depend on this plugin.

        Args:
            plugin_id: Plugin ID
            recursive: Include transitive dependents

        Returns:
            Set of dependent plugin IDs
        """
        node = self._nodes.get(plugin_id)
        if not node:
            return set()

        dependents = set(node.reverse_dependencies)

        if recursive:
            to_visit = list(dependents)
            while to_visit:
                dep_id = to_visit.pop()
                dep_node = self._nodes.get(dep_id)
                if dep_node:
                    new_deps = set(dep_node.reverse_dependencies) - dependents
                    dependents.update(new_deps)
                    to_visit.extend(new_deps)

        return dependents

    def detect_cycles(self) -> list[list[str]]:
        """Detect all cycles in the dependency graph.

        Uses DFS-based cycle detection.

        Returns:
            List of cycles, each cycle is a list of plugin IDs
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            node = self._nodes.get(node_id)
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        dfs(dep_id)
                    elif dep_id in rec_stack:
                        # Found cycle
                        cycle_start = path.index(dep_id)
                        cycle = path[cycle_start:] + [dep_id]
                        cycles.append(cycle)

            path.pop()
            rec_stack.remove(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles

    def get_load_order(self) -> list[str]:
        """Get topological order for loading plugins.

        Dependencies are loaded before dependents.

        Returns:
            List of plugin IDs in load order

        Raises:
            ValueError: If graph contains cycles
        """
        cycles = self.detect_cycles()
        if cycles:
            cycle_str = " -> ".join(cycles[0])
            raise ValueError(f"Circular dependency detected: {cycle_str}")

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {
            node_id: len(node.dependencies)
            for node_id, node in self._nodes.items()
        }

        queue = deque([
            node_id for node_id, degree in in_degree.items()
            if degree == 0
        ])

        result: list[str] = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            node = self._nodes.get(node_id)
            if node:
                for dependent in node.reverse_dependencies:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return result

    def get_unload_order(self) -> list[str]:
        """Get order for unloading plugins (reverse of load order).

        Dependents are unloaded before dependencies.

        Returns:
            List of plugin IDs in unload order
        """
        return list(reversed(self.get_load_order()))

    def validate(self) -> list[str]:
        """Validate the dependency graph.

        Checks for:
        - Missing dependencies
        - Circular dependencies

        Returns:
            List of validation errors
        """
        errors: list[str] = []

        # Check for missing dependencies
        for node_id, node in self._nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self._nodes:
                    errors.append(
                        f"Plugin '{node_id}' depends on missing plugin '{dep_id}'"
                    )

        # Check for cycles
        cycles = self.detect_cycles()
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            errors.append(f"Circular dependency: {cycle_str}")

        return errors

    def __len__(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[str]:
        """Iterate over node IDs."""
        return iter(self._nodes)

    def __contains__(self, plugin_id: str) -> bool:
        """Check if plugin is in graph."""
        return plugin_id in self._nodes

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": {
                node_id: {
                    "version": node.version,
                    "dependencies": {
                        dep_id: dep_type.value
                        for dep_id, dep_type in node.dependencies.items()
                    },
                    "metadata": node.metadata,
                }
                for node_id, node in self._nodes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DependencyGraph":
        """Create graph from dictionary."""
        graph = cls()

        for node_id, node_data in data.get("nodes", {}).items():
            dependencies = {
                dep_id: DependencyType(dep_type)
                for dep_id, dep_type in node_data.get("dependencies", {}).items()
            }
            graph.add_node(
                plugin_id=node_id,
                version=node_data.get("version", "0.0.0"),
                dependencies=dependencies,
                metadata=node_data.get("metadata", {}),
            )

        return graph
