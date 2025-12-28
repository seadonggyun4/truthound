"""Dependency resolver for plugins.

This module provides dependency resolution and conflict detection
for plugin dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from truthound.plugins.dependencies.graph import (
    DependencyGraph,
    DependencyNode,
    DependencyType,
)
from truthound.plugins.versioning.constraints import VersionConstraint

if TYPE_CHECKING:
    from truthound.plugins.base import Plugin, PluginInfo

logger = logging.getLogger(__name__)


@dataclass
class DependencyConflict:
    """Represents a dependency conflict.

    Attributes:
        plugin_id: Plugin with the conflict
        dependency_id: Conflicting dependency
        required_by: List of plugins requiring this dependency
        versions_required: Map of plugin ID to required version
        conflict_type: Type of conflict (version mismatch, missing, etc.)
        message: Human-readable conflict description
    """

    plugin_id: str
    dependency_id: str
    required_by: tuple[str, ...]
    versions_required: dict[str, str]
    conflict_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plugin_id": self.plugin_id,
            "dependency_id": self.dependency_id,
            "required_by": list(self.required_by),
            "versions_required": self.versions_required,
            "conflict_type": self.conflict_type,
            "message": self.message,
        }


@dataclass
class ResolutionResult:
    """Result of dependency resolution.

    Attributes:
        success: Whether resolution succeeded
        graph: Resolved dependency graph
        load_order: Order to load plugins
        conflicts: List of conflicts found
        missing: List of missing dependencies
        warnings: Warning messages
    """

    success: bool
    graph: DependencyGraph
    load_order: list[str] = field(default_factory=list)
    conflicts: list[DependencyConflict] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)  # (plugin, dep)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def successful(
        cls,
        graph: DependencyGraph,
        load_order: list[str],
        warnings: list[str] | None = None,
    ) -> "ResolutionResult":
        """Create a successful resolution result."""
        return cls(
            success=True,
            graph=graph,
            load_order=load_order,
            warnings=warnings or [],
        )

    @classmethod
    def failed(
        cls,
        graph: DependencyGraph,
        conflicts: list[DependencyConflict] | None = None,
        missing: list[tuple[str, str]] | None = None,
    ) -> "ResolutionResult":
        """Create a failed resolution result."""
        return cls(
            success=False,
            graph=graph,
            conflicts=conflicts or [],
            missing=missing or [],
        )


class DependencyResolver:
    """Resolves plugin dependencies.

    Builds dependency graphs, detects conflicts, and determines
    the correct load order for plugins.

    Example:
        >>> resolver = DependencyResolver()
        >>> result = resolver.resolve(plugin_infos)
        >>> if result.success:
        ...     for plugin_id in result.load_order:
        ...         load_plugin(plugin_id)
    """

    def __init__(
        self,
        strict: bool = True,
        allow_missing_optional: bool = True,
    ) -> None:
        """Initialize resolver.

        Args:
            strict: Fail on any conflict
            allow_missing_optional: Allow missing optional dependencies
        """
        self.strict = strict
        self.allow_missing_optional = allow_missing_optional

    def build_graph(
        self,
        plugins: list["PluginInfo"],
    ) -> DependencyGraph:
        """Build dependency graph from plugin info list.

        Args:
            plugins: List of PluginInfo objects

        Returns:
            DependencyGraph
        """
        graph = DependencyGraph()

        for info in plugins:
            dependencies: dict[str, DependencyType] = {}

            # Add required dependencies
            for dep in info.dependencies:
                dependencies[dep] = DependencyType.REQUIRED

            graph.add_node(
                plugin_id=info.name,
                version=info.version,
                dependencies=dependencies,
                metadata={
                    "type": info.plugin_type.value,
                    "description": info.description,
                },
            )

        # Update reverse dependencies
        for node_id in graph:
            node = graph.get_node(node_id)
            if node:
                for dep_id in node.dependencies:
                    dep_node = graph.get_node(dep_id)
                    if dep_node:
                        dep_node.add_reverse_dependency(node_id)

        return graph

    def resolve(
        self,
        plugins: list["PluginInfo"],
    ) -> ResolutionResult:
        """Resolve dependencies for a set of plugins.

        Args:
            plugins: List of PluginInfo objects

        Returns:
            ResolutionResult with graph and load order
        """
        graph = self.build_graph(plugins)

        # Check for missing dependencies
        missing: list[tuple[str, str]] = []
        warnings: list[str] = []

        for node_id in graph:
            node = graph.get_node(node_id)
            if not node:
                continue

            for dep_id, dep_type in node.dependencies.items():
                if not graph.has_node(dep_id):
                    if dep_type == DependencyType.REQUIRED:
                        missing.append((node_id, dep_id))
                    elif dep_type == DependencyType.OPTIONAL:
                        if not self.allow_missing_optional:
                            missing.append((node_id, dep_id))
                        else:
                            warnings.append(
                                f"Optional dependency '{dep_id}' for '{node_id}' not found"
                            )

        if missing and self.strict:
            return ResolutionResult.failed(graph, missing=missing)

        # Check for conflicts
        conflicts = self.find_conflicts(graph)
        if conflicts and self.strict:
            return ResolutionResult.failed(graph, conflicts=conflicts)

        # Check for cycles
        cycles = graph.detect_cycles()
        if cycles:
            for cycle in cycles:
                conflict = DependencyConflict(
                    plugin_id=cycle[0],
                    dependency_id=cycle[-1] if len(cycle) > 1 else cycle[0],
                    required_by=tuple(cycle),
                    versions_required={},
                    conflict_type="circular",
                    message=f"Circular dependency: {' -> '.join(cycle)}",
                )
                conflicts.append(conflict)

            if self.strict:
                return ResolutionResult.failed(graph, conflicts=conflicts)

        # Get load order
        try:
            load_order = graph.get_load_order()
        except ValueError as e:
            return ResolutionResult.failed(
                graph,
                conflicts=[
                    DependencyConflict(
                        plugin_id="",
                        dependency_id="",
                        required_by=(),
                        versions_required={},
                        conflict_type="cycle",
                        message=str(e),
                    )
                ],
            )

        return ResolutionResult.successful(
            graph=graph,
            load_order=load_order,
            warnings=warnings,
        )

    def find_conflicts(
        self,
        graph: DependencyGraph,
    ) -> list[DependencyConflict]:
        """Find all conflicts in a dependency graph.

        Args:
            graph: Dependency graph to check

        Returns:
            List of conflicts found
        """
        conflicts: list[DependencyConflict] = []

        # Group nodes by dependency requirements
        dependency_requirements: dict[str, dict[str, str]] = {}

        for node_id in graph:
            node = graph.get_node(node_id)
            if not node:
                continue

            for dep_id in node.dependencies:
                if dep_id not in dependency_requirements:
                    dependency_requirements[dep_id] = {}
                # Store which version is required by which plugin
                # In a full implementation, this would track version constraints
                dependency_requirements[dep_id][node_id] = "any"

        # Check for version conflicts (simplified - would need version constraints)
        for dep_id, requirers in dependency_requirements.items():
            dep_node = graph.get_node(dep_id)
            if not dep_node:
                continue

            # Check if all requirers can use the same version
            # This is a simplified check - full implementation would
            # compare version constraints
            versions = set(requirers.values())
            if len(versions) > 1 and "any" not in versions:
                conflicts.append(
                    DependencyConflict(
                        plugin_id=dep_id,
                        dependency_id=dep_id,
                        required_by=tuple(requirers.keys()),
                        versions_required=requirers,
                        conflict_type="version",
                        message=(
                            f"Conflicting version requirements for '{dep_id}': "
                            f"{requirers}"
                        ),
                    )
                )

        return conflicts

    def can_load(
        self,
        plugin_id: str,
        graph: DependencyGraph,
        loaded: set[str],
    ) -> tuple[bool, list[str]]:
        """Check if a plugin can be loaded given already loaded plugins.

        Args:
            plugin_id: Plugin to check
            graph: Dependency graph
            loaded: Set of already loaded plugin IDs

        Returns:
            Tuple of (can_load, missing_dependencies)
        """
        node = graph.get_node(plugin_id)
        if not node:
            return False, [plugin_id]

        missing = []
        for dep_id, dep_type in node.dependencies.items():
            if dep_id not in loaded:
                if dep_type == DependencyType.REQUIRED:
                    missing.append(dep_id)
                elif dep_type == DependencyType.OPTIONAL:
                    if not self.allow_missing_optional:
                        missing.append(dep_id)

        return len(missing) == 0, missing

    def get_install_order(
        self,
        plugin_ids: list[str],
        graph: DependencyGraph,
    ) -> list[str]:
        """Get the order to install a subset of plugins.

        Args:
            plugin_ids: Plugins to install
            graph: Full dependency graph

        Returns:
            Installation order including dependencies
        """
        to_install: set[str] = set()
        to_visit = list(plugin_ids)

        # Gather all required plugins (including transitive dependencies)
        while to_visit:
            plugin_id = to_visit.pop()
            if plugin_id in to_install:
                continue

            to_install.add(plugin_id)
            deps = graph.get_dependencies(plugin_id, recursive=False)
            for dep_id in deps:
                if dep_id not in to_install:
                    to_visit.append(dep_id)

        # Get load order for just these plugins
        full_order = graph.get_load_order()
        return [p for p in full_order if p in to_install]

    def get_uninstall_order(
        self,
        plugin_ids: list[str],
        graph: DependencyGraph,
        force: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Get the order to uninstall plugins.

        Args:
            plugin_ids: Plugins to uninstall
            graph: Dependency graph
            force: Force uninstall even if other plugins depend on these

        Returns:
            Tuple of (uninstall_order, affected_plugins)
        """
        to_uninstall: set[str] = set(plugin_ids)
        affected: set[str] = set()

        # Find plugins that depend on those being uninstalled
        for plugin_id in plugin_ids:
            dependents = graph.get_dependents(plugin_id, recursive=True)
            for dep in dependents:
                if dep not in to_uninstall:
                    affected.add(dep)

        if force:
            to_uninstall.update(affected)
            affected = set()

        # Get unload order (reverse of load order)
        full_order = graph.get_unload_order()
        uninstall_order = [p for p in full_order if p in to_uninstall]

        return uninstall_order, list(affected)
