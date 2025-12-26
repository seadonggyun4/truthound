"""Graph traversal optimization algorithms.

This module provides memory-efficient and stack-safe graph traversal
implementations for hierarchy and relationship validators.

Key Optimizations:
    - Iterative (non-recursive) DFS to avoid Python stack overflow
    - Tarjan's algorithm for SCC detection O(V+E)
    - Memory-efficient adjacency representation
    - Early termination for specific queries

Usage:
    class OptimizedHierarchyValidator(HierarchyValidator, GraphTraversalMixin):
        def _find_cycles(self, df):
            adj = self.build_adjacency_list(df, 'id', 'parent_id')
            return self.find_all_cycles(adj)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, TypeVar, Generic, Hashable
from enum import Enum, auto

T = TypeVar("T", bound=Hashable)


class NodeState(Enum):
    """Node state during DFS traversal."""

    WHITE = auto()  # Not visited
    GRAY = auto()  # In progress (on current path)
    BLACK = auto()  # Finished


@dataclass
class CycleInfo(Generic[T]):
    """Information about a detected cycle.

    Attributes:
        nodes: List of nodes in the cycle (start repeats at end)
        length: Number of edges in the cycle
        is_self_loop: Whether this is a self-referencing node
    """

    nodes: list[T]
    length: int = field(init=False)
    is_self_loop: bool = field(init=False)

    def __post_init__(self) -> None:
        self.length = len(self.nodes) - 1  # Exclude repeated start node
        self.is_self_loop = self.length == 1

    def __str__(self) -> str:
        return " -> ".join(str(n) for n in self.nodes)


class IterativeDFS(Generic[T]):
    """Iterative Depth-First Search implementation.

    Uses explicit stack to avoid Python's recursion limit (default ~1000).
    Can handle graphs with millions of nodes.

    Example:
        dfs = IterativeDFS(adjacency_list)
        for node in dfs.traverse(start_node):
            process(node)
    """

    def __init__(self, adjacency: dict[T, list[T]]):
        """Initialize with adjacency list.

        Args:
            adjacency: Map from node to list of neighbors
        """
        self.adjacency = adjacency
        self._all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            self._all_nodes.update(neighbors)

    def traverse(
        self,
        start: T | None = None,
        order: str = "preorder",
    ) -> Iterator[T]:
        """Traverse graph in DFS order.

        Args:
            start: Starting node (None = traverse all components)
            order: 'preorder' or 'postorder'

        Yields:
            Nodes in specified order
        """
        visited: set[T] = set()

        def dfs_from(node: T) -> Iterator[T]:
            stack: list[tuple[T, bool]] = [(node, False)]  # (node, is_returning)

            while stack:
                current, returning = stack.pop()

                if returning:
                    if order == "postorder":
                        yield current
                    continue

                if current in visited:
                    continue

                visited.add(current)

                if order == "preorder":
                    yield current

                # Mark that we need to emit in postorder after processing children
                stack.append((current, True))

                # Add children in reverse order for correct traversal order
                for neighbor in reversed(self.adjacency.get(current, [])):
                    if neighbor not in visited:
                        stack.append((neighbor, False))

        if start is not None:
            yield from dfs_from(start)
        else:
            for node in self._all_nodes:
                if node not in visited:
                    yield from dfs_from(node)

    def find_path(self, start: T, end: T) -> list[T] | None:
        """Find path from start to end using DFS.

        Args:
            start: Starting node
            end: Target node

        Returns:
            Path as list of nodes, or None if no path exists
        """
        if start == end:
            return [start]

        visited: set[T] = set()
        # Stack contains (node, path_to_node)
        stack: list[tuple[T, list[T]]] = [(start, [start])]

        while stack:
            current, path = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.adjacency.get(current, []):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None

    def compute_depths(
        self,
        roots: list[T] | None = None,
    ) -> dict[T, int]:
        """Compute depth of each node from roots.

        Args:
            roots: Root nodes (None = auto-detect nodes with no parents)

        Returns:
            Map from node to depth
        """
        # Find roots if not specified
        if roots is None:
            has_parent: set[T] = set()
            for neighbors in self.adjacency.values():
                has_parent.update(neighbors)
            roots = [n for n in self._all_nodes if n not in has_parent]

        depths: dict[T, int] = {}
        stack: list[tuple[T, int]] = [(r, 0) for r in roots]

        while stack:
            node, depth = stack.pop()

            if node in depths:
                continue

            depths[node] = depth

            for neighbor in self.adjacency.get(node, []):
                if neighbor not in depths:
                    stack.append((neighbor, depth + 1))

        return depths


class TarjanSCC(Generic[T]):
    """Tarjan's Strongly Connected Components algorithm.

    Finds all SCCs in O(V + E) time using a single DFS pass.
    SCCs with size > 1 indicate cycles.

    Iterative implementation to handle large graphs.

    Example:
        tarjan = TarjanSCC(adjacency_list)
        sccs = tarjan.find_sccs()
        cycles = [scc for scc in sccs if len(scc) > 1]
    """

    def __init__(self, adjacency: dict[T, list[T]]):
        """Initialize with adjacency list."""
        self.adjacency = adjacency
        self._all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            self._all_nodes.update(neighbors)

    def find_sccs(self) -> list[list[T]]:
        """Find all strongly connected components.

        Returns:
            List of SCCs (each SCC is a list of nodes)
        """
        index_counter = [0]
        stack: list[T] = []
        lowlinks: dict[T, int] = {}
        index: dict[T, int] = {}
        on_stack: set[T] = set()
        sccs: list[list[T]] = []

        def strongconnect(node: T) -> None:
            # Use iterative approach with explicit call stack
            call_stack: list[tuple[T, int, Iterator[T]]] = []
            call_stack.append((node, 0, iter(self.adjacency.get(node, []))))

            while call_stack:
                v, phase, neighbors = call_stack[-1]

                if phase == 0:
                    # First visit to this node
                    index[v] = index_counter[0]
                    lowlinks[v] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(v)
                    on_stack.add(v)
                    call_stack[-1] = (v, 1, neighbors)
                    continue

                # Process neighbors
                try:
                    w = next(neighbors)
                    if w not in index:
                        # Recurse
                        call_stack.append((w, 0, iter(self.adjacency.get(w, []))))
                    elif w in on_stack:
                        lowlinks[v] = min(lowlinks[v], index[w])
                except StopIteration:
                    # Done with neighbors, check if SCC root
                    call_stack.pop()

                    if lowlinks[v] == index[v]:
                        # Root of SCC
                        scc: list[T] = []
                        while True:
                            w = stack.pop()
                            on_stack.remove(w)
                            scc.append(w)
                            if w == v:
                                break
                        sccs.append(scc)

                    # Update parent's lowlink
                    if call_stack:
                        parent, _, _ = call_stack[-1]
                        lowlinks[parent] = min(lowlinks[parent], lowlinks[v])

        for node in self._all_nodes:
            if node not in index:
                strongconnect(node)

        return sccs

    def find_cycles(self) -> list[CycleInfo[T]]:
        """Find all cycles in the graph.

        Returns:
            List of CycleInfo objects
        """
        sccs = self.find_sccs()
        cycles: list[CycleInfo[T]] = []

        for scc in sccs:
            if len(scc) > 1:
                # Multi-node SCC is a cycle
                # Reconstruct one cycle path
                scc_set = set(scc)
                start = scc[0]
                path = self._find_cycle_path(start, scc_set)
                if path:
                    cycles.append(CycleInfo(nodes=path))
            elif len(scc) == 1:
                # Check for self-loop
                node = scc[0]
                if node in self.adjacency.get(node, []):
                    cycles.append(CycleInfo(nodes=[node, node]))

        return cycles

    def _find_cycle_path(self, start: T, scc_nodes: set[T]) -> list[T] | None:
        """Find a cycle path within an SCC."""
        visited: set[T] = set()
        stack: list[tuple[T, list[T]]] = [(start, [start])]

        while stack:
            current, path = stack.pop()

            if current in visited and current == start and len(path) > 1:
                return path

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.adjacency.get(current, []):
                if neighbor in scc_nodes:
                    if neighbor == start and len(path) > 1:
                        return path + [neighbor]
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

        return None


class TopologicalSort(Generic[T]):
    """Topological sorting with cycle detection.

    Uses Kahn's algorithm for BFS-based topological sort.
    More intuitive for dependency ordering.

    Example:
        sorter = TopologicalSort(dependencies)
        try:
            order = sorter.sort()
        except CycleDetectedError:
            print("Cyclic dependencies")
    """

    def __init__(self, adjacency: dict[T, list[T]]):
        """Initialize with adjacency list (edges point from dependency to dependent)."""
        self.adjacency = adjacency

    def sort(self) -> list[T]:
        """Perform topological sort.

        Returns:
            List of nodes in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        # Compute in-degrees
        in_degree: dict[T, int] = defaultdict(int)
        all_nodes: set[T] = set(self.adjacency.keys())

        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                in_degree[neighbor] += 1
                all_nodes.add(neighbor)

        # Initialize queue with zero in-degree nodes
        queue: list[T] = [n for n in all_nodes if in_degree[n] == 0]
        result: list[T] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in self.adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(all_nodes):
            raise ValueError("Graph contains cycles - topological sort not possible")

        return result

    def has_cycles(self) -> bool:
        """Check if graph has cycles."""
        try:
            self.sort()
            return False
        except ValueError:
            return True


class GraphTraversalMixin:
    """Mixin providing optimized graph traversal operations.

    Use this mixin in validators that work with hierarchical or
    graph-structured data.

    Features:
        - Stack-safe iterative algorithms (no recursion limit)
        - Efficient SCC-based cycle detection
        - Memory-efficient adjacency representation
        - Caching for repeated queries

    Example:
        class MyHierarchyValidator(ReferentialValidator, GraphTraversalMixin):
            def _find_cycles(self, df):
                adj = self.build_adjacency_list(df, 'id', 'parent_id')
                return self.find_all_cycles(adj)
    """

    # Cache for adjacency lists
    _adjacency_cache: dict[str, dict[Any, list[Any]]] = {}

    def build_adjacency_list(
        self,
        data: Any,  # DataFrame or dict
        id_column: str,
        parent_column: str,
        cache_key: str | None = None,
    ) -> dict[Any, list[Any]]:
        """Build adjacency list from data.

        Args:
            data: DataFrame or existing dict
            id_column: Column containing node IDs
            parent_column: Column containing parent IDs
            cache_key: Optional key for caching

        Returns:
            Adjacency list (parent -> children mapping)
        """
        if cache_key and cache_key in self._adjacency_cache:
            return self._adjacency_cache[cache_key]

        if isinstance(data, dict):
            adjacency = data
        else:
            # Assume Polars DataFrame
            adjacency: dict[Any, list[Any]] = defaultdict(list)

            for row in data.select([id_column, parent_column]).iter_rows():
                node_id, parent_id = row
                if parent_id is not None:
                    adjacency[parent_id].append(node_id)
                # Ensure all nodes are in adjacency
                if node_id not in adjacency:
                    adjacency[node_id] = []

            adjacency = dict(adjacency)

        if cache_key:
            self._adjacency_cache[cache_key] = adjacency

        return adjacency

    def build_child_to_parent(
        self,
        data: Any,
        id_column: str,
        parent_column: str,
    ) -> dict[Any, Any]:
        """Build child-to-parent mapping.

        Args:
            data: DataFrame
            id_column: Column containing node IDs
            parent_column: Column containing parent IDs

        Returns:
            Dict mapping child ID to parent ID
        """
        mapping: dict[Any, Any] = {}

        for row in data.select([id_column, parent_column]).iter_rows():
            node_id, parent_id = row
            if parent_id is not None:
                mapping[node_id] = parent_id

        return mapping

    def find_all_cycles(
        self,
        adjacency: dict[Any, list[Any]],
    ) -> list[CycleInfo[Any]]:
        """Find all cycles in a graph using Tarjan's SCC.

        Args:
            adjacency: Adjacency list

        Returns:
            List of CycleInfo objects
        """
        tarjan = TarjanSCC(adjacency)
        return tarjan.find_cycles()

    def find_hierarchy_cycles(
        self,
        child_to_parent: dict[Any, Any],
        max_depth: int = 1000,
    ) -> list[CycleInfo[Any]]:
        """Find cycles in parent-child hierarchy.

        Optimized for tree-like structures with potential cycles.

        Args:
            child_to_parent: Mapping from child to parent
            max_depth: Maximum traversal depth

        Returns:
            List of detected cycles
        """
        cycles: list[CycleInfo[Any]] = []
        checked: set[Any] = set()

        for start_node in child_to_parent:
            if start_node in checked:
                continue

            path: list[Any] = []
            visited: set[Any] = set()
            current = start_node

            while current is not None and len(path) < max_depth:
                if current in visited:
                    # Found cycle - find where it starts
                    cycle_start_idx = path.index(current)
                    cycle_path = path[cycle_start_idx:] + [current]
                    cycles.append(CycleInfo(nodes=cycle_path))
                    break

                visited.add(current)
                path.append(current)
                current = child_to_parent.get(current)

            checked.update(visited)

        return cycles

    def compute_node_depths(
        self,
        adjacency: dict[Any, list[Any]],
        roots: list[Any] | None = None,
    ) -> dict[Any, int]:
        """Compute depth of each node from roots.

        Args:
            adjacency: Parent -> children adjacency list
            roots: Root nodes (auto-detected if None)

        Returns:
            Dict mapping node to depth
        """
        dfs = IterativeDFS(adjacency)
        return dfs.compute_depths(roots)

    def topological_sort(
        self,
        adjacency: dict[Any, list[Any]],
    ) -> list[Any] | None:
        """Perform topological sort if possible.

        Args:
            adjacency: Adjacency list

        Returns:
            Sorted nodes or None if cycles exist
        """
        sorter = TopologicalSort(adjacency)
        try:
            return sorter.sort()
        except ValueError:
            return None

    def clear_adjacency_cache(self) -> None:
        """Clear cached adjacency lists."""
        self._adjacency_cache.clear()
