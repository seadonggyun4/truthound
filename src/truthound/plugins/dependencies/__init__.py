"""Plugin dependency management module.

This module provides dependency graph management and resolution
for plugins.

Components:
    - DependencyGraph: Represents plugin dependencies
    - DependencyResolver: Resolves and validates dependencies
    - DependencyConflict: Represents dependency conflicts

Example:
    >>> from truthound.plugins.dependencies import (
    ...     DependencyGraph,
    ...     DependencyResolver,
    ... )
    >>>
    >>> resolver = DependencyResolver()
    >>> graph = resolver.build_graph(plugins)
    >>> load_order = graph.get_load_order()
"""

from __future__ import annotations

from truthound.plugins.dependencies.graph import (
    DependencyGraph,
    DependencyNode,
    DependencyType,
)
from truthound.plugins.dependencies.resolver import (
    DependencyResolver,
    DependencyConflict,
    ResolutionResult,
)

__all__ = [
    "DependencyGraph",
    "DependencyNode",
    "DependencyType",
    "DependencyResolver",
    "DependencyConflict",
    "ResolutionResult",
]
