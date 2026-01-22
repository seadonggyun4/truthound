"""Protocols for lineage visualization.

Defines interfaces for graph rendering and visualization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.lineage.base import LineageGraph


class RenderFormat(str, Enum):
    """Supported render formats."""

    JSON = "json"  # Generic JSON
    D3_JSON = "d3_json"  # D3.js force-directed graph
    CYTOSCAPE_JSON = "cytoscape_json"  # Cytoscape.js format
    DOT = "dot"  # Graphviz DOT format
    MERMAID = "mermaid"  # Mermaid diagram syntax
    SVG = "svg"  # SVG image
    PNG = "png"  # PNG image


@dataclass
class RenderConfig:
    """Configuration for graph rendering.

    Attributes:
        format: Output format
        include_metadata: Include node/edge metadata
        max_depth: Maximum depth to render (-1 for unlimited)
        layout: Graph layout algorithm
        theme: Color theme ('light' or 'dark')
        node_colors: Node type to color mapping
        edge_colors: Edge type to color mapping
        node_sizes: Node type to size mapping
        highlight_nodes: Node IDs to highlight
        filter_node_types: Only include these node types
        filter_edge_types: Only include these edge types
        orientation: Graph orientation (TB, BT, LR, RL)
        width: Output width (for image formats)
        height: Output height (for image formats)
    """

    format: RenderFormat = RenderFormat.D3_JSON
    include_metadata: bool = True
    max_depth: int = -1
    layout: str = "force"  # force, hierarchical, circular, grid
    theme: str = "light"  # light, dark
    node_colors: dict[str, str] = field(default_factory=dict)
    edge_colors: dict[str, str] = field(default_factory=dict)
    node_sizes: dict[str, int] = field(default_factory=dict)
    highlight_nodes: list[str] = field(default_factory=list)
    filter_node_types: list[str] | None = None
    filter_edge_types: list[str] | None = None
    orientation: str = "TB"  # TB (top-bottom), BT, LR (left-right), RL
    width: int = 1200
    height: int = 800

    # Light theme colors
    DEFAULT_NODE_COLORS: dict[str, str] = field(default_factory=lambda: {
        "SOURCE": "#4CAF50",
        "TABLE": "#2196F3",
        "FILE": "#9C27B0",
        "STREAM": "#FF9800",
        "TRANSFORMATION": "#607D8B",
        "VALIDATION": "#E91E63",
        "MODEL": "#00BCD4",
        "REPORT": "#795548",
        "EXTERNAL": "#9E9E9E",
        "VIRTUAL": "#CDDC39",
    })

    DEFAULT_EDGE_COLORS: dict[str, str] = field(default_factory=lambda: {
        "DERIVED_FROM": "#2196F3",
        "VALIDATED_BY": "#E91E63",
        "USED_BY": "#4CAF50",
        "TRANSFORMED_TO": "#FF9800",
        "JOINED_WITH": "#9C27B0",
        "AGGREGATED_TO": "#00BCD4",
        "FILTERED_TO": "#607D8B",
        "DEPENDS_ON": "#795548",
    })

    # Dark theme colors (brighter for dark backgrounds)
    DARK_NODE_COLORS: dict[str, str] = field(default_factory=lambda: {
        "SOURCE": "#66BB6A",
        "TABLE": "#42A5F5",
        "FILE": "#AB47BC",
        "STREAM": "#FFA726",
        "TRANSFORMATION": "#78909C",
        "VALIDATION": "#EC407A",
        "MODEL": "#26C6DA",
        "REPORT": "#8D6E63",
        "EXTERNAL": "#BDBDBD",
        "VIRTUAL": "#D4E157",
    })

    DARK_EDGE_COLORS: dict[str, str] = field(default_factory=lambda: {
        "DERIVED_FROM": "#64B5F6",
        "VALIDATED_BY": "#F06292",
        "USED_BY": "#81C784",
        "TRANSFORMED_TO": "#FFB74D",
        "JOINED_WITH": "#BA68C8",
        "AGGREGATED_TO": "#4DD0E1",
        "FILTERED_TO": "#90A4AE",
        "DEPENDS_ON": "#A1887F",
    })

    def get_node_color(self, node_type: str) -> str:
        """Get color for node type (case-insensitive)."""
        upper_type = node_type.upper()
        # Check user-defined colors first
        if node_type in self.node_colors:
            return self.node_colors[node_type]
        if upper_type in self.node_colors:
            return self.node_colors[upper_type]
        # Use theme-appropriate defaults
        if self.theme == "dark":
            return self.DARK_NODE_COLORS.get(upper_type, "#BDBDBD")
        return self.DEFAULT_NODE_COLORS.get(upper_type, "#9E9E9E")

    def get_edge_color(self, edge_type: str) -> str:
        """Get color for edge type (case-insensitive)."""
        upper_type = edge_type.upper()
        # Check user-defined colors first
        if edge_type in self.edge_colors:
            return self.edge_colors[edge_type]
        if upper_type in self.edge_colors:
            return self.edge_colors[upper_type]
        # Use theme-appropriate defaults
        if self.theme == "dark":
            return self.DARK_EDGE_COLORS.get(upper_type, "#BDBDBD")
        return self.DEFAULT_EDGE_COLORS.get(upper_type, "#9E9E9E")


@runtime_checkable
class IGraphRenderer(Protocol):
    """Protocol for graph renderers.

    Renderers convert LineageGraph to various output formats.
    """

    @property
    def format(self) -> RenderFormat:
        """Get renderer output format."""
        ...

    def render(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str | bytes:
        """Render graph to output format.

        Args:
            graph: Lineage graph to render
            config: Render configuration

        Returns:
            Rendered output (string or bytes)
        """
        ...

    def render_subgraph(
        self,
        graph: "LineageGraph",
        root_node_id: str,
        direction: str = "downstream",
        max_depth: int = -1,
        config: RenderConfig | None = None,
    ) -> str | bytes:
        """Render a subgraph from a root node.

        Args:
            graph: Full lineage graph
            root_node_id: Root node for subgraph
            direction: "upstream" or "downstream"
            max_depth: Maximum depth (-1 for unlimited)
            config: Render configuration

        Returns:
            Rendered output
        """
        ...
