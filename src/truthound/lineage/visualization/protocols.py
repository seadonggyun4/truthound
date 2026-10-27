"""Protocols for lineage visualization.

Defines interfaces for graph rendering and visualization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
        max_depth: Maximum rooted subgraph depth (-1 for unlimited)
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
    DEFAULT_NODE_COLORS: dict[str, str] = field(
        default_factory=lambda: {
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
        }
    )

    DEFAULT_EDGE_COLORS: dict[str, str] = field(
        default_factory=lambda: {
            "DERIVED_FROM": "#2196F3",
            "VALIDATED_BY": "#E91E63",
            "USED_BY": "#4CAF50",
            "TRANSFORMED_TO": "#FF9800",
            "JOINED_WITH": "#9C27B0",
            "AGGREGATED_TO": "#00BCD4",
            "FILTERED_TO": "#607D8B",
            "DEPENDS_ON": "#795548",
        }
    )

    # Dark theme colors (brighter for dark backgrounds)
    DARK_NODE_COLORS: dict[str, str] = field(
        default_factory=lambda: {
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
        }
    )

    DARK_EDGE_COLORS: dict[str, str] = field(
        default_factory=lambda: {
            "DERIVED_FROM": "#64B5F6",
            "VALIDATED_BY": "#F06292",
            "USED_BY": "#81C784",
            "TRANSFORMED_TO": "#FFB74D",
            "JOINED_WITH": "#BA68C8",
            "AGGREGATED_TO": "#4DD0E1",
            "FILTERED_TO": "#90A4AE",
            "DEPENDS_ON": "#A1887F",
        }
    )

    def __post_init__(self) -> None:
        try:
            self.format = RenderFormat(self.format)
        except (ValueError, TypeError):
            raise ValueError("Unsupported render format") from None
        if self.theme not in {"light", "dark"} or self.orientation not in {"TB", "BT", "LR", "RL"}:
            raise ValueError("Unsupported theme or orientation")
        if not isinstance(self.layout, str) or not self.layout:
            raise ValueError("Layout must be a nonempty string")
        if type(self.include_metadata) is not bool:
            raise ValueError("include_metadata must be a boolean")
        if any(
            type(getattr(self, key)) is not int or getattr(self, key) < 1
            for key in ("width", "height")
        ):
            raise ValueError("Image dimensions must be positive integers")
        if type(self.max_depth) is not int or self.max_depth < -1:
            raise ValueError("max_depth must be -1 or a nonnegative integer")
        for key in (
            "node_colors",
            "edge_colors",
            "DEFAULT_NODE_COLORS",
            "DEFAULT_EDGE_COLORS",
            "DARK_NODE_COLORS",
            "DARK_EDGE_COLORS",
        ):
            mapping = getattr(self, key)
            if not isinstance(mapping, dict) or any(
                not isinstance(name, str)
                or not isinstance(value, str)
                or not value
                or any(ord(character) < 32 for character in value)
                for name, value in mapping.items()
            ):
                raise ValueError("Color mappings require string names and values")
            setattr(self, key, dict(mapping))
        if not isinstance(self.node_sizes, dict) or any(
            not isinstance(name, str) or type(value) is not int or value < 1
            for name, value in self.node_sizes.items()
        ):
            raise ValueError("Node sizes must be positive integers")
        self.node_sizes = dict(self.node_sizes)
        for key in ("highlight_nodes", "filter_node_types", "filter_edge_types"):
            values = getattr(self, key)
            if values is None and key != "highlight_nodes":
                continue
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) for value in values
            ):
                raise ValueError("Node and edge selectors must be sequences of strings")
            setattr(self, key, list(values))

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


def resolve_render_depth(depth: int | None, config: RenderConfig) -> int:
    """An omitted native depth inherits config; explicit -1 stays unlimited."""
    value = config.max_depth if depth is None else depth
    if type(value) is not int or value < -1:
        raise ValueError("Depth must be -1 or a nonnegative integer")
    return value


def capture_render_config(config: RenderConfig | None, *, theme: str = "light") -> RenderConfig:
    """Validate an owned snapshot without changing the caller's settings."""
    if config is None:
        return RenderConfig(theme=theme)
    snapshot = deepcopy(config)
    snapshot.__post_init__()
    if snapshot.theme == "light" and theme == "dark":
        snapshot.theme = theme
    return snapshot


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
        graph: LineageGraph,
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
        graph: LineageGraph,
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
