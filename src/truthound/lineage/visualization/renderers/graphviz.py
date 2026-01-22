"""Graphviz DOT renderer for lineage graphs.

Produces DOT format output for use with Graphviz tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from truthound.lineage.visualization.protocols import (
    IGraphRenderer,
    RenderFormat,
    RenderConfig,
)

if TYPE_CHECKING:
    from truthound.lineage.base import LineageGraph


class GraphvizRenderer(IGraphRenderer):
    """Renders lineage graphs to Graphviz DOT format.

    DOT format can be used with Graphviz tools to generate
    SVG, PNG, PDF and other image formats.

    Example:
        >>> renderer = GraphvizRenderer()
        >>> dot_output = renderer.render(graph)
        >>> # Save and render: dot -Tpng output.dot -o output.png

        >>> # Dark theme
        >>> renderer = GraphvizRenderer(theme="dark")
        >>> svg_output = renderer.render_svg(graph)
    """

    def __init__(self, theme: str = "light") -> None:
        """Initialize renderer with theme.

        Args:
            theme: Color theme ('light' or 'dark')
        """
        self._theme = theme

    # Node shapes by type
    NODE_SHAPES = {
        "SOURCE": "cylinder",
        "TABLE": "box3d",
        "FILE": "note",
        "STREAM": "parallelogram",
        "TRANSFORMATION": "component",
        "VALIDATION": "diamond",
        "MODEL": "hexagon",
        "REPORT": "tab",
        "EXTERNAL": "oval",
        "VIRTUAL": "none",
    }

    # Edge styles by type
    EDGE_STYLES = {
        "DERIVED_FROM": "solid",
        "VALIDATED_BY": "dashed",
        "USED_BY": "dotted",
        "TRANSFORMED_TO": "solid",
        "JOINED_WITH": "bold",
        "AGGREGATED_TO": "solid",
        "FILTERED_TO": "dashed",
        "DEPENDS_ON": "dotted",
    }

    @property
    def format(self) -> RenderFormat:
        return RenderFormat.DOT

    def render(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render graph to DOT format.

        Args:
            graph: Lineage graph to render
            config: Render configuration

        Returns:
            DOT format string
        """
        config = config or RenderConfig(theme=self._theme)
        if config.theme == "light" and self._theme == "dark":
            config.theme = self._theme
        return self._build_dot(graph, config)

    def render_subgraph(
        self,
        graph: "LineageGraph",
        root_node_id: str,
        direction: str = "downstream",
        max_depth: int = -1,
        config: RenderConfig | None = None,
    ) -> str:
        """Render subgraph from root node.

        Args:
            graph: Full lineage graph
            root_node_id: Root node ID
            direction: "upstream", "downstream", or "both"
            max_depth: Maximum depth
            config: Render configuration

        Returns:
            DOT format string
        """
        config = config or RenderConfig()

        # Collect subgraph nodes
        subgraph_node_ids: set[str] = {root_node_id}

        if direction in ("upstream", "both"):
            for node in graph.get_upstream(root_node_id, max_depth):
                subgraph_node_ids.add(node.id)

        if direction in ("downstream", "both"):
            for node in graph.get_downstream(root_node_id, max_depth):
                subgraph_node_ids.add(node.id)

        # Build filtered node list
        subgraph_nodes = [graph.get_node(nid) for nid in subgraph_node_ids]

        # Filter edges to only those within subgraph
        subgraph_edges = [
            edge for edge in graph.edges
            if edge.source in subgraph_node_ids and edge.target in subgraph_node_ids
        ]

        config.highlight_nodes = config.highlight_nodes + [root_node_id]

        return self._build_dot_from_lists(subgraph_nodes, subgraph_edges, config)

    def _build_dot_from_lists(
        self,
        node_list: list,
        edge_list: list,
        config: RenderConfig,
    ) -> str:
        """Build DOT format string from node and edge lists.

        Args:
            node_list: List of LineageNode objects
            edge_list: List of LineageEdge objects
            config: Render configuration

        Returns:
            DOT format string
        """
        lines = []

        # Theme-specific settings
        is_dark = config.theme == "dark"
        bg_color = "#1a1a2e" if is_dark else "white"
        font_color = "#e0e0e0" if is_dark else "#333333"
        edge_font_color = "#b0b0b0" if is_dark else "#666666"

        # Graph header
        lines.append("digraph LineageGraph {")
        lines.append("    // Graph settings")
        lines.append(f"    rankdir={config.orientation};")
        lines.append(f'    bgcolor="{bg_color}";')
        lines.append(f'    node [fontname="Arial", fontsize=10, fontcolor="{font_color}"];')
        lines.append(f'    edge [fontname="Arial", fontsize=8, fontcolor="{edge_font_color}"];')
        lines.append("")

        # Node definitions
        lines.append("    // Nodes")
        node_ids = set()

        for node in node_list:
            if config.filter_node_types and node.node_type.value not in config.filter_node_types:
                continue

            node_ids.add(node.id)

            shape = self.NODE_SHAPES.get(node.node_type.value, "ellipse")
            color = config.get_node_color(node.node_type.value)
            label = self._escape_label(node.name)

            attrs = [
                f'label="{label}"',
                f'shape={shape}',
                f'fillcolor="{color}"',
                'style=filled',
            ]

            if node.id in config.highlight_nodes:
                attrs.append('penwidth=3')
                attrs.append('color="red"')

            if config.include_metadata and hasattr(node, "metadata"):
                tooltip = f"{node.name}\\nType: {node.node_type.value}"
                if node.metadata.description:
                    tooltip += f"\\n{node.metadata.description}"
                attrs.append(f'tooltip="{self._escape_label(tooltip)}"')

            safe_id = self._safe_id(node.id)
            lines.append(f'    {safe_id} [{", ".join(attrs)}];')

        lines.append("")

        # Edge definitions
        lines.append("    // Edges")

        for edge in edge_list:
            if config.filter_edge_types and edge.edge_type.value not in config.filter_edge_types:
                continue

            source_id = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            target_id = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

            if source_id not in node_ids or target_id not in node_ids:
                continue

            style = self.EDGE_STYLES.get(edge.edge_type.value, "solid")
            color = config.get_edge_color(edge.edge_type.value)

            attrs = [
                f'style={style}',
                f'color="{color}"',
                f'label="{edge.edge_type.value}"',
            ]

            safe_source = self._safe_id(source_id)
            safe_target = self._safe_id(target_id)
            lines.append(f'    {safe_source} -> {safe_target} [{", ".join(attrs)}];')

        lines.append("}")

        return "\n".join(lines)

    def _build_dot(
        self,
        graph: "LineageGraph",
        config: RenderConfig,
    ) -> str:
        """Build DOT format string."""
        lines = []

        # Theme-specific settings
        is_dark = config.theme == "dark"
        bg_color = "#1a1a2e" if is_dark else "white"
        font_color = "#e0e0e0" if is_dark else "#333333"
        edge_font_color = "#b0b0b0" if is_dark else "#666666"

        # Graph header
        lines.append("digraph LineageGraph {")
        lines.append("    // Graph settings")
        lines.append(f"    rankdir={config.orientation};")
        lines.append(f"    bgcolor=\"{bg_color}\";")
        lines.append(f"    node [fontname=\"Arial\", fontsize=10, fontcolor=\"{font_color}\"];")
        lines.append(f"    edge [fontname=\"Arial\", fontsize=8, fontcolor=\"{edge_font_color}\"];")
        lines.append("")

        # Node definitions
        lines.append("    // Nodes")
        node_ids = set()

        # Handle both list and dict node formats
        if isinstance(graph.nodes, dict):
            node_items = graph.nodes.items()
        else:
            node_items = [(node.id, node) for node in graph.nodes]

        for node_id, node in node_items:
            if config.filter_node_types and node.node_type.value not in config.filter_node_types:
                continue

            node_ids.add(node_id)

            shape = self.NODE_SHAPES.get(node.node_type.value, "ellipse")
            color = config.get_node_color(node.node_type.value)
            label = self._escape_label(node.name)

            # Build node attributes
            attrs = [
                f'label="{label}"',
                f'shape={shape}',
                f'fillcolor="{color}"',
                'style=filled',
            ]

            # Highlight
            if node_id in config.highlight_nodes:
                attrs.append('penwidth=3')
                attrs.append('color="red"')

            # Tooltip with metadata
            if config.include_metadata and hasattr(node, "metadata"):
                tooltip = f"{node.name}\\nType: {node.node_type.value}"
                if node.metadata.description:
                    tooltip += f"\\n{node.metadata.description}"
                attrs.append(f'tooltip="{self._escape_label(tooltip)}"')

            safe_id = self._safe_id(node_id)
            lines.append(f'    {safe_id} [{", ".join(attrs)}];')

        lines.append("")

        # Edge definitions
        lines.append("    // Edges")

        for edge in graph.edges:
            if config.filter_edge_types and edge.edge_type.value not in config.filter_edge_types:
                continue

            # Handle both source/source_id attribute names
            source_id = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            target_id = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

            if source_id not in node_ids or target_id not in node_ids:
                continue

            style = self.EDGE_STYLES.get(edge.edge_type.value, "solid")
            color = config.get_edge_color(edge.edge_type.value)

            attrs = [
                f'style={style}',
                f'color="{color}"',
                f'label="{edge.edge_type.value}"',
            ]

            safe_source = self._safe_id(source_id)
            safe_target = self._safe_id(target_id)
            lines.append(f'    {safe_source} -> {safe_target} [{", ".join(attrs)}];')

        lines.append("}")

        return "\n".join(lines)

    def _escape_label(self, text: str) -> str:
        """Escape text for DOT labels."""
        return text.replace('"', '\\"').replace("\n", "\\n")

    def _safe_id(self, node_id: str) -> str:
        """Convert node ID to safe DOT identifier."""
        # Replace non-alphanumeric with underscore
        safe = "".join(c if c.isalnum() else "_" for c in node_id)
        # Ensure it starts with letter
        if safe and not safe[0].isalpha():
            safe = "n_" + safe
        return safe or "node"

    def render_svg(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render graph to SVG using Graphviz.

        Requires graphviz package to be installed.

        Args:
            graph: Lineage graph
            config: Render configuration

        Returns:
            SVG string
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz package is required for SVG rendering. "
                "Install with: pip install graphviz"
            )

        dot = self.render(graph, config)

        # Create graphviz source
        source = graphviz.Source(dot)
        svg = source.pipe(format="svg").decode("utf-8")

        return svg

    def render_png(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> bytes:
        """Render graph to PNG using Graphviz.

        Requires graphviz package to be installed.

        Args:
            graph: Lineage graph
            config: Render configuration

        Returns:
            PNG bytes
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz package is required for PNG rendering. "
                "Install with: pip install graphviz"
            )

        dot = self.render(graph, config)

        source = graphviz.Source(dot)
        png = source.pipe(format="png")

        return png
