"""Mermaid renderer for lineage graphs.

Produces Mermaid diagram syntax for Markdown-friendly visualization.
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


class MermaidRenderer(IGraphRenderer):
    """Renders lineage graphs to Mermaid diagram syntax.

    Mermaid diagrams can be embedded in Markdown files and
    rendered by GitHub, GitLab, and other platforms.

    Example:
        >>> renderer = MermaidRenderer()
        >>> mermaid = renderer.render(graph)
        >>> # Embed in Markdown: ```mermaid\\n{mermaid}\\n```
    """

    # Node shapes in Mermaid syntax
    NODE_SHAPES = {
        "SOURCE": ("[(", ")]"),  # Cylindrical
        "TABLE": ("[", "]"),  # Rectangle
        "FILE": ("{{", "}}"),  # Hexagon
        "STREAM": ("[/", "/]"),  # Parallelogram
        "TRANSFORMATION": ("[[", "]]"),  # Subroutine
        "VALIDATION": ("{", "}"),  # Diamond-like
        "MODEL": ("([", "])"),  # Stadium
        "REPORT": ("[", "]"),  # Rectangle
        "EXTERNAL": ("((", "))"),  # Circle
        "VIRTUAL": ("[", "]"),  # Rectangle
    }

    # Edge arrow styles
    EDGE_ARROWS = {
        "DERIVED_FROM": "-->",
        "VALIDATED_BY": "-.->",
        "USED_BY": "-.->",
        "TRANSFORMED_TO": "==>",
        "JOINED_WITH": "<-->",
        "AGGREGATED_TO": "-->",
        "FILTERED_TO": "-.->",
        "DEPENDS_ON": "-.-",
    }

    @property
    def format(self) -> RenderFormat:
        return RenderFormat.MERMAID

    def render(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render graph to Mermaid syntax.

        Args:
            graph: Lineage graph to render
            config: Render configuration

        Returns:
            Mermaid diagram string
        """
        config = config or RenderConfig()
        return self._build_mermaid(graph, config)

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
            direction: "upstream" or "downstream"
            max_depth: Maximum depth
            config: Render configuration

        Returns:
            Mermaid diagram string
        """
        config = config or RenderConfig()

        if direction == "upstream":
            subgraph = graph.get_upstream(root_node_id, max_depth)
        else:
            subgraph = graph.get_downstream(root_node_id, max_depth)

        config.highlight_nodes = config.highlight_nodes + [root_node_id]

        return self._build_mermaid(subgraph, config)

    def _build_mermaid(
        self,
        graph: "LineageGraph",
        config: RenderConfig,
    ) -> str:
        """Build Mermaid diagram string."""
        lines = []

        # Diagram header
        lines.append(f"graph {config.orientation}")
        lines.append("")

        # Group nodes by type for styling
        node_by_type: dict[str, list[str]] = {}
        node_ids = set()

        # Handle both list and dict node formats
        if isinstance(graph.nodes, dict):
            node_items = graph.nodes.items()
        else:
            node_items = [(node.id, node) for node in graph.nodes]

        # Node definitions
        for node_id, node in node_items:
            if config.filter_node_types and node.node_type.value not in config.filter_node_types:
                continue

            node_ids.add(node_id)

            node_type = node.node_type.value
            if node_type not in node_by_type:
                node_by_type[node_type] = []
            node_by_type[node_type].append(node_id)

            # Build node definition
            safe_id = self._safe_id(node_id)
            label = self._escape_label(node.name)
            left, right = self.NODE_SHAPES.get(node_type, ("[", "]"))

            lines.append(f"    {safe_id}{left}{label}{right}")

        lines.append("")

        # Edge definitions
        for edge in graph.edges:
            if config.filter_edge_types and edge.edge_type.value not in config.filter_edge_types:
                continue

            # Handle both source/source_id attribute names
            source_id = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            target_id = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

            if source_id not in node_ids or target_id not in node_ids:
                continue

            safe_source = self._safe_id(source_id)
            safe_target = self._safe_id(target_id)
            arrow = self.EDGE_ARROWS.get(edge.edge_type.value, "-->")

            # Add edge label if not too long
            edge_label = edge.edge_type.value
            if len(edge_label) <= 15:
                lines.append(f"    {safe_source} {arrow}|{edge_label}| {safe_target}")
            else:
                lines.append(f"    {safe_source} {arrow} {safe_target}")

        lines.append("")

        # Styling by node type
        lines.append("    %% Styling")
        for node_type, ids in node_by_type.items():
            color = config.get_node_color(node_type)
            safe_ids = ",".join(self._safe_id(nid) for nid in ids)
            lines.append(f"    style {safe_ids} fill:{color}")

        # Highlight nodes
        if config.highlight_nodes:
            for node_id in config.highlight_nodes:
                if node_id in node_ids:
                    safe_id = self._safe_id(node_id)
                    lines.append(f"    style {safe_id} stroke:#ff0000,stroke-width:3px")

        return "\n".join(lines)

    def _escape_label(self, text: str) -> str:
        """Escape text for Mermaid labels."""
        # Remove special characters that break Mermaid syntax
        return text.replace('"', "'").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")

    def _safe_id(self, node_id: str) -> str:
        """Convert node ID to safe Mermaid identifier."""
        # Replace non-alphanumeric with underscore
        safe = "".join(c if c.isalnum() else "_" for c in node_id)
        return safe or "node"

    def render_markdown(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
        title: str = "Lineage Graph",
    ) -> str:
        """Render graph as Markdown with embedded Mermaid diagram.

        Args:
            graph: Lineage graph
            config: Render configuration
            title: Document title

        Returns:
            Markdown string
        """
        config = config or RenderConfig()
        mermaid = self.render(graph, config)

        md_lines = [
            f"# {title}",
            "",
            "```mermaid",
            mermaid,
            "```",
            "",
            "## Legend",
            "",
            "| Symbol | Node Type |",
            "|--------|-----------|",
        ]

        # Add legend
        for node_type, (left, right) in self.NODE_SHAPES.items():
            md_lines.append(f"| `{left}...{right}` | {node_type} |")

        md_lines.extend([
            "",
            "## Edge Types",
            "",
            "| Arrow | Relationship |",
            "|-------|--------------|",
        ])

        for edge_type, arrow in self.EDGE_ARROWS.items():
            md_lines.append(f"| `{arrow}` | {edge_type} |")

        return "\n".join(md_lines)

    def render_html(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render complete HTML page with Mermaid diagram.

        Args:
            graph: Lineage graph
            config: Render configuration

        Returns:
            Complete HTML page
        """
        config = config or RenderConfig()
        mermaid = self.render(graph, config)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Lineage Graph - Mermaid</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
            }}
        }});
    </script>
</body>
</html>"""
