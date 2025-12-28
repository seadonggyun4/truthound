"""D3.js renderer for lineage graphs.

Produces JSON output compatible with D3.js force-directed graphs.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import json

from truthound.lineage.visualization.protocols import (
    IGraphRenderer,
    RenderFormat,
    RenderConfig,
)

if TYPE_CHECKING:
    from truthound.lineage.base import LineageGraph, LineageNode, LineageEdge


class D3Renderer(IGraphRenderer):
    """Renders lineage graphs to D3.js JSON format.

    Produces output compatible with D3.js force-directed graph
    visualization library.

    Example:
        >>> renderer = D3Renderer()
        >>> json_output = renderer.render(graph)
        >>> # Use in D3.js: d3.forceSimulation(data.nodes)...
    """

    @property
    def format(self) -> RenderFormat:
        return RenderFormat.D3_JSON

    def render(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render graph to D3.js JSON.

        Args:
            graph: Lineage graph to render
            config: Render configuration

        Returns:
            JSON string for D3.js
        """
        config = config or RenderConfig()
        data = self._build_d3_data(graph, config)
        return json.dumps(data, indent=2, default=str)

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
            JSON string for D3.js
        """
        config = config or RenderConfig()

        # Get subgraph nodes
        if direction == "upstream":
            subgraph = graph.get_upstream(root_node_id, max_depth)
        else:
            subgraph = graph.get_downstream(root_node_id, max_depth)

        # Add root to highlights
        config.highlight_nodes = config.highlight_nodes + [root_node_id]

        data = self._build_d3_data(subgraph, config)
        return json.dumps(data, indent=2, default=str)

    def _build_d3_data(
        self,
        graph: "LineageGraph",
        config: RenderConfig,
    ) -> dict[str, Any]:
        """Build D3.js data structure."""
        nodes = []
        links = []
        node_index: dict[str, int] = {}

        # Handle both list and dict node formats
        if isinstance(graph.nodes, dict):
            node_items = graph.nodes.items()
        else:
            # Assume list of nodes with id attribute
            node_items = [(node.id, node) for node in graph.nodes]

        # Build nodes
        for i, (node_id, node) in enumerate(node_items):
            # Apply filters
            if config.filter_node_types and node.node_type.value not in config.filter_node_types:
                continue

            node_index[node_id] = len(nodes)

            node_data: dict[str, Any] = {
                "id": node_id,
                "name": node.name,
                "type": node.node_type.value,
                "color": config.get_node_color(node.node_type.value),
                "size": config.node_sizes.get(node.node_type.value, 10),
                "highlighted": node_id in config.highlight_nodes,
            }

            if config.include_metadata:
                node_data["metadata"] = {
                    "description": node.metadata.description if hasattr(node, "metadata") else None,
                    "owner": node.metadata.owner if hasattr(node, "metadata") else None,
                    "tags": list(node.metadata.tags) if hasattr(node, "metadata") and node.metadata.tags else [],
                }
                if hasattr(node, "schema") and node.schema:
                    node_data["schema"] = node.schema
                if hasattr(node, "statistics") and node.statistics:
                    node_data["statistics"] = node.statistics

            nodes.append(node_data)

        # Build links (edges)
        for edge in graph.edges:
            # Apply filters
            if config.filter_edge_types and edge.edge_type.value not in config.filter_edge_types:
                continue

            # Handle both source/source_id attribute names
            source_id = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            target_id = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

            # Only include if both nodes are in the graph
            if source_id not in node_index or target_id not in node_index:
                continue

            link_data: dict[str, Any] = {
                "source": node_index[source_id],
                "target": node_index[target_id],
                "type": edge.edge_type.value,
                "color": config.get_edge_color(edge.edge_type.value),
            }

            if config.include_metadata and hasattr(edge, "metadata"):
                link_data["metadata"] = edge.metadata.to_dict() if hasattr(edge.metadata, "to_dict") else {}

            links.append(link_data)

        return {
            "nodes": nodes,
            "links": links,
            "config": {
                "layout": config.layout,
                "width": config.width,
                "height": config.height,
            },
        }

    def render_html(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render complete HTML page with D3.js visualization.

        Args:
            graph: Lineage graph
            config: Render configuration

        Returns:
            Complete HTML page
        """
        config = config or RenderConfig()
        data_json = self.render(graph, config)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Lineage Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        svg {{ width: 100%; height: 100vh; }}
        .node {{ cursor: pointer; }}
        .node text {{ font-size: 12px; }}
        .link {{ stroke-opacity: 0.6; fill: none; }}
        .highlighted {{ stroke: #ff0000; stroke-width: 3px; }}
        .tooltip {{
            position: absolute;
            background: white;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            pointer-events: none;
            display: none;
        }}
    </style>
</head>
<body>
    <div id="tooltip" class="tooltip"></div>
    <svg></svg>
    <script>
        const data = {data_json};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("svg")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g");

        // Zoom behavior
        svg.call(d3.zoom()
            .extent([[0, 0], [width, height]])
            .scaleExtent([0.1, 8])
            .on("zoom", (event) => g.attr("transform", event.transform)));

        // Force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id((d, i) => i).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        // Arrow markers
        svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("fill", "#999")
            .attr("d", "M0,-5L10,0L0,5");

        // Links
        const link = g.append("g")
            .selectAll("path")
            .data(data.links)
            .enter().append("path")
            .attr("class", "link")
            .attr("stroke", d => d.color)
            .attr("stroke-width", 2)
            .attr("marker-end", "url(#end)");

        // Nodes
        const node = g.append("g")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", d => d.size || 10)
            .attr("fill", d => d.color)
            .attr("class", d => d.highlighted ? "highlighted" : "");

        node.append("text")
            .attr("dx", 15)
            .attr("dy", 5)
            .text(d => d.name);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        node.on("mouseover", (event, d) => {{
            tooltip.style("display", "block")
                .html(`<strong>${{d.name}}</strong><br/>Type: ${{d.type}}`);
        }})
        .on("mousemove", (event) => {{
            tooltip.style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", () => {{
            tooltip.style("display", "none");
        }});

        // Simulation tick
        simulation.on("tick", () => {{
            link.attr("d", d => {{
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                return `M${{d.source.x}},${{d.source.y}}L${{d.target.x}},${{d.target.y}}`;
            }});
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""
