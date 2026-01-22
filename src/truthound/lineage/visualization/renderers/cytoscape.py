"""Cytoscape.js renderer for lineage graphs.

Produces JSON output compatible with Cytoscape.js graph library.
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
    from truthound.lineage.base import LineageGraph


class CytoscapeRenderer(IGraphRenderer):
    """Renders lineage graphs to Cytoscape.js JSON format.

    Cytoscape.js is a graph theory library for analysis and
    visualization with extensive layout algorithms.

    Example:
        >>> renderer = CytoscapeRenderer()
        >>> json_output = renderer.render(graph)
        >>> # Use in Cytoscape: cytoscape({elements: data.elements})

        >>> # Dark theme
        >>> renderer = CytoscapeRenderer(theme="dark")
        >>> html_output = renderer.render_html(graph)
    """

    def __init__(self, theme: str = "light") -> None:
        """Initialize renderer with theme.

        Args:
            theme: Color theme ('light' or 'dark')
        """
        self._theme = theme

    # Layout algorithm mapping
    LAYOUTS = {
        "force": "cose",
        "hierarchical": "dagre",
        "circular": "circle",
        "grid": "grid",
        "breadthfirst": "breadthfirst",
        "concentric": "concentric",
    }

    @property
    def format(self) -> RenderFormat:
        return RenderFormat.CYTOSCAPE_JSON

    def render(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render graph to Cytoscape.js JSON.

        Args:
            graph: Lineage graph to render
            config: Render configuration

        Returns:
            JSON string for Cytoscape.js
        """
        config = config or RenderConfig(theme=self._theme)
        if config.theme == "light" and self._theme == "dark":
            config.theme = self._theme
        data = self._build_cytoscape_data(graph, config)
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
            direction: "upstream", "downstream", or "both"
            max_depth: Maximum depth
            config: Render configuration

        Returns:
            JSON string for Cytoscape.js
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

        data = self._build_cytoscape_data_from_lists(subgraph_nodes, subgraph_edges, config)
        return json.dumps(data, indent=2, default=str)

    def _build_cytoscape_data_from_lists(
        self,
        node_list: list,
        edge_list: list,
        config: RenderConfig,
    ) -> dict[str, Any]:
        """Build Cytoscape.js data structure from node and edge lists.

        Args:
            node_list: List of LineageNode objects
            edge_list: List of LineageEdge objects
            config: Render configuration

        Returns:
            Cytoscape.js compatible data structure
        """
        elements = []
        node_ids = set()

        # Build nodes
        for node in node_list:
            if config.filter_node_types and node.node_type.value not in config.filter_node_types:
                continue

            node_ids.add(node.id)

            node_data: dict[str, Any] = {
                "group": "nodes",
                "data": {
                    "id": node.id,
                    "label": node.name,
                    "type": node.node_type.value,
                    "color": config.get_node_color(node.node_type.value),
                    "highlighted": node.id in config.highlight_nodes,
                },
                "classes": node.node_type.value.lower(),
            }

            if config.include_metadata:
                if hasattr(node, "metadata"):
                    node_data["data"]["description"] = node.metadata.description
                    node_data["data"]["owner"] = node.metadata.owner
                    node_data["data"]["tags"] = list(node.metadata.tags) if node.metadata.tags else []
                if hasattr(node, "schema") and node.schema:
                    node_data["data"]["schema"] = node.schema

            if node.id in config.highlight_nodes:
                node_data["classes"] += " highlighted"

            elements.append(node_data)

        # Build edges
        for edge in edge_list:
            if config.filter_edge_types and edge.edge_type.value not in config.filter_edge_types:
                continue

            source_id = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            target_id = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

            if source_id not in node_ids or target_id not in node_ids:
                continue

            edge_data: dict[str, Any] = {
                "group": "edges",
                "data": {
                    "id": f"{source_id}-{target_id}",
                    "source": source_id,
                    "target": target_id,
                    "type": edge.edge_type.value,
                    "color": config.get_edge_color(edge.edge_type.value),
                },
                "classes": edge.edge_type.value.lower(),
            }

            if config.include_metadata and hasattr(edge, "metadata"):
                edge_data["data"]["metadata"] = edge.metadata.to_dict() if hasattr(edge.metadata, "to_dict") else {}

            elements.append(edge_data)

        # Layout configuration
        layout_name = self.LAYOUTS.get(config.layout, "cose")

        return {
            "elements": elements,
            "layout": {
                "name": layout_name,
                "rankDir": config.orientation,
                "animate": True,
                "animationDuration": 500,
            },
            "style": self._get_default_style(config),
        }

    def _build_cytoscape_data(
        self,
        graph: "LineageGraph",
        config: RenderConfig,
    ) -> dict[str, Any]:
        """Build Cytoscape.js data structure."""
        elements = []
        node_ids = set()

        # Handle both list and dict node formats
        if isinstance(graph.nodes, dict):
            node_items = graph.nodes.items()
        else:
            node_items = [(node.id, node) for node in graph.nodes]

        # Build nodes
        for node_id, node in node_items:
            if config.filter_node_types and node.node_type.value not in config.filter_node_types:
                continue

            node_ids.add(node_id)

            node_data: dict[str, Any] = {
                "group": "nodes",
                "data": {
                    "id": node_id,
                    "label": node.name,
                    "type": node.node_type.value,
                    "color": config.get_node_color(node.node_type.value),
                    "highlighted": node_id in config.highlight_nodes,
                },
                "classes": node.node_type.value.lower(),
            }

            if config.include_metadata:
                if hasattr(node, "metadata"):
                    node_data["data"]["description"] = node.metadata.description
                    node_data["data"]["owner"] = node.metadata.owner
                    node_data["data"]["tags"] = list(node.metadata.tags) if node.metadata.tags else []
                if hasattr(node, "schema") and node.schema:
                    node_data["data"]["schema"] = node.schema

            if node_id in config.highlight_nodes:
                node_data["classes"] += " highlighted"

            elements.append(node_data)

        # Build edges
        for edge in graph.edges:
            if config.filter_edge_types and edge.edge_type.value not in config.filter_edge_types:
                continue

            # Handle both source/source_id attribute names
            source_id = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            target_id = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)

            if source_id not in node_ids or target_id not in node_ids:
                continue

            edge_data: dict[str, Any] = {
                "group": "edges",
                "data": {
                    "id": f"{source_id}-{target_id}",
                    "source": source_id,
                    "target": target_id,
                    "type": edge.edge_type.value,
                    "color": config.get_edge_color(edge.edge_type.value),
                },
                "classes": edge.edge_type.value.lower(),
            }

            if config.include_metadata and hasattr(edge, "metadata"):
                edge_data["data"]["metadata"] = edge.metadata.to_dict() if hasattr(edge.metadata, "to_dict") else {}

            elements.append(edge_data)

        # Layout configuration
        layout_name = self.LAYOUTS.get(config.layout, "cose")

        return {
            "elements": elements,
            "layout": {
                "name": layout_name,
                "rankDir": config.orientation,
                "animate": True,
                "animationDuration": 500,
            },
            "style": self._get_default_style(config),
        }

    def _get_default_style(self, config: RenderConfig) -> list[dict[str, Any]]:
        """Get default Cytoscape.js styles."""
        is_dark = config.theme == "dark"
        text_color = "#e0e0e0" if is_dark else "#333"

        return [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "data(color)",
                    "text-valign": "bottom",
                    "text-margin-y": 5,
                    "font-size": 12,
                    "width": 30,
                    "height": 30,
                    "color": text_color,
                },
            },
            {
                "selector": "node.highlighted",
                "style": {
                    "border-width": 3,
                    "border-color": "#ff0000",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "data(color)",
                    "target-arrow-color": "data(color)",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                },
            },
            {
                "selector": "edge:selected",
                "style": {
                    "width": 4,
                    "line-color": "#ff0000",
                },
            },
        ]

    def render_html(
        self,
        graph: "LineageGraph",
        config: RenderConfig | None = None,
    ) -> str:
        """Render complete HTML page with Cytoscape.js visualization.

        Args:
            graph: Lineage graph
            config: Render configuration

        Returns:
            Complete HTML page
        """
        config = config or RenderConfig(theme=self._theme)
        if config.theme == "light" and self._theme == "dark":
            config.theme = self._theme
        data = self._build_cytoscape_data(graph, config)
        data_json = json.dumps(data, indent=2, default=str)

        # Theme-specific styles
        is_dark = config.theme == "dark"
        bg_color = "#1a1a2e" if is_dark else "white"
        text_color = "#e0e0e0" if is_dark else "#333"
        panel_bg = "#16213e" if is_dark else "white"
        panel_border = "#0f3460" if is_dark else "#ccc"

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Lineage Graph - Cytoscape.js</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.25.0/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; background: {bg_color}; color: {text_color}; }}
        #cy {{ width: 100%; height: 100vh; }}
        #controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: {panel_bg};
            color: {text_color};
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border: 1px solid {panel_border};
        }}
        #controls select {{ padding: 5px; margin: 5px 0; background: {bg_color}; color: {text_color}; border: 1px solid {panel_border}; }}
        #info {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: {panel_bg};
            color: {text_color};
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border: 1px solid {panel_border};
            max-width: 300px;
            display: none;
        }}
    </style>
</head>
<body>
    <div id="cy"></div>
    <div id="controls">
        <label>Layout:</label>
        <select id="layout-select">
            <option value="dagre">Hierarchical</option>
            <option value="cose">Force</option>
            <option value="circle">Circular</option>
            <option value="grid">Grid</option>
            <option value="breadthfirst">Breadth First</option>
        </select>
    </div>
    <div id="info"></div>
    <script>
        const graphData = {data_json};

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: graphData.elements,
            style: graphData.style,
            layout: graphData.layout,
        }});

        // Layout selector
        document.getElementById('layout-select').addEventListener('change', (e) => {{
            cy.layout({{ name: e.target.value }}).run();
        }});

        // Node info on click
        const infoPanel = document.getElementById('info');
        cy.on('tap', 'node', (event) => {{
            const node = event.target;
            const data = node.data();
            infoPanel.style.display = 'block';
            infoPanel.innerHTML = `
                <h4>${{data.label}}</h4>
                <p><strong>Type:</strong> ${{data.type}}</p>
                ${{data.description ? `<p><strong>Description:</strong> ${{data.description}}</p>` : ''}}
                ${{data.owner ? `<p><strong>Owner:</strong> ${{data.owner}}</p>` : ''}}
            `;
        }});

        cy.on('tap', (event) => {{
            if (event.target === cy) {{
                infoPanel.style.display = 'none';
            }}
        }});
    </script>
</body>
</html>"""
