"""Data lineage commands.

This module implements data lineage tracking and analysis commands (Phase 10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file

# Lineage app for subcommands
app = typer.Typer(
    name="lineage",
    help="Data lineage tracking and analysis commands",
)


@app.command(name="show")
@error_boundary
def show_cmd(
    lineage_file: Annotated[Path, typer.Argument(help="Path to lineage JSON file")],
    node: Annotated[
        Optional[str],
        typer.Option("--node", "-n", help="Show lineage for specific node"),
    ] = None,
    direction: Annotated[
        str,
        typer.Option("--direction", "-d", help="Direction (upstream, downstream, both)"),
    ] = "both",
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, dot)"),
    ] = "console",
) -> None:
    """Display lineage information.

    Shows data lineage relationships including upstream sources
    and downstream consumers of data assets.

    Examples:
        truthound lineage show lineage.json
        truthound lineage show lineage.json --node my_table --direction upstream
        truthound lineage show lineage.json --format dot > lineage.dot
    """
    from truthound.lineage import LineageGraph

    require_file(lineage_file, "Lineage file")

    try:
        graph = LineageGraph.load(lineage_file)

        if node:
            if not graph.has_node(node):
                typer.echo(f"Error: Node '{node}' not found", err=True)
                raise typer.Exit(1)

            node_obj = graph.get_node(node)
            typer.echo(f"\nLineage for: {node}")
            typer.echo(f"Type: {node_obj.node_type.value}")

            if direction in ("upstream", "both"):
                upstream = graph.get_upstream(node)
                typer.echo(f"\nUpstream ({len(upstream)} nodes):")
                for n in upstream:
                    typer.echo(f"  <- {n.name} ({n.node_type.value})")

            if direction in ("downstream", "both"):
                downstream = graph.get_downstream(node)
                typer.echo(f"\nDownstream ({len(downstream)} nodes):")
                for n in downstream:
                    typer.echo(f"  -> {n.name} ({n.node_type.value})")
        else:
            typer.echo("\nLineage Graph Summary")
            typer.echo("=" * 40)
            typer.echo(f"Nodes: {graph.node_count}")
            typer.echo(f"Edges: {graph.edge_count}")

            roots = graph.get_roots()
            typer.echo(f"\nRoot nodes ({len(roots)}):")
            for r in roots[:10]:
                typer.echo(f"  {r.name} ({r.node_type.value})")

            leaves = graph.get_leaves()
            typer.echo(f"\nLeaf nodes ({len(leaves)}):")
            for leaf in leaves[:10]:
                typer.echo(f"  {leaf.name} ({leaf.node_type.value})")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="impact")
@error_boundary
def impact_cmd(
    lineage_file: Annotated[Path, typer.Argument(help="Path to lineage JSON file")],
    node: Annotated[str, typer.Argument(help="Node to analyze impact for")],
    max_depth: Annotated[
        int,
        typer.Option("--max-depth", help="Maximum depth for impact analysis"),
    ] = -1,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file for results"),
    ] = None,
) -> None:
    """Analyze impact of changes to a data asset.

    This command traces downstream dependencies to understand
    what would be affected if a data asset changes.

    Examples:
        truthound lineage impact lineage.json raw_data
        truthound lineage impact lineage.json my_table --max-depth 3
    """
    from truthound.lineage import LineageGraph, ImpactAnalyzer

    require_file(lineage_file, "Lineage file")

    try:
        graph = LineageGraph.load(lineage_file)
        analyzer = ImpactAnalyzer(graph)

        result = analyzer.analyze_impact(node, max_depth=max_depth)

        typer.echo(result.summary())

        if result.affected_nodes:
            typer.echo("\nAffected nodes:")
            for affected in result.affected_nodes[:20]:
                level_marker = {
                    "critical": "[!!!]",
                    "high": "[!!]",
                    "medium": "[!]",
                    "low": "[-]",
                    "none": "[ ]",
                }.get(affected.impact_level.value, "")
                typer.echo(
                    f"  {level_marker} {affected.node.name} (depth={affected.distance})"
                )

        if output:
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            typer.echo(f"\nResults saved to {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="visualize")
@error_boundary
def visualize_cmd(
    lineage_file: Annotated[Path, typer.Argument(help="Path to lineage JSON file")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ],
    renderer: Annotated[
        str,
        typer.Option("--renderer", "-r", help="Renderer (d3, cytoscape, graphviz, mermaid)"),
    ] = "d3",
    theme: Annotated[
        str,
        typer.Option("--theme", "-t", help="Theme (light, dark)"),
    ] = "light",
    focus: Annotated[
        Optional[str],
        typer.Option("--focus", "-f", help="Focus on specific node"),
    ] = None,
) -> None:
    """Generate visual representation of lineage graph.

    Supported renderers:
        - d3: Interactive D3.js visualization (HTML)
        - cytoscape: Cytoscape.js visualization (HTML)
        - graphviz: Static Graphviz diagram (SVG/PNG)
        - mermaid: Mermaid diagram (Markdown)

    Examples:
        truthound lineage visualize lineage.json -o graph.html
        truthound lineage visualize lineage.json -o graph.html --renderer cytoscape
        truthound lineage visualize lineage.json -o graph.svg --renderer graphviz
    """
    from truthound.lineage import LineageGraph
    from truthound.lineage.visualization import get_renderer, RenderConfig

    require_file(lineage_file, "Lineage file")

    try:
        graph = LineageGraph.load(lineage_file)

        typer.echo(f"Generating {renderer} visualization...")

        renderer_instance = get_renderer(renderer, theme=theme)

        # Determine output format based on file extension
        output_suffix = output.suffix.lower()

        # Use subgraph rendering if focus node is specified
        if focus:
            if not graph.has_node(focus):
                typer.echo(f"Error: Focus node '{focus}' not found in graph", err=True)
                raise typer.Exit(1)
            # render_subgraph returns JSON, wrap in HTML if needed
            json_content = renderer_instance.render_subgraph(
                graph, focus, direction="both", max_depth=-1
            )
            if output_suffix == ".html" and hasattr(renderer_instance, "render_html"):
                # For focus mode, we need to render full HTML with subgraph data
                # Get subgraph and render as HTML
                subgraph_nodes = set()
                subgraph_nodes.add(focus)
                for node in graph.get_upstream(focus, max_depth=-1):
                    subgraph_nodes.add(node.id)
                for node in graph.get_downstream(focus, max_depth=-1):
                    subgraph_nodes.add(node.id)
                # Use render_html for the full graph (focus highlighting handled by config)
                content = renderer_instance.render_html(graph)
            else:
                content = json_content
        else:
            # Use render_html for HTML output if available
            if output_suffix == ".html" and hasattr(renderer_instance, "render_html"):
                content = renderer_instance.render_html(graph)
            elif output_suffix == ".svg" and hasattr(renderer_instance, "render_svg"):
                content = renderer_instance.render_svg(graph)
            elif output_suffix == ".png" and hasattr(renderer_instance, "render_png"):
                content = renderer_instance.render_png(graph)
            elif output_suffix == ".md" and hasattr(renderer_instance, "render_markdown"):
                content = renderer_instance.render_markdown(graph)
            else:
                content = renderer_instance.render(graph)

        # Handle bytes output (PNG)
        if isinstance(content, bytes):
            output.write_bytes(content)
        else:
            output.write_text(content, encoding="utf-8")
        typer.echo(f"Visualization saved to: {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
