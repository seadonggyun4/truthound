"""Lineage visualization module.

Provides graph rendering and visualization capabilities:
- D3.js: Interactive web visualizations
- Cytoscape.js: Advanced graph layouts
- Graphviz: DOT format export
- Mermaid: Markdown-friendly diagrams
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from truthound.lineage.visualization.protocols import (
    IGraphRenderer,
    RenderFormat,
    RenderConfig,
)

from truthound.lineage.visualization.renderers import (
    D3Renderer,
    CytoscapeRenderer,
    GraphvizRenderer,
    MermaidRenderer,
)

if TYPE_CHECKING:
    pass


def get_renderer(
    renderer_type: str,
    *,
    theme: str = "light",
    **kwargs,
) -> IGraphRenderer:
    """Get a renderer instance by type.

    Args:
        renderer_type: Type of renderer ('d3', 'cytoscape', 'graphviz', 'mermaid')
        theme: Color theme ('light', 'dark')
        **kwargs: Additional renderer configuration

    Returns:
        Configured renderer instance

    Raises:
        ValueError: If renderer type is unknown

    Example:
        >>> renderer = get_renderer("d3", theme="dark")
        >>> output = renderer.render(graph)
    """
    renderer_map: dict[str, type[IGraphRenderer]] = {
        "d3": D3Renderer,
        "cytoscape": CytoscapeRenderer,
        "graphviz": GraphvizRenderer,
        "mermaid": MermaidRenderer,
    }

    if renderer_type not in renderer_map:
        available = ", ".join(sorted(renderer_map.keys()))
        raise ValueError(
            f"Unknown renderer type: '{renderer_type}'. "
            f"Available renderers: {available}"
        )

    renderer_class = renderer_map[renderer_type]
    return renderer_class(theme=theme)


__all__ = [
    # Protocols
    "IGraphRenderer",
    "RenderFormat",
    "RenderConfig",
    # Renderers
    "D3Renderer",
    "CytoscapeRenderer",
    "GraphvizRenderer",
    "MermaidRenderer",
    # Factory function
    "get_renderer",
]
