"""Renderers for the Data Docs report pipeline.

Renderers convert the transformed report data into intermediate HTML
that can then be exported to various formats.

Available Renderers:
- JinjaRenderer: Jinja2-based template rendering
- SimpleRenderer: Basic HTML rendering without templates
- CustomRenderer: Support for user-defined templates
- CallableRenderer: Render using any Python callable
"""

from truthound.datadocs.renderers.base import (
    Renderer,
    BaseRenderer,
    RenderResult,
)
from truthound.datadocs.renderers.jinja import (
    JinjaRenderer,
    JinjaTemplateLoader,
)
from truthound.datadocs.renderers.custom import (
    CustomRenderer,
    StringTemplateRenderer,
    FileTemplateRenderer,
    CallableRenderer,
)

__all__ = [
    # Base
    "Renderer",
    "BaseRenderer",
    "RenderResult",
    # Jinja
    "JinjaRenderer",
    "JinjaTemplateLoader",
    # Custom
    "CustomRenderer",
    "StringTemplateRenderer",
    "FileTemplateRenderer",
    "CallableRenderer",
]
