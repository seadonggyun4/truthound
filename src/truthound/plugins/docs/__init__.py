"""Plugin documentation generation module.

This module provides automatic documentation generation for plugins
by extracting information from source code and metadata.

Components:
    - DocumentationExtractor: Extracts docs from plugin source
    - DocumentationRenderer: Renders docs to various formats
    - PluginDocumentation: Documentation data structure

Example:
    >>> from truthound.plugins.docs import (
    ...     DocumentationExtractor,
    ...     DocumentationRenderer,
    ... )
    >>>
    >>> extractor = DocumentationExtractor()
    >>> docs = extractor.extract(MyPlugin)
    >>> html = DocumentationRenderer().render(docs, "html")
"""

from __future__ import annotations

from truthound.plugins.docs.extractor import (
    DocumentationExtractor,
    PluginDocumentation,
    HookDocumentation,
    ConfigSchema,
    CodeExample,
)
from truthound.plugins.docs.renderer import (
    DocumentationRenderer,
    MarkdownRenderer,
    HtmlRenderer,
)

__all__ = [
    "DocumentationExtractor",
    "PluginDocumentation",
    "HookDocumentation",
    "ConfigSchema",
    "CodeExample",
    "DocumentationRenderer",
    "MarkdownRenderer",
    "HtmlRenderer",
]
