"""Documentation rendering for various output formats.

This module provides renderers that convert PluginDocumentation
to different formats like Markdown, HTML, and RST.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from truthound.plugins.docs.extractor import PluginDocumentation

logger = logging.getLogger(__name__)


class DocumentationRenderer(ABC):
    """Abstract base class for documentation renderers.

    Subclasses implement rendering to specific formats.
    """

    @abstractmethod
    def render(self, doc: PluginDocumentation) -> str:
        """Render documentation to string.

        Args:
            doc: Documentation to render

        Returns:
            Rendered documentation string
        """
        ...

    def render_to_file(
        self,
        doc: PluginDocumentation,
        path: Path,
    ) -> None:
        """Render documentation to a file.

        Args:
            doc: Documentation to render
            path: Output file path
        """
        content = self.render(doc)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


class MarkdownRenderer(DocumentationRenderer):
    """Renders documentation to Markdown format.

    Produces GitHub-flavored Markdown suitable for README files
    and documentation sites like MkDocs.
    """

    def __init__(
        self,
        include_toc: bool = True,
        include_badges: bool = True,
    ) -> None:
        """Initialize Markdown renderer.

        Args:
            include_toc: Include table of contents
            include_badges: Include version/type badges
        """
        self.include_toc = include_toc
        self.include_badges = include_badges

    def render(self, doc: PluginDocumentation) -> str:
        """Render documentation to Markdown."""
        sections: list[str] = []

        # Title
        sections.append(f"# {doc.name}")
        sections.append("")

        # Badges
        if self.include_badges:
            sections.append(self._render_badges(doc))
            sections.append("")

        # Description
        if doc.description:
            sections.append(doc.description)
            sections.append("")

        # TOC
        if self.include_toc:
            sections.append(self._render_toc(doc))
            sections.append("")

        # Installation
        sections.append("## Installation")
        sections.append("")
        sections.append("```bash")
        sections.append(f"pip install truthound-plugin-{doc.name}")
        sections.append("```")
        sections.append("")

        # Configuration
        if doc.configuration:
            sections.append("## Configuration")
            sections.append("")
            sections.append(self._render_config(doc))
            sections.append("")

        # Hooks
        if doc.hooks:
            sections.append("## Hooks")
            sections.append("")
            sections.append(self._render_hooks(doc))
            sections.append("")

        # Examples
        if doc.examples:
            sections.append("## Examples")
            sections.append("")
            sections.append(self._render_examples(doc))
            sections.append("")

        # Changelog
        if doc.changelog:
            sections.append("## Changelog")
            sections.append("")
            sections.append(self._render_changelog(doc))
            sections.append("")

        # License
        if doc.license:
            sections.append("## License")
            sections.append("")
            sections.append(doc.license)
            sections.append("")

        return "\n".join(sections)

    def _render_badges(self, doc: PluginDocumentation) -> str:
        """Render version and type badges."""
        badges = []
        badges.append(f"![Version](https://img.shields.io/badge/version-{doc.version}-blue)")
        badges.append(f"![Type](https://img.shields.io/badge/type-{doc.plugin_type}-green)")
        if doc.license:
            badges.append(f"![License](https://img.shields.io/badge/license-{doc.license}-yellow)")
        return " ".join(badges)

    def _render_toc(self, doc: PluginDocumentation) -> str:
        """Render table of contents."""
        lines = ["## Table of Contents", ""]
        lines.append("- [Installation](#installation)")
        if doc.configuration:
            lines.append("- [Configuration](#configuration)")
        if doc.hooks:
            lines.append("- [Hooks](#hooks)")
        if doc.examples:
            lines.append("- [Examples](#examples)")
        if doc.changelog:
            lines.append("- [Changelog](#changelog)")
        return "\n".join(lines)

    def _render_config(self, doc: PluginDocumentation) -> str:
        """Render configuration section."""
        lines = ["| Name | Type | Default | Required | Description |",
                 "|------|------|---------|----------|-------------|"]
        for config in doc.configuration:
            default = config.default if config.default is not None else "-"
            required = "Yes" if config.required else "No"
            lines.append(
                f"| `{config.name}` | `{config.type_hint}` | "
                f"`{default}` | {required} | {config.description} |"
            )
        return "\n".join(lines)

    def _render_hooks(self, doc: PluginDocumentation) -> str:
        """Render hooks section."""
        lines = []
        for hook in doc.hooks:
            lines.append(f"### {hook.name}")
            lines.append("")
            lines.append(f"**Type:** `{hook.hook_type}`")
            lines.append("")
            if hook.description:
                lines.append(hook.description)
                lines.append("")
            if hook.parameters:
                lines.append("**Parameters:**")
                lines.append("")
                for name, type_hint, desc in hook.parameters:
                    lines.append(f"- `{name}` ({type_hint}): {desc}")
                lines.append("")
            if hook.example:
                lines.append("**Example:**")
                lines.append("")
                lines.append("```python")
                lines.append(hook.example)
                lines.append("```")
                lines.append("")
        return "\n".join(lines)

    def _render_examples(self, doc: PluginDocumentation) -> str:
        """Render examples section."""
        lines = []
        for example in doc.examples:
            lines.append(f"### {example.title}")
            lines.append("")
            if example.description:
                lines.append(example.description)
                lines.append("")
            lines.append(f"```{example.language}")
            lines.append(example.code)
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    def _render_changelog(self, doc: PluginDocumentation) -> str:
        """Render changelog section."""
        lines = []
        for version, changes in doc.changelog:
            lines.append(f"### {version}")
            lines.append("")
            lines.append(changes)
            lines.append("")
        return "\n".join(lines)


class HtmlRenderer(DocumentationRenderer):
    """Renders documentation to HTML format.

    Produces standalone HTML suitable for web viewing.
    """

    def __init__(
        self,
        template: str | None = None,
        css: str | None = None,
    ) -> None:
        """Initialize HTML renderer.

        Args:
            template: Custom HTML template
            css: Custom CSS styles
        """
        self.template = template
        self.css = css or self._default_css()

    def render(self, doc: PluginDocumentation) -> str:
        """Render documentation to HTML."""
        content = self._render_content(doc)

        if self.template:
            return self.template.format(
                title=doc.name,
                content=content,
                css=self.css,
            )

        return self._wrap_html(doc.name, content)

    def _render_content(self, doc: PluginDocumentation) -> str:
        """Render the main content."""
        sections: list[str] = []

        # Header
        sections.append(f"<h1>{doc.name}</h1>")
        sections.append(f"<p class='version'>Version {doc.version}</p>")
        sections.append(f"<p class='type'>Type: {doc.plugin_type}</p>")

        # Description
        if doc.description:
            sections.append(f"<div class='description'>{doc.description}</div>")

        # Configuration
        if doc.configuration:
            sections.append("<h2>Configuration</h2>")
            sections.append(self._render_config_html(doc))

        # Hooks
        if doc.hooks:
            sections.append("<h2>Hooks</h2>")
            sections.append(self._render_hooks_html(doc))

        # Examples
        if doc.examples:
            sections.append("<h2>Examples</h2>")
            sections.append(self._render_examples_html(doc))

        return "\n".join(sections)

    def _render_config_html(self, doc: PluginDocumentation) -> str:
        """Render configuration as HTML table."""
        rows = []
        for config in doc.configuration:
            default = config.default if config.default is not None else "-"
            required = "Yes" if config.required else "No"
            rows.append(f"""
            <tr>
                <td><code>{config.name}</code></td>
                <td><code>{config.type_hint}</code></td>
                <td><code>{default}</code></td>
                <td>{required}</td>
                <td>{config.description}</td>
            </tr>
            """)

        return f"""
        <table class='config-table'>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Default</th>
                    <th>Required</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _render_hooks_html(self, doc: PluginDocumentation) -> str:
        """Render hooks as HTML."""
        sections = []
        for hook in doc.hooks:
            sections.append(f"<div class='hook'>")
            sections.append(f"<h3>{hook.name}</h3>")
            sections.append(f"<p class='hook-type'>Type: {hook.hook_type}</p>")
            if hook.description:
                sections.append(f"<p>{hook.description}</p>")
            if hook.example:
                sections.append(f"<pre><code>{hook.example}</code></pre>")
            sections.append("</div>")
        return "\n".join(sections)

    def _render_examples_html(self, doc: PluginDocumentation) -> str:
        """Render examples as HTML."""
        sections = []
        for example in doc.examples:
            sections.append(f"<div class='example'>")
            sections.append(f"<h3>{example.title}</h3>")
            if example.description:
                sections.append(f"<p>{example.description}</p>")
            sections.append(f"<pre><code class='language-{example.language}'>{example.code}</code></pre>")
            sections.append("</div>")
        return "\n".join(sections)

    def _wrap_html(self, title: str, content: str) -> str:
        """Wrap content in full HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Plugin Documentation</title>
    <style>{self.css}</style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

    def _default_css(self) -> str:
        """Return default CSS styles."""
        return """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2, h3 {
    color: #2c3e50;
}

.version, .type {
    color: #7f8c8d;
    font-size: 0.9em;
}

.description {
    margin: 20px 0;
    padding: 15px;
    background: #f7f9fc;
    border-radius: 5px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background: #f5f5f5;
}

code {
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', monospace;
}

pre {
    background: #282c34;
    color: #abb2bf;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
}

pre code {
    background: none;
    padding: 0;
}

.hook, .example {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
}

.hook-type {
    color: #27ae60;
    font-weight: bold;
}
"""


class JsonRenderer(DocumentationRenderer):
    """Renders documentation to JSON format.

    Useful for API responses or further processing.
    """

    def __init__(self, indent: int = 2) -> None:
        """Initialize JSON renderer.

        Args:
            indent: Indentation level for pretty-printing
        """
        self.indent = indent

    def render(self, doc: PluginDocumentation) -> str:
        """Render documentation to JSON."""
        return json.dumps(doc.to_dict(), indent=self.indent)


def render_documentation(
    doc: PluginDocumentation,
    format: str = "markdown",
    **kwargs: Any,
) -> str:
    """Convenience function to render documentation.

    Args:
        doc: Documentation to render
        format: Output format ("markdown", "html", "json")
        **kwargs: Arguments passed to renderer

    Returns:
        Rendered documentation string

    Raises:
        ValueError: If format is unknown
    """
    renderers = {
        "markdown": MarkdownRenderer,
        "md": MarkdownRenderer,
        "html": HtmlRenderer,
        "json": JsonRenderer,
    }

    renderer_class = renderers.get(format.lower())
    if not renderer_class:
        raise ValueError(f"Unknown format: {format}. Available: {list(renderers.keys())}")

    renderer = renderer_class(**kwargs)
    return renderer.render(doc)
