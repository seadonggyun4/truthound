"""Documentation renderers for multiple output formats.

This module provides renderers for generating documentation in
various formats including MkDocs, Sphinx, and standalone HTML.

Key Features:
- Multiple output format support
- Template-based rendering
- Cross-reference generation
- Navigation structure generation
- Search index generation

Example:
    from truthound.docs.renderer import MkDocsRenderer, RenderConfig

    renderer = MkDocsRenderer(RenderConfig(output_dir="docs"))
    renderer.render(package_info)
"""

from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from truthound.docs.extractor import (
    ClassInfo,
    FunctionInfo,
    ModuleInfo,
    PackageInfo,
)


class DocFormat(str, Enum):
    """Documentation output formats."""

    MKDOCS = "mkdocs"
    SPHINX = "sphinx"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class RenderConfig:
    """Configuration for documentation rendering.

    Attributes:
        output_dir: Output directory for generated docs
        format: Output format
        title: Documentation title
        description: Documentation description
        include_source: Include source code links
        include_toc: Include table of contents
        include_search: Include search functionality
        syntax_highlight: Enable syntax highlighting
        theme: Theme name (format-specific)
        extra_css: Additional CSS files
        extra_js: Additional JavaScript files
        templates_dir: Custom templates directory
        nav_depth: Navigation tree depth
        show_type_annotations: Show type annotations
        show_source_links: Show links to source code
    """

    output_dir: str = "docs"
    format: DocFormat = DocFormat.MKDOCS
    title: str = "API Documentation"
    description: str = ""
    include_source: bool = True
    include_toc: bool = True
    include_search: bool = True
    syntax_highlight: bool = True
    theme: str = "material"
    extra_css: list[str] = field(default_factory=list)
    extra_js: list[str] = field(default_factory=list)
    templates_dir: str | None = None
    nav_depth: int = 3
    show_type_annotations: bool = True
    show_source_links: bool = True


class DocRenderer(ABC):
    """Abstract base class for documentation renderers.

    Implement this to create custom output formats.
    """

    format: DocFormat = DocFormat.MARKDOWN

    def __init__(self, config: RenderConfig | None = None):
        """Initialize renderer.

        Args:
            config: Rendering configuration
        """
        self.config = config or RenderConfig()

    @abstractmethod
    def render_module(self, module: ModuleInfo) -> str:
        """Render a single module.

        Args:
            module: Module information

        Returns:
            Rendered content string
        """
        pass

    @abstractmethod
    def render_class(self, cls: ClassInfo) -> str:
        """Render a single class.

        Args:
            cls: Class information

        Returns:
            Rendered content string
        """
        pass

    @abstractmethod
    def render_function(self, func: FunctionInfo) -> str:
        """Render a single function.

        Args:
            func: Function information

        Returns:
            Rendered content string
        """
        pass

    def render_package(self, package: PackageInfo) -> dict[str, str]:
        """Render an entire package.

        Args:
            package: Package information

        Returns:
            Dictionary mapping file paths to content
        """
        files: dict[str, str] = {}

        for module in package.modules:
            path = self._module_path(module)
            files[path] = self.render_module(module)

        # Index page
        files["index.md"] = self._render_index(package)

        return files

    def write(
        self,
        package: PackageInfo | ModuleInfo,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Write rendered documentation to files.

        Args:
            package: Package or module information
            output_dir: Output directory (uses config if not specified)

        Returns:
            List of written file paths
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written_files = []

        if isinstance(package, PackageInfo):
            files = self.render_package(package)
        else:
            files = {self._module_path(package): self.render_module(package)}

        for rel_path, content in files.items():
            file_path = output_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            written_files.append(file_path)

        # Write additional config files
        written_files.extend(self._write_config_files(output_dir))

        return written_files

    def _module_path(self, module: ModuleInfo) -> str:
        """Get output path for a module."""
        parts = module.qualified_name.split(".")
        return "/".join(parts) + ".md"

    def _render_index(self, package: PackageInfo) -> str:
        """Render package index page."""
        lines = [
            f"# {package.name}",
            "",
        ]

        if package.description:
            lines.extend([package.description, ""])

        lines.extend([
            "## Modules",
            "",
        ])

        for module in sorted(package.modules, key=lambda m: m.qualified_name):
            path = self._module_path(module)
            desc = module.docstring.short_description or ""
            lines.append(f"- [{module.qualified_name}]({path}): {desc}")

        return "\n".join(lines)

    def _write_config_files(self, output_dir: Path) -> list[Path]:
        """Write format-specific configuration files."""
        return []


class MarkdownRenderer(DocRenderer):
    """Renderer for plain Markdown output."""

    format = DocFormat.MARKDOWN

    def render_module(self, module: ModuleInfo) -> str:
        """Render module to Markdown."""
        lines = [
            f"# {module.qualified_name}",
            "",
        ]

        # Module docstring
        if module.docstring.short_description:
            lines.extend([
                module.docstring.short_description,
                "",
            ])

        if module.docstring.long_description:
            lines.extend([
                module.docstring.long_description,
                "",
            ])

        # Classes
        if module.classes:
            lines.extend(["## Classes", ""])

            for cls in module.classes:
                lines.append(self.render_class(cls))
                lines.append("")

        # Functions
        if module.functions:
            lines.extend(["## Functions", ""])

            for func in module.functions:
                lines.append(self.render_function(func))
                lines.append("")

        # Constants
        if module.constants:
            lines.extend(["## Constants", ""])

            for const in module.constants:
                lines.append(f"### `{const['name']}`")
                if const.get("value"):
                    lines.append(f"```python\n{const['name']} = {const['value']}\n```")
                lines.append("")

        return "\n".join(lines)

    def render_class(self, cls: ClassInfo) -> str:
        """Render class to Markdown."""
        lines = [f"### `class {cls.name}`"]

        # Bases
        if cls.bases:
            bases = ", ".join(cls.bases)
            lines.append(f"Inherits from: {bases}")

        lines.append("")

        # Docstring
        if cls.docstring.short_description:
            lines.extend([cls.docstring.short_description, ""])

        if cls.docstring.long_description:
            lines.extend([cls.docstring.long_description, ""])

        # Attributes from docstring
        if cls.docstring.attributes:
            lines.append("**Attributes:**")
            for attr in cls.docstring.attributes:
                type_str = f" ({attr.type_hint})" if attr.type_hint else ""
                lines.append(f"- `{attr.name}`{type_str}: {attr.description}")
            lines.append("")

        # Methods
        all_methods = cls.methods + cls.class_methods + cls.static_methods

        if all_methods:
            lines.append("**Methods:**")
            lines.append("")

            for method in all_methods:
                lines.append(self._render_method(method))

        # Properties
        if cls.properties:
            lines.append("**Properties:**")
            lines.append("")

            for prop in cls.properties:
                lines.append(self._render_property(prop))

        return "\n".join(lines)

    def render_function(self, func: FunctionInfo) -> str:
        """Render function to Markdown."""
        # Function header
        prefix = "async " if func.is_async else ""
        lines = [f"### `{prefix}def {func.name}{func.signature}`"]
        lines.append("")

        # Docstring
        if func.docstring.short_description:
            lines.extend([func.docstring.short_description, ""])

        if func.docstring.long_description:
            lines.extend([func.docstring.long_description, ""])

        # Parameters
        if func.docstring.params:
            lines.append("**Parameters:**")
            lines.append("")
            for param in func.docstring.params:
                type_str = f" ({param.type_hint})" if param.type_hint else ""
                lines.append(f"- `{param.name}`{type_str}: {param.description}")
            lines.append("")

        # Returns
        if func.docstring.returns:
            lines.append("**Returns:**")
            type_str = f" ({func.docstring.returns.type_hint})" if func.docstring.returns.type_hint else ""
            lines.append(f"- {type_str} {func.docstring.returns.description}")
            lines.append("")

        # Raises
        if func.docstring.raises:
            lines.append("**Raises:**")
            for exc in func.docstring.raises:
                lines.append(f"- `{exc.exception_type}`: {exc.description}")
            lines.append("")

        # Examples
        if func.docstring.examples:
            lines.append("**Example:**")
            lines.append("")
            for example in func.docstring.examples:
                lines.append(f"```{example.language}")
                lines.append(example.code)
                lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def _render_method(self, method: FunctionInfo) -> str:
        """Render method documentation."""
        prefix = ""
        if method.is_classmethod:
            prefix = "@classmethod "
        elif method.is_staticmethod:
            prefix = "@staticmethod "
        elif method.is_async:
            prefix = "async "

        lines = [f"#### `{prefix}{method.name}{method.signature}`"]

        if method.docstring.short_description:
            lines.append(method.docstring.short_description)

        lines.append("")
        return "\n".join(lines)

    def _render_property(self, prop: FunctionInfo) -> str:
        """Render property documentation."""
        lines = [f"#### `{prop.name}` (property)"]

        if prop.docstring.short_description:
            lines.append(prop.docstring.short_description)

        lines.append("")
        return "\n".join(lines)


class MkDocsRenderer(MarkdownRenderer):
    """Renderer for MkDocs-compatible output."""

    format = DocFormat.MKDOCS

    def render_module(self, module: ModuleInfo) -> str:
        """Render module with MkDocs-specific features."""
        lines = [
            "---",
            f"title: {module.name}",
            "---",
            "",
        ]

        # Add breadcrumb
        parts = module.qualified_name.split(".")
        if len(parts) > 1:
            breadcrumb = " / ".join(parts[:-1])
            lines.extend([
                f"*{breadcrumb}*",
                "",
            ])

        lines.append(super().render_module(module))
        return "\n".join(lines)

    def render_class(self, cls: ClassInfo) -> str:
        """Render class with MkDocs admonitions."""
        content = super().render_class(cls)

        # Add deprecated warning if applicable
        if cls.docstring.deprecated:
            warning = f'!!! warning "Deprecated"\n    {cls.docstring.deprecated}\n\n'
            content = warning + content

        return content

    def _write_config_files(self, output_dir: Path) -> list[Path]:
        """Write mkdocs.yml configuration."""
        written = []

        # mkdocs.yml
        config = {
            "site_name": self.config.title,
            "site_description": self.config.description,
            "theme": {
                "name": self.config.theme,
                "features": [
                    "navigation.tabs",
                    "navigation.sections",
                    "navigation.expand",
                    "search.suggest",
                    "content.code.copy",
                ],
            },
            "markdown_extensions": [
                "admonition",
                "pymdownx.details",
                "pymdownx.superfences",
                "pymdownx.highlight",
                "pymdownx.inlinehilite",
                "toc",
            ],
        }

        if self.config.extra_css:
            config["extra_css"] = self.config.extra_css

        if self.config.extra_js:
            config["extra_javascript"] = self.config.extra_js

        # Write as YAML
        yaml_path = output_dir.parent / "mkdocs.yml"

        try:
            import yaml
            yaml_content = yaml.dump(config, default_flow_style=False)
        except ImportError:
            # Simple YAML generation
            yaml_content = self._simple_yaml_dump(config)

        yaml_path.write_text(yaml_content, encoding="utf-8")
        written.append(yaml_path)

        return written

    def _simple_yaml_dump(self, data: Any, indent: int = 0) -> str:
        """Simple YAML serialization without external dependencies."""
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._simple_yaml_dump(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {self._yaml_value(value)}")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.append(f"{prefix}-")
                    for key, value in item.items():
                        lines.append(f"{prefix}  {key}: {self._yaml_value(value)}")
                else:
                    lines.append(f"{prefix}- {self._yaml_value(item)}")

        return "\n".join(lines)

    def _yaml_value(self, value: Any) -> str:
        """Format a value for YAML."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            if any(c in value for c in ":#{}[]&*!|>'\""):
                return f'"{value}"'
            return value
        return str(value)


class SphinxRenderer(MarkdownRenderer):
    """Renderer for Sphinx-compatible reStructuredText output."""

    format = DocFormat.SPHINX

    def render_module(self, module: ModuleInfo) -> str:
        """Render module to RST format."""
        lines = [
            self._rst_title(module.qualified_name, "="),
            "",
        ]

        if module.docstring.short_description:
            lines.extend([module.docstring.short_description, ""])

        if module.docstring.long_description:
            lines.extend([module.docstring.long_description, ""])

        # Add autodoc directive
        lines.extend([
            f".. automodule:: {module.qualified_name}",
            "   :members:",
            "   :undoc-members:",
            "   :show-inheritance:",
            "",
        ])

        return "\n".join(lines)

    def render_class(self, cls: ClassInfo) -> str:
        """Render class to RST format."""
        lines = [
            self._rst_title(f"class {cls.name}", "-"),
            "",
        ]

        # Class directive
        lines.extend([
            f".. py:class:: {cls.name}",
            "",
        ])

        if cls.docstring.short_description:
            lines.extend([f"   {cls.docstring.short_description}", ""])

        return "\n".join(lines)

    def render_function(self, func: FunctionInfo) -> str:
        """Render function to RST format."""
        lines = [
            self._rst_title(f"{func.name}()", "-"),
            "",
        ]

        prefix = "async " if func.is_async else ""
        lines.extend([
            f".. py:function:: {prefix}{func.name}{func.signature}",
            "",
        ])

        if func.docstring.short_description:
            lines.extend([f"   {func.docstring.short_description}", ""])

        # Parameters
        for param in func.docstring.params:
            type_str = f" ({param.type_hint})" if param.type_hint else ""
            lines.append(f"   :param {param.name}{type_str}: {param.description}")

        # Returns
        if func.docstring.returns:
            lines.append(f"   :returns: {func.docstring.returns.description}")
            if func.docstring.returns.type_hint:
                lines.append(f"   :rtype: {func.docstring.returns.type_hint}")

        # Raises
        for exc in func.docstring.raises:
            lines.append(f"   :raises {exc.exception_type}: {exc.description}")

        lines.append("")
        return "\n".join(lines)

    def _rst_title(self, title: str, char: str) -> str:
        """Create RST title with underline."""
        return f"{title}\n{char * len(title)}"

    def _module_path(self, module: ModuleInfo) -> str:
        """Get RST file path for module."""
        parts = module.qualified_name.split(".")
        return "/".join(parts) + ".rst"

    def _write_config_files(self, output_dir: Path) -> list[Path]:
        """Write Sphinx configuration files."""
        written = []

        # conf.py
        conf_content = f'''
"""Sphinx configuration file."""

project = "{self.config.title}"
copyright = "2024"
author = "Truthound"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_static_path = ["_static"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
'''

        conf_path = output_dir / "conf.py"
        conf_path.write_text(conf_content, encoding="utf-8")
        written.append(conf_path)

        # index.rst
        index_content = f'''
{self.config.title}
{"=" * len(self.config.title)}

{self.config.description}

.. toctree::
   :maxdepth: {self.config.nav_depth}
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
'''

        index_path = output_dir / "index.rst"
        index_path.write_text(index_content, encoding="utf-8")
        written.append(index_path)

        return written


class HTMLRenderer(MarkdownRenderer):
    """Renderer for standalone HTML output."""

    format = DocFormat.HTML

    def __init__(self, config: RenderConfig | None = None):
        super().__init__(config)
        self._css = self._default_css()

    def render_module(self, module: ModuleInfo) -> str:
        """Render module to HTML."""
        md_content = super().render_module(module)

        # Convert markdown to HTML
        html_content = self._md_to_html(md_content)

        return self._wrap_html(module.qualified_name, html_content)

    def _md_to_html(self, md: str) -> str:
        """Convert markdown to HTML."""
        # Try using markdown library
        try:
            import markdown
            return markdown.markdown(
                md,
                extensions=["fenced_code", "tables", "toc"],
            )
        except ImportError:
            pass

        # Simple fallback conversion
        html = md

        # Headers
        html = self._convert_headers(html)

        # Code blocks
        html = self._convert_code_blocks(html)

        # Inline code
        html = html.replace("`", "<code>", 1).replace("`", "</code>", 1)

        # Bold
        html = html.replace("**", "<strong>", 1).replace("**", "</strong>", 1)

        # Lists
        html = self._convert_lists(html)

        # Paragraphs
        paragraphs = html.split("\n\n")
        html = "\n".join(f"<p>{p}</p>" if not p.startswith("<") else p for p in paragraphs)

        return html

    def _convert_headers(self, text: str) -> str:
        """Convert markdown headers to HTML."""
        import re
        text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
        text = re.sub(r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
        return text

    def _convert_code_blocks(self, text: str) -> str:
        """Convert fenced code blocks to HTML."""
        import re
        pattern = r"```(\w*)\n(.*?)```"
        return re.sub(
            pattern,
            r'<pre><code class="language-\1">\2</code></pre>',
            text,
            flags=re.DOTALL,
        )

    def _convert_lists(self, text: str) -> str:
        """Convert markdown lists to HTML."""
        import re
        lines = text.split("\n")
        in_list = False
        result = []

        for line in lines:
            if line.strip().startswith("- "):
                if not in_list:
                    result.append("<ul>")
                    in_list = True
                result.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    result.append("</ul>")
                    in_list = False
                result.append(line)

        if in_list:
            result.append("</ul>")

        return "\n".join(result)

    def _wrap_html(self, title: str, content: str) -> str:
        """Wrap content in HTML document."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {self.config.title}</title>
    <style>
{self._css}
    </style>
</head>
<body>
    <div class="container">
        <nav class="nav">
            <a href="index.html">{self.config.title}</a>
        </nav>
        <main class="content">
{content}
        </main>
        <footer>
            Generated by Truthound Documentation System
        </footer>
    </div>
</body>
</html>'''

    def _default_css(self) -> str:
        """Default CSS styles."""
        return '''
:root {
    --primary: #3498db;
    --text: #2c3e50;
    --bg: #ffffff;
    --code-bg: #f8f9fa;
    --border: #e9ecef;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    color: var(--text);
    background: var(--bg);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.nav {
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.nav a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 600;
}

h1, h2, h3, h4 {
    margin: 1.5rem 0 0.5rem;
}

h1 { font-size: 2rem; }
h2 { font-size: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
h3 { font-size: 1.25rem; }

p {
    margin: 0.5rem 0;
}

code {
    background: var(--code-bg);
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: "Monaco", "Menlo", monospace;
    font-size: 0.9em;
}

pre {
    background: var(--code-bg);
    padding: 1rem;
    border-radius: 5px;
    overflow-x: auto;
    margin: 1rem 0;
}

pre code {
    padding: 0;
    background: none;
}

ul {
    margin: 0.5rem 0;
    padding-left: 2rem;
}

li {
    margin: 0.25rem 0;
}

footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: #6c757d;
    font-size: 0.9rem;
}
'''

    def _module_path(self, module: ModuleInfo) -> str:
        """Get HTML file path for module."""
        parts = module.qualified_name.split(".")
        return "/".join(parts) + ".html"


class JSONRenderer(DocRenderer):
    """Renderer for JSON output (for programmatic access)."""

    format = DocFormat.JSON

    def render_module(self, module: ModuleInfo) -> str:
        """Render module to JSON."""
        return json.dumps(module.to_dict(), indent=2, ensure_ascii=False)

    def render_class(self, cls: ClassInfo) -> str:
        """Render class to JSON."""
        return json.dumps(cls.to_dict(), indent=2, ensure_ascii=False)

    def render_function(self, func: FunctionInfo) -> str:
        """Render function to JSON."""
        return json.dumps(func.to_dict(), indent=2, ensure_ascii=False)

    def render_package(self, package: PackageInfo) -> dict[str, str]:
        """Render package to JSON files."""
        files = {}

        # Full package JSON
        files["api.json"] = json.dumps(
            package.to_dict(),
            indent=2,
            ensure_ascii=False,
        )

        # Search index
        files["search-index.json"] = self._generate_search_index(package)

        return files

    def _generate_search_index(self, package: PackageInfo) -> str:
        """Generate search index for documentation."""
        index = []

        for module in package.modules:
            # Module entry
            index.append({
                "type": "module",
                "name": module.qualified_name,
                "title": module.name,
                "description": module.docstring.short_description,
                "path": self._module_path(module).replace(".md", ".html"),
            })

            # Class entries
            for cls in module.classes:
                index.append({
                    "type": "class",
                    "name": cls.qualified_name,
                    "title": cls.name,
                    "description": cls.docstring.short_description,
                    "path": self._module_path(module).replace(".md", ".html") + f"#{cls.name.lower()}",
                })

            # Function entries
            for func in module.functions:
                index.append({
                    "type": "function",
                    "name": func.qualified_name,
                    "title": func.name,
                    "description": func.docstring.short_description,
                    "path": self._module_path(module).replace(".md", ".html") + f"#{func.name.lower()}",
                })

        return json.dumps(index, indent=2, ensure_ascii=False)


# Renderer registry
RENDERERS: dict[DocFormat, type[DocRenderer]] = {
    DocFormat.MARKDOWN: MarkdownRenderer,
    DocFormat.MKDOCS: MkDocsRenderer,
    DocFormat.SPHINX: SphinxRenderer,
    DocFormat.HTML: HTMLRenderer,
    DocFormat.JSON: JSONRenderer,
}


def get_renderer(
    format: DocFormat | str,
    config: RenderConfig | None = None,
) -> DocRenderer:
    """Get a renderer for the specified format.

    Args:
        format: Output format
        config: Rendering configuration

    Returns:
        Appropriate renderer instance
    """
    if isinstance(format, str):
        format = DocFormat(format.lower())

    renderer_class = RENDERERS.get(format)
    if not renderer_class:
        raise ValueError(f"Unknown format: {format}. Available: {list(RENDERERS.keys())}")

    return renderer_class(config)
