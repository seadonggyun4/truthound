"""Jinja2-based template renderer for Data Docs.

This module provides a full-featured template renderer using Jinja2.
"""

from __future__ import annotations

import html as html_module
from pathlib import Path
from typing import Any, TYPE_CHECKING

from truthound.datadocs.renderers.base import BaseRenderer, RenderResult

if TYPE_CHECKING:
    from truthound.datadocs.engine.context import ReportContext
    from truthound.datadocs.themes.base import Theme


class JinjaTemplateLoader:
    """Loader for Jinja2 templates from various sources.

    Supports loading templates from:
    - Package resources (default templates)
    - File system paths
    - String templates
    """

    def __init__(
        self,
        template_dirs: list[Path] | None = None,
        package_path: str | None = None,
    ) -> None:
        """Initialize the template loader.

        Args:
            template_dirs: Additional directories to search for templates.
            package_path: Package path for default templates.
        """
        self._template_dirs = template_dirs or []
        self._package_path = package_path
        self._env = None

    def get_environment(self) -> Any:
        """Get or create the Jinja2 environment.

        Returns:
            Jinja2 Environment instance.
        """
        if self._env is not None:
            return self._env

        try:
            from jinja2 import Environment, FileSystemLoader, BaseLoader, ChoiceLoader
        except ImportError:
            raise ImportError(
                "Jinja2 is required for JinjaRenderer. "
                "Install with: pip install jinja2"
            )

        loaders = []

        # Add custom template directories
        for template_dir in self._template_dirs:
            if template_dir.exists():
                loaders.append(FileSystemLoader(str(template_dir)))

        # Add package templates if available
        if self._package_path:
            try:
                import importlib.resources as pkg_resources
                # For Python 3.9+
                ref = pkg_resources.files(self._package_path) / "templates"
                if hasattr(ref, "__fspath__"):
                    loaders.append(FileSystemLoader(str(ref)))
            except (ImportError, TypeError, AttributeError):
                pass

        # Create environment with appropriate loader
        if loaders:
            loader = ChoiceLoader(loaders) if len(loaders) > 1 else loaders[0]
            self._env = Environment(loader=loader, autoescape=True)
        else:
            # No file loaders, use string templates
            self._env = Environment(autoescape=True)

        # Add custom filters
        self._add_custom_filters(self._env)

        return self._env

    def _add_custom_filters(self, env: Any) -> None:
        """Add custom Jinja2 filters.

        Args:
            env: Jinja2 Environment.
        """
        def format_number(value: float | int, decimals: int = 2) -> str:
            """Format a number with thousands separator."""
            if isinstance(value, int):
                return f"{value:,}"
            return f"{value:,.{decimals}f}"

        def format_percent(value: float, decimals: int = 1) -> str:
            """Format a ratio as percentage."""
            return f"{value * 100:.{decimals}f}%"

        def format_bytes(value: int) -> str:
            """Format bytes in human-readable form."""
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if abs(value) < 1024.0:
                    return f"{value:.1f} {unit}"
                value /= 1024.0
            return f"{value:.1f} PB"

        def severity_class(severity: str) -> str:
            """Get CSS class for severity level."""
            return f"alert-{severity.lower()}"

        def truncate_text(text: str, length: int = 100) -> str:
            """Truncate text with ellipsis."""
            if len(text) <= length:
                return text
            return text[:length-3] + "..."

        env.filters["format_number"] = format_number
        env.filters["format_percent"] = format_percent
        env.filters["format_bytes"] = format_bytes
        env.filters["severity_class"] = severity_class
        env.filters["truncate"] = truncate_text

    def render_string(self, template_str: str, context: dict[str, Any]) -> str:
        """Render a string template.

        Args:
            template_str: Template string.
            context: Template context.

        Returns:
            Rendered string.
        """
        env = self.get_environment()
        template = env.from_string(template_str)
        return template.render(**context)

    def render_file(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template file.

        Args:
            template_name: Template file name.
            context: Template context.

        Returns:
            Rendered string.
        """
        env = self.get_environment()
        template = env.get_template(template_name)
        return template.render(**context)


class JinjaRenderer(BaseRenderer):
    """Jinja2-based template renderer.

    This is the primary renderer for production use, providing
    full template support with Jinja2.

    Example:
        renderer = JinjaRenderer(
            template_dirs=[Path("./templates")],
            default_template="report.html.j2",
        )
        html = renderer.render(ctx, theme)
    """

    # Default template for reports
    DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="{{ locale }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <meta name="generator" content="Truthound">
    <style>
{{ theme_css }}
{{ base_css }}
{{ custom_css }}
    </style>
    {{ head_scripts }}
</head>
<body class="theme-{{ theme_name }}">
    <div class="report-container">
        {% if show_header %}
        <header class="report-header">
            <div class="report-header-main">
                <div>
                    <h1 class="report-title">{{ title }}</h1>
                    {% if subtitle %}
                    <p class="report-subtitle">{{ subtitle }}</p>
                    {% endif %}
                </div>
                {% if logo_url %}
                <img src="{{ logo_url }}" alt="Logo" class="report-logo">
                {% endif %}
            </div>
            {% if metadata.generated_at %}
            <div class="report-meta">
                <span class="report-meta-item">Generated: {{ metadata.generated_at }}</span>
            </div>
            {% endif %}
        </header>
        {% endif %}

        {% if show_toc and sections %}
        <nav class="report-toc">
            <h3 class="toc-title">Contents</h3>
            <ul class="toc-list">
                {% for section_name in sections.keys() %}
                <li class="toc-item">
                    <a href="#section-{{ section_name }}">{{ section_name | replace('_', ' ') | title }}</a>
                </li>
                {% endfor %}
            </ul>
        </nav>
        {% endif %}

        <main class="report-main">
            {% if quality_score %}
            <section class="report-section quality-overview" id="section-quality-score">
                <h2>Quality Score</h2>
                <div class="quality-score-display">
                    <div class="score-circle score-{{ quality_score.grade | lower }}">
                        <span class="score-value">{{ quality_score.overall }}</span>
                        <span class="score-grade">{{ quality_score.grade }}</span>
                    </div>
                    <div class="score-dimensions">
                        {% for dim, value in quality_score.dimensions.items() %}
                        <div class="dimension">
                            <span class="dimension-name">{{ dim | title }}</span>
                            <div class="dimension-bar">
                                <div class="dimension-fill" style="width: {{ value }}%"></div>
                            </div>
                            <span class="dimension-value">{{ value }}%</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </section>
            {% endif %}

            {% for section_name, section_data in sections.items() %}
            <section class="report-section" id="section-{{ section_name }}">
                <h2>{{ section_name | replace('_', ' ') | title }}</h2>
                <div class="section-content">
                    {% if section_data is mapping %}
                    <dl class="data-list">
                        {% for key, value in section_data.items() %}
                        <dt>{{ key | replace('_', ' ') | title }}</dt>
                        <dd>
                            {% if value is number %}
                                {{ value | format_number }}
                            {% elif value is iterable and value is not string %}
                                <ul>
                                    {% for item in value %}
                                    <li>{{ item }}</li>
                                    {% endfor %}
                                </ul>
                            {% else %}
                                {{ value }}
                            {% endif %}
                        </dd>
                        {% endfor %}
                    </dl>
                    {% elif section_data is iterable and section_data is not string %}
                    <ul class="data-list">
                        {% for item in section_data %}
                        <li>{{ item }}</li>
                        {% endfor %}
                    </ul>
                    {% else %}
                    <p>{{ section_data }}</p>
                    {% endif %}
                </div>
            </section>
            {% endfor %}

            {% if alerts %}
            <section class="report-section" id="section-alerts">
                <h2>Alerts</h2>
                <div class="alerts-container">
                    {% for alert in alerts %}
                    <div class="alert {{ alert.severity | severity_class }}">
                        <div class="alert-header">
                            <span class="alert-title">{{ alert.title }}</span>
                            <span class="alert-severity">{{ alert.severity | upper }}</span>
                        </div>
                        <div class="alert-message">{{ alert.message }}</div>
                        {% if alert.suggestion %}
                        <div class="alert-suggestion">{{ alert.suggestion }}</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </section>
            {% endif %}

            {% if recommendations %}
            <section class="report-section" id="section-recommendations">
                <h2>Recommendations</h2>
                <ul class="recommendations-list">
                    {% for rec in recommendations %}
                    <li class="recommendation-item">{{ rec }}</li>
                    {% endfor %}
                </ul>
            </section>
            {% endif %}
        </main>

        {% if show_footer %}
        <footer class="report-footer">
            <p>{{ footer_text }}</p>
        </footer>
        {% endif %}
    </div>
    {{ body_scripts }}
</body>
</html>"""

    # Default CSS
    DEFAULT_CSS = """
* { box-sizing: border-box; }
body {
    font-family: var(--font-family, system-ui, -apple-system, sans-serif);
    line-height: 1.6;
    color: var(--color-text-primary, #1a1a2e);
    background: var(--color-background, #f5f5f5);
    margin: 0;
    padding: 0;
}
.report-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}
.report-header {
    background: var(--color-surface, white);
    padding: 2rem;
    border-radius: var(--border-radius-lg, 12px);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md, 0 4px 6px rgba(0,0,0,0.1));
}
.report-header-main {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.report-title {
    margin: 0;
    font-size: 2rem;
    color: var(--color-text-primary, #1a1a2e);
}
.report-subtitle {
    color: var(--color-text-secondary, #666);
    margin-top: 0.5rem;
    margin-bottom: 0;
}
.report-logo {
    max-height: 60px;
    max-width: 200px;
}
.report-meta {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--color-border, #e5e7eb);
    color: var(--color-text-secondary, #666);
    font-size: 0.875rem;
}
.report-toc {
    background: var(--color-surface, white);
    padding: 1.5rem;
    border-radius: var(--border-radius-md, 8px);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-sm, 0 2px 4px rgba(0,0,0,0.05));
}
.toc-title {
    margin: 0 0 1rem 0;
    font-size: 1rem;
    color: var(--color-text-secondary, #666);
}
.toc-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.toc-item a {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    background: var(--color-background, #f5f5f5);
    border-radius: var(--border-radius-sm, 4px);
    color: var(--color-primary, #2563eb);
    text-decoration: none;
    font-size: 0.875rem;
}
.toc-item a:hover {
    background: var(--color-primary, #2563eb);
    color: white;
}
.report-section {
    background: var(--color-surface, white);
    padding: 1.5rem;
    border-radius: var(--border-radius-md, 8px);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm, 0 2px 4px rgba(0,0,0,0.05));
}
.report-section h2 {
    margin-top: 0;
    color: var(--color-primary, #2563eb);
    border-bottom: 2px solid var(--color-border, #e5e7eb);
    padding-bottom: 0.5rem;
    font-size: 1.25rem;
}
.quality-score-display {
    display: flex;
    gap: 2rem;
    align-items: center;
}
.score-circle {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: var(--color-success, #22c55e);
}
.score-circle.score-a { background: var(--color-success, #22c55e); }
.score-circle.score-b { background: #84cc16; }
.score-circle.score-c { background: var(--color-warning, #f59e0b); }
.score-circle.score-d { background: var(--color-error, #ef4444); }
.score-value {
    font-size: 2rem;
    font-weight: bold;
    color: white;
}
.score-grade {
    font-size: 1rem;
    color: rgba(255,255,255,0.9);
}
.score-dimensions {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.dimension {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.dimension-name {
    width: 100px;
    font-size: 0.875rem;
    color: var(--color-text-secondary, #666);
}
.dimension-bar {
    flex: 1;
    height: 8px;
    background: var(--color-border, #e5e7eb);
    border-radius: 4px;
    overflow: hidden;
}
.dimension-fill {
    height: 100%;
    background: var(--color-primary, #2563eb);
    border-radius: 4px;
}
.dimension-value {
    width: 50px;
    text-align: right;
    font-size: 0.875rem;
    font-weight: 500;
}
.data-list {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.5rem 1rem;
}
.data-list dt {
    font-weight: 600;
    color: var(--color-text-secondary, #666);
}
.data-list dd {
    margin: 0;
}
.alerts-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.alert {
    padding: 1rem;
    border-radius: var(--border-radius-sm, 4px);
    border-left: 4px solid;
}
.alert-info {
    background: #e0f2fe;
    border-color: #0ea5e9;
    color: #0369a1;
}
.alert-warning {
    background: #fef3c7;
    border-color: #f59e0b;
    color: #92400e;
}
.alert-error {
    background: #fee2e2;
    border-color: #ef4444;
    color: #991b1b;
}
.alert-critical {
    background: #fce7f3;
    border-color: #ec4899;
    color: #9d174d;
}
.alert-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.alert-title {
    font-weight: 600;
}
.alert-severity {
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.125rem 0.5rem;
    border-radius: 9999px;
    background: rgba(0,0,0,0.1);
}
.alert-message {
    font-size: 0.875rem;
}
.alert-suggestion {
    margin-top: 0.5rem;
    font-size: 0.875rem;
    font-style: italic;
    opacity: 0.9;
}
.recommendations-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.recommendation-item {
    padding: 0.75rem;
    background: var(--color-background, #f5f5f5);
    border-radius: var(--border-radius-sm, 4px);
    margin-bottom: 0.5rem;
}
.recommendation-item::before {
    content: "💡";
    margin-right: 0.5rem;
}
.report-footer {
    text-align: center;
    padding: 2rem;
    color: var(--color-text-secondary, #666);
    font-size: 0.875rem;
}
@media print {
    .report-container { max-width: none; padding: 0; }
    .report-section { break-inside: avoid; box-shadow: none; border: 1px solid #ddd; }
    .report-toc { display: none; }
}
"""

    def __init__(
        self,
        template_dirs: list[Path] | None = None,
        default_template: str | None = None,
        custom_css: str = "",
        show_header: bool = True,
        show_footer: bool = True,
        show_toc: bool = True,
        footer_text: str = "Generated by Truthound",
        name: str | None = None,
    ) -> None:
        """Initialize the Jinja renderer.

        Args:
            template_dirs: Directories to search for templates.
            default_template: Default template name or string.
            custom_css: Additional CSS to include.
            show_header: Show report header.
            show_footer: Show report footer.
            show_toc: Show table of contents.
            footer_text: Footer text.
            name: Renderer name.
        """
        super().__init__(name=name or "JinjaRenderer")
        self._loader = JinjaTemplateLoader(template_dirs=template_dirs)
        self._default_template = default_template or self.DEFAULT_TEMPLATE
        self._custom_css = custom_css
        self._show_header = show_header
        self._show_footer = show_footer
        self._show_toc = show_toc
        self._footer_text = footer_text

    def _do_render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> str:
        """Render using Jinja2.

        Args:
            ctx: Report context.
            theme: Theme configuration.

        Returns:
            Rendered HTML.
        """
        data = ctx.data

        # Get theme CSS and configuration
        theme_css = ""
        theme_name = ctx.theme
        logo_url = ctx.get_option("logo_url")

        if theme:
            try:
                theme_css = theme.get_css()
                theme_name = getattr(theme, "name", theme_name)
                if not logo_url:
                    logo_url = getattr(theme, "logo_url", None)
            except (AttributeError, Exception):
                pass

        # Build template context
        template_context = {
            # Basic info
            "title": ctx.title,
            "subtitle": ctx.subtitle,
            "locale": ctx.locale,
            "theme_name": theme_name,
            # Styling
            "theme_css": theme_css,
            "base_css": self.DEFAULT_CSS,
            "custom_css": self._custom_css,
            # Layout options
            "show_header": self._show_header,
            "show_footer": self._show_footer,
            "show_toc": self._show_toc,
            "footer_text": self._footer_text,
            "logo_url": logo_url,
            # Data
            "metadata": data.metadata,
            "sections": data.sections,
            "alerts": data.alerts,
            "recommendations": data.recommendations,
            "charts": data.charts,
            "tables": data.tables,
            "raw": data.raw,
            # Quality score (if present)
            "quality_score": data.metadata.get("quality_score"),
            # Context options
            "options": ctx.options,
            # Scripts (empty by default)
            "head_scripts": "",
            "body_scripts": "",
        }

        # Render template
        return self._loader.render_string(self._default_template, template_context)

    def with_template(self, template: str) -> "JinjaRenderer":
        """Create a new renderer with a different template.

        Args:
            template: Template string or file name.

        Returns:
            New renderer instance.
        """
        new_renderer = JinjaRenderer(
            template_dirs=self._loader._template_dirs,
            default_template=template,
            custom_css=self._custom_css,
            show_header=self._show_header,
            show_footer=self._show_footer,
            show_toc=self._show_toc,
            footer_text=self._footer_text,
        )
        return new_renderer
