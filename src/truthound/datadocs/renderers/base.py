"""Base classes and protocols for renderers.

This module defines the core abstractions for template renderers
in the report generation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.datadocs.engine.context import ReportContext
    from truthound.datadocs.themes.base import Theme


@dataclass
class RenderResult:
    """Result of a rendering operation.

    Attributes:
        content: The rendered HTML content.
        metadata: Additional metadata about rendering.
        warnings: Any warnings generated during rendering.
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Renderer(Protocol):
    """Protocol for template renderers.

    Renderers receive a ReportContext and Theme, and produce HTML output.
    """

    def render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None" = None,
    ) -> str:
        """Render the report context to HTML.

        Args:
            ctx: Report context with data.
            theme: Optional theme for styling.

        Returns:
            Rendered HTML string.
        """
        ...


class BaseRenderer(ABC):
    """Abstract base class for renderers.

    Provides common functionality and ensures consistent interface.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the renderer.

        Args:
            name: Optional name for this renderer instance.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the renderer name."""
        return self._name

    def render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None" = None,
    ) -> str:
        """Render the report context to HTML.

        Args:
            ctx: Report context with data.
            theme: Optional theme for styling.

        Returns:
            Rendered HTML string.
        """
        try:
            result = self._do_render(ctx, theme)

            if isinstance(result, RenderResult):
                return result.content

            return result

        except Exception as e:
            return self._render_error(ctx, e)

    @abstractmethod
    def _do_render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> str | RenderResult:
        """Perform the actual rendering.

        Subclasses implement this method.

        Args:
            ctx: Report context.
            theme: Theme configuration.

        Returns:
            Rendered HTML or RenderResult.
        """
        pass

    def _render_error(
        self,
        ctx: "ReportContext",
        error: Exception,
    ) -> str:
        """Render an error page when rendering fails.

        Args:
            ctx: Original context.
            error: The exception that occurred.

        Returns:
            Error HTML.
        """
        return f"""<!DOCTYPE html>
<html lang="{ctx.locale}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rendering Error</title>
    <style>
        body {{
            font-family: system-ui, sans-serif;
            padding: 2rem;
            background: #fef2f2;
            color: #991b1b;
        }}
        .error-container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1 {{ color: #dc2626; }}
        pre {{
            background: #fee2e2;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <h1>Rendering Error</h1>
        <p>An error occurred while rendering the report.</p>
        <pre>{type(error).__name__}: {str(error)}</pre>
    </div>
</body>
</html>"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


class SimpleRenderer(BaseRenderer):
    """Simple HTML renderer without template engine.

    Generates basic HTML directly from the context data.
    Useful for testing or when no template engine is available.
    """

    def __init__(
        self,
        include_styles: bool = True,
        include_scripts: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize the simple renderer.

        Args:
            include_styles: Include basic CSS styles.
            include_scripts: Include JavaScript (if any).
            name: Renderer name.
        """
        super().__init__(name=name or "SimpleRenderer")
        self._include_styles = include_styles
        self._include_scripts = include_scripts

    def _do_render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> str:
        """Render simple HTML.

        Args:
            ctx: Report context.
            theme: Theme (used for colors if provided).

        Returns:
            Basic HTML string.
        """
        import html as html_module

        data = ctx.data
        title = html_module.escape(ctx.title)
        subtitle = html_module.escape(ctx.subtitle)

        # Get theme CSS if available
        theme_css = ""
        if theme:
            try:
                theme_css = theme.get_css()
            except (AttributeError, Exception):
                pass

        # Build sections
        sections_html = self._render_sections(data.sections)

        # Build alerts
        alerts_html = self._render_alerts(data.alerts)

        # Build recommendations
        recs_html = self._render_recommendations(data.recommendations)

        # Basic styles
        styles = self._get_basic_styles() if self._include_styles else ""

        return f"""<!DOCTYPE html>
<html lang="{ctx.locale}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{theme_css}
{styles}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1 class="report-title">{title}</h1>
            {f'<p class="report-subtitle">{subtitle}</p>' if subtitle else ''}
        </header>

        <main class="report-main">
            {sections_html}
            {alerts_html}
            {recs_html}
        </main>

        <footer class="report-footer">
            <p>Generated by Truthound</p>
        </footer>
    </div>
</body>
</html>"""

    def _render_sections(self, sections: dict[str, Any]) -> str:
        """Render all sections."""
        if not sections:
            return ""

        html_parts = []
        for name, section in sections.items():
            html_parts.append(f"""
            <section class="report-section" id="section-{name}">
                <h2>{name.replace('_', ' ').title()}</h2>
                <div class="section-content">
                    {self._render_section_content(section)}
                </div>
            </section>
            """)

        return "\n".join(html_parts)

    def _render_section_content(self, section: Any) -> str:
        """Render section content."""
        if isinstance(section, dict):
            items = []
            for key, value in section.items():
                items.append(f"<dt>{key}</dt><dd>{value}</dd>")
            return f"<dl>{''.join(items)}</dl>"
        elif isinstance(section, list):
            items = [f"<li>{item}</li>" for item in section]
            return f"<ul>{''.join(items)}</ul>"
        else:
            return f"<p>{section}</p>"

    def _render_alerts(self, alerts: list[dict[str, Any]]) -> str:
        """Render alerts section."""
        if not alerts:
            return ""

        alert_items = []
        for alert in alerts:
            severity = alert.get("severity", "info")
            title = alert.get("title", "Alert")
            message = alert.get("message", "")
            alert_items.append(f"""
            <div class="alert alert-{severity}">
                <strong>{title}</strong>
                <p>{message}</p>
            </div>
            """)

        return f"""
        <section class="report-section" id="section-alerts">
            <h2>Alerts</h2>
            {''.join(alert_items)}
        </section>
        """

    def _render_recommendations(self, recommendations: list[str]) -> str:
        """Render recommendations section."""
        if not recommendations:
            return ""

        items = [f"<li>{rec}</li>" for rec in recommendations]
        return f"""
        <section class="report-section" id="section-recommendations">
            <h2>Recommendations</h2>
            <ul>{''.join(items)}</ul>
        </section>
        """

    def _get_basic_styles(self) -> str:
        """Get basic CSS styles."""
        return """
        * { box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .report-header {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .report-title {
            margin: 0;
            color: #1a1a2e;
        }
        .report-subtitle {
            color: #666;
            margin-top: 0.5rem;
        }
        .report-section {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .report-section h2 {
            margin-top: 0;
            color: #2563eb;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 0.5rem;
        }
        .alert {
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        .alert-info { background: #e0f2fe; color: #0369a1; }
        .alert-warning { background: #fef3c7; color: #92400e; }
        .alert-error { background: #fee2e2; color: #991b1b; }
        .alert-critical { background: #fce7f3; color: #9d174d; }
        .report-footer {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        dl { display: grid; grid-template-columns: auto 1fr; gap: 0.5rem 1rem; }
        dt { font-weight: 600; }
        dd { margin: 0; }
        """
