"""Custom template renderers for Data Docs.

This module provides renderers that support user-defined templates
from various sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from truthound.datadocs.renderers.base import BaseRenderer

if TYPE_CHECKING:
    from truthound.datadocs.engine.context import ReportContext
    from truthound.datadocs.themes.base import Theme


class CustomRenderer(BaseRenderer):
    """Base class for custom template renderers.

    Provides a framework for implementing custom template rendering
    with any template engine.

    Example:
        class MyRenderer(CustomRenderer):
            def _do_render(self, ctx, theme):
                return my_template_engine.render(
                    self._template,
                    self._build_context(ctx, theme),
                )
    """

    def __init__(
        self,
        name: str | None = None,
        context_builder: Callable[["ReportContext", "Theme | None"], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize custom renderer.

        Args:
            name: Renderer name.
            context_builder: Optional function to build template context.
        """
        super().__init__(name=name or "CustomRenderer")
        self._context_builder = context_builder

    def _build_context(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> dict[str, Any]:
        """Build the template context.

        Can be overridden by subclasses or customized via context_builder.

        Args:
            ctx: Report context.
            theme: Theme configuration.

        Returns:
            Template context dictionary.
        """
        if self._context_builder:
            return self._context_builder(ctx, theme)

        data = ctx.data
        theme_css = ""
        if theme:
            try:
                theme_css = theme.get_css()
            except (AttributeError, Exception):
                pass

        return {
            "title": ctx.title,
            "subtitle": ctx.subtitle,
            "locale": ctx.locale,
            "theme": ctx.theme,
            "theme_css": theme_css,
            "metadata": data.metadata,
            "sections": data.sections,
            "alerts": data.alerts,
            "recommendations": data.recommendations,
            "charts": data.charts,
            "tables": data.tables,
            "raw": data.raw,
            "options": ctx.options,
        }


class StringTemplateRenderer(CustomRenderer):
    """Renderer that uses a string template.

    Supports simple string formatting with {key} placeholders.

    Example:
        renderer = StringTemplateRenderer(
            template=\"\"\"
            <html>
                <head><title>{title}</title></head>
                <body>{content}</body>
            </html>
            \"\"\",
        )
    """

    def __init__(
        self,
        template: str,
        name: str | None = None,
        safe_mode: bool = True,
    ) -> None:
        """Initialize with a template string.

        Args:
            template: Template string with {key} placeholders.
            name: Renderer name.
            safe_mode: If True, HTML-escape values.
        """
        super().__init__(name=name or "StringTemplateRenderer")
        self._template = template
        self._safe_mode = safe_mode

    def _do_render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> str:
        """Render using string formatting.

        Args:
            ctx: Report context.
            theme: Theme configuration.

        Returns:
            Rendered string.
        """
        import html as html_module

        context = self._build_context(ctx, theme)

        if self._safe_mode:
            # Escape string values for HTML
            safe_context = {}
            for key, value in context.items():
                if isinstance(value, str):
                    safe_context[key] = html_module.escape(value)
                else:
                    safe_context[key] = value
            context = safe_context

        # Simple string formatting
        try:
            return self._template.format(**context)
        except KeyError as e:
            # Return template with error message for missing keys
            return f"<!-- Template error: missing key {e} -->\n{self._template}"


class FileTemplateRenderer(CustomRenderer):
    """Renderer that loads templates from files.

    Supports both Jinja2 and simple string templates.

    Example:
        renderer = FileTemplateRenderer(
            template_path=Path("./templates/report.html.j2"),
            engine="jinja2",
        )
    """

    def __init__(
        self,
        template_path: Path | str,
        engine: str = "auto",
        name: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Initialize with a template file path.

        Args:
            template_path: Path to the template file.
            engine: Template engine ("jinja2", "string", or "auto").
            name: Renderer name.
            encoding: File encoding.
        """
        super().__init__(name=name or "FileTemplateRenderer")
        self._template_path = Path(template_path)
        self._engine = engine
        self._encoding = encoding
        self._template_content: str | None = None
        self._jinja_env: Any = None

    def _load_template(self) -> str:
        """Load the template content from file.

        Returns:
            Template string.
        """
        if self._template_content is None:
            self._template_content = self._template_path.read_text(
                encoding=self._encoding
            )
        return self._template_content

    def _detect_engine(self) -> str:
        """Detect the template engine based on file extension.

        Returns:
            Engine name.
        """
        if self._engine != "auto":
            return self._engine

        suffix = self._template_path.suffix.lower()
        if suffix in (".j2", ".jinja", ".jinja2"):
            return "jinja2"
        elif suffix in (".html", ".htm"):
            # Check for Jinja2 syntax
            content = self._load_template()
            if "{{" in content or "{%" in content:
                return "jinja2"
        return "string"

    def _do_render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> str:
        """Render using the appropriate engine.

        Args:
            ctx: Report context.
            theme: Theme configuration.

        Returns:
            Rendered string.
        """
        engine = self._detect_engine()
        context = self._build_context(ctx, theme)

        if engine == "jinja2":
            return self._render_jinja2(context)
        else:
            return self._render_string(context)

    def _render_jinja2(self, context: dict[str, Any]) -> str:
        """Render using Jinja2.

        Args:
            context: Template context.

        Returns:
            Rendered string.
        """
        try:
            from jinja2 import Environment, FileSystemLoader
        except ImportError:
            raise ImportError(
                "Jinja2 is required for .j2 templates. "
                "Install with: pip install jinja2"
            )

        if self._jinja_env is None:
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self._template_path.parent)),
                autoescape=True,
            )

        template = self._jinja_env.get_template(self._template_path.name)
        return template.render(**context)

    def _render_string(self, context: dict[str, Any]) -> str:
        """Render using string formatting.

        Args:
            context: Template context.

        Returns:
            Rendered string.
        """
        import html as html_module

        template = self._load_template()

        # Escape string values
        safe_context = {}
        for key, value in context.items():
            if isinstance(value, str):
                safe_context[key] = html_module.escape(value)
            else:
                safe_context[key] = value

        try:
            return template.format(**safe_context)
        except KeyError as e:
            return f"<!-- Template error: missing key {e} -->\n{template}"


class CallableRenderer(CustomRenderer):
    """Renderer that uses a callable for rendering.

    Provides maximum flexibility by accepting any callable
    that takes context and theme and returns HTML.

    Example:
        def my_render_func(ctx, theme):
            return f"<html><body>{ctx.title}</body></html>"

        renderer = CallableRenderer(render_func=my_render_func)
    """

    def __init__(
        self,
        render_func: Callable[["ReportContext", "Theme | None"], str],
        name: str | None = None,
    ) -> None:
        """Initialize with a render function.

        Args:
            render_func: Function that performs rendering.
            name: Renderer name.
        """
        super().__init__(name=name or "CallableRenderer")
        self._render_func = render_func

    def _do_render(
        self,
        ctx: "ReportContext",
        theme: "Theme | None",
    ) -> str:
        """Delegate to the render function.

        Args:
            ctx: Report context.
            theme: Theme configuration.

        Returns:
            Rendered string.
        """
        return self._render_func(ctx, theme)
