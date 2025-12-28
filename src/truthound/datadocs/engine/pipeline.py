"""Report Generation Pipeline for Data Docs.

This module provides the core pipeline orchestration for report generation.
The pipeline follows a composable, fluent API pattern.

Pipeline Stages:
1. Transform: Apply data transformations (i18n, filtering, enrichment)
2. Render: Generate intermediate HTML using templates and themes
3. Export: Convert to final format (HTML, PDF, Markdown, JSON)

Example:
    # Simple usage
    report = (
        ReportPipeline()
        .generate(ReportContext.from_profile(profile_data))
    )

    # Advanced usage with customization
    report = (
        ReportPipeline()
        .transform(I18nTransformer(catalog))
        .transform(FilterTransformer(sections=["overview", "quality"]))
        .render_with("jinja")
        .theme("enterprise")
        .export_as("pdf")
        .with_option("page_size", "A4")
        .generate(ctx)
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from truthound.datadocs.engine.context import ReportContext
from truthound.datadocs.engine.registry import component_registry, ComponentRegistry

if TYPE_CHECKING:
    from truthound.datadocs.transformers.base import Transformer
    from truthound.datadocs.renderers.base import Renderer
    from truthound.datadocs.themes.base import Theme
    from truthound.datadocs.exporters.base import Exporter


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        content: Generated report content (bytes or string).
        format: Output format (html, pdf, markdown, json).
        context: Final context after all transformations.
        metadata: Additional metadata about the generation.
        elapsed_ms: Time taken for generation in milliseconds.
        success: Whether generation was successful.
        error: Error message if generation failed.
    """
    content: bytes | str
    format: str
    context: ReportContext
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def is_binary(self) -> bool:
        """Check if content is binary (e.g., PDF)."""
        return isinstance(self.content, bytes)

    def save(self, path: str) -> None:
        """Save the result to a file.

        Args:
            path: Output file path.
        """
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if self.is_binary:
            p.write_bytes(self.content)
        else:
            p.write_text(self.content, encoding="utf-8")


class ReportPipeline:
    """Composable report generation pipeline.

    The pipeline orchestrates the flow of data through transformers,
    renderer, and exporter to produce the final report.

    The pipeline is immutable - each modification returns a new pipeline instance.
    This allows for safe reuse and composition.

    Example:
        # Create a base pipeline
        base_pipeline = (
            ReportPipeline()
            .theme("enterprise")
            .export_as("html")
        )

        # Derive specific pipelines
        korean_pipeline = base_pipeline.transform(I18nTransformer(locale="ko"))
        english_pipeline = base_pipeline.transform(I18nTransformer(locale="en"))
    """

    def __init__(
        self,
        registry: ComponentRegistry | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            registry: Component registry to use (default: global registry).
        """
        self._registry = registry or component_registry
        self._transformers: list[Any] = []
        self._renderer_name: str = "jinja"
        self._renderer_kwargs: dict[str, Any] = {}
        self._theme_name: str = "default"
        self._theme_kwargs: dict[str, Any] = {}
        self._exporter_name: str = "html"
        self._exporter_kwargs: dict[str, Any] = {}
        self._options: dict[str, Any] = {}

    def _copy(self) -> "ReportPipeline":
        """Create a copy of this pipeline.

        Returns:
            New pipeline with same configuration.
        """
        new = ReportPipeline(registry=self._registry)
        new._transformers = list(self._transformers)
        new._renderer_name = self._renderer_name
        new._renderer_kwargs = dict(self._renderer_kwargs)
        new._theme_name = self._theme_name
        new._theme_kwargs = dict(self._theme_kwargs)
        new._exporter_name = self._exporter_name
        new._exporter_kwargs = dict(self._exporter_kwargs)
        new._options = dict(self._options)
        return new

    # Fluent API methods

    def transform(self, transformer: Any) -> "ReportPipeline":
        """Add a transformer to the pipeline.

        Transformers are applied in order during the transform phase.

        Args:
            transformer: Transformer instance or name string.

        Returns:
            New pipeline with the transformer added.
        """
        new = self._copy()

        if isinstance(transformer, str):
            # Look up by name
            transformer = self._registry.get_transformer(transformer)

        new._transformers.append(transformer)
        return new

    def transform_with(
        self,
        name: str,
        **kwargs: Any,
    ) -> "ReportPipeline":
        """Add a transformer by name with configuration.

        Args:
            name: Transformer name in registry.
            **kwargs: Arguments for transformer constructor.

        Returns:
            New pipeline with the transformer added.
        """
        new = self._copy()
        transformer = self._registry.get_transformer(name, **kwargs)
        new._transformers.append(transformer)
        return new

    def render_with(
        self,
        renderer: str | Any,
        **kwargs: Any,
    ) -> "ReportPipeline":
        """Set the renderer for the pipeline.

        Args:
            renderer: Renderer name or instance.
            **kwargs: Arguments for renderer constructor.

        Returns:
            New pipeline with the renderer set.
        """
        new = self._copy()

        if isinstance(renderer, str):
            new._renderer_name = renderer
            new._renderer_kwargs = kwargs
        else:
            # Store instance directly - we'll use it in generate()
            new._renderer_name = "__instance__"
            new._renderer_kwargs = {"instance": renderer}

        return new

    def theme(
        self,
        theme: str | Any,
        **kwargs: Any,
    ) -> "ReportPipeline":
        """Set the theme for the pipeline.

        Args:
            theme: Theme name or instance.
            **kwargs: Arguments for theme constructor or customization.

        Returns:
            New pipeline with the theme set.
        """
        new = self._copy()

        if isinstance(theme, str):
            new._theme_name = theme
            new._theme_kwargs = kwargs
        else:
            new._theme_name = "__instance__"
            new._theme_kwargs = {"instance": theme}

        return new

    def export_as(
        self,
        format: str,
        **kwargs: Any,
    ) -> "ReportPipeline":
        """Set the export format for the pipeline.

        Args:
            format: Output format (html, pdf, markdown, json).
            **kwargs: Arguments for exporter constructor.

        Returns:
            New pipeline with the exporter set.
        """
        new = self._copy()
        new._exporter_name = format
        new._exporter_kwargs = kwargs
        return new

    def with_option(self, key: str, value: Any) -> "ReportPipeline":
        """Add a pipeline option.

        Args:
            key: Option key.
            value: Option value.

        Returns:
            New pipeline with the option added.
        """
        new = self._copy()
        new._options[key] = value
        return new

    def with_options(self, **options: Any) -> "ReportPipeline":
        """Add multiple pipeline options.

        Args:
            **options: Option key-value pairs.

        Returns:
            New pipeline with the options added.
        """
        new = self._copy()
        new._options.update(options)
        return new

    # Execution methods

    def generate(self, ctx: ReportContext) -> PipelineResult:
        """Execute the pipeline and generate the report.

        This is the main entry point for report generation.

        Args:
            ctx: Report context with data and configuration.

        Returns:
            PipelineResult with the generated report.
        """
        start_time = datetime.now()

        try:
            # Apply context-level options
            for key, value in self._options.items():
                ctx = ctx.with_option(key, value)

            # Set output format from exporter
            ctx = ctx.with_output_format(self._exporter_name)

            # Set theme in context
            ctx = ctx.with_theme(self._theme_name)

            # Phase 1: Transform
            ctx = self._apply_transformers(ctx)

            # Phase 2: Render
            rendered_content = self._apply_renderer(ctx)

            # Phase 3: Export
            exported_content = self._apply_exporter(rendered_content, ctx)

            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            return PipelineResult(
                content=exported_content,
                format=self._exporter_name,
                context=ctx,
                metadata={
                    "transformers": [type(t).__name__ for t in self._transformers],
                    "renderer": self._renderer_name,
                    "theme": self._theme_name,
                    "exporter": self._exporter_name,
                },
                elapsed_ms=elapsed,
                success=True,
            )

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return PipelineResult(
                content="",
                format=self._exporter_name,
                context=ctx,
                elapsed_ms=elapsed,
                success=False,
                error=str(e),
            )

    def generate_html(self, ctx: ReportContext) -> str:
        """Convenience method to generate HTML report.

        Args:
            ctx: Report context.

        Returns:
            HTML string.
        """
        result = self.export_as("html").generate(ctx)
        if not result.success:
            raise RuntimeError(f"Report generation failed: {result.error}")
        return result.content if isinstance(result.content, str) else result.content.decode()

    def generate_pdf(self, ctx: ReportContext) -> bytes:
        """Convenience method to generate PDF report.

        Args:
            ctx: Report context.

        Returns:
            PDF bytes.
        """
        result = self.export_as("pdf").generate(ctx)
        if not result.success:
            raise RuntimeError(f"Report generation failed: {result.error}")
        return result.content if isinstance(result.content, bytes) else result.content.encode()

    # Internal methods

    def _apply_transformers(self, ctx: ReportContext) -> ReportContext:
        """Apply all transformers in order.

        Args:
            ctx: Input context.

        Returns:
            Transformed context.
        """
        for transformer in self._transformers:
            ctx = transformer.transform(ctx)
            ctx = ctx.with_trace(type(transformer).__name__)
        return ctx

    def _apply_renderer(self, ctx: ReportContext) -> str:
        """Apply the renderer to produce HTML.

        Args:
            ctx: Transformed context.

        Returns:
            Rendered HTML string.
        """
        # Get renderer instance
        if self._renderer_name == "__instance__":
            renderer = self._renderer_kwargs["instance"]
        elif self._registry.has_renderer(self._renderer_name):
            renderer = self._registry.get_renderer(
                self._renderer_name,
                **self._renderer_kwargs,
            )
        else:
            # Fall back to a simple renderer
            return self._simple_render(ctx)

        # Get theme instance
        if self._theme_name == "__instance__":
            theme = self._theme_kwargs["instance"]
        elif self._registry.has_theme(self._theme_name):
            theme = self._registry.get_theme(
                self._theme_name,
                **self._theme_kwargs,
            )
        else:
            theme = None

        return renderer.render(ctx, theme)

    def _apply_exporter(self, content: str, ctx: ReportContext) -> bytes | str:
        """Apply the exporter to produce final output.

        Args:
            content: Rendered HTML content.
            ctx: Pipeline context.

        Returns:
            Exported content (bytes for binary formats, str for text).
        """
        if self._registry.has_exporter(self._exporter_name):
            exporter = self._registry.get_exporter(
                self._exporter_name,
                **self._exporter_kwargs,
            )
            return exporter.export(content, ctx)
        else:
            # Fall back to returning HTML as-is
            return content

    def _simple_render(self, ctx: ReportContext) -> str:
        """Simple fallback renderer when no renderer is registered.

        Args:
            ctx: Report context.

        Returns:
            Basic HTML string.
        """
        title = ctx.title
        data = ctx.data

        sections_html = []
        for name, section_data in data.sections.items():
            sections_html.append(f"""
                <section class="report-section" id="section-{name}">
                    <h2>{name.replace('_', ' ').title()}</h2>
                    <pre>{section_data}</pre>
                </section>
            """)

        return f"""<!DOCTYPE html>
<html lang="{ctx.locale}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <header>
        <h1>{title}</h1>
    </header>
    <main>
        {"".join(sections_html)}
    </main>
</body>
</html>"""


class PipelineBuilder:
    """Builder for creating customized pipelines.

    This provides a more verbose but explicit way to construct pipelines.

    Example:
        pipeline = (
            PipelineBuilder()
            .with_registry(custom_registry)
            .add_transformer(I18nTransformer(locale="ko"))
            .add_transformer(FilterTransformer())
            .set_renderer("jinja")
            .set_theme("enterprise")
            .set_exporter("pdf", page_size="A4")
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._registry: ComponentRegistry | None = None
        self._transformers: list[Any] = []
        self._renderer_name: str = "jinja"
        self._renderer_kwargs: dict[str, Any] = {}
        self._theme_name: str = "default"
        self._theme_kwargs: dict[str, Any] = {}
        self._exporter_name: str = "html"
        self._exporter_kwargs: dict[str, Any] = {}
        self._options: dict[str, Any] = {}

    def with_registry(self, registry: ComponentRegistry) -> "PipelineBuilder":
        """Set the component registry.

        Args:
            registry: Registry to use.

        Returns:
            Self for chaining.
        """
        self._registry = registry
        return self

    def add_transformer(self, transformer: Any) -> "PipelineBuilder":
        """Add a transformer.

        Args:
            transformer: Transformer instance.

        Returns:
            Self for chaining.
        """
        self._transformers.append(transformer)
        return self

    def set_renderer(self, name: str, **kwargs: Any) -> "PipelineBuilder":
        """Set the renderer.

        Args:
            name: Renderer name.
            **kwargs: Renderer configuration.

        Returns:
            Self for chaining.
        """
        self._renderer_name = name
        self._renderer_kwargs = kwargs
        return self

    def set_theme(self, name: str, **kwargs: Any) -> "PipelineBuilder":
        """Set the theme.

        Args:
            name: Theme name.
            **kwargs: Theme configuration.

        Returns:
            Self for chaining.
        """
        self._theme_name = name
        self._theme_kwargs = kwargs
        return self

    def set_exporter(self, name: str, **kwargs: Any) -> "PipelineBuilder":
        """Set the exporter.

        Args:
            name: Exporter name (format).
            **kwargs: Exporter configuration.

        Returns:
            Self for chaining.
        """
        self._exporter_name = name
        self._exporter_kwargs = kwargs
        return self

    def set_option(self, key: str, value: Any) -> "PipelineBuilder":
        """Set an option.

        Args:
            key: Option key.
            value: Option value.

        Returns:
            Self for chaining.
        """
        self._options[key] = value
        return self

    def build(self) -> ReportPipeline:
        """Build the pipeline.

        Returns:
            Configured ReportPipeline.
        """
        pipeline = ReportPipeline(registry=self._registry)

        # Add transformers
        for t in self._transformers:
            pipeline = pipeline.transform(t)

        # Set renderer
        pipeline = pipeline.render_with(self._renderer_name, **self._renderer_kwargs)

        # Set theme
        pipeline = pipeline.theme(self._theme_name, **self._theme_kwargs)

        # Set exporter
        pipeline = pipeline.export_as(self._exporter_name, **self._exporter_kwargs)

        # Set options
        for key, value in self._options.items():
            pipeline = pipeline.with_option(key, value)

        return pipeline
