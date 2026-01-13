"""Tests for engine pipeline module."""

import pytest
from truthound.datadocs.engine.context import ReportContext, ReportData
from truthound.datadocs.engine.pipeline import (
    ReportPipeline,
    PipelineBuilder,
    PipelineResult,
)


class MockTransformer:
    """Mock transformer for testing."""

    def __init__(self, suffix: str = "_transformed"):
        self.suffix = suffix

    def transform(self, ctx: ReportContext) -> ReportContext:
        return ctx.with_option("transformed", True)


class MockRenderer:
    """Mock renderer for testing."""

    def __init__(self, name: str = "MockRenderer"):
        self.name = name

    def render(self, ctx: ReportContext, theme=None) -> str:
        return f"<html>Rendered by {self.name}</html>"


class MockExporter:
    """Mock exporter for testing."""

    @property
    def format(self) -> str:
        return "mock"

    def export(self, content: str, ctx: ReportContext) -> bytes:
        return f"Exported: {content}".encode()


class TestReportPipeline:
    """Tests for ReportPipeline class."""

    def test_create_empty_pipeline(self):
        """Test creating empty pipeline."""
        pipeline = ReportPipeline()
        assert len(pipeline._transformers) == 0

    def test_add_transformer(self):
        """Test adding a transformer."""
        pipeline = ReportPipeline()
        transformer = MockTransformer()
        result = pipeline.transform(transformer)
        # Fluent API returns a new pipeline (immutable)
        assert result is not pipeline
        assert len(result._transformers) == 1

    def test_set_renderer(self):
        """Test setting a renderer."""
        pipeline = ReportPipeline()
        renderer = MockRenderer()
        result = pipeline.render_with(renderer)
        # Fluent API returns a new pipeline (immutable)
        assert result is not pipeline
        assert result._renderer_name == "__instance__"

    def test_set_exporter(self):
        """Test setting an exporter."""
        pipeline = ReportPipeline()
        result = pipeline.export_as("pdf")
        # Fluent API returns a new pipeline (immutable)
        assert result is not pipeline
        assert result._exporter_name == "pdf"

    def test_generate_basic(self):
        """Test basic pipeline generation."""
        pipeline = ReportPipeline()
        pipeline = pipeline.render_with(MockRenderer())

        ctx = ReportContext(data=ReportData())
        result = pipeline.generate(ctx)

        assert result.success
        assert "<html>" in result.content
        assert result.error is None

    def test_generate_with_transformer(self):
        """Test pipeline with transformer."""
        pipeline = ReportPipeline()
        pipeline = pipeline.transform(MockTransformer())
        pipeline = pipeline.render_with(MockRenderer())

        ctx = ReportContext(data=ReportData())
        result = pipeline.generate(ctx)

        assert result.success
        assert result.context.options.get("transformed") is True

    def test_generate_with_multiple_transformers(self):
        """Test pipeline with multiple transformers."""
        pipeline = ReportPipeline()
        pipeline = pipeline.transform(MockTransformer("_1"))
        pipeline = pipeline.transform(MockTransformer("_2"))
        pipeline = pipeline.render_with(MockRenderer())

        ctx = ReportContext(data=ReportData())
        result = pipeline.generate(ctx)

        assert result.success

    def test_generate_without_renderer_uses_fallback(self):
        """Test that generation without renderer uses simple fallback."""
        pipeline = ReportPipeline()
        ctx = ReportContext(data=ReportData())

        result = pipeline.generate(ctx)

        # Pipeline uses simple fallback renderer when no renderer is registered
        assert result.success
        assert "<!DOCTYPE html>" in result.content


class TestPipelineBuilder:
    """Tests for PipelineBuilder class."""

    def test_create_builder(self):
        """Test creating a builder."""
        builder = PipelineBuilder()
        assert isinstance(builder, PipelineBuilder)

    def test_builder_set_theme(self):
        """Test builder with theme."""
        builder = PipelineBuilder()
        builder.set_theme("dark")
        pipeline = builder.build()
        assert pipeline._theme_name == "dark"

    def test_builder_set_exporter(self):
        """Test builder with exporter."""
        builder = PipelineBuilder()
        builder.set_exporter("pdf")
        pipeline = builder.build()
        assert pipeline._exporter_name == "pdf"

    def test_builder_fluent_chain(self):
        """Test builder fluent API."""
        pipeline = (
            PipelineBuilder()
            .set_theme("dark")
            .set_renderer("jinja")
            .set_exporter("html")
            .add_transformer(MockTransformer())
            .build()
        )

        assert pipeline._theme_name == "dark"
        assert pipeline._renderer_name == "jinja"
        assert len(pipeline._transformers) == 1

    def test_builder_with_options(self):
        """Test builder with options."""
        builder = PipelineBuilder()
        builder.set_option("debug", True)
        pipeline = builder.build()
        assert pipeline._options.get("debug") is True


class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_success_result(self):
        """Test successful result."""
        ctx = ReportContext(data=ReportData())
        result = PipelineResult(
            content="<html></html>",
            format="html",
            context=ctx,
            success=True,
        )
        assert result.success
        assert result.content == "<html></html>"
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        ctx = ReportContext(data=ReportData())
        result = PipelineResult(
            content="",
            format="html",
            context=ctx,
            success=False,
            error="Renderer not configured",
        )
        assert not result.success
        assert result.error == "Renderer not configured"

    def test_result_is_binary(self):
        """Test result with binary content."""
        ctx = ReportContext(data=ReportData())
        result = PipelineResult(
            content=b"PDF content",
            format="pdf",
            context=ctx,
            success=True,
        )
        assert result.is_binary
        assert result.format == "pdf"
