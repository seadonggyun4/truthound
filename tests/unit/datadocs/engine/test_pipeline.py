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
        assert result is pipeline  # Fluent API
        assert len(pipeline._transformers) == 1

    def test_set_renderer(self):
        """Test setting a renderer."""
        pipeline = ReportPipeline()
        renderer = MockRenderer()
        result = pipeline.render_with(renderer)
        assert result is pipeline
        assert pipeline._renderer is renderer

    def test_set_exporter(self):
        """Test setting an exporter."""
        pipeline = ReportPipeline()
        exporter = MockExporter()
        result = pipeline.export_as(exporter)
        assert result is pipeline

    def test_generate_basic(self):
        """Test basic pipeline generation."""
        pipeline = ReportPipeline()
        pipeline.render_with(MockRenderer())

        ctx = ReportContext(data=ReportData())
        result = pipeline.generate(ctx)

        assert result.success
        assert "<html>" in result.content
        assert result.error is None

    def test_generate_with_transformer(self):
        """Test pipeline with transformer."""
        pipeline = ReportPipeline()
        pipeline.transform(MockTransformer())
        pipeline.render_with(MockRenderer())

        ctx = ReportContext(data=ReportData())
        result = pipeline.generate(ctx)

        assert result.success
        assert result.context.options.get("transformed") is True

    def test_generate_with_multiple_transformers(self):
        """Test pipeline with multiple transformers."""
        pipeline = ReportPipeline()
        pipeline.transform(MockTransformer("_1"))
        pipeline.transform(MockTransformer("_2"))
        pipeline.render_with(MockRenderer())

        ctx = ReportContext(data=ReportData())
        result = pipeline.generate(ctx)

        assert result.success

    def test_generate_without_renderer_fails(self):
        """Test that generation without renderer fails."""
        pipeline = ReportPipeline()
        ctx = ReportContext(data=ReportData())

        result = pipeline.generate(ctx)

        assert not result.success
        assert result.error is not None


class TestPipelineBuilder:
    """Tests for PipelineBuilder class."""

    def test_create_builder(self):
        """Test creating a builder."""
        builder = PipelineBuilder()
        assert isinstance(builder, PipelineBuilder)

    def test_builder_with_locale(self):
        """Test builder with locale."""
        builder = PipelineBuilder().with_locale("ko")
        pipeline = builder.build()
        assert pipeline._default_locale == "ko"

    def test_builder_with_theme(self):
        """Test builder with theme."""
        builder = PipelineBuilder().with_theme("dark")
        pipeline = builder.build()
        assert pipeline._default_theme == "dark"

    def test_builder_fluent_chain(self):
        """Test builder fluent API."""
        pipeline = (
            PipelineBuilder()
            .with_locale("ja")
            .with_theme("corporate")
            .with_transformer(MockTransformer())
            .with_renderer(MockRenderer())
            .build()
        )

        assert pipeline._default_locale == "ja"
        assert pipeline._default_theme == "corporate"
        assert len(pipeline._transformers) == 1
        assert pipeline._renderer is not None

    def test_builder_with_options(self):
        """Test builder with options."""
        builder = PipelineBuilder().with_option("debug", True)
        pipeline = builder.build()
        assert pipeline._default_options.get("debug") is True


class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_success_result(self):
        """Test successful result."""
        ctx = ReportContext(data=ReportData())
        result = PipelineResult(
            success=True,
            content="<html></html>",
            context=ctx,
        )
        assert result.success
        assert result.content == "<html></html>"
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        ctx = ReportContext(data=ReportData())
        result = PipelineResult(
            success=False,
            content="",
            context=ctx,
            error="Renderer not configured",
        )
        assert not result.success
        assert result.error == "Renderer not configured"

    def test_result_with_exported_bytes(self):
        """Test result with exported bytes."""
        ctx = ReportContext(data=ReportData())
        result = PipelineResult(
            success=True,
            content="<html></html>",
            context=ctx,
            exported_bytes=b"PDF content",
            export_format="pdf",
        )
        assert result.exported_bytes == b"PDF content"
        assert result.export_format == "pdf"
