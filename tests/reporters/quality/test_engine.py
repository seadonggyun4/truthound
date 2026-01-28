"""Tests for Quality Report Engine and Pipeline."""

import json
import pytest
from pathlib import Path

from truthound.profiler.quality import (
    QualityLevel,
    QualityMetrics,
    RuleType,
    RuleQualityScore,
)
from truthound.reporters.quality.engine import (
    QualityReportEngine,
    QualityReportContext,
    QualityReportPipeline,
    FilterStage,
    SortStage,
    LimitStage,
    StatisticsStage,
    RenderStage,
    WriteStage,
    generate_quality_report,
    filter_quality_scores,
    compare_quality_scores,
)
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityReportEngineConfig,
    QualityFilterConfig,
    ReportSortOrder,
)
from truthound.reporters.quality.filters import QualityFilter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics_list() -> list[QualityMetrics]:
    """Create a list of sample metrics."""
    return [
        QualityMetrics(
            precision=0.96, recall=0.95, f1_score=0.955,
            accuracy=0.97, confidence=0.9, quality_level=QualityLevel.EXCELLENT,
        ),
        QualityMetrics(
            precision=0.88, recall=0.86, f1_score=0.87,
            accuracy=0.89, confidence=0.85, quality_level=QualityLevel.GOOD,
        ),
        QualityMetrics(
            precision=0.75, recall=0.72, f1_score=0.735,
            accuracy=0.78, confidence=0.7, quality_level=QualityLevel.ACCEPTABLE,
        ),
        QualityMetrics(
            precision=0.55, recall=0.52, f1_score=0.534,
            accuracy=0.6, confidence=0.5, quality_level=QualityLevel.POOR,
        ),
        QualityMetrics(
            precision=0.35, recall=0.32, f1_score=0.334,
            accuracy=0.4, confidence=0.3, quality_level=QualityLevel.UNACCEPTABLE,
        ),
    ]


@pytest.fixture
def sample_scores(sample_metrics_list) -> list[RuleQualityScore]:
    """Create sample quality scores."""
    rule_names = ["email_format", "age_range", "name_not_null", "phone_pattern", "status_enum"]
    rule_types = [RuleType.PATTERN, RuleType.RANGE, RuleType.COMPLETENESS, RuleType.PATTERN, RuleType.CUSTOM]
    columns = ["email", "age", "name", "phone", "status"]
    recommendations = [
        "Excellent quality. Safe to use.",
        "Good quality. Recommended for use.",
        "Acceptable quality. Monitor for issues.",
        "Poor quality. Consider improvements.",
        "Unacceptable quality. Do not use.",
    ]
    should_use_list = [True, True, True, False, False]

    return [
        RuleQualityScore(
            rule_name=rule_names[i],
            rule_type=rule_types[i],
            column=columns[i],
            metrics=sample_metrics_list[i],
            recommendation=recommendations[i],
            should_use=should_use_list[i],
        )
        for i in range(5)
    ]


# =============================================================================
# Test Pipeline Context
# =============================================================================


class TestQualityReportContext:
    """Tests for QualityReportContext."""

    def test_create_context(self, sample_scores):
        """Test creating a context."""
        context = QualityReportContext(
            scores=sample_scores,
            original_count=len(sample_scores),
        )

        assert len(context.scores) == 5
        assert context.original_count == 5
        assert context.filtered_count == 0

    def test_clone_context(self, sample_scores):
        """Test cloning a context."""
        original = QualityReportContext(
            scores=sample_scores,
            original_count=len(sample_scores),
        )
        original.metadata["test"] = "value"

        cloned = original.clone()

        assert len(cloned.scores) == len(original.scores)
        assert cloned.metadata["test"] == "value"

        # Modify clone shouldn't affect original
        cloned.scores.pop()
        assert len(original.scores) == 5


# =============================================================================
# Test Pipeline Stages
# =============================================================================


class TestFilterStage:
    """Tests for FilterStage."""

    def test_filter_stage_with_filter(self, sample_scores):
        """Test filter stage with explicit filter."""
        filter_obj = QualityFilter.by_level(min_level="good")
        stage = FilterStage(filter_obj=filter_obj)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert len(result.scores) == 2
        assert result.filtered_count == 3

    def test_filter_stage_with_config(self, sample_scores):
        """Test filter stage with filter config."""
        config = QualityFilterConfig(min_level="acceptable")
        stage = FilterStage(filter_config=config)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert len(result.scores) == 3

    def test_filter_stage_no_filter(self, sample_scores):
        """Test filter stage without filter."""
        stage = FilterStage()

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert len(result.scores) == 5


class TestSortStage:
    """Tests for SortStage."""

    def test_sort_by_f1_desc(self, sample_scores):
        """Test sorting by F1 descending."""
        stage = SortStage(order=ReportSortOrder.F1_DESC)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        f1_scores = [s.metrics.f1_score for s in result.scores]
        assert f1_scores == sorted(f1_scores, reverse=True)

    def test_sort_by_f1_asc(self, sample_scores):
        """Test sorting by F1 ascending."""
        stage = SortStage(order=ReportSortOrder.F1_ASC)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        f1_scores = [s.metrics.f1_score for s in result.scores]
        assert f1_scores == sorted(f1_scores)

    def test_sort_by_name(self, sample_scores):
        """Test sorting by name."""
        stage = SortStage(order=ReportSortOrder.NAME_ASC)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        names = [s.rule_name for s in result.scores]
        assert names == sorted(names)


class TestLimitStage:
    """Tests for LimitStage."""

    def test_limit_scores(self, sample_scores):
        """Test limiting scores."""
        stage = LimitStage(max_scores=3)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert len(result.scores) == 3

    def test_limit_larger_than_count(self, sample_scores):
        """Test limit larger than score count."""
        stage = LimitStage(max_scores=10)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert len(result.scores) == 5

    def test_no_limit(self, sample_scores):
        """Test no limit."""
        stage = LimitStage()

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert len(result.scores) == 5


class TestStatisticsStage:
    """Tests for StatisticsStage."""

    def test_calculate_statistics(self, sample_scores):
        """Test statistics calculation."""
        stage = StatisticsStage()

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert result.statistics is not None
        assert result.statistics.total_count == 5
        assert result.statistics.excellent_count == 1
        assert result.statistics.should_use_count == 3


class TestRenderStage:
    """Tests for RenderStage."""

    def test_render_json(self, sample_scores):
        """Test rendering to JSON."""
        stage = RenderStage(format="json")

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert result.rendered_content
        data = json.loads(result.rendered_content)
        assert "scores" in data

    def test_render_console(self, sample_scores):
        """Test rendering to console."""
        stage = RenderStage(format="console")

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        result = stage.transform(context)

        assert result.rendered_content
        assert "email_format" in result.rendered_content


class TestWriteStage:
    """Tests for WriteStage."""

    def test_write_to_file(self, sample_scores, tmp_path):
        """Test writing to file."""
        output_path = tmp_path / "test_report.json"
        render_stage = RenderStage(format="json")
        write_stage = WriteStage(path=output_path)

        context = QualityReportContext(scores=sample_scores, original_count=len(sample_scores))
        context = render_stage.transform(context)
        result = write_stage.transform(context)

        assert result.output_path == output_path
        assert output_path.exists()


# =============================================================================
# Test Pipeline
# =============================================================================


class TestQualityReportPipeline:
    """Tests for QualityReportPipeline."""

    def test_simple_pipeline(self, sample_scores):
        """Test simple pipeline."""
        pipeline = (
            QualityReportPipeline()
            .sort(ReportSortOrder.F1_DESC)
            .statistics()
            .render("json")
        )

        context = pipeline.execute(sample_scores)

        assert context.statistics is not None
        assert context.rendered_content

    def test_pipeline_with_filter(self, sample_scores):
        """Test pipeline with filtering."""
        pipeline = (
            QualityReportPipeline()
            .filter(QualityFilter.by_level(min_level="good"))
            .sort(ReportSortOrder.F1_DESC)
            .render("json")
        )

        context = pipeline.execute(sample_scores)
        data = json.loads(context.rendered_content)

        assert len(data["scores"]) == 2

    def test_pipeline_with_limit(self, sample_scores):
        """Test pipeline with limit."""
        pipeline = (
            QualityReportPipeline()
            .sort(ReportSortOrder.F1_DESC)
            .limit(2)
            .render("json")
        )

        context = pipeline.execute(sample_scores)
        data = json.loads(context.rendered_content)

        assert len(data["scores"]) == 2

    def test_pipeline_with_write(self, sample_scores, tmp_path):
        """Test pipeline with file write."""
        output_path = tmp_path / "pipeline_report.json"
        pipeline = (
            QualityReportPipeline()
            .render("json")
            .write(output_path)
        )

        context = pipeline.execute(sample_scores)

        assert output_path.exists()
        assert context.output_path == output_path

    def test_pipeline_tracks_stage_times(self, sample_scores):
        """Test pipeline tracks stage times."""
        pipeline = (
            QualityReportPipeline()
            .filter(QualityFilter.by_level(min_level="acceptable"))
            .sort()
            .statistics()
            .render("json")
        )

        context = pipeline.execute(sample_scores)

        assert "filter" in context.stage_times
        assert "sort" in context.stage_times
        assert "statistics" in context.stage_times
        assert "render" in context.stage_times

    def test_full_pipeline(self, sample_scores, tmp_path):
        """Test full pipeline with all stages."""
        output_path = tmp_path / "full_report.html"
        pipeline = (
            QualityReportPipeline()
            .filter(QualityFilter.by_level(min_level="acceptable"))
            .sort(ReportSortOrder.F1_DESC)
            .limit(10)
            .statistics()
            .render("html")
            .write(output_path)
        )

        context = pipeline.execute(sample_scores)

        assert len(context.scores) == 3
        assert context.statistics is not None
        assert output_path.exists()


# =============================================================================
# Test Engine
# =============================================================================


class TestQualityReportEngine:
    """Tests for QualityReportEngine."""

    def test_generate_json_report(self, sample_scores):
        """Test generating JSON report."""
        engine = QualityReportEngine()
        result = engine.generate(sample_scores, format="json")

        assert result.format == "json"
        assert result.scores_count == 5
        data = json.loads(result.content)
        assert "scores" in data

    def test_generate_html_report(self, sample_scores):
        """Test generating HTML report."""
        engine = QualityReportEngine()
        result = engine.generate(sample_scores, format="html")

        assert result.format == "html"
        assert "<!DOCTYPE html>" in result.content

    def test_generate_with_filter(self, sample_scores):
        """Test generating with filter."""
        engine = QualityReportEngine()
        result = engine.generate(
            sample_scores,
            format="json",
            filter=QualityFilter.by_level(min_level="good"),
        )

        data = json.loads(result.content)
        assert len(data["scores"]) == 2

    def test_generate_with_output_path(self, sample_scores, tmp_path):
        """Test generating with output path."""
        output_path = tmp_path / "engine_report.json"
        engine = QualityReportEngine()
        result = engine.generate(
            sample_scores,
            format="json",
            output_path=output_path,
        )

        assert result.output_path == output_path
        assert output_path.exists()

    def test_generate_tracks_time(self, sample_scores):
        """Test generation tracks time."""
        engine = QualityReportEngine()
        result = engine.generate(sample_scores, format="json")

        assert result.generation_time_ms > 0

    def test_caching(self, sample_scores):
        """Test result caching."""
        config = QualityReportEngineConfig(enable_caching=True)
        engine = QualityReportEngine(config=config)

        # First call
        result1 = engine.generate(sample_scores, format="json")
        # Second call should use cache
        result2 = engine.generate(sample_scores, format="json")

        # Both should have same content
        assert result1.content == result2.content

    def test_clear_cache(self, sample_scores):
        """Test clearing cache."""
        config = QualityReportEngineConfig(enable_caching=True)
        engine = QualityReportEngine(config=config)

        engine.generate(sample_scores, format="json")
        engine.clear_cache()

        # Should work without issues
        result = engine.generate(sample_scores, format="json")
        assert result.content

    def test_execute_pipeline(self, sample_scores):
        """Test executing custom pipeline."""
        engine = QualityReportEngine()
        pipeline = (
            QualityReportPipeline()
            .filter(QualityFilter.by_level(min_level="good"))
            .sort()
            .render("json")
        )

        context = engine.execute_pipeline(sample_scores, pipeline)

        assert len(context.scores) == 2


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_quality_report(self, sample_scores):
        """Test generate_quality_report function."""
        result = generate_quality_report(sample_scores, format="json")

        assert result.format == "json"
        data = json.loads(result.content)
        assert len(data["scores"]) == 5

    def test_generate_quality_report_with_output(self, sample_scores, tmp_path):
        """Test generate_quality_report with output path."""
        output_path = tmp_path / "convenience_report.json"
        result = generate_quality_report(
            sample_scores,
            format="json",
            output_path=output_path,
        )

        assert output_path.exists()

    def test_filter_quality_scores_with_filter(self, sample_scores):
        """Test filter_quality_scores with filter object."""
        filtered = filter_quality_scores(
            sample_scores,
            filter=QualityFilter.by_level(min_level="good"),
        )

        assert len(filtered) == 2

    def test_filter_quality_scores_with_min_level(self, sample_scores):
        """Test filter_quality_scores with min_level."""
        filtered = filter_quality_scores(
            sample_scores,
            min_level="acceptable",
        )

        assert len(filtered) == 3

    def test_filter_quality_scores_with_min_f1(self, sample_scores):
        """Test filter_quality_scores with min_f1."""
        filtered = filter_quality_scores(
            sample_scores,
            min_f1=0.8,
        )

        assert len(filtered) == 2

    def test_filter_quality_scores_should_use_only(self, sample_scores):
        """Test filter_quality_scores with should_use_only."""
        filtered = filter_quality_scores(
            sample_scores,
            should_use_only=True,
        )

        assert len(filtered) == 3
        assert all(s.should_use for s in filtered)

    def test_compare_quality_scores(self, sample_scores):
        """Test compare_quality_scores function."""
        sorted_scores = compare_quality_scores(sample_scores, sort_by="f1_score")

        f1_scores = [s.metrics.f1_score for s in sorted_scores]
        assert f1_scores == sorted(f1_scores, reverse=True)

    def test_compare_quality_scores_ascending(self, sample_scores):
        """Test compare_quality_scores ascending."""
        sorted_scores = compare_quality_scores(
            sample_scores,
            sort_by="f1_score",
            descending=False,
        )

        f1_scores = [s.metrics.f1_score for s in sorted_scores]
        assert f1_scores == sorted(f1_scores)

    def test_compare_quality_scores_by_precision(self, sample_scores):
        """Test compare_quality_scores by precision."""
        sorted_scores = compare_quality_scores(sample_scores, sort_by="precision")

        precisions = [s.metrics.precision for s in sorted_scores]
        assert precisions == sorted(precisions, reverse=True)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_scores(self):
        """Test with empty scores list."""
        engine = QualityReportEngine()
        result = engine.generate([], format="json")

        data = json.loads(result.content)
        assert data["count"] == 0

    def test_single_score(self, sample_scores):
        """Test with single score."""
        engine = QualityReportEngine()
        result = engine.generate([sample_scores[0]], format="json")

        data = json.loads(result.content)
        assert len(data["scores"]) == 1

    def test_all_filtered_out(self, sample_scores):
        """Test when all scores are filtered out."""
        engine = QualityReportEngine()
        result = engine.generate(
            sample_scores,
            format="json",
            filter=QualityFilter.by_metric("f1_score", ">=", 1.0),
        )

        data = json.loads(result.content)
        assert len(data["scores"]) == 0
