"""Tests for P3 profiler modules: Distributed, ML Inference, Auto Threshold, Visualization.

These tests verify the core functionality of P3 modules with simplified API tests.
"""

import json
import tempfile
from pathlib import Path

import pytest
import polars as pl

# =============================================================================
# Distributed Processing Tests
# =============================================================================

from truthound.profiler.distributed import (
    BackendType,
    PartitionStrategy,
    PartitionInfo,
    WorkerResult,
    BackendConfig,
    LocalBackend,
    BackendRegistry,
    get_available_backends,
)


class TestDistributed:
    """Tests for distributed processing module."""

    def test_backend_types(self):
        """Test backend type enum."""
        assert BackendType.LOCAL.value == "local"
        assert BackendType.SPARK.value == "spark"
        assert BackendType.DASK.value == "dask"
        assert BackendType.RAY.value == "ray"

    def test_partition_strategy(self):
        """Test partition strategy enum."""
        assert PartitionStrategy.ROW_BASED.value == "row_based"
        assert PartitionStrategy.COLUMN_BASED.value == "column_based"

    def test_partition_info(self):
        """Test PartitionInfo dataclass."""
        partition = PartitionInfo(
            partition_id=0,
            total_partitions=4,
            start_row=0,
            end_row=1000,
        )
        assert partition.partition_id == 0
        assert partition.total_partitions == 4

    def test_worker_result(self):
        """Test WorkerResult dataclass."""
        result = WorkerResult(
            partition_id=0,
            column_stats={"col1": {"mean": 10.5}},
            row_count=1000,
            processing_time_ms=150.0,
        )
        assert result.partition_id == 0
        assert result.row_count == 1000

    def test_backend_config(self):
        """Test BackendConfig defaults."""
        config = BackendConfig()
        assert config.backend_type == BackendType.LOCAL

    def test_local_backend(self):
        """Test LocalBackend creation and availability."""
        backend = LocalBackend()
        assert backend.name == "local"
        assert backend.is_available() is True

    def test_backend_registry(self):
        """Test BackendRegistry."""
        registry = BackendRegistry()
        # Registry may be empty at first, but should be able to get local backend
        assert registry is not None

    def test_get_available_backends(self):
        """Test get_available_backends function."""
        backends = get_available_backends()
        assert isinstance(backends, list)
        assert "local" in backends


# =============================================================================
# ML-based Type Inference Tests
# =============================================================================

from truthound.profiler.ml_inference import (
    FeatureType,
    Feature,
    FeatureVector,
    InferenceResult,
    NameFeatureExtractor,
    ValueFeatureExtractor,
    StatisticalFeatureExtractor,
    RuleBasedModel,
    NaiveBayesModel,
    EnsembleModel,
    MLTypeInferrer,
    create_inference_model,
)


class TestMLInference:
    """Tests for ML inference module."""

    def test_feature_type_enum(self):
        """Test FeatureType enum values."""
        # Check that enum has expected members
        members = list(FeatureType)
        assert len(members) > 0

    def test_feature_dataclass(self):
        """Test Feature dataclass."""
        feature = Feature(
            name="test_feature",
            feature_type=list(FeatureType)[0],  # Use first available type
            value=0.95,
        )
        assert feature.name == "test_feature"
        assert feature.value == 0.95

    def test_feature_vector(self):
        """Test FeatureVector dataclass."""
        feature = Feature(
            name="f1",
            feature_type=list(FeatureType)[0],
            value=1.0,
        )
        vector = FeatureVector(
            column_name="email",
            features=[feature],
        )
        assert vector.column_name == "email"
        assert len(vector.features) == 1

    def test_inference_result(self):
        """Test InferenceResult dataclass."""
        result = InferenceResult(
            column_name="email",
            inferred_type="email",
            confidence=0.95,
        )
        assert result.column_name == "email"
        assert result.inferred_type == "email"
        assert result.confidence == 0.95

    def test_name_feature_extractor(self):
        """Test NameFeatureExtractor creation."""
        extractor = NameFeatureExtractor()
        assert extractor is not None

    def test_value_feature_extractor(self):
        """Test ValueFeatureExtractor creation."""
        extractor = ValueFeatureExtractor()
        assert extractor is not None

    def test_statistical_feature_extractor(self):
        """Test StatisticalFeatureExtractor creation."""
        extractor = StatisticalFeatureExtractor()
        assert extractor is not None

    def test_rule_based_model(self):
        """Test RuleBasedModel creation."""
        model = RuleBasedModel()
        assert model is not None

    def test_naive_bayes_model(self):
        """Test NaiveBayesModel creation."""
        model = NaiveBayesModel()
        assert model is not None

    def test_ensemble_model(self):
        """Test EnsembleModel creation."""
        model = EnsembleModel()
        assert model is not None

    def test_ml_type_inferrer(self):
        """Test MLTypeInferrer creation."""
        inferrer = MLTypeInferrer()
        assert inferrer is not None

    def test_create_inference_model(self):
        """Test create_inference_model function."""
        model = create_inference_model("rule_based")
        assert model is not None


# =============================================================================
# Automatic Threshold Tuning Tests
# =============================================================================

from truthound.profiler.auto_threshold import (
    TuningStrategy,
    ThresholdType,
    ColumnThresholds,
    TableThresholds,
    StrictnessPreset,
    ConservativeStrategy,
    BalancedStrategy,
    PermissiveStrategy,
    AdaptiveStrategy,
    StatisticalStrategy,
    DomainAwareStrategy,
    StrategyRegistry,
    ThresholdTuner,
    ThresholdTester,
    tune_thresholds,
    get_available_strategies,
    create_tuner,
)


class TestAutoThreshold:
    """Tests for auto threshold tuning module."""

    def test_tuning_strategy_enum(self):
        """Test TuningStrategy enum."""
        assert TuningStrategy.CONSERVATIVE is not None
        assert TuningStrategy.BALANCED is not None
        assert TuningStrategy.PERMISSIVE is not None
        assert TuningStrategy.ADAPTIVE is not None

    def test_threshold_type_enum(self):
        """Test ThresholdType enum."""
        # Check enum has expected members
        members = list(ThresholdType)
        assert len(members) > 0
        # Check NULL_RATIO exists
        assert ThresholdType.NULL_RATIO is not None

    def test_column_thresholds(self):
        """Test ColumnThresholds dataclass."""
        thresholds = ColumnThresholds(
            column_name="age",
            null_threshold=0.05,
        )
        assert thresholds.column_name == "age"
        assert thresholds.null_threshold == 0.05

    def test_table_thresholds(self):
        """Test TableThresholds dataclass."""
        col = ColumnThresholds(column_name="a", null_threshold=0.05)
        table = TableThresholds(table_name="test", columns=[col])
        assert len(table.columns) == 1

    def test_strictness_preset(self):
        """Test StrictnessPreset dataclass."""
        # StrictnessPreset may have different fields depending on implementation
        preset = StrictnessPreset()
        assert preset is not None

    def test_strategy_implementations(self):
        """Test all strategy implementations."""
        strategies = [
            ConservativeStrategy(),
            BalancedStrategy(),
            PermissiveStrategy(),
            AdaptiveStrategy(),
            StatisticalStrategy(),
            DomainAwareStrategy(),
        ]
        for strategy in strategies:
            assert strategy is not None
            assert hasattr(strategy, 'name')

    def test_strategy_registry(self):
        """Test StrategyRegistry."""
        registry = StrategyRegistry()
        assert registry is not None
        # Registry exists and can be used
        assert hasattr(registry, 'list_strategies')

    def test_threshold_tuner(self):
        """Test ThresholdTuner creation."""
        tuner = ThresholdTuner()
        assert tuner is not None

    def test_threshold_tester(self):
        """Test ThresholdTester creation."""
        tester = ThresholdTester()
        assert tester is not None

    def test_get_available_strategies(self):
        """Test get_available_strategies function."""
        strategies = get_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_create_tuner(self):
        """Test create_tuner function."""
        tuner = create_tuner(strategy=TuningStrategy.BALANCED)
        assert tuner is not None


# =============================================================================
# Profile Visualization Tests
# =============================================================================

from truthound.profiler.visualization import (
    ChartType,
    ColorScheme,
    ReportTheme,
    SectionType,
    COLOR_PALETTES,
    ChartData,
    ChartConfig,
    ThemeConfig,
    SectionContent,
    ReportConfig,
    ProfileData,
    THEME_CONFIGS,
    ChartRendererRegistry,
    ThemeRegistry,
    SVGChartRenderer,
    ReportTemplate,
    ProfileDataConverter,
    HTMLReportGenerator,
    ReportExporter,
    generate_report,
    compare_profiles as compare_profile_reports,
)


class TestVisualizationEnums:
    """Tests for visualization enums."""

    def test_chart_types(self):
        """Test ChartType enum."""
        assert ChartType.BAR is not None
        assert ChartType.PIE is not None
        assert ChartType.LINE is not None
        assert ChartType.GAUGE is not None

    def test_color_schemes(self):
        """Test ColorScheme enum."""
        assert ColorScheme.DEFAULT is not None
        assert ColorScheme.CATEGORICAL is not None
        for scheme in ColorScheme:
            assert scheme in COLOR_PALETTES

    def test_report_themes(self):
        """Test ReportTheme enum."""
        assert ReportTheme.LIGHT is not None
        assert ReportTheme.DARK is not None
        for theme in ReportTheme:
            assert theme in THEME_CONFIGS

    def test_section_types(self):
        """Test SectionType enum."""
        assert SectionType.OVERVIEW is not None
        assert SectionType.DATA_QUALITY is not None


class TestVisualizationDataClasses:
    """Tests for visualization data classes."""

    def test_chart_data(self):
        """Test ChartData dataclass."""
        data = ChartData(
            labels=["A", "B", "C"],
            values=[10, 20, 30],
            title="Test Chart",
        )
        assert len(data.labels) == 3
        assert sum(data.values) == 60

    def test_chart_config(self):
        """Test ChartConfig dataclass."""
        config = ChartConfig()
        assert config.chart_type == ChartType.BAR
        assert config.width == 600
        assert config.height == 400

    def test_theme_config(self):
        """Test ThemeConfig dataclass."""
        theme = ThemeConfig(
            name="custom",
            background_color="#ffffff",
            text_color="#333333",
            primary_color="#4e79a7",
        )
        assert theme.name == "custom"

    def test_section_content(self):
        """Test SectionContent dataclass."""
        section = SectionContent(
            section_type=SectionType.OVERVIEW,
            title="Overview",
        )
        assert section.section_type == SectionType.OVERVIEW

    def test_report_config(self):
        """Test ReportConfig dataclass."""
        config = ReportConfig(
            title="Test Report",
            theme=ReportTheme.DARK,
        )
        assert config.title == "Test Report"
        assert config.theme == ReportTheme.DARK

    def test_profile_data(self):
        """Test ProfileData dataclass."""
        profile = ProfileData(
            table_name="users",
            row_count=1000,
            column_count=5,
            columns=[
                {"name": "id", "data_type": "integer", "null_count": 0},
                {"name": "email", "data_type": "string", "null_count": 10},
            ],
        )
        assert profile.table_name == "users"
        assert profile.row_count == 1000


class TestSVGChartRenderer:
    """Tests for SVGChartRenderer."""

    def test_creation(self):
        """Test SVGChartRenderer creation."""
        renderer = SVGChartRenderer()
        assert renderer is not None

    def test_supports_chart_types(self):
        """Test chart type support."""
        renderer = SVGChartRenderer()
        assert renderer.supports_chart_type(ChartType.BAR)
        assert renderer.supports_chart_type(ChartType.PIE)
        assert renderer.supports_chart_type(ChartType.GAUGE)

    def test_render_bar(self):
        """Test rendering bar chart."""
        renderer = SVGChartRenderer()
        data = ChartData(labels=["A", "B", "C"], values=[10, 20, 30], title="Bar")
        config = ChartConfig(chart_type=ChartType.BAR)
        svg = renderer.render(data, config)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "rect" in svg

    def test_render_pie(self):
        """Test rendering pie chart."""
        renderer = SVGChartRenderer()
        data = ChartData(labels=["A", "B", "C"], values=[30, 40, 30], title="Pie")
        config = ChartConfig(chart_type=ChartType.PIE)
        svg = renderer.render(data, config)
        assert "<svg" in svg
        assert "path" in svg

    def test_render_gauge(self):
        """Test rendering gauge chart."""
        renderer = SVGChartRenderer()
        data = ChartData(values=[75], title="Score", metadata={"min": 0, "max": 100})
        config = ChartConfig(chart_type=ChartType.GAUGE)
        svg = renderer.render(data, config)
        assert "<svg" in svg
        assert "75" in svg

    def test_render_line(self):
        """Test rendering line chart."""
        renderer = SVGChartRenderer()
        data = ChartData(labels=["1", "2", "3", "4"], values=[10, 20, 15, 25], title="Line")
        config = ChartConfig(chart_type=ChartType.LINE)
        svg = renderer.render(data, config)
        assert "<svg" in svg
        assert "polyline" in svg

    def test_empty_data(self):
        """Test rendering with empty data."""
        renderer = SVGChartRenderer()
        data = ChartData(labels=[], values=[])
        config = ChartConfig(chart_type=ChartType.BAR)
        result = renderer.render(data, config)
        assert "No data" in result


class TestRegistries:
    """Tests for registries."""

    def test_chart_renderer_registry(self):
        """Test ChartRendererRegistry."""
        registry = ChartRendererRegistry()
        svg = registry.get("svg")
        assert svg is not None

    def test_theme_registry(self):
        """Test ThemeRegistry."""
        registry = ThemeRegistry()
        light = registry.get(ReportTheme.LIGHT)
        dark = registry.get(ReportTheme.DARK)
        assert light is not None
        assert dark is not None


class TestReportTemplate:
    """Tests for ReportTemplate."""

    def test_creation(self):
        """Test ReportTemplate creation."""
        theme = THEME_CONFIGS[ReportTheme.LIGHT]
        template = ReportTemplate(theme)
        assert template is not None

    def test_get_css(self):
        """Test getting CSS."""
        theme = THEME_CONFIGS[ReportTheme.LIGHT]
        template = ReportTemplate(theme)
        css = template.get_css()
        assert "--bg-color" in css
        assert "--primary-color" in css

    def test_get_js(self):
        """Test getting JavaScript."""
        theme = THEME_CONFIGS[ReportTheme.LIGHT]
        template = ReportTemplate(theme)
        js = template.get_js()
        assert "addEventListener" in js

    def test_render_header(self):
        """Test rendering header."""
        theme = THEME_CONFIGS[ReportTheme.LIGHT]
        template = ReportTemplate(theme)
        config = ReportConfig(title="Test Report")
        header = template.render_header(config)
        assert "Test Report" in header


class TestProfileDataConverter:
    """Tests for ProfileDataConverter."""

    def test_creation(self):
        """Test ProfileDataConverter creation."""
        profile = ProfileData(table_name="test", row_count=100, column_count=3, columns=[])
        converter = ProfileDataConverter(profile)
        assert converter is not None

    def test_overview_section(self):
        """Test creating overview section."""
        profile = ProfileData(
            table_name="users",
            row_count=1000,
            column_count=5,
            columns=[
                {"name": "id", "data_type": "integer"},
                {"name": "email", "data_type": "string"},
            ],
        )
        converter = ProfileDataConverter(profile)
        section = converter.create_overview_section()
        assert section.section_type == SectionType.OVERVIEW

    def test_quality_section(self):
        """Test creating quality section."""
        profile = ProfileData(
            table_name="test",
            row_count=100,
            column_count=2,
            columns=[{"name": "a", "null_count": 5}],
            quality_scores={"overall": 0.95},
        )
        converter = ProfileDataConverter(profile)
        section = converter.create_quality_section()
        assert section.section_type == SectionType.DATA_QUALITY


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator."""

    def test_creation(self):
        """Test generator creation."""
        generator = HTMLReportGenerator()
        assert generator is not None

    def test_generate(self):
        """Test generating HTML report."""
        generator = HTMLReportGenerator()
        profile = ProfileData(
            table_name="users",
            row_count=1000,
            column_count=3,
            columns=[
                {"name": "id", "data_type": "integer", "null_count": 0, "unique_count": 1000},
                {"name": "name", "data_type": "string", "null_count": 10, "unique_count": 900},
            ],
        )
        html = generator.generate(profile)
        assert "<!DOCTYPE html>" in html
        assert "users" in html
        assert "</html>" in html

    def test_generate_with_themes(self):
        """Test generating with different themes."""
        generator = HTMLReportGenerator()
        profile = ProfileData(table_name="test", row_count=100, column_count=1, columns=[])

        for theme in [ReportTheme.LIGHT, ReportTheme.DARK, ReportTheme.PROFESSIONAL]:
            config = ReportConfig(title="Test", theme=theme)
            html = generator.generate(profile, config)
            assert "<!DOCTYPE html>" in html

    def test_save(self):
        """Test saving report to file."""
        generator = HTMLReportGenerator()
        profile = ProfileData(
            table_name="test",
            row_count=50,
            column_count=1,
            columns=[{"name": "a", "data_type": "string"}],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result = generator.save(profile, output_path)
            assert result.exists()
            content = result.read_text()
            assert "<!DOCTYPE html>" in content


class TestReportExporter:
    """Tests for ReportExporter."""

    def test_creation(self):
        """Test exporter creation."""
        exporter = ReportExporter()
        assert exporter is not None

    def test_to_html(self):
        """Test exporting to HTML."""
        exporter = ReportExporter()
        profile = ProfileData(table_name="test", row_count=100, column_count=2, columns=[])
        html = exporter.to_html(profile)
        assert "<!DOCTYPE html>" in html

    def test_to_json(self):
        """Test exporting to JSON."""
        exporter = ReportExporter()
        profile = ProfileData(
            table_name="test",
            row_count=100,
            column_count=2,
            columns=[{"name": "a", "data_type": "string"}],
            quality_scores={"overall": 0.9},
        )
        json_str = exporter.to_json(profile)
        data = json.loads(json_str)
        assert data["table_name"] == "test"
        assert data["row_count"] == 100

    def test_to_file(self):
        """Test exporting to file."""
        exporter = ReportExporter()
        profile = ProfileData(table_name="test", row_count=50, column_count=1, columns=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            exporter.to_file(profile, html_path, format="html")
            assert html_path.exists()

            json_path = Path(tmpdir) / "report.json"
            exporter.to_file(profile, json_path, format="json")
            assert json_path.exists()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_report(self):
        """Test generate_report function."""
        profile = ProfileData(table_name="test", row_count=100, column_count=2, columns=[])
        html = generate_report(profile)
        assert "<!DOCTYPE html>" in html

    def test_generate_report_with_dict(self):
        """Test generate_report with dict input."""
        profile_dict = {
            "table_name": "users",
            "row_count": 500,
            "column_count": 3,
            "columns": [{"name": "id", "data_type": "integer"}],
        }
        html = generate_report(profile_dict)
        assert "users" in html

    def test_generate_report_save(self):
        """Test generate_report with save."""
        profile = ProfileData(table_name="test", row_count=50, column_count=1, columns=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"
            result = generate_report(profile, output_path=output_path)
            assert isinstance(result, Path)
            assert result.exists()

    def test_compare_profile_reports(self):
        """Test compare_profile_reports function."""
        profiles = [
            ProfileData("table1", 100, 3, []),
            ProfileData("table2", 200, 4, []),
        ]
        html = compare_profile_reports(profiles, labels=["Before", "After"])
        assert "<!DOCTYPE html>" in html
        assert "Before" in html
        assert "After" in html


# =============================================================================
# Integration Tests
# =============================================================================

class TestP3Integration:
    """Integration tests for P3 modules."""

    def test_visualization_with_recommendations(self):
        """Test visualization with recommendations."""
        recommendations = [
            "Apply 5% null threshold for id column",
            "Apply 10% null threshold for name column",
        ]

        profile = ProfileData(
            table_name="users",
            row_count=1000,
            column_count=2,
            columns=[
                {"name": "id", "null_count": 0, "unique_count": 1000},
                {"name": "name", "null_count": 50, "unique_count": 900},
            ],
            recommendations=recommendations,
        )

        html = generate_report(profile)
        assert "Recommendations" in html

    def test_visualization_with_quality_scores(self):
        """Test visualization with quality scores."""
        profile = ProfileData(
            table_name="orders",
            row_count=1000,
            column_count=3,
            columns=[
                {"name": "id", "null_count": 0, "unique_count": 1000},
                {"name": "amount", "null_count": 10, "unique_count": 800},
                {"name": "status", "null_count": 0, "unique_count": 5},
            ],
            quality_scores={"overall": 0.95, "completeness": 0.97},
        )

        config = ReportConfig(
            title="Order Data Profile",
            theme=ReportTheme.PROFESSIONAL,
            include_toc=True,
        )
        html = generate_report(profile, config=config)
        assert "<!DOCTYPE html>" in html
        assert "orders" in html.lower()

    def test_all_themes_render(self):
        """Test all themes render correctly."""
        profile = ProfileData(
            table_name="test",
            row_count=100,
            column_count=2,
            columns=[
                {"name": "a", "data_type": "integer", "null_count": 0},
                {"name": "b", "data_type": "string", "null_count": 5},
            ],
        )

        for theme in ReportTheme:
            config = ReportConfig(title=f"Test {theme.value}", theme=theme)
            html = generate_report(profile, config=config)
            assert "<!DOCTYPE html>" in html
            assert "</html>" in html

    def test_all_chart_types_render(self):
        """Test all chart types render."""
        renderer = SVGChartRenderer()
        data = ChartData(labels=["A", "B", "C"], values=[10, 20, 30], title="Test")

        for chart_type in [ChartType.BAR, ChartType.PIE, ChartType.LINE, ChartType.HISTOGRAM]:
            if renderer.supports_chart_type(chart_type):
                config = ChartConfig(chart_type=chart_type)
                svg = renderer.render(data, config)
                assert "<svg" in svg or "No data" in svg
