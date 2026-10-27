"""Tests for P3 profiler modules: Distributed, ML Inference, Auto Threshold, Visualization.

These tests verify the core functionality of P3 modules with simplified API tests.
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

# =============================================================================
# Distributed Processing Tests
# =============================================================================
from truthound.profiler.distributed import (
    BackendConfig,
    BackendRegistry,
    BackendType,
    LocalBackend,
    PartitionInfo,
    PartitionStrategy,
    WorkerResult,
    get_available_backends,
)
from truthound.profiler.visualization import (
    compare_profiles as compare_profile_reports,
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
    EnsembleModel,
    Feature,
    FeatureType,
    FeatureVector,
    InferenceResult,
    MLTypeInferrer,
    NaiveBayesModel,
    NameFeatureExtractor,
    RuleBasedModel,
    StatisticalFeatureExtractor,
    ValueFeatureExtractor,
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
    AdaptiveStrategy,
    BalancedStrategy,
    ColumnThresholds,
    ConservativeStrategy,
    DomainAwareStrategy,
    PermissiveStrategy,
    StatisticalStrategy,
    StrategyRegistry,
    StrictnessPreset,
    TableThresholds,
    ThresholdTester,
    ThresholdTuner,
    ThresholdType,
    TuningStrategy,
    create_tuner,
    get_available_strategies,
    tune_thresholds,
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
    COLOR_PALETTES,
    THEME_CONFIGS,
    ChartConfig,
    ChartData,
    ChartRendererRegistry,
    ChartType,
    ColorScheme,
    HTMLReportGenerator,
    ProfileData,
    ProfileDataConverter,
    ReportConfig,
    ReportExporter,
    ReportTemplate,
    ReportTheme,
    SectionContent,
    SectionType,
    SVGChartRenderer,
    ThemeConfig,
    ThemeRegistry,
    generate_report,
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


def test_ml_inference_config_threshold_and_context_are_honored(monkeypatch):
    from truthound.profiler.base import DataType
    from truthound.profiler.ml_inference import InferenceModel, InferrerConfig, model_registry
    class SelectedModel(InferenceModel):
        def predict(self, features):
            return [(DataType.INTEGER, .4), (DataType.STRING, .3)]
        def train(self, data):
            pass
    monkeypatch.setitem(model_registry._models, 'selected-test-model', SelectedModel)
    inferrer = MLTypeInferrer(config=InferrerConfig(model='selected-test-model', confidence_threshold=.8))
    context = {'table_name': 'selected'}
    result = inferrer.infer(pl.Series('x', [1, 2]), context)
    assert result.inferred_type == DataType.UNKNOWN
    assert result.alternatives[0] == (DataType.INTEGER, .4)
    assert context == {'table_name': 'selected'}


def test_ml_inference_does_not_cache_only_column_head_or_mutable_custom_state():
    from truthound.profiler.base import DataType
    from truthound.profiler.ml_inference import InferenceModel
    class CountingModel(InferenceModel):
        def __init__(self): self.calls = 0
        def predict(self, features):
            self.calls += 1
            return [(DataType.INTEGER if self.calls == 1 else DataType.STRING, 1.)]
        def train(self, data): pass
    model = CountingModel()
    inferrer = MLTypeInferrer(model=model)
    first = inferrer.infer(pl.Series('x', list(range(10)) + [1]))
    second = inferrer.infer(pl.Series('x', list(range(10)) + [None]), {'table_name': 'new'})
    assert first.inferred_type == DataType.INTEGER
    assert second.inferred_type == DataType.STRING and model.calls == 2


def test_ml_extractor_failure_is_not_silent_success():
    from truthound.profiler.ml_inference import FeatureExtractor, FeatureExtractorRegistry
    class FailedExtractor(FeatureExtractor):
        name = 'failed'
        def extract(self, column, context):
            raise RuntimeError('synthetic-private-extractor')
    registry = FeatureExtractorRegistry()
    registry.register(FailedExtractor())
    with pytest.raises(RuntimeError):
        registry.extract_all(pl.Series('x', [1]), {})


def test_ml_ensemble_persists_native_children_and_rejects_corruption(tmp_path):
    from truthound.profiler.base import DataType
    from truthound.profiler.ml_inference import Feature, FeatureType, FeatureVector
    model = EnsembleModel()
    features = FeatureVector('x', [Feature('selected', .7, FeatureType.STATISTICAL)])
    model.train([(features, DataType.INTEGER), (features, DataType.INTEGER)])
    expected = model.predict(features)
    path = tmp_path / 'ensemble.json'
    model.save(path)
    assert path.is_file()
    restored = EnsembleModel()
    restored.load(path)
    assert restored.predict(features) == expected
    path.write_text('{broken')
    with pytest.raises(ValueError):
        restored.load(path)
    assert restored.predict(features) == expected


def test_ml_failed_training_fences_inference_and_keeps_failed_feedback():
    from truthound.profiler.base import DataType
    from truthound.profiler.ml_inference import InferenceModel
    class FailedModel(InferenceModel):
        def predict(self, features):
            return [(DataType.INTEGER, 1.)]
        def train(self, data):
            raise RuntimeError('synthetic-train-failure')
    inferrer = MLTypeInferrer(model=FailedModel())
    column = pl.Series('x', [1, 2])
    for _ in range(99):
        inferrer.provide_feedback(column, DataType.INTEGER)
    with pytest.raises(RuntimeError, match='synthetic-train-failure'):
        inferrer.provide_feedback(column, DataType.INTEGER)
    with pytest.raises(RuntimeError, match='Reopen'):
        inferrer.infer(column)
    assert len(inferrer._feedback_buffer) == 100


def test_ml_cooperating_writers_reject_stale_model_before_training(tmp_path):
    from truthound.profiler.base import DataType
    from truthound.profiler.ml_inference import InferrerConfig
    path = tmp_path / 'native.json'
    config = InferrerConfig(model='naive_bayes', model_path=str(path), feedback_batch_size=1)
    first, stale = MLTypeInferrer(config=config), MLTypeInferrer(config=config)
    values = pl.Series('x', [1, 2])
    first.provide_feedback(values, DataType.INTEGER)
    with pytest.raises(RuntimeError, match='snapshot changed'):
        stale.provide_feedback(values, DataType.STRING)
    restored = MLTypeInferrer(config=config)
    assert restored.infer(values).inferred_type == DataType.INTEGER
    with pytest.raises(RuntimeError, match='Reopen'):
        stale.infer(values)


def test_profile_report_escapes_names_and_restores_override_after_failure(monkeypatch):
    from truthound.profiler.visualization import (
        HTMLReportGenerator,
        ProfileData,
        ReportConfig,
        ReportTheme,
    )
    data = ProfileData('<img src=x onerror=alert(1)>', 1, 0, [])
    generator = HTMLReportGenerator(ReportConfig(renderer='svg', title='Original'))
    original = generator.config
    text = generator.generate(data, ReportConfig(renderer='svg', theme=ReportTheme.DARK, title='Override'))
    assert '<img src=x onerror=alert(1)>' not in text
    assert generator.config is original
    def fail(*args):
        raise RuntimeError('render-failed')
    monkeypatch.setattr(HTMLReportGenerator, '_render_summary', fail)
    with pytest.raises(RuntimeError):
        generator.generate(data, ReportConfig(renderer='svg'))
    assert generator.config is original


def test_native_chart_script_cannot_be_terminated_by_profile_labels():
    from truthound.profiler.visualization import (
        ChartConfig,
        ChartData,
        ChartType,
        EChartsChartRenderer,
    )
    renderer = EChartsChartRenderer()
    html = renderer.render(ChartData(labels=['</script><img src=x onerror=alert(1)>'], values=[1]), ChartConfig(chart_type=ChartType.BAR))
    assert '</script><img' not in html
    assert html.count('</script>') == 1


@pytest.mark.parametrize('backend', ['plotly', 'echarts'])
def test_interactive_chart_rejects_misaligned_series_and_nonfinite_values(backend):
    from truthound.profiler.visualization import (
        ChartConfig,
        ChartData,
        ChartType,
        EChartsChartRenderer,
        PlotlyChartRenderer,
    )
    renderer = PlotlyChartRenderer() if backend == 'plotly' else EChartsChartRenderer()
    with pytest.raises(ValueError):
        renderer.render(ChartData(labels=['a', 'b'], values=[1]), ChartConfig(chart_type=ChartType.BAR))
    with pytest.raises(ValueError):
        renderer.render(ChartData(labels=['a'], values=[float('nan')]), ChartConfig(chart_type=ChartType.BAR))


@pytest.mark.parametrize('format', ['html', 'json'])
def test_report_export_uses_atomic_publication_and_preserves_previous_file(tmp_path, monkeypatch, format):
    import truthound.stores._filesystem_io as native_io
    target = tmp_path / ('report.' + format)
    target.write_bytes(b'previous')
    def fail(path, payload):
        raise OSError('synthetic publication failure')
    monkeypatch.setattr(native_io, 'publish', fail)
    with pytest.raises(OSError):
        ReportExporter().to_file(ProfileData(table_name='atomic', row_count=1, column_count=0, columns=[]), target, format=format)
    assert target.read_bytes() == b'previous'


@pytest.mark.parametrize('kind', [ChartType.BAR, ChartType.HORIZONTAL_BAR])
def test_svg_negative_values_have_valid_visible_geometry(kind):
    import xml.etree.ElementTree as ET
    document = SVGChartRenderer().render(ChartData(labels=['debit', 'credit'], values=[-2, 3]), ChartConfig(chart_type=kind))
    rectangles = ET.fromstring(document).findall('.//{http://www.w3.org/2000/svg}rect')
    assert len(rectangles) == 2
    assert all(float(item.attrib['height']) > 0 and float(item.attrib['width']) > 0 for item in rectangles)


def test_svg_rejects_mismatched_series_instead_of_silently_dropping_values():
    with pytest.raises(ValueError, match='matching'):
        SVGChartRenderer().render(ChartData(labels=['one'], values=[1, 2]), ChartConfig(chart_type=ChartType.BAR))


def test_unknown_explicit_renderer_is_rejected_instead_of_silent_svg_fallback():
    with pytest.raises(ValueError, match='renderer'):
        HTMLReportGenerator(ReportConfig(renderer='unregistered-renderer'))
