"""Tests for suite generation system.

Tests cover:
- Suite export formatters (YAML, JSON, Python, TOML, Checkpoint)
- Configuration management (presets, file I/O)
- CLI handlers
- Export post-processors
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_column_profile():
    """Create a sample column profile for testing."""
    from truthound.profiler import ColumnProfile, DataType, PatternMatch

    return ColumnProfile(
        name="email",
        physical_type="Utf8",
        inferred_type=DataType.EMAIL,
        row_count=1000,
        null_count=10,
        null_ratio=0.01,
        distinct_count=900,
        unique_ratio=0.9,
        detected_patterns=(
            PatternMatch(
                pattern="email",
                regex=r"^[\w\.-]+@[\w\.-]+\.\w+$",
                match_ratio=0.99,
                sample_matches=("test@example.com",),
            ),
        ),
    )


@pytest.fixture
def sample_table_profile(sample_column_profile):
    """Create a sample table profile for testing."""
    from truthound.profiler import TableProfile, ColumnProfile, DataType

    return TableProfile(
        name="test_data",
        source="test.csv",
        row_count=1000,
        column_count=3,
        columns=(
            sample_column_profile,
            ColumnProfile(
                name="age",
                physical_type="Int64",
                inferred_type=DataType.INTEGER,
                row_count=1000,
                null_count=0,
                null_ratio=0.0,
                distinct_count=80,
                unique_ratio=0.08,
            ),
            ColumnProfile(
                name="name",
                physical_type="Utf8",
                inferred_type=DataType.STRING,
                row_count=1000,
                null_count=50,
                null_ratio=0.05,
                distinct_count=950,
                unique_ratio=0.95,
            ),
        ),
    )


@pytest.fixture
def sample_suite(sample_table_profile):
    """Create a sample validation suite for testing."""
    from truthound.profiler import generate_suite

    return generate_suite(
        sample_table_profile,
        strictness="medium",
        name="test_suite",
    )


# =============================================================================
# Tests: Suite Export Formatters
# =============================================================================


class TestYAMLFormatter:
    """Tests for YAML formatter."""

    def test_format_basic(self, sample_suite):
        """Test basic YAML formatting."""
        from truthound.profiler import YAMLFormatter, ExportConfig

        formatter = YAMLFormatter()
        config = ExportConfig()

        result = formatter.format(sample_suite, config)

        assert "# Validation Suite: test_suite" in result
        assert "rules:" in result
        assert "validator:" in result

    def test_format_with_grouping(self, sample_suite):
        """Test YAML formatting with category grouping."""
        from truthound.profiler import YAMLFormatter, ExportConfig

        formatter = YAMLFormatter()
        config = ExportConfig(group_by_category=True)

        result = formatter.format(sample_suite, config)

        assert "# Category:" in result

    def test_format_minimal(self, sample_suite):
        """Test minimal YAML formatting."""
        from truthound.profiler import YAMLFormatter, MINIMAL_CONFIG

        formatter = YAMLFormatter()
        result = formatter.format(sample_suite, MINIMAL_CONFIG)

        # Should not include rationale
        assert "rationale:" not in result


class TestJSONFormatter:
    """Tests for JSON formatter."""

    def test_format_basic(self, sample_suite):
        """Test basic JSON formatting."""
        from truthound.profiler import JSONFormatter, ExportConfig

        formatter = JSONFormatter()
        config = ExportConfig()

        result = formatter.format(sample_suite, config)
        data = json.loads(result)

        assert data["name"] == "test_suite"
        assert "rules" in data
        assert isinstance(data["rules"], list)

    def test_format_with_grouping(self, sample_suite):
        """Test JSON formatting with category grouping."""
        from truthound.profiler import JSONFormatter, ExportConfig

        formatter = JSONFormatter()
        config = ExportConfig(group_by_category=True)

        result = formatter.format(sample_suite, config)
        data = json.loads(result)

        # Rules should be a dict when grouped
        assert isinstance(data["rules"], dict)

    def test_format_includes_summary(self, sample_suite):
        """Test that JSON includes summary."""
        from truthound.profiler import JSONFormatter, ExportConfig

        formatter = JSONFormatter()
        config = ExportConfig(include_summary=True)

        result = formatter.format(sample_suite, config)
        data = json.loads(result)

        assert "summary" in data
        assert "total_rules" in data["summary"]


class TestPythonFormatter:
    """Tests for Python code formatter."""

    def test_format_functional(self, sample_suite):
        """Test functional-style Python code generation."""
        from truthound.profiler import PythonFormatter, ExportConfig, CodeStyle

        formatter = PythonFormatter()
        config = ExportConfig(code_style=CodeStyle.FUNCTIONAL)

        result = formatter.format(sample_suite, config)

        assert "def create_validators()" in result
        assert "validators = []" in result
        assert "validators.append(" in result
        assert "return validators" in result

    def test_format_class_based(self, sample_suite):
        """Test class-based Python code generation."""
        from truthound.profiler import PythonFormatter, ExportConfig, CodeStyle

        formatter = PythonFormatter()
        config = ExportConfig(code_style=CodeStyle.CLASS_BASED)

        result = formatter.format(sample_suite, config)

        assert "class " in result
        assert "ValidationSuite:" in result
        assert "def create_validators(self)" in result

    def test_format_declarative(self, sample_suite):
        """Test declarative-style Python code generation."""
        from truthound.profiler import PythonFormatter, ExportConfig, CodeStyle

        formatter = PythonFormatter()
        config = ExportConfig(code_style=CodeStyle.DECLARATIVE)

        result = formatter.format(sample_suite, config)

        assert "SUITE_CONFIG = {" in result
        assert '"rules": [' in result

    def test_includes_imports(self, sample_suite):
        """Test that Python code includes imports."""
        from truthound.profiler import PythonFormatter, ExportConfig

        formatter = PythonFormatter()
        config = ExportConfig(include_imports=True)

        result = formatter.format(sample_suite, config)

        assert "from truthound.validators import" in result


class TestTOMLFormatter:
    """Tests for TOML formatter."""

    def test_format_basic(self, sample_suite):
        """Test basic TOML formatting."""
        from truthound.profiler import TOMLFormatter, ExportConfig

        formatter = TOMLFormatter()
        config = ExportConfig()

        result = formatter.format(sample_suite, config)

        assert "[suite]" in result
        assert 'name = "test_suite"' in result
        assert "[[rules]]" in result


class TestCheckpointFormatter:
    """Tests for Checkpoint formatter."""

    def test_format_basic(self, sample_suite):
        """Test basic Checkpoint formatting."""
        from truthound.profiler import CheckpointFormatter, ExportConfig

        formatter = CheckpointFormatter()
        config = ExportConfig()

        result = formatter.format(sample_suite, config)

        assert "checkpoints:" in result
        assert "validators:" in result
        assert "data_source:" in result


# =============================================================================
# Tests: Suite Exporter
# =============================================================================


class TestSuiteExporter:
    """Tests for SuiteExporter."""

    def test_export_to_string(self, sample_suite):
        """Test exporting to string."""
        from truthound.profiler import SuiteExporter

        exporter = SuiteExporter(format="yaml")
        result = exporter.export_to_string(sample_suite)

        assert "# Validation Suite:" in result
        assert "rules:" in result

    def test_export_to_file(self, sample_suite):
        """Test exporting to file."""
        from truthound.profiler import SuiteExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rules.yaml"
            exporter = SuiteExporter(format="yaml")
            result = exporter.export(sample_suite, output_path)

            assert result.success
            assert output_path.exists()
            content = output_path.read_text()
            assert "rules:" in content

    def test_export_creates_parent_dirs(self, sample_suite):
        """Test that export creates parent directories."""
        from truthound.profiler import SuiteExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "rules.json"
            exporter = SuiteExporter(format="json")
            result = exporter.export(sample_suite, output_path)

            assert result.success
            assert output_path.exists()

    def test_export_with_post_processor(self, sample_suite):
        """Test export with post-processor."""
        from truthound.profiler import (
            SuiteExporter,
            AddHeaderPostProcessor,
        )

        exporter = SuiteExporter(format="yaml")
        exporter.add_post_processor(
            AddHeaderPostProcessor("# Custom Header")
        )

        result = exporter.export_to_string(sample_suite)
        assert result.startswith("# Custom Header")


# =============================================================================
# Tests: Formatter Registry
# =============================================================================


class TestFormatterRegistry:
    """Tests for FormatterRegistry."""

    def test_get_builtin_formatters(self):
        """Test getting built-in formatters."""
        from truthound.profiler import formatter_registry

        for name in ["yaml", "json", "python", "toml", "checkpoint"]:
            formatter = formatter_registry.get(name)
            assert formatter is not None

    def test_create_formatter(self):
        """Test creating formatter instance."""
        from truthound.profiler import formatter_registry

        formatter = formatter_registry.create("yaml")
        assert formatter.format_name == "yaml"

    def test_list_all_formatters(self):
        """Test listing all formatters."""
        from truthound.profiler import formatter_registry

        all_formatters = formatter_registry.list_all()

        assert "yaml" in all_formatters
        assert "json" in all_formatters
        assert "python" in all_formatters

    def test_register_custom_formatter(self):
        """Test registering custom formatter."""
        from truthound.profiler import (
            SuiteFormatter,
            formatter_registry,
            ExportConfig,
        )

        class CustomFormatter(SuiteFormatter):
            format_name = "custom_test"
            file_extension = ".custom"

            def format(self, suite, config):
                return f"Custom: {suite.name}"

        formatter_registry.register(CustomFormatter)
        formatter = formatter_registry.create("custom_test")

        assert formatter.format_name == "custom_test"


# =============================================================================
# Tests: Configuration
# =============================================================================


class TestSuiteGeneratorConfig:
    """Tests for SuiteGeneratorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from truthound.profiler import SuiteGeneratorConfig

        config = SuiteGeneratorConfig()

        assert config.strictness == "medium"
        assert config.output.format == "yaml"

    def test_from_preset(self):
        """Test creating config from preset."""
        from truthound.profiler import SuiteGeneratorConfig, ConfigPreset

        config = SuiteGeneratorConfig.from_preset(ConfigPreset.STRICT)

        assert config.strictness == "strict"
        assert config.confidence.min_level == "medium"

    def test_from_preset_string(self):
        """Test creating config from preset string."""
        from truthound.profiler import SuiteGeneratorConfig

        config = SuiteGeneratorConfig.from_preset("ci_cd")

        assert config.name == "ci_cd_validation"
        assert config.output.format == "checkpoint"

    def test_with_overrides(self):
        """Test config with overrides."""
        from truthound.profiler import SuiteGeneratorConfig

        config = SuiteGeneratorConfig(strictness="loose")
        new_config = config.with_overrides(strictness="strict")

        assert new_config.strictness == "strict"
        assert config.strictness == "loose"  # Original unchanged

    def test_to_dict(self):
        """Test converting config to dict."""
        from truthound.profiler import SuiteGeneratorConfig

        config = SuiteGeneratorConfig(name="test")
        data = config.to_dict()

        assert data["name"] == "test"
        assert "strictness" in data
        assert "categories" in data

    def test_from_dict(self):
        """Test creating config from dict."""
        from truthound.profiler import SuiteGeneratorConfig

        data = {
            "name": "from_dict",
            "strictness": "strict",
            "categories": {"include": ["schema"]},
        }
        config = SuiteGeneratorConfig.from_dict(data)

        assert config.name == "from_dict"
        assert config.strictness == "strict"
        assert config.categories.include == ["schema"]

    def test_invalid_strictness_raises(self):
        """Test that invalid strictness raises error."""
        from truthound.profiler import SuiteGeneratorConfig

        with pytest.raises(ValueError, match="Invalid strictness"):
            SuiteGeneratorConfig(strictness="invalid")


class TestCategoryConfig:
    """Tests for CategoryConfig."""

    def test_should_include(self):
        """Test category inclusion logic."""
        from truthound.profiler import CategoryConfig

        config = CategoryConfig(include=["schema", "format"])

        assert config.should_include("schema")
        assert config.should_include("format")
        assert not config.should_include("pattern")

    def test_should_exclude(self):
        """Test category exclusion logic."""
        from truthound.profiler import CategoryConfig

        config = CategoryConfig(exclude=["anomaly"])

        assert config.should_include("schema")
        assert not config.should_include("anomaly")


class TestConfigPresets:
    """Tests for configuration presets."""

    def test_all_presets_valid(self):
        """Test that all presets create valid configs."""
        from truthound.profiler import ConfigPreset, PRESETS

        for preset in ConfigPreset:
            config = PRESETS[preset]
            assert config.strictness in {"loose", "medium", "strict"}

    def test_ci_cd_preset(self):
        """Test CI/CD preset configuration."""
        from truthound.profiler import SuiteGeneratorConfig, ConfigPreset

        config = SuiteGeneratorConfig.from_preset(ConfigPreset.CI_CD)

        assert config.output.format == "checkpoint"

    def test_minimal_preset(self):
        """Test minimal preset configuration."""
        from truthound.profiler import SuiteGeneratorConfig, ConfigPreset

        config = SuiteGeneratorConfig.from_preset(ConfigPreset.MINIMAL)

        assert config.categories.include == ["schema"]
        assert config.confidence.min_level == "high"


class TestConfigFileIO:
    """Tests for configuration file I/O."""

    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        from truthound.profiler import (
            SuiteGeneratorConfig,
            save_suite_config,
            load_suite_config,
        )

        config = SuiteGeneratorConfig(
            name="test_config",
            strictness="strict",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_suite_config(config, path)

            loaded = load_suite_config(path)

            assert loaded.name == "test_config"
            assert loaded.strictness == "strict"


# =============================================================================
# Tests: CLI Handlers
# =============================================================================


class TestSuiteGenerationHandler:
    """Tests for SuiteGenerationHandler."""

    def test_generate_with_valid_profile(self, sample_table_profile):
        """Test generation with valid profile."""
        from truthound.profiler import (
            SuiteGenerationHandler,
            SuiteGenerationProgress,
            save_profile,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save profile
            profile_path = Path(tmpdir) / "profile.json"
            save_profile(sample_table_profile, profile_path)

            # Generate
            progress = SuiteGenerationProgress(verbose=False)
            handler = SuiteGenerationHandler(progress=progress)

            result = handler.generate(
                profile_path=profile_path,
                format="yaml",
                strictness="medium",
            )

            assert result.success
            assert result.suite is not None
            assert result.rule_count > 0

    def test_generate_with_output_file(self, sample_table_profile):
        """Test generation with output file."""
        from truthound.profiler import (
            SuiteGenerationHandler,
            SuiteGenerationProgress,
            save_profile,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "rules.yaml"
            save_profile(sample_table_profile, profile_path)

            progress = SuiteGenerationProgress(verbose=False)
            handler = SuiteGenerationHandler(progress=progress)

            result = handler.generate(
                profile_path=profile_path,
                output_path=output_path,
                format="yaml",
            )

            assert result.success
            assert output_path.exists()

    def test_generate_with_preset(self, sample_table_profile):
        """Test generation with preset."""
        from truthound.profiler import (
            SuiteGenerationHandler,
            SuiteGenerationProgress,
            save_profile,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"
            save_profile(sample_table_profile, profile_path)

            progress = SuiteGenerationProgress(verbose=False)
            handler = SuiteGenerationHandler(progress=progress)

            result = handler.generate(
                profile_path=profile_path,
                preset="strict",
            )

            assert result.success

    def test_generate_nonexistent_profile(self):
        """Test error handling for nonexistent profile."""
        from truthound.profiler import (
            SuiteGenerationHandler,
            SuiteGenerationProgress,
        )

        progress = SuiteGenerationProgress(verbose=False)
        handler = SuiteGenerationHandler(progress=progress)

        result = handler.generate(
            profile_path=Path("/nonexistent/profile.json"),
        )

        assert not result.success
        assert "not found" in result.message.lower()


# =============================================================================
# Tests: Post-Processors
# =============================================================================


class TestPostProcessors:
    """Tests for export post-processors."""

    def test_add_header(self, sample_suite):
        """Test AddHeaderPostProcessor."""
        from truthound.profiler import AddHeaderPostProcessor

        processor = AddHeaderPostProcessor("# My Header")
        result = processor.process("content", sample_suite)

        assert result.startswith("# My Header")

    def test_add_footer(self, sample_suite):
        """Test AddFooterPostProcessor."""
        from truthound.profiler import AddFooterPostProcessor

        processor = AddFooterPostProcessor("# Footer")
        result = processor.process("content", sample_suite)

        assert result.endswith("# Footer")

    def test_template_substitution(self, sample_suite):
        """Test TemplatePostProcessor."""
        from truthound.profiler import TemplatePostProcessor

        processor = TemplatePostProcessor({
            "VERSION": "1.0.0",
            "DATE": "2025-01-01",
        })
        result = processor.process(
            "Version: ${VERSION}, Date: ${DATE}",
            sample_suite,
        )

        assert "Version: 1.0.0" in result
        assert "Date: 2025-01-01" in result


# =============================================================================
# Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_available_formats(self):
        """Test get_available_formats."""
        from truthound.profiler import get_available_formats

        formats = get_available_formats()

        assert "yaml" in formats
        assert "json" in formats
        assert "python" in formats

    def test_get_available_presets(self):
        """Test get_available_presets."""
        from truthound.profiler import get_available_presets

        presets = get_available_presets()

        assert "default" in presets
        assert "strict" in presets
        assert "ci_cd" in presets

    def test_get_available_categories(self):
        """Test get_available_categories."""
        from truthound.profiler import get_available_categories

        categories = get_available_categories()

        assert "schema" in categories
        assert "format" in categories

    def test_format_suite(self, sample_suite):
        """Test format_suite convenience function."""
        from truthound.profiler import format_suite

        result = format_suite(sample_suite, format="yaml")

        assert "rules:" in result

    def test_export_suite(self, sample_suite):
        """Test export_suite convenience function."""
        from truthound.profiler import export_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rules.yaml"
            result = export_suite(sample_suite, path)

            assert result.success
            assert path.exists()

    def test_export_suite_infers_format(self, sample_suite):
        """Test that export_suite infers format from extension."""
        from truthound.profiler import export_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test .json extension
            json_path = Path(tmpdir) / "rules.json"
            result = export_suite(sample_suite, json_path)

            assert result.success
            content = json_path.read_text()
            data = json.loads(content)
            assert "rules" in data


# =============================================================================
# Tests: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for the suite generation system."""

    def test_full_workflow(self, sample_table_profile):
        """Test full workflow from profile to exported suite."""
        from truthound.profiler import (
            generate_suite,
            SuiteExporter,
            ExportConfig,
        )

        # Generate suite
        suite = generate_suite(
            sample_table_profile,
            strictness="medium",
            include_categories=["schema", "completeness"],
            name="integration_test",
        )

        assert len(suite) > 0

        # Export to multiple formats
        config = ExportConfig(include_metadata=True)

        yaml_exporter = SuiteExporter(format="yaml", config=config)
        yaml_output = yaml_exporter.export_to_string(suite)
        assert "rules:" in yaml_output

        json_exporter = SuiteExporter(format="json", config=config)
        json_output = json_exporter.export_to_string(suite)
        data = json.loads(json_output)
        assert data["name"] == "integration_test"

        python_exporter = SuiteExporter(format="python", config=config)
        python_output = python_exporter.export_to_string(suite)
        assert "def create_validators" in python_output

    def test_preset_workflow(self, sample_table_profile):
        """Test workflow using presets."""
        from truthound.profiler import (
            SuiteGeneratorConfig,
            ConfigPreset,
            generate_suite,
            SuiteExporter,
        )

        # Get preset
        config = SuiteGeneratorConfig.from_preset(ConfigPreset.CI_CD)

        # Generate suite with preset settings
        suite = generate_suite(
            sample_table_profile,
            strictness=config.strictness,
            name=config.name,
        )

        # Export using preset format
        exporter = SuiteExporter(format=config.output.format)
        output = exporter.export_to_string(suite)

        assert "checkpoints:" in output
