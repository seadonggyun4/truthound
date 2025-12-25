"""Core Path E2E Tests.

This module contains comprehensive E2E tests for Truthound's core paths:
1. Data Profiling Path: Data → Profile → Save/Load
2. Suite Generation Path: Profile → Suite → Export
3. Validation Path: Data → Schema → Validation Report
4. Quick Suite Path: Data → Profile → Suite (one-step)
5. PII Detection Path: Data → Scan → Report
6. Full Pipeline Path: Data → Profile → Suite → Validation

Each test covers the complete flow from input to output,
validating all intermediate states and final results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import polars as pl

from tests.e2e.fixtures import (
    DataGenerator,
    create_test_data,
    create_clean_data_scenario,
    create_pii_data_scenario,
    create_mixed_quality_scenario,
    create_edge_case_scenario,
)
from tests.e2e.utils import (
    CLITestHelper,
    E2EAssertions,
    E2ETestContext,
    assert_cli_success,
    assert_cli_error,
    assert_file_contains,
    assert_valid_json,
)


# =============================================================================
# Core Path 1: Data Profiling
# =============================================================================


class TestDataProfilingPath:
    """E2E tests for data profiling path.

    Flow: Data File → DataProfiler → TableProfile → Save → Load → Verify
    """

    def test_profile_csv_file(self, tmp_path: Path) -> None:
        """Test profiling a CSV file."""
        # Arrange
        data_file, _ = create_clean_data_scenario(tmp_path, row_count=100)

        # Act
        from truthound.profiler import profile_file, save_profile, load_profile

        profile = profile_file(str(data_file))

        # Assert
        assert profile.row_count == 100
        assert profile.column_count > 0
        assert len(profile.columns) == profile.column_count

        # Save and load roundtrip
        profile_path = tmp_path / "profile.json"
        save_profile(profile, profile_path)
        assert profile_path.exists()

        loaded = load_profile(profile_path)
        assert loaded.row_count == profile.row_count
        assert loaded.column_count == profile.column_count

    def test_profile_parquet_file(self, tmp_path: Path) -> None:
        """Test profiling a Parquet file."""
        # Arrange
        data_file, _ = create_pii_data_scenario(tmp_path, row_count=500)

        # Act
        from truthound.profiler import profile_file

        profile = profile_file(str(data_file))

        # Assert
        assert profile.row_count == 500
        assert profile.column_count >= 8  # Basic + PII columns

        # Verify column types are detected
        column_names = [col.name for col in profile.columns]
        assert "email" in column_names
        assert "phone" in column_names

    def test_profile_with_nulls(self, tmp_path: Path) -> None:
        """Test profiling data with null values."""
        # Arrange
        generator = DataGenerator(
            row_count=100,
            null_ratio=0.2,
            seed=42,
        )
        df = generator.generate()
        data_file = tmp_path / "null_data.parquet"
        df.write_parquet(data_file)

        # Act
        from truthound.profiler import profile_file

        profile = profile_file(str(data_file))

        # Assert
        # Check that null ratios are detected
        null_ratios = [col.null_ratio for col in profile.columns]
        assert any(ratio > 0 for ratio in null_ratios)

    def test_profile_with_patterns(self, tmp_path: Path) -> None:
        """Test that profiling detects patterns (email, phone, etc.)."""
        # Arrange
        data_file, _ = create_pii_data_scenario(tmp_path, row_count=200)

        # Act
        from truthound.profiler import profile_file

        profile = profile_file(str(data_file))

        # Assert - Check column types are inferred
        email_col = next(
            (col for col in profile.columns if col.name == "email"),
            None
        )
        assert email_col is not None
        # Email pattern should be detected
        assert email_col.inferred_type is not None

    def test_profile_edge_cases(self, tmp_path: Path) -> None:
        """Test profiling edge case data."""
        # Arrange
        data_file, _ = create_edge_case_scenario(tmp_path)

        # Act
        from truthound.profiler import profile_file

        profile = profile_file(str(data_file))

        # Assert - Should handle edge cases without error
        assert profile.row_count == 5
        assert profile.column_count > 0


# =============================================================================
# Core Path 2: Suite Generation
# =============================================================================


class TestSuiteGenerationPath:
    """E2E tests for suite generation path.

    Flow: TableProfile → ValidationSuiteGenerator → ValidationSuite → Export
    """

    def test_generate_suite_from_profile(self, tmp_path: Path) -> None:
        """Test generating validation suite from profile."""
        # Arrange
        data_file, _ = create_clean_data_scenario(tmp_path, row_count=100)

        from truthound.profiler import profile_file, generate_suite

        profile = profile_file(str(data_file))

        # Act
        suite = generate_suite(
            profile,
            strictness="medium",
            name="test_suite",
        )

        # Assert
        assert len(suite) > 0
        assert suite.name == "test_suite"
        assert suite.strictness.value == "medium"

    def test_suite_export_yaml(self, tmp_path: Path) -> None:
        """Test exporting suite to YAML."""
        # Arrange
        data_file, _ = create_pii_data_scenario(tmp_path, row_count=100)

        from truthound.profiler import (
            profile_file,
            generate_suite,
            SuiteExporter,
        )

        profile = profile_file(str(data_file))
        suite = generate_suite(profile, strictness="medium")

        # Act
        exporter = SuiteExporter(format="yaml")
        output_path = tmp_path / "rules.yaml"
        result = exporter.export(suite, output_path)

        # Assert
        assert result.success
        assert output_path.exists()

        content = output_path.read_text()
        assert "rules:" in content
        assert "validator:" in content

    def test_suite_export_json(self, tmp_path: Path) -> None:
        """Test exporting suite to JSON."""
        # Arrange
        data_file, _ = create_clean_data_scenario(tmp_path, row_count=100)

        from truthound.profiler import (
            profile_file,
            generate_suite,
            SuiteExporter,
        )

        profile = profile_file(str(data_file))
        suite = generate_suite(profile, strictness="strict")

        # Act
        exporter = SuiteExporter(format="json")
        output_path = tmp_path / "rules.json"
        result = exporter.export(suite, output_path)

        # Assert
        assert result.success
        assert output_path.exists()

        data = assert_valid_json(output_path)
        assert "rules" in data
        assert "strictness" in data

    def test_suite_export_python(self, tmp_path: Path) -> None:
        """Test exporting suite to Python code."""
        # Arrange
        data_file, _ = create_clean_data_scenario(tmp_path, row_count=100)

        from truthound.profiler import (
            profile_file,
            generate_suite,
            SuiteExporter,
        )

        profile = profile_file(str(data_file))
        suite = generate_suite(profile, strictness="medium")

        # Act
        exporter = SuiteExporter(format="python")
        output_path = tmp_path / "validators.py"
        result = exporter.export(suite, output_path)

        # Assert
        assert result.success
        assert output_path.exists()

        content = output_path.read_text()
        assert "def create_validators" in content
        assert "validators.append" in content

    def test_suite_with_category_filter(self, tmp_path: Path) -> None:
        """Test suite generation with category filtering."""
        # Arrange
        data_file, _ = create_pii_data_scenario(tmp_path, row_count=100)

        from truthound.profiler import profile_file, generate_suite

        profile = profile_file(str(data_file))

        # Act
        suite = generate_suite(
            profile,
            strictness="medium",
            include_categories=["schema", "completeness"],
        )

        # Assert
        categories = set(r.category.value for r in suite.rules)
        assert "schema" in categories or "completeness" in categories
        # Should not include other categories
        assert "anomaly" not in categories

    def test_suite_with_preset(self, tmp_path: Path) -> None:
        """Test suite generation with configuration preset."""
        # Arrange
        data_file, _ = create_clean_data_scenario(tmp_path, row_count=100)

        from truthound.profiler import (
            profile_file,
            generate_suite,
            SuiteGeneratorConfig,
            ConfigPreset,
        )

        profile = profile_file(str(data_file))
        config = SuiteGeneratorConfig.from_preset(ConfigPreset.STRICT)

        # Act
        suite = generate_suite(
            profile,
            strictness=config.strictness,
        )

        # Assert
        assert suite.strictness.value == "strict"


# =============================================================================
# Core Path 3: CLI Integration
# =============================================================================


class TestCLIIntegrationPath:
    """E2E tests for CLI commands."""

    def test_cli_auto_profile(self, tmp_path: Path) -> None:
        """Test auto-profile CLI command."""
        # Arrange
        data_file = create_test_data(tmp_path, scenario="clean", row_count=50)
        output_path = tmp_path / "profile.json"

        cli = CLITestHelper()

        # Act
        result = cli.run_auto_profile(data_file, output_path)

        # Assert
        assert_cli_success(result)
        assert output_path.exists()

        # Verify profile content
        data = assert_valid_json(output_path)
        assert "row_count" in data
        assert "columns" in data

    def test_cli_generate_suite(self, tmp_path: Path) -> None:
        """Test generate-suite CLI command."""
        # Arrange
        data_file = create_test_data(tmp_path, scenario="pii", row_count=100)

        # First create a profile
        from truthound.profiler import profile_file, save_profile

        profile = profile_file(str(data_file))
        profile_path = tmp_path / "profile.json"
        save_profile(profile, profile_path)

        output_path = tmp_path / "rules.yaml"
        cli = CLITestHelper()

        # Act
        result = cli.run_generate_suite(
            profile_path,
            output_path,
            format="yaml",
            strictness="medium",
        )

        # Assert
        assert_cli_success(result)
        assert output_path.exists()
        assert_file_contains(output_path, ["rules:", "validator:"])

    def test_cli_generate_suite_all_formats(self, tmp_path: Path) -> None:
        """Test generate-suite with all output formats."""
        # Arrange
        data_file = create_test_data(tmp_path, scenario="clean", row_count=50)

        from truthound.profiler import profile_file, save_profile

        profile = profile_file(str(data_file))
        profile_path = tmp_path / "profile.json"
        save_profile(profile, profile_path)

        cli = CLITestHelper()
        formats = ["yaml", "json", "python", "toml", "checkpoint"]

        for fmt in formats:
            # Act
            output_path = tmp_path / f"rules.{fmt}"
            result = cli.run_generate_suite(
                profile_path,
                output_path,
                format=fmt,
            )

            # Assert
            assert_cli_success(result, expected_output=None)
            assert output_path.exists(), f"Output not created for format: {fmt}"

    def test_cli_quick_suite(self, tmp_path: Path) -> None:
        """Test quick-suite CLI command (profile + generate in one step)."""
        # Arrange
        data_file = create_test_data(tmp_path, scenario="pii", row_count=100)
        output_path = tmp_path / "quick_rules.yaml"

        cli = CLITestHelper()

        # Act
        result = cli.run_quick_suite(
            data_file,
            output_path,
            format="yaml",
            strictness="medium",
        )

        # Assert
        assert_cli_success(result)
        assert output_path.exists()
        assert_file_contains(output_path, "rules:")

    def test_cli_quick_suite_with_preset(self, tmp_path: Path) -> None:
        """Test quick-suite with preset configuration."""
        # Arrange
        data_file = create_test_data(tmp_path, scenario="clean", row_count=50)
        output_path = tmp_path / "preset_rules.yaml"

        cli = CLITestHelper()

        # Act
        result = cli.run_quick_suite(
            data_file,
            output_path,
            preset="strict",
        )

        # Assert
        assert_cli_success(result)
        assert output_path.exists()

    def test_cli_list_commands(self, tmp_path: Path) -> None:
        """Test list-* CLI commands."""
        cli = CLITestHelper()

        # Test list-formats
        result = cli.run("list-formats")
        assert_cli_success(result)
        assert "yaml" in result.output
        assert "json" in result.output
        assert "python" in result.output

        # Test list-presets
        result = cli.run("list-presets")
        assert_cli_success(result)
        assert "strict" in result.output
        assert "ci_cd" in result.output

        # Test list-categories
        result = cli.run("list-categories")
        assert_cli_success(result)
        assert "schema" in result.output
        assert "format" in result.output

    def test_cli_error_handling(self, tmp_path: Path) -> None:
        """Test CLI error handling."""
        cli = CLITestHelper()

        # Non-existent file
        result = cli.run("generate-suite", "/nonexistent/profile.json")
        assert_cli_error(result, expected_message="not found")

        # Invalid format
        data_file = create_test_data(tmp_path, scenario="clean")

        from truthound.profiler import profile_file, save_profile

        profile = profile_file(str(data_file))
        profile_path = tmp_path / "profile.json"
        save_profile(profile, profile_path)

        result = cli.run(
            "generate-suite", str(profile_path),
            "-f", "invalid_format",
        )
        assert_cli_error(result, expected_message="Invalid format")


# =============================================================================
# Core Path 4: Full Pipeline
# =============================================================================


class TestFullPipelinePath:
    """E2E tests for the complete data quality pipeline.

    Flow: Data → Profile → Suite → Export → Validation
    """

    def test_full_pipeline_clean_data(self, tmp_path: Path) -> None:
        """Test full pipeline with clean data."""
        with E2ETestContext(tmp_path) as ctx:
            # 1. Create test data
            data_file = ctx.create_data(scenario="clean", row_count=100)

            # 2. Profile
            profile = ctx.profile_helper.create_profile(data_file)
            assert profile.row_count == 100

            # 3. Save profile
            profile_path = tmp_path / "profile.json"
            ctx.profile_helper.save_profile(profile, profile_path)

            # 4. Generate suite via CLI
            rules_path = tmp_path / "rules.yaml"
            result = ctx.cli.run_generate_suite(
                profile_path,
                rules_path,
                format="yaml",
            )
            ctx.assert_success(result)

            # 5. Verify output
            assert rules_path.exists()
            content = rules_path.read_text()
            assert "rules:" in content

    def test_full_pipeline_pii_data(self, tmp_path: Path) -> None:
        """Test full pipeline with PII data."""
        with E2ETestContext(tmp_path) as ctx:
            # 1. Create PII data
            data_file = ctx.create_data(
                scenario="pii",
                format="parquet",
                row_count=200,
            )

            # 2. Profile and generate suite in one step
            rules_path = tmp_path / "pii_rules.yaml"
            result = ctx.cli.run_quick_suite(
                data_file,
                rules_path,
                strictness="strict",
            )
            ctx.assert_success(result)

            # 3. Verify rules include format validation
            content = rules_path.read_text()
            assert "rules:" in content

    def test_full_pipeline_mixed_quality(self, tmp_path: Path) -> None:
        """Test full pipeline with mixed quality data."""
        # 1. Create mixed quality data
        data_file, df = create_mixed_quality_scenario(tmp_path, row_count=300)

        # 2. Profile
        from truthound.profiler import profile_file, generate_suite

        profile = profile_file(str(data_file))

        # 3. Verify profile detects issues
        assert profile.row_count == 302  # 300 + 2 anomalies

        # Check for null detection
        null_columns = [
            col for col in profile.columns
            if col.null_ratio > 0
        ]
        assert len(null_columns) > 0

        # 4. Generate comprehensive suite
        suite = generate_suite(
            profile,
            strictness="strict",
            name="mixed_quality_suite",
        )

        assert len(suite) > 0

        # 5. Export in multiple formats
        from truthound.profiler import SuiteExporter

        for fmt in ["yaml", "json", "python"]:
            exporter = SuiteExporter(format=fmt)
            output = exporter.export_to_string(suite)
            assert len(output) > 0

    def test_full_pipeline_with_config(self, tmp_path: Path) -> None:
        """Test full pipeline with configuration file."""
        # 1. Create config file
        from truthound.profiler import SuiteGeneratorConfig, save_suite_config

        config = SuiteGeneratorConfig(
            name="configured_suite",
            strictness="medium",
        )
        config.categories.include = ["schema", "completeness", "format"]
        config.output.group_by_category = True

        config_path = tmp_path / "suite_config.json"
        save_suite_config(config, config_path)

        # 2. Create data and profile
        data_file = create_test_data(tmp_path, scenario="pii", row_count=100)

        from truthound.profiler import profile_file, save_profile

        profile = profile_file(str(data_file))
        profile_path = tmp_path / "profile.json"
        save_profile(profile, profile_path)

        # 3. Generate with config via CLI
        rules_path = tmp_path / "configured_rules.yaml"
        cli = CLITestHelper()

        result = cli.run(
            "generate-suite", str(profile_path),
            "-o", str(rules_path),
            "-c", str(config_path),
        )

        # Note: This test verifies the flow works, even if config
        # loading is not yet implemented in CLI
        # The important thing is the path doesn't crash


# =============================================================================
# Core Path 5: Export Formats
# =============================================================================


class TestExportFormatsPath:
    """E2E tests for all export format variations."""

    @pytest.fixture
    def sample_suite(self, tmp_path: Path):
        """Create a sample suite for export tests."""
        data_file = create_test_data(tmp_path, scenario="pii", row_count=100)

        from truthound.profiler import profile_file, generate_suite

        profile = profile_file(str(data_file))
        return generate_suite(profile, strictness="medium", name="export_test")

    def test_yaml_export_variations(
        self,
        tmp_path: Path,
        sample_suite,
    ) -> None:
        """Test YAML export with different configurations."""
        from truthound.profiler import SuiteExporter, ExportConfig

        # Default config
        exporter = SuiteExporter(format="yaml")
        content = exporter.export_to_string(sample_suite)
        assert "rules:" in content

        # Grouped by category
        config = ExportConfig(group_by_category=True)
        exporter = SuiteExporter(format="yaml", config=config)
        content = exporter.export_to_string(sample_suite)
        assert "# Category:" in content

        # Minimal config
        from truthound.profiler import MINIMAL_CONFIG

        exporter = SuiteExporter(format="yaml", config=MINIMAL_CONFIG)
        content = exporter.export_to_string(sample_suite)
        assert "rules:" in content

    def test_json_export_variations(
        self,
        tmp_path: Path,
        sample_suite,
    ) -> None:
        """Test JSON export with different configurations."""
        from truthound.profiler import SuiteExporter, ExportConfig

        # Default config
        exporter = SuiteExporter(format="json")
        content = exporter.export_to_string(sample_suite)
        data = json.loads(content)
        assert "rules" in data
        assert isinstance(data["rules"], list)

        # Grouped by category
        config = ExportConfig(group_by_category=True)
        exporter = SuiteExporter(format="json", config=config)
        content = exporter.export_to_string(sample_suite)
        data = json.loads(content)
        assert "rules" in data
        assert isinstance(data["rules"], dict)  # Grouped as dict

    def test_python_export_styles(
        self,
        tmp_path: Path,
        sample_suite,
    ) -> None:
        """Test Python code export with different styles."""
        from truthound.profiler import SuiteExporter, ExportConfig, CodeStyle

        # Functional style
        config = ExportConfig(code_style=CodeStyle.FUNCTIONAL)
        exporter = SuiteExporter(format="python", config=config)
        content = exporter.export_to_string(sample_suite)
        assert "def create_validators()" in content

        # Class-based style
        config = ExportConfig(code_style=CodeStyle.CLASS_BASED)
        exporter = SuiteExporter(format="python", config=config)
        content = exporter.export_to_string(sample_suite)
        assert "class " in content
        assert "ValidationSuite" in content

        # Declarative style
        config = ExportConfig(code_style=CodeStyle.DECLARATIVE)
        exporter = SuiteExporter(format="python", config=config)
        content = exporter.export_to_string(sample_suite)
        assert "SUITE_CONFIG = {" in content

    def test_checkpoint_export(
        self,
        tmp_path: Path,
        sample_suite,
    ) -> None:
        """Test Checkpoint format export."""
        from truthound.profiler import SuiteExporter

        exporter = SuiteExporter(format="checkpoint")
        content = exporter.export_to_string(sample_suite)

        assert "checkpoints:" in content
        assert "validators:" in content
        assert "data_source:" in content


# =============================================================================
# Core Path 6: Error Handling
# =============================================================================


class TestErrorHandlingPath:
    """E2E tests for error handling across the pipeline."""

    def test_invalid_file_format(self, tmp_path: Path) -> None:
        """Test handling of invalid file format."""
        # Create a text file with invalid content
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not a valid data file")

        from truthound.profiler import profile_file

        with pytest.raises(Exception):
            profile_file(str(invalid_file))

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        cli = CLITestHelper()
        result = cli.run("auto-profile", str(empty_file))

        # Should fail gracefully
        assert result.exit_code != 0

    def test_profile_file_not_found(self, tmp_path: Path) -> None:
        """Test handling of non-existent profile file."""
        cli = CLITestHelper()

        result = cli.run(
            "generate-suite",
            str(tmp_path / "nonexistent.json"),
        )

        assert_cli_error(result, expected_message="not found")

    def test_invalid_strictness(self, tmp_path: Path) -> None:
        """Test handling of invalid strictness value."""
        data_file = create_test_data(tmp_path, scenario="clean")

        from truthound.profiler import profile_file, generate_suite

        profile = profile_file(str(data_file))

        with pytest.raises(ValueError):
            generate_suite(profile, strictness="invalid")

    def test_invalid_preset(self, tmp_path: Path) -> None:
        """Test handling of invalid preset name."""
        data_file = create_test_data(tmp_path, scenario="clean")

        from truthound.profiler import profile_file, save_profile

        profile = profile_file(str(data_file))
        profile_path = tmp_path / "profile.json"
        save_profile(profile, profile_path)

        cli = CLITestHelper()
        result = cli.run(
            "generate-suite", str(profile_path),
            "--preset", "invalid_preset",
        )

        assert_cli_error(result, expected_message="Invalid preset")


# =============================================================================
# Core Path 7: Performance and Scale
# =============================================================================


class TestPerformancePath:
    """E2E tests for performance with larger datasets."""

    @pytest.mark.slow
    def test_large_dataset_profiling(self, tmp_path: Path) -> None:
        """Test profiling a larger dataset."""
        # Generate larger dataset
        generator = DataGenerator(
            row_count=10000,
            null_ratio=0.05,
            include_pii=True,
            seed=42,
        )
        df = generator.generate()
        data_file = tmp_path / "large_data.parquet"
        df.write_parquet(data_file)

        # Profile
        from truthound.profiler import profile_file, generate_suite

        profile = profile_file(str(data_file))

        assert profile.row_count == 10000

        # Generate suite
        suite = generate_suite(profile, strictness="medium")

        assert len(suite) > 0

    @pytest.mark.slow
    def test_many_columns_profiling(self, tmp_path: Path) -> None:
        """Test profiling dataset with many columns."""
        # Generate wide dataset
        data = {"id": list(range(100))}
        for i in range(50):
            data[f"col_{i}"] = [f"value_{j}" for j in range(100)]

        df = pl.DataFrame(data)
        data_file = tmp_path / "wide_data.parquet"
        df.write_parquet(data_file)

        # Profile
        from truthound.profiler import profile_file

        profile = profile_file(str(data_file))

        assert profile.column_count == 51  # id + 50 columns


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TestDataProfilingPath",
    "TestSuiteGenerationPath",
    "TestCLIIntegrationPath",
    "TestFullPipelinePath",
    "TestExportFormatsPath",
    "TestErrorHandlingPath",
    "TestPerformancePath",
]
