"""Unit tests for scaffolding system.

This module tests the scaffolding infrastructure including:
    - ScaffoldConfig validation
    - ScaffoldResult operations
    - ScaffoldRegistry functionality
    - Scaffold implementations
"""

from __future__ import annotations

from pathlib import Path

import pytest

from truthound.cli_modules.scaffolding import (
    ScaffoldConfig,
    ScaffoldResult,
    ScaffoldFile,
    ScaffoldRegistry,
    ScaffoldType,
    ValidatorScaffold,
    ReporterScaffold,
    PluginScaffold,
    get_registry,
    snake_to_pascal,
    snake_to_kebab,
    pascal_to_snake,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_snake_to_pascal(self):
        """Test snake_case to PascalCase conversion."""
        assert snake_to_pascal("my_validator") == "MyValidator"
        assert snake_to_pascal("simple") == "Simple"
        assert snake_to_pascal("my_long_name") == "MyLongName"
        assert snake_to_pascal("a_b_c") == "ABC"

    def test_snake_to_kebab(self):
        """Test snake_case to kebab-case conversion."""
        assert snake_to_kebab("my_validator") == "my-validator"
        assert snake_to_kebab("simple") == "simple"
        assert snake_to_kebab("my_long_name") == "my-long-name"

    def test_pascal_to_snake(self):
        """Test PascalCase to snake_case conversion."""
        assert pascal_to_snake("MyValidator") == "my_validator"
        assert pascal_to_snake("Simple") == "simple"
        assert pascal_to_snake("MyLongName") == "my_long_name"
        # All caps stays lowercase (standard convention)
        assert pascal_to_snake("ABC") == "abc"
        # Mixed case with acronym
        assert pascal_to_snake("ABCValidator") == "abc_validator"


class TestScaffoldConfig:
    """Tests for ScaffoldConfig."""

    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = ScaffoldConfig(name="my_validator")
        assert config.name == "my_validator"
        assert config.class_name == "MyValidator"
        assert config.kebab_name == "my-validator"
        assert config.title_name == "My Validator"

    def test_invalid_name_raises_error(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid name"):
            ScaffoldConfig(name="MyValidator")  # PascalCase not allowed

        with pytest.raises(ValueError, match="Invalid name"):
            ScaffoldConfig(name="123validator")  # Can't start with number

        with pytest.raises(ValueError, match="Invalid name"):
            ScaffoldConfig(name="my-validator")  # Hyphens not allowed

    def test_config_defaults(self):
        """Test default values."""
        config = ScaffoldConfig(name="test")
        assert config.scaffold_type == ScaffoldType.VALIDATOR
        assert config.template_variant == "basic"
        assert config.author == ""
        assert config.version == "0.1.0"
        assert config.category == "custom"
        assert config.license_type == "MIT"
        assert config.include_tests is True
        assert config.include_docs is False

    def test_config_output_dir_conversion(self):
        """Test that string output_dir is converted to Path."""
        config = ScaffoldConfig(name="test", output_dir=Path("/tmp"))
        assert isinstance(config.output_dir, Path)


class TestScaffoldResult:
    """Tests for ScaffoldResult."""

    def test_empty_result(self):
        """Test empty result."""
        result = ScaffoldResult(success=True)
        assert result.success is True
        assert result.file_count == 0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_file(self):
        """Test adding files to result."""
        result = ScaffoldResult(success=True)
        result.add_file("test.py", "print('hello')")
        result.add_file(Path("dir/test2.py"), "print('world')")

        assert result.file_count == 2
        assert result.files[0].path == Path("test.py")
        assert result.files[1].path == Path("dir/test2.py")

    def test_add_error(self):
        """Test adding errors."""
        result = ScaffoldResult(success=True)
        result.add_error("Something went wrong")

        assert result.success is False
        assert len(result.errors) == 1
        assert "Something went wrong" in result.errors

    def test_add_warning(self):
        """Test adding warnings."""
        result = ScaffoldResult(success=True)
        result.add_warning("Be careful")

        assert result.success is True  # Warnings don't fail
        assert len(result.warnings) == 1

    def test_write_files(self, tmp_path: Path):
        """Test writing files to disk."""
        result = ScaffoldResult(success=True)
        result.add_file("test.py", "print('hello')")
        result.add_file("subdir/test2.py", "print('world')")

        written = result.write_files(tmp_path)

        assert len(written) == 2
        assert (tmp_path / "test.py").exists()
        assert (tmp_path / "test.py").read_text() == "print('hello')"
        assert (tmp_path / "subdir/test2.py").exists()

    def test_write_files_skips_existing(self, tmp_path: Path):
        """Test that existing files are not overwritten by default."""
        # Create existing file
        existing = tmp_path / "test.py"
        existing.write_text("original")

        result = ScaffoldResult(success=True)
        result.add_file("test.py", "new content", overwrite=False)

        written = result.write_files(tmp_path)

        assert len(written) == 0
        assert existing.read_text() == "original"

    def test_write_files_overwrites_when_enabled(self, tmp_path: Path):
        """Test that files can be overwritten when enabled."""
        existing = tmp_path / "test.py"
        existing.write_text("original")

        result = ScaffoldResult(success=True)
        result.add_file("test.py", "new content", overwrite=True)

        written = result.write_files(tmp_path)

        assert len(written) == 1
        assert existing.read_text() == "new content"


class TestScaffoldRegistry:
    """Tests for ScaffoldRegistry."""

    def test_get_global_registry(self):
        """Test getting the global registry."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_registry_has_default_scaffolds(self):
        """Test that default scaffolds are registered."""
        registry = get_registry()
        assert "validator" in registry
        assert "reporter" in registry
        assert "plugin" in registry

    def test_list_scaffolds(self):
        """Test listing scaffolds."""
        registry = get_registry()
        scaffolds = registry.list_scaffolds()

        assert len(scaffolds) >= 3
        names = [name for name, _ in scaffolds]
        assert "validator" in names
        assert "reporter" in names
        assert "plugin" in names

    def test_get_scaffold_by_alias(self):
        """Test getting scaffold by alias."""
        registry = get_registry()

        # Validator has aliases "val" and "v"
        scaffold_by_name = registry.get("validator")
        scaffold_by_alias = registry.get("val")
        scaffold_by_short = registry.get("v")

        assert scaffold_by_name is scaffold_by_alias
        assert scaffold_by_name is scaffold_by_short

    def test_get_nonexistent_scaffold(self):
        """Test getting a nonexistent scaffold."""
        registry = get_registry()
        assert registry.get("nonexistent") is None

    def test_registry_contains(self):
        """Test __contains__ method."""
        registry = get_registry()
        assert "validator" in registry
        assert "val" in registry  # Alias
        assert "nonexistent" not in registry


class TestValidatorScaffold:
    """Tests for ValidatorScaffold."""

    def test_scaffold_metadata(self):
        """Test scaffold metadata."""
        scaffold = ValidatorScaffold()
        assert scaffold.name == "validator"
        assert "val" in scaffold.aliases

    def test_template_variants(self):
        """Test available template variants."""
        scaffold = ValidatorScaffold()
        variants = scaffold.get_template_variants()

        assert "basic" in variants
        assert "column" in variants
        assert "pattern" in variants
        assert "range" in variants
        assert "composite" in variants
        assert "full" in variants

    def test_generate_basic_validator(self):
        """Test generating a basic validator."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="test_validator",
            template_variant="basic",
            author="Test Author",
            description="A test validator",
        )

        result = scaffold.generate(config)

        assert result.success is True
        assert result.file_count >= 2  # At least validator.py and __init__.py

        # Check file paths
        paths = [f.path.as_posix() for f in result.files]
        assert "test_validator/validator.py" in paths
        assert "test_validator/__init__.py" in paths

        # Check content
        validator_file = next(f for f in result.files if "validator.py" in str(f.path))
        assert "TestValidatorValidator" in validator_file.content
        assert "@custom_validator" in validator_file.content
        assert "Test Author" in validator_file.content

    def test_generate_column_validator(self):
        """Test generating a column validator."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="column_check",
            template_variant="column",
        )

        result = scaffold.generate(config)

        assert result.success is True

        validator_file = next(f for f in result.files if "validator.py" in str(f.path))
        assert "ColumnValidator" in validator_file.content
        assert "check_column" in validator_file.content

    def test_generate_pattern_validator(self):
        """Test generating a pattern validator."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="email_format",
            template_variant="pattern",
            extra={"pattern": r"^[a-z@.]+$"},
        )

        result = scaffold.generate(config)

        assert result.success is True

        validator_file = next(f for f in result.files if "validator.py" in str(f.path))
        assert "RegexValidatorMixin" in validator_file.content
        assert "pattern" in validator_file.content

    def test_generate_range_validator(self):
        """Test generating a range validator."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="percentage",
            template_variant="range",
            extra={"min_value": 0, "max_value": 100},
        )

        result = scaffold.generate(config)

        assert result.success is True

        validator_file = next(f for f in result.files if "validator.py" in str(f.path))
        assert "min_value" in validator_file.content
        assert "max_value" in validator_file.content
        assert "_build_violation_expr" in validator_file.content

    def test_generate_with_tests(self):
        """Test generating validator with tests."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="test_val",
            template_variant="basic",
            include_tests=True,
        )

        result = scaffold.generate(config)

        assert result.success is True

        paths = [f.path.as_posix() for f in result.files]
        assert any("test_" in p for p in paths)

    def test_generate_with_docs(self):
        """Test generating validator with documentation."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="documented_val",
            template_variant="full",
            include_docs=True,
        )

        result = scaffold.generate(config)

        assert result.success is True

        paths = [f.path.as_posix() for f in result.files]
        assert any("docs" in p for p in paths)

    def test_invalid_template_variant(self):
        """Test that invalid template variant fails."""
        scaffold = ValidatorScaffold()
        config = ScaffoldConfig(
            name="test",
            template_variant="invalid_template",
        )

        result = scaffold.generate(config)

        assert result.success is False
        assert len(result.errors) > 0


class TestReporterScaffold:
    """Tests for ReporterScaffold."""

    def test_scaffold_metadata(self):
        """Test scaffold metadata."""
        scaffold = ReporterScaffold()
        assert scaffold.name == "reporter"
        assert "rep" in scaffold.aliases

    def test_generate_basic_reporter(self):
        """Test generating a basic reporter."""
        scaffold = ReporterScaffold()
        config = ScaffoldConfig(
            name="custom_report",
            template_variant="basic",
            extra={"file_extension": ".txt", "content_type": "text/plain"},
        )

        result = scaffold.generate(config)

        assert result.success is True

        reporter_file = next(f for f in result.files if "reporter.py" in str(f.path))
        assert "CustomReportReporter" in reporter_file.content
        assert "@register_reporter" in reporter_file.content
        assert "render" in reporter_file.content

    def test_generate_full_reporter(self):
        """Test generating a full-featured reporter."""
        scaffold = ReporterScaffold()
        config = ScaffoldConfig(
            name="json_export",
            template_variant="full",
            extra={"file_extension": ".json", "content_type": "application/json"},
        )

        result = scaffold.generate(config)

        assert result.success is True

        reporter_file = next(f for f in result.files if "reporter.py" in str(f.path))
        assert "AggregationMixin" in reporter_file.content or "FilteringMixin" in reporter_file.content


class TestPluginScaffold:
    """Tests for PluginScaffold."""

    def test_scaffold_metadata(self):
        """Test scaffold metadata."""
        scaffold = PluginScaffold()
        assert scaffold.name == "plugin"
        assert "plug" in scaffold.aliases

    def test_template_variants(self):
        """Test available template variants."""
        scaffold = PluginScaffold()
        variants = scaffold.get_template_variants()

        assert "validator" in variants
        assert "reporter" in variants
        assert "hook" in variants
        assert "datasource" in variants
        assert "action" in variants
        assert "full" in variants

    def test_generate_validator_plugin(self):
        """Test generating a validator plugin."""
        scaffold = PluginScaffold()
        config = ScaffoldConfig(
            name="my_validators",
            template_variant="validator",
            author="Test Author",
        )

        result = scaffold.generate(config)

        assert result.success is True

        # Check pyproject.toml
        pyproject = next(f for f in result.files if "pyproject.toml" in str(f.path))
        assert "truthound-plugin-my_validators" in pyproject.content
        assert "truthound.plugins" in pyproject.content

        # Check plugin.py
        plugin_file = next(f for f in result.files if "plugin.py" in str(f.path))
        assert "ValidatorPlugin" in plugin_file.content
        assert "get_validators" in plugin_file.content

    def test_generate_reporter_plugin(self):
        """Test generating a reporter plugin."""
        scaffold = PluginScaffold()
        config = ScaffoldConfig(
            name="custom_reports",
            template_variant="reporter",
        )

        result = scaffold.generate(config)

        assert result.success is True

        plugin_file = next(f for f in result.files if "plugin.py" in str(f.path))
        assert "ReporterPlugin" in plugin_file.content
        assert "get_reporters" in plugin_file.content

    def test_generate_hook_plugin(self):
        """Test generating a hook plugin."""
        scaffold = PluginScaffold()
        config = ScaffoldConfig(
            name="my_hooks",
            template_variant="hook",
        )

        result = scaffold.generate(config)

        assert result.success is True

        plugin_file = next(f for f in result.files if "plugin.py" in str(f.path))
        assert "HookPlugin" in plugin_file.content
        assert "get_hooks" in plugin_file.content
        assert "on_validation_start" in plugin_file.content

    def test_generate_datasource_plugin(self):
        """Test generating a datasource plugin."""
        scaffold = PluginScaffold()
        config = ScaffoldConfig(
            name="custom_db",
            template_variant="datasource",
        )

        result = scaffold.generate(config)

        assert result.success is True

        plugin_file = next(f for f in result.files if "plugin.py" in str(f.path))
        assert "DataSourcePlugin" in plugin_file.content
        assert "get_datasources" in plugin_file.content
        assert "connect" in plugin_file.content
        assert "read" in plugin_file.content

    def test_generate_action_plugin(self):
        """Test generating an action plugin."""
        scaffold = PluginScaffold()
        config = ScaffoldConfig(
            name="custom_action",
            template_variant="action",
        )

        result = scaffold.generate(config)

        assert result.success is True

        plugin_file = next(f for f in result.files if "plugin.py" in str(f.path))
        assert "ActionPlugin" in plugin_file.content
        assert "get_actions" in plugin_file.content
        assert "execute" in plugin_file.content

    def test_generate_full_plugin(self):
        """Test generating a full-featured plugin."""
        scaffold = PluginScaffold()
        config = ScaffoldConfig(
            name="enterprise",
            template_variant="full",
        )

        result = scaffold.generate(config)

        assert result.success is True

        plugin_file = next(f for f in result.files if "plugin.py" in str(f.path))
        # Full plugin should have multiple components
        assert "Validator" in plugin_file.content
        assert "Reporter" in plugin_file.content
        assert "register" in plugin_file.content


class TestScaffoldIntegration:
    """Integration tests for scaffolding system."""

    def test_end_to_end_validator_generation(self, tmp_path: Path):
        """Test generating and writing a complete validator."""
        registry = get_registry()
        config = ScaffoldConfig(
            name="integration_test",
            template_variant="column",
            author="Test",
            include_tests=True,
        )

        result = registry.generate("validator", config)
        assert result.success is True

        written = result.write_files(tmp_path)
        assert len(written) >= 3

        # Verify structure
        assert (tmp_path / "integration_test" / "validator.py").exists()
        assert (tmp_path / "integration_test" / "__init__.py").exists()
        assert (tmp_path / "integration_test" / "tests").is_dir()

    def test_end_to_end_plugin_generation(self, tmp_path: Path):
        """Test generating and writing a complete plugin."""
        registry = get_registry()
        config = ScaffoldConfig(
            name="test_plugin",
            template_variant="validator",
            author="Test",
            include_tests=True,
        )

        result = registry.generate("plugin", config)
        assert result.success is True

        plugin_dir = tmp_path / "truthound-plugin-test_plugin"
        written = result.write_files(plugin_dir)

        # Verify structure
        assert (plugin_dir / "pyproject.toml").exists()
        assert (plugin_dir / "README.md").exists()
        assert (plugin_dir / "test_plugin" / "__init__.py").exists()
        assert (plugin_dir / "test_plugin" / "plugin.py").exists()
