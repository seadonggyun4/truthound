"""Suite generation CLI handlers.

This module provides the CLI interface for suite generation commands.
It integrates with the configuration and export systems for a
clean, extensible command-line experience.

Commands:
    generate-suite: Generate validation rules from a profile
    quick-suite: Profile data and generate rules in one step
    list-formats: List available output formats
    list-presets: List available configuration presets

Example:
    # Basic usage
    truthound generate-suite profile.json -o rules.yaml

    # With preset
    truthound generate-suite profile.json --preset strict -o rules.yaml

    # Quick suite
    truthound quick-suite data.parquet -o rules.yaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.profiler.generators.suite_generator import ValidationSuite
    from truthound.profiler.base import TableProfile


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class GenerateSuiteResult:
    """Result of suite generation."""

    success: bool
    suite: "ValidationSuite | None" = None
    output_path: Path | None = None
    content: str = ""
    message: str = ""
    error: Exception | None = None
    rule_count: int = 0
    category_counts: dict[str, int] | None = None


@dataclass
class QuickSuiteResult:
    """Result of quick suite generation."""

    success: bool
    profile: "TableProfile | None" = None
    suite: "ValidationSuite | None" = None
    output_path: Path | None = None
    content: str = ""
    message: str = ""
    error: Exception | None = None


# =============================================================================
# Progress Callbacks
# =============================================================================


class SuiteGenerationProgress:
    """Progress tracker for suite generation."""

    def __init__(
        self,
        echo: Callable[[str], None] | None = None,
        verbose: bool = True,
    ):
        self.echo = echo or print
        self.verbose = verbose

    def on_start(self, profile_path: Path) -> None:
        """Called when generation starts."""
        if self.verbose:
            self.echo(f"Loading profile from {profile_path}...")

    def on_profile_loaded(self, profile: "TableProfile") -> None:
        """Called when profile is loaded."""
        if self.verbose:
            self.echo(
                f"Profile loaded: {profile.row_count} rows, "
                f"{profile.column_count} columns"
            )

    def on_generation_start(self, strictness: str) -> None:
        """Called when rule generation starts."""
        if self.verbose:
            self.echo(f"Generating validation suite (strictness: {strictness})...")

    def on_generation_complete(
        self,
        suite: "ValidationSuite",
        counts: dict[str, int],
    ) -> None:
        """Called when generation completes."""
        if self.verbose:
            self.echo(f"Generated {len(suite)} rules")
            self.echo("\nRules by category:")
            for cat, count in sorted(counts.items()):
                self.echo(f"  {cat}: {count}")

    def on_export_start(self, format: str) -> None:
        """Called when export starts."""
        if self.verbose:
            self.echo(f"Exporting to {format} format...")

    def on_export_complete(self, output_path: Path | None) -> None:
        """Called when export completes."""
        if self.verbose and output_path:
            self.echo(f"Suite saved to {output_path}")

    def on_error(self, error: Exception) -> None:
        """Called on error."""
        self.echo(f"Error: {error}")


# =============================================================================
# Core Handlers
# =============================================================================


class SuiteGenerationHandler:
    """Handler for suite generation operations.

    This class encapsulates all the logic for generating validation
    suites from profiles, making it easy to test and extend.

    Example:
        handler = SuiteGenerationHandler()
        result = handler.generate(
            profile_path=Path("profile.json"),
            output_path=Path("rules.yaml"),
            format="yaml",
            strictness="medium",
        )
    """

    def __init__(
        self,
        progress: SuiteGenerationProgress | None = None,
    ):
        self.progress = progress or SuiteGenerationProgress()

    def generate(
        self,
        profile_path: Path,
        output_path: Path | None = None,
        format: str = "yaml",
        strictness: str = "medium",
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        min_confidence: str | None = None,
        name: str | None = None,
        preset: str | None = None,
        config_file: Path | None = None,
        group_by_category: bool = False,
    ) -> GenerateSuiteResult:
        """Generate validation suite from profile.

        Args:
            profile_path: Path to profile JSON file
            output_path: Output file path
            format: Output format (yaml, json, python, toml, checkpoint)
            strictness: Rule strictness (loose, medium, strict)
            include_categories: Only include these categories
            exclude_categories: Exclude these categories
            min_confidence: Minimum confidence level
            name: Suite name
            preset: Configuration preset name
            config_file: Path to configuration file
            group_by_category: Group rules by category in output

        Returns:
            Generation result
        """
        try:
            # Import dependencies
            from truthound.profiler import load_profile, generate_suite
            from truthound.profiler.suite_config import (
                SuiteGeneratorConfig,
                load_config as load_suite_config,
            )
            from truthound.profiler.suite_export import (
                SuiteExporter,
                ExportConfig,
                CodeStyle,
            )

            # Validate input
            if not profile_path.exists():
                return GenerateSuiteResult(
                    success=False,
                    message=f"Profile file not found: {profile_path}",
                )

            # Load profile
            self.progress.on_start(profile_path)
            profile = load_profile(profile_path)
            self.progress.on_profile_loaded(profile)

            # Build configuration
            if config_file and config_file.exists():
                config = load_suite_config(config_file)
            elif preset:
                config = SuiteGeneratorConfig.from_preset(preset)
            else:
                config = SuiteGeneratorConfig(strictness=strictness)

            # Apply overrides
            if include_categories:
                config.categories.include = include_categories
            if exclude_categories:
                config.categories.exclude = exclude_categories
            if min_confidence:
                config.confidence.min_level = min_confidence
            if name:
                config = config.with_overrides(name=name)

            # Update output config
            config.output.format = format
            config.output.group_by_category = group_by_category

            # Generate suite
            self.progress.on_generation_start(config.strictness)

            suite = generate_suite(
                profile,
                strictness=config.strictness,
                include_categories=config.categories.include or None,
                exclude_categories=config.categories.exclude or None,
                min_confidence=config.confidence.min_level,
                name=config.name,
            )

            category_counts = suite._count_by_category()
            self.progress.on_generation_complete(suite, category_counts)

            # Export
            self.progress.on_export_start(format)

            export_config = ExportConfig(
                include_metadata=config.output.include_metadata,
                include_rationale=config.confidence.include_rationale,
                include_confidence=config.confidence.show_in_output,
                include_summary=config.output.include_summary,
                include_description=config.output.include_description,
                sort_rules=config.output.sort_rules,
                group_by_category=config.output.group_by_category,
                indent=config.output.indent,
                code_style=CodeStyle(config.output.code_style),
                include_docstrings=config.output.include_docstrings,
                include_type_hints=config.output.include_type_hints,
                max_line_length=config.output.max_line_length,
            )

            exporter = SuiteExporter(format=format, config=export_config)
            content = exporter.export_to_string(suite)

            # Write output
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.progress.on_export_complete(output_path)

            return GenerateSuiteResult(
                success=True,
                suite=suite,
                output_path=output_path,
                content=content,
                message="Suite generated successfully",
                rule_count=len(suite),
                category_counts=category_counts,
            )

        except Exception as e:
            self.progress.on_error(e)
            return GenerateSuiteResult(
                success=False,
                message=f"Generation failed: {e}",
                error=e,
            )

    def quick_suite(
        self,
        data_path: Path,
        output_path: Path | None = None,
        format: str = "yaml",
        strictness: str = "medium",
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        min_confidence: str | None = None,
        name: str | None = None,
        preset: str | None = None,
        sample_size: int | None = None,
    ) -> QuickSuiteResult:
        """Profile data and generate suite in one step.

        Args:
            data_path: Path to data file
            output_path: Output file path
            format: Output format
            strictness: Rule strictness
            include_categories: Categories to include
            exclude_categories: Categories to exclude
            min_confidence: Minimum confidence
            name: Suite name
            preset: Configuration preset
            sample_size: Sample size for profiling

        Returns:
            Quick suite result
        """
        try:
            from truthound.profiler import profile_file, generate_suite
            from truthound.profiler.suite_export import (
                SuiteExporter,
                ExportConfig,
            )

            # Validate input
            if not data_path.exists():
                return QuickSuiteResult(
                    success=False,
                    message=f"Data file not found: {data_path}",
                )

            # Profile data
            if self.progress.verbose:
                self.progress.echo(f"Profiling {data_path}...")

            profile = profile_file(
                str(data_path),
                sample_size=sample_size,
            )

            if self.progress.verbose:
                self.progress.echo(
                    f"Profile complete: {profile.row_count} rows, "
                    f"{profile.column_count} columns"
                )

            # Generate suite
            suite_name = name or data_path.stem
            suite = generate_suite(
                profile,
                strictness=strictness,
                include_categories=include_categories,
                exclude_categories=exclude_categories,
                min_confidence=min_confidence,
                name=suite_name,
            )

            if self.progress.verbose:
                self.progress.echo(f"Generated {len(suite)} rules")

            # Export
            exporter = SuiteExporter(format=format)
            content = exporter.export_to_string(suite)

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

                if self.progress.verbose:
                    self.progress.echo(f"Suite saved to {output_path}")

            return QuickSuiteResult(
                success=True,
                profile=profile,
                suite=suite,
                output_path=output_path,
                content=content,
                message="Quick suite generated successfully",
            )

        except Exception as e:
            self.progress.on_error(e)
            return QuickSuiteResult(
                success=False,
                message=f"Quick suite failed: {e}",
                error=e,
            )


# =============================================================================
# CLI Integration Functions
# =============================================================================


def get_available_formats() -> list[str]:
    """Get list of available output formats."""
    from truthound.profiler.suite_export import get_available_formats
    return get_available_formats()


def get_available_presets() -> list[str]:
    """Get list of available configuration presets."""
    from truthound.profiler.suite_config import ConfigPreset
    return [p.value for p in ConfigPreset]


def get_available_categories() -> list[str]:
    """Get list of available rule categories."""
    from truthound.profiler.generators.base import RuleCategory
    return [c.value for c in RuleCategory]


def format_category_help() -> str:
    """Generate help text for categories."""
    categories = get_available_categories()
    return "Available categories: " + ", ".join(categories)


def format_preset_help() -> str:
    """Generate help text for presets."""
    presets = get_available_presets()
    return "Available presets: " + ", ".join(presets)


def validate_format(format: str) -> bool:
    """Validate output format."""
    return format in get_available_formats()


def validate_preset(preset: str) -> bool:
    """Validate preset name."""
    return preset in get_available_presets()


def validate_strictness(strictness: str) -> bool:
    """Validate strictness level."""
    return strictness in {"loose", "medium", "strict"}


def validate_confidence(confidence: str) -> bool:
    """Validate confidence level."""
    return confidence in {"low", "medium", "high"}


# =============================================================================
# CLI Command Wrapper
# =============================================================================


def create_cli_handler(
    echo: Callable[[str], None] | None = None,
    exit_on_error: Callable[[int], None] | None = None,
    verbose: bool = True,
) -> SuiteGenerationHandler:
    """Create a CLI-ready handler with proper callbacks.

    Args:
        echo: Function to output messages
        exit_on_error: Function to exit on error
        verbose: Enable verbose output

    Returns:
        Configured handler
    """
    progress = SuiteGenerationProgress(
        echo=echo,
        verbose=verbose,
    )
    return SuiteGenerationHandler(progress=progress)


def run_generate_suite(
    profile_file: Path,
    output: Path | None = None,
    format: str = "yaml",
    strictness: str = "medium",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    min_confidence: str | None = None,
    name: str | None = None,
    preset: str | None = None,
    config: Path | None = None,
    group_by_category: bool = False,
    echo: Callable[[str], None] | None = None,
    verbose: bool = True,
) -> int:
    """Run generate-suite command.

    This is the main entry point for CLI integration.

    Args:
        profile_file: Path to profile JSON file
        output: Output file path
        format: Output format
        strictness: Rule strictness
        include: Categories to include
        exclude: Categories to exclude
        min_confidence: Minimum confidence level
        name: Suite name
        preset: Configuration preset
        config: Path to config file
        group_by_category: Group rules by category
        echo: Echo function for output
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    handler = create_cli_handler(echo=echo, verbose=verbose)

    result = handler.generate(
        profile_path=profile_file,
        output_path=output,
        format=format,
        strictness=strictness,
        include_categories=include,
        exclude_categories=exclude,
        min_confidence=min_confidence,
        name=name,
        preset=preset,
        config_file=config,
        group_by_category=group_by_category,
    )

    if result.success:
        if not output and result.content:
            (echo or print)(result.content)
        return 0
    else:
        return 1


def run_quick_suite(
    file: Path,
    output: Path | None = None,
    format: str = "yaml",
    strictness: str = "medium",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    min_confidence: str | None = None,
    name: str | None = None,
    preset: str | None = None,
    sample_size: int | None = None,
    echo: Callable[[str], None] | None = None,
    verbose: bool = True,
) -> int:
    """Run quick-suite command.

    Args:
        file: Path to data file
        output: Output file path
        format: Output format
        strictness: Rule strictness
        include: Categories to include
        exclude: Categories to exclude
        min_confidence: Minimum confidence level
        name: Suite name
        preset: Configuration preset
        sample_size: Sample size for profiling
        echo: Echo function for output
        verbose: Enable verbose output

    Returns:
        Exit code
    """
    handler = create_cli_handler(echo=echo, verbose=verbose)

    result = handler.quick_suite(
        data_path=file,
        output_path=output,
        format=format,
        strictness=strictness,
        include_categories=include,
        exclude_categories=exclude,
        min_confidence=min_confidence,
        name=name,
        preset=preset,
        sample_size=sample_size,
    )

    if result.success:
        if not output and result.content:
            (echo or print)(result.content)
        return 0
    else:
        return 1


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Result types
    "GenerateSuiteResult",
    "QuickSuiteResult",
    # Progress tracking
    "SuiteGenerationProgress",
    # Handlers
    "SuiteGenerationHandler",
    # Helpers
    "get_available_formats",
    "get_available_presets",
    "get_available_categories",
    "format_category_help",
    "format_preset_help",
    "validate_format",
    "validate_preset",
    "validate_strictness",
    "validate_confidence",
    # CLI integration
    "create_cli_handler",
    "run_generate_suite",
    "run_quick_suite",
]
