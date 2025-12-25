"""Suite generation configuration management.

This module provides a comprehensive configuration system for
validation suite generation with support for:
- File-based configuration (YAML/JSON)
- Environment variable overrides
- Presets for common use cases
- Validation and defaults

Example:
    from truthound.profiler.suite_config import (
        SuiteGeneratorConfig,
        load_config,
        ConfigPreset,
    )

    # Use preset
    config = SuiteGeneratorConfig.from_preset(ConfigPreset.STRICT)

    # Load from file
    config = load_config("suite_config.yaml")

    # Programmatic configuration
    config = SuiteGeneratorConfig(
        strictness="strict",
        include_categories=["schema", "format"],
        output_format="yaml",
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.profiler.base import Strictness
    from truthound.profiler.generators.base import RuleCategory, RuleConfidence


# =============================================================================
# Enums
# =============================================================================


class ConfigPreset(str, Enum):
    """Pre-defined configuration presets."""

    DEFAULT = "default"
    STRICT = "strict"
    LOOSE = "loose"
    MINIMAL = "minimal"
    COMPREHENSIVE = "comprehensive"
    SCHEMA_ONLY = "schema_only"
    FORMAT_ONLY = "format_only"
    CI_CD = "ci_cd"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class OutputFormat(str, Enum):
    """Supported output formats."""

    YAML = "yaml"
    JSON = "json"
    PYTHON = "python"
    TOML = "toml"
    CHECKPOINT = "checkpoint"


class GeneratorMode(str, Enum):
    """Generator execution mode."""

    FULL = "full"  # Run all generators
    FAST = "fast"  # Skip expensive generators
    CUSTOM = "custom"  # Only specified generators


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class CategoryConfig:
    """Configuration for rule categories.

    Controls which categories are included/excluded
    and provides category-specific settings.
    """

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    priority_order: list[str] = field(default_factory=list)

    def should_include(self, category: str) -> bool:
        """Check if a category should be included."""
        if self.exclude and category in self.exclude:
            return False
        if self.include and category not in self.include:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "include": self.include,
            "exclude": self.exclude,
            "priority_order": self.priority_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoryConfig":
        return cls(
            include=data.get("include", []),
            exclude=data.get("exclude", []),
            priority_order=data.get("priority_order", []),
        )


@dataclass
class ConfidenceConfig:
    """Configuration for confidence filtering."""

    min_level: str = "low"  # low, medium, high
    include_rationale: bool = True
    show_in_output: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_level": self.min_level,
            "include_rationale": self.include_rationale,
            "show_in_output": self.show_in_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfidenceConfig":
        return cls(
            min_level=data.get("min_level", "low"),
            include_rationale=data.get("include_rationale", True),
            show_in_output=data.get("show_in_output", True),
        )


@dataclass
class OutputConfig:
    """Configuration for output generation."""

    format: str = "yaml"
    include_metadata: bool = True
    include_summary: bool = True
    include_description: bool = True
    group_by_category: bool = False
    sort_rules: bool = True
    indent: int = 2

    # Python-specific
    code_style: str = "functional"  # functional, class_based, declarative
    include_docstrings: bool = True
    include_type_hints: bool = True
    max_line_length: int = 88

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "include_metadata": self.include_metadata,
            "include_summary": self.include_summary,
            "include_description": self.include_description,
            "group_by_category": self.group_by_category,
            "sort_rules": self.sort_rules,
            "indent": self.indent,
            "code_style": self.code_style,
            "include_docstrings": self.include_docstrings,
            "include_type_hints": self.include_type_hints,
            "max_line_length": self.max_line_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputConfig":
        return cls(
            format=data.get("format", "yaml"),
            include_metadata=data.get("include_metadata", True),
            include_summary=data.get("include_summary", True),
            include_description=data.get("include_description", True),
            group_by_category=data.get("group_by_category", False),
            sort_rules=data.get("sort_rules", True),
            indent=data.get("indent", 2),
            code_style=data.get("code_style", "functional"),
            include_docstrings=data.get("include_docstrings", True),
            include_type_hints=data.get("include_type_hints", True),
            max_line_length=data.get("max_line_length", 88),
        )


@dataclass
class GeneratorConfig:
    """Configuration for rule generators."""

    mode: str = "full"  # full, fast, custom
    enabled_generators: list[str] = field(default_factory=list)
    disabled_generators: list[str] = field(default_factory=list)
    generator_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def should_use_generator(self, name: str) -> bool:
        """Check if a generator should be used."""
        if self.disabled_generators and name in self.disabled_generators:
            return False
        if self.mode == "custom" and self.enabled_generators:
            return name in self.enabled_generators
        return True

    def get_generator_options(self, name: str) -> dict[str, Any]:
        """Get options for a specific generator."""
        return self.generator_options.get(name, {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled_generators": self.enabled_generators,
            "disabled_generators": self.disabled_generators,
            "generator_options": self.generator_options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorConfig":
        return cls(
            mode=data.get("mode", "full"),
            enabled_generators=data.get("enabled_generators", []),
            disabled_generators=data.get("disabled_generators", []),
            generator_options=data.get("generator_options", {}),
        )


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class SuiteGeneratorConfig:
    """Complete configuration for suite generation.

    This is the main configuration class that combines all
    sub-configurations and provides preset support.

    Attributes:
        name: Optional name for the generated suite
        strictness: Rule strictness level (loose, medium, strict)
        categories: Category inclusion/exclusion config
        confidence: Confidence filtering config
        output: Output formatting config
        generators: Generator execution config
        custom_options: Additional custom options

    Example:
        # Basic usage
        config = SuiteGeneratorConfig(strictness="strict")

        # From preset
        config = SuiteGeneratorConfig.from_preset(ConfigPreset.CI_CD)

        # Full customization
        config = SuiteGeneratorConfig(
            name="production_rules",
            strictness="strict",
            categories=CategoryConfig(
                include=["schema", "format", "completeness"]
            ),
            confidence=ConfidenceConfig(min_level="medium"),
            output=OutputConfig(format="yaml", group_by_category=True),
        )
    """

    # Core settings
    name: str | None = None
    strictness: str = "medium"

    # Sub-configurations
    categories: CategoryConfig = field(default_factory=CategoryConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    generators: GeneratorConfig = field(default_factory=GeneratorConfig)

    # Custom options
    custom_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        valid_strictness = {"loose", "medium", "strict"}
        if self.strictness not in valid_strictness:
            raise ValueError(
                f"Invalid strictness: {self.strictness}. "
                f"Must be one of: {valid_strictness}"
            )

        valid_confidence = {"low", "medium", "high"}
        if self.confidence.min_level not in valid_confidence:
            raise ValueError(
                f"Invalid min_confidence: {self.confidence.min_level}. "
                f"Must be one of: {valid_confidence}"
            )

    def with_overrides(self, **kwargs: Any) -> "SuiteGeneratorConfig":
        """Create new config with overrides."""
        data = self.to_dict()
        data.update(kwargs)
        return SuiteGeneratorConfig.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "strictness": self.strictness,
            "categories": self.categories.to_dict(),
            "confidence": self.confidence.to_dict(),
            "output": self.output.to_dict(),
            "generators": self.generators.to_dict(),
            "custom_options": self.custom_options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteGeneratorConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name"),
            strictness=data.get("strictness", "medium"),
            categories=CategoryConfig.from_dict(data.get("categories", {})),
            confidence=ConfidenceConfig.from_dict(data.get("confidence", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
            generators=GeneratorConfig.from_dict(data.get("generators", {})),
            custom_options=data.get("custom_options", {}),
        )

    @classmethod
    def from_preset(cls, preset: ConfigPreset | str) -> "SuiteGeneratorConfig":
        """Create configuration from preset.

        Args:
            preset: Preset name or enum value

        Returns:
            Configured instance
        """
        if isinstance(preset, str):
            preset = ConfigPreset(preset)

        return PRESETS[preset]

    @classmethod
    def from_env(cls, prefix: str = "TRUTHOUND_SUITE_") -> "SuiteGeneratorConfig":
        """Create configuration from environment variables.

        Environment variables:
            {prefix}STRICTNESS: Strictness level
            {prefix}MIN_CONFIDENCE: Minimum confidence
            {prefix}FORMAT: Output format
            {prefix}INCLUDE_CATEGORIES: Comma-separated categories
            {prefix}EXCLUDE_CATEGORIES: Comma-separated categories

        Args:
            prefix: Environment variable prefix

        Returns:
            Configured instance
        """
        config = cls()

        # Strictness
        if val := os.environ.get(f"{prefix}STRICTNESS"):
            config = config.with_overrides(strictness=val)

        # Confidence
        if val := os.environ.get(f"{prefix}MIN_CONFIDENCE"):
            config.confidence.min_level = val

        # Format
        if val := os.environ.get(f"{prefix}FORMAT"):
            config.output.format = val

        # Categories
        if val := os.environ.get(f"{prefix}INCLUDE_CATEGORIES"):
            config.categories.include = [c.strip() for c in val.split(",")]

        if val := os.environ.get(f"{prefix}EXCLUDE_CATEGORIES"):
            config.categories.exclude = [c.strip() for c in val.split(",")]

        return config


# =============================================================================
# Presets
# =============================================================================


PRESETS: dict[ConfigPreset, SuiteGeneratorConfig] = {
    ConfigPreset.DEFAULT: SuiteGeneratorConfig(
        strictness="medium",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="low"),
        output=OutputConfig(format="yaml"),
    ),

    ConfigPreset.STRICT: SuiteGeneratorConfig(
        strictness="strict",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="medium"),
        output=OutputConfig(format="yaml", group_by_category=True),
    ),

    ConfigPreset.LOOSE: SuiteGeneratorConfig(
        strictness="loose",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="low"),
        output=OutputConfig(format="yaml"),
    ),

    ConfigPreset.MINIMAL: SuiteGeneratorConfig(
        strictness="loose",
        categories=CategoryConfig(include=["schema"]),
        confidence=ConfidenceConfig(min_level="high"),
        output=OutputConfig(
            format="yaml",
            include_metadata=False,
            include_summary=False,
        ),
        generators=GeneratorConfig(
            mode="fast",
            enabled_generators=["schema"],
        ),
    ),

    ConfigPreset.COMPREHENSIVE: SuiteGeneratorConfig(
        strictness="strict",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="low"),
        output=OutputConfig(
            format="yaml",
            include_metadata=True,
            include_summary=True,
            group_by_category=True,
        ),
        generators=GeneratorConfig(mode="full"),
    ),

    ConfigPreset.SCHEMA_ONLY: SuiteGeneratorConfig(
        strictness="medium",
        categories=CategoryConfig(include=["schema", "completeness"]),
        confidence=ConfidenceConfig(min_level="medium"),
        output=OutputConfig(format="yaml"),
        generators=GeneratorConfig(
            enabled_generators=["schema"],
        ),
    ),

    ConfigPreset.FORMAT_ONLY: SuiteGeneratorConfig(
        strictness="medium",
        categories=CategoryConfig(include=["format", "pattern"]),
        confidence=ConfidenceConfig(min_level="medium"),
        output=OutputConfig(format="yaml"),
        generators=GeneratorConfig(
            enabled_generators=["pattern"],
        ),
    ),

    ConfigPreset.CI_CD: SuiteGeneratorConfig(
        name="ci_cd_validation",
        strictness="medium",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="medium"),
        output=OutputConfig(
            format="checkpoint",
            include_metadata=True,
            include_summary=True,
        ),
    ),

    ConfigPreset.DEVELOPMENT: SuiteGeneratorConfig(
        strictness="loose",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="low"),
        output=OutputConfig(
            format="python",
            code_style="functional",
            include_docstrings=True,
        ),
    ),

    ConfigPreset.PRODUCTION: SuiteGeneratorConfig(
        strictness="strict",
        categories=CategoryConfig(),
        confidence=ConfidenceConfig(min_level="high"),
        output=OutputConfig(
            format="yaml",
            include_metadata=True,
            include_summary=True,
            group_by_category=True,
        ),
    ),
}


# =============================================================================
# File I/O
# =============================================================================


def load_config(path: str | Path) -> SuiteGeneratorConfig:
    """Load configuration from file.

    Supports YAML and JSON formats based on file extension.

    Args:
        path: Path to configuration file

    Returns:
        Loaded configuration
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if path.suffix in {".yaml", ".yml"}:
        # Simple YAML parsing (for basic configs)
        data = _parse_simple_yaml(content)
    else:
        data = json.loads(content)

    return SuiteGeneratorConfig.from_dict(data)


def save_config(config: SuiteGeneratorConfig, path: str | Path) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    path = Path(path)
    data = config.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            f.write(_to_simple_yaml(data))
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    """Simple YAML parser for basic configuration.

    Note: For complex YAML, consider using PyYAML.
    """
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    section_name: str | None = None

    for line in content.split("\n"):
        line = line.rstrip()

        # Skip comments and empty lines
        if not line or line.strip().startswith("#"):
            continue

        # Count indentation
        indent = len(line) - len(line.lstrip())

        # Parse key-value
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if indent == 0:
                # Top-level key
                if value:
                    result[key] = _parse_value(value)
                else:
                    # Start of section
                    result[key] = {}
                    current_section = result[key]
                    section_name = key
            elif current_section is not None:
                # Nested key
                if value:
                    current_section[key] = _parse_value(value)
                else:
                    current_section[key] = {}

    return result


def _parse_value(value: str) -> Any:
    """Parse a YAML value."""
    value = value.strip()

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # None
    if value.lower() in {"null", "~", ""}:
        return None

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # List (simple inline)
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [_parse_value(item) for item in items]

    # String (remove quotes if present)
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    return value


def _to_simple_yaml(data: dict[str, Any], indent: int = 0) -> str:
    """Convert dictionary to simple YAML."""
    lines: list[str] = []
    prefix = "  " * indent

    for key, value in data.items():
        if value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {str(value).lower()}")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_to_simple_yaml(value, indent + 1))
        elif isinstance(value, list):
            if all(isinstance(v, (str, int, float)) for v in value):
                # Inline list
                list_str = ", ".join(str(v) for v in value)
                lines.append(f"{prefix}{key}: [{list_str}]")
            else:
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        lines.append(_to_simple_yaml(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


# =============================================================================
# CLI Argument Builder
# =============================================================================


@dataclass
class CLIArguments:
    """Parsed CLI arguments for suite generation."""

    profile_path: Path
    output_path: Path | None = None
    format: str = "yaml"
    strictness: str = "medium"
    include_categories: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)
    min_confidence: str | None = None
    name: str | None = None
    config_file: Path | None = None
    preset: str | None = None

    def to_config(self) -> SuiteGeneratorConfig:
        """Convert CLI arguments to configuration."""
        # Start with default or preset
        if self.preset:
            config = SuiteGeneratorConfig.from_preset(self.preset)
        elif self.config_file:
            config = load_config(self.config_file)
        else:
            config = SuiteGeneratorConfig()

        # Apply CLI overrides
        if self.name:
            config = config.with_overrides(name=self.name)

        config = config.with_overrides(strictness=self.strictness)

        if self.include_categories:
            config.categories.include = self.include_categories
        if self.exclude_categories:
            config.categories.exclude = self.exclude_categories
        if self.min_confidence:
            config.confidence.min_level = self.min_confidence

        config.output.format = self.format

        return config


def build_config_from_cli(
    profile_path: str | Path,
    output: str | Path | None = None,
    format: str = "yaml",
    strictness: str = "medium",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    min_confidence: str | None = None,
    name: str | None = None,
    config_file: str | Path | None = None,
    preset: str | None = None,
) -> tuple[CLIArguments, SuiteGeneratorConfig]:
    """Build configuration from CLI arguments.

    Args:
        profile_path: Path to profile file
        output: Output path
        format: Output format
        strictness: Strictness level
        include: Categories to include
        exclude: Categories to exclude
        min_confidence: Minimum confidence level
        name: Suite name
        config_file: Path to config file
        preset: Preset name

    Returns:
        Tuple of (CLIArguments, SuiteGeneratorConfig)
    """
    args = CLIArguments(
        profile_path=Path(profile_path),
        output_path=Path(output) if output else None,
        format=format,
        strictness=strictness,
        include_categories=include or [],
        exclude_categories=exclude or [],
        min_confidence=min_confidence,
        name=name,
        config_file=Path(config_file) if config_file else None,
        preset=preset,
    )

    config = args.to_config()
    return args, config


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ConfigPreset",
    "OutputFormat",
    "GeneratorMode",
    # Sub-configurations
    "CategoryConfig",
    "ConfidenceConfig",
    "OutputConfig",
    "GeneratorConfig",
    # Main configuration
    "SuiteGeneratorConfig",
    # Presets
    "PRESETS",
    # File I/O
    "load_config",
    "save_config",
    # CLI support
    "CLIArguments",
    "build_config_from_cli",
]
