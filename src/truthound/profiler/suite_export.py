"""Suite export system with extensible formatters and exporters.

This module provides a highly extensible architecture for exporting
validation suites to various formats and destinations.

Key Features:
- Plugin architecture for output formats (YAML, JSON, Python, etc.)
- Configurable export options
- Template-based code generation
- Post-processing hooks

Example:
    from truthound.profiler.suite_export import (
        SuiteExporter,
        create_exporter,
        export_suite,
    )

    # Simple export
    export_suite(suite, "rules.yaml", format="yaml")

    # Customized export
    exporter = create_exporter(
        format="python",
        config=ExportConfig(
            include_docstrings=True,
            include_type_hints=True,
        )
    )
    exporter.export(suite, "validators.py")

    # Custom formatter registration
    @register_formatter("custom")
    class CustomFormatter(SuiteFormatter):
        ...
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    Sequence,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from truthound.profiler.generators.suite_generator import ValidationSuite
    from truthound.profiler.generators.base import GeneratedRule


# =============================================================================
# Enums and Types
# =============================================================================


class ExportFormat(str, Enum):
    """Supported export formats."""

    YAML = "yaml"
    JSON = "json"
    PYTHON = "python"
    TOML = "toml"
    CHECKPOINT = "checkpoint"  # Truthound checkpoint format
    GREAT_EXPECTATIONS = "great_expectations"
    DEEQU = "deequ"
    CUSTOM = "custom"


class CodeStyle(str, Enum):
    """Python code generation style."""

    FUNCTIONAL = "functional"  # List of validator instances
    CLASS_BASED = "class_based"  # Validator class with methods
    DECLARATIVE = "declarative"  # Declarative configuration


class OutputMode(str, Enum):
    """Output mode for the exporter."""

    FILE = "file"
    STDOUT = "stdout"
    STRING = "string"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for suite export.

    This immutable configuration controls how validation suites
    are exported to various formats.

    Attributes:
        include_metadata: Include suite metadata in output
        include_rationale: Include rule rationale/explanation
        include_confidence: Include confidence levels
        include_summary: Include summary statistics
        sort_rules: Sort rules by category/name
        group_by_category: Group rules by category
        indent: Indentation for formatted output
        code_style: Style for Python code generation
        include_docstrings: Include docstrings in Python code
        include_type_hints: Include type hints in Python code
        include_imports: Include import statements in Python code
        max_line_length: Max line length for formatted output
        custom_options: Additional format-specific options
    """

    # Content options
    include_metadata: bool = True
    include_rationale: bool = True
    include_confidence: bool = True
    include_summary: bool = True
    include_description: bool = True

    # Formatting options
    sort_rules: bool = True
    group_by_category: bool = False
    indent: int = 2

    # Python code generation options
    code_style: CodeStyle = CodeStyle.FUNCTIONAL
    include_docstrings: bool = True
    include_type_hints: bool = True
    include_imports: bool = True
    max_line_length: int = 88

    # Custom options
    custom_options: dict[str, Any] = field(default_factory=dict)

    def with_options(self, **kwargs: Any) -> "ExportConfig":
        """Create new config with updated options."""
        current = {
            "include_metadata": self.include_metadata,
            "include_rationale": self.include_rationale,
            "include_confidence": self.include_confidence,
            "include_summary": self.include_summary,
            "include_description": self.include_description,
            "sort_rules": self.sort_rules,
            "group_by_category": self.group_by_category,
            "indent": self.indent,
            "code_style": self.code_style,
            "include_docstrings": self.include_docstrings,
            "include_type_hints": self.include_type_hints,
            "include_imports": self.include_imports,
            "max_line_length": self.max_line_length,
            "custom_options": dict(self.custom_options),
        }
        current.update(kwargs)
        return ExportConfig(**current)


# Default configurations for common use cases
DEFAULT_CONFIG = ExportConfig()
MINIMAL_CONFIG = ExportConfig(
    include_metadata=False,
    include_rationale=False,
    include_confidence=False,
    include_summary=False,
    include_docstrings=False,
)
VERBOSE_CONFIG = ExportConfig(
    include_metadata=True,
    include_rationale=True,
    include_confidence=True,
    include_summary=True,
    group_by_category=True,
)


# =============================================================================
# Formatter Protocol and Base Class
# =============================================================================


class SuiteFormatterProtocol(Protocol):
    """Protocol for suite formatters."""

    format_name: str
    file_extension: str

    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        """Format a validation suite to string output."""
        ...


class SuiteFormatter(ABC):
    """Abstract base class for suite formatters.

    Formatters are responsible for converting a ValidationSuite
    to a specific output format (YAML, JSON, Python, etc.).

    Subclasses must implement:
    - format_name: Name of the format
    - file_extension: Default file extension
    - format(): Main formatting method

    Example:
        class MyFormatter(SuiteFormatter):
            format_name = "my_format"
            file_extension = ".mf"

            def format(self, suite, config):
                return "..."
    """

    format_name: str = "base"
    file_extension: str = ".txt"

    def __init__(self, **kwargs: Any):
        """Initialize formatter with optional configuration."""
        self.options = kwargs

    @abstractmethod
    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        """Format the validation suite to string output.

        Args:
            suite: Validation suite to format
            config: Export configuration

        Returns:
            Formatted string output
        """
        pass

    def _sort_rules(
        self,
        rules: Sequence["GeneratedRule"],
        config: ExportConfig,
    ) -> list["GeneratedRule"]:
        """Sort rules according to configuration."""
        if not config.sort_rules:
            return list(rules)

        return sorted(
            rules,
            key=lambda r: (r.category.value, r.name),
        )

    def _group_rules(
        self,
        rules: Sequence["GeneratedRule"],
    ) -> dict[str, list["GeneratedRule"]]:
        """Group rules by category."""
        groups: dict[str, list["GeneratedRule"]] = {}
        for rule in rules:
            cat = rule.category.value
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(rule)
        return groups


# =============================================================================
# Built-in Formatters
# =============================================================================


class YAMLFormatter(SuiteFormatter):
    """YAML format output for validation suites."""

    format_name = "yaml"
    file_extension = ".yaml"

    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        lines: list[str] = []

        # Header
        lines.append(f"# Validation Suite: {suite.name}")
        lines.append(f"# Strictness: {suite.strictness.value}")
        lines.append(f"# Total rules: {len(suite.rules)}")
        lines.append("")

        # Metadata
        if config.include_metadata and suite.metadata:
            lines.append("metadata:")
            for key, value in suite.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Summary
        if config.include_summary:
            lines.append("summary:")
            counts = suite._count_by_category()
            lines.append("  by_category:")
            for cat, count in sorted(counts.items()):
                lines.append(f"    {cat}: {count}")
            conf_counts = suite._count_by_confidence()
            lines.append("  by_confidence:")
            for conf, count in sorted(conf_counts.items()):
                lines.append(f"    {conf}: {count}")
            lines.append("")

        # Rules
        rules = self._sort_rules(suite.rules, config)

        if config.group_by_category:
            lines.extend(self._format_grouped_rules(rules, config))
        else:
            lines.extend(self._format_flat_rules(rules, config))

        return "\n".join(lines)

    def _format_flat_rules(
        self,
        rules: list["GeneratedRule"],
        config: ExportConfig,
    ) -> list[str]:
        """Format rules as flat list."""
        lines = ["rules:"]

        for rule in rules:
            lines.extend(self._format_rule(rule, config, indent=2))
            lines.append("")

        return lines

    def _format_grouped_rules(
        self,
        rules: list["GeneratedRule"],
        config: ExportConfig,
    ) -> list[str]:
        """Format rules grouped by category."""
        lines = ["rules:"]
        groups = self._group_rules(rules)

        for category, cat_rules in sorted(groups.items()):
            lines.append(f"  # Category: {category}")
            lines.append(f"  {category}:")
            for rule in cat_rules:
                lines.extend(self._format_rule(rule, config, indent=4))
                lines.append("")

        return lines

    def _format_rule(
        self,
        rule: "GeneratedRule",
        config: ExportConfig,
        indent: int = 2,
    ) -> list[str]:
        """Format a single rule."""
        prefix = " " * indent
        lines = [
            f"{prefix}- name: {rule.name}",
            f"{prefix}  validator: {rule.validator_class}",
            f"{prefix}  category: {rule.category.value}",
        ]

        if config.include_confidence:
            lines.append(f"{prefix}  confidence: {rule.confidence.value}")

        if rule.columns:
            lines.append(f"{prefix}  columns: {list(rule.columns)}")

        if rule.parameters:
            lines.append(f"{prefix}  parameters:")
            for k, v in rule.parameters.items():
                if isinstance(v, str):
                    lines.append(f'{prefix}    {k}: "{v}"')
                else:
                    lines.append(f"{prefix}    {k}: {v}")

        if rule.mostly is not None:
            lines.append(f"{prefix}  mostly: {rule.mostly}")

        if config.include_description and rule.description:
            lines.append(f'{prefix}  description: "{rule.description}"')

        if config.include_rationale and rule.rationale:
            lines.append(f'{prefix}  rationale: "{rule.rationale}"')

        return lines


class JSONFormatter(SuiteFormatter):
    """JSON format output for validation suites."""

    format_name = "json"
    file_extension = ".json"

    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        data = self._build_dict(suite, config)
        return json.dumps(
            data,
            indent=config.indent,
            ensure_ascii=False,
            default=str,
        )

    def _build_dict(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> dict[str, Any]:
        """Build dictionary representation."""
        rules = self._sort_rules(suite.rules, config)

        data: dict[str, Any] = {
            "name": suite.name,
            "strictness": suite.strictness.value,
        }

        if config.include_metadata:
            data["metadata"] = suite.metadata
            data["source_profile"] = suite.source_profile

        if config.group_by_category:
            groups = self._group_rules(rules)
            data["rules"] = {
                cat: [self._rule_to_dict(r, config) for r in cat_rules]
                for cat, cat_rules in sorted(groups.items())
            }
        else:
            data["rules"] = [self._rule_to_dict(r, config) for r in rules]

        if config.include_summary:
            data["summary"] = {
                "total_rules": len(rules),
                "by_category": suite._count_by_category(),
                "by_confidence": suite._count_by_confidence(),
            }

        return data

    def _rule_to_dict(
        self,
        rule: "GeneratedRule",
        config: ExportConfig,
    ) -> dict[str, Any]:
        """Convert rule to dictionary."""
        data: dict[str, Any] = {
            "name": rule.name,
            "validator_class": rule.validator_class,
            "category": rule.category.value,
        }

        if config.include_confidence:
            data["confidence"] = rule.confidence.value

        if rule.columns:
            data["columns"] = list(rule.columns)

        if rule.parameters:
            data["parameters"] = rule.parameters

        if rule.mostly is not None:
            data["mostly"] = rule.mostly

        if config.include_description and rule.description:
            data["description"] = rule.description

        if config.include_rationale and rule.rationale:
            data["rationale"] = rule.rationale

        return data


class PythonFormatter(SuiteFormatter):
    """Python code generation for validation suites."""

    format_name = "python"
    file_extension = ".py"

    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        if config.code_style == CodeStyle.CLASS_BASED:
            return self._format_class_based(suite, config)
        elif config.code_style == CodeStyle.DECLARATIVE:
            return self._format_declarative(suite, config)
        else:
            return self._format_functional(suite, config)

    def _format_functional(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        """Generate functional-style Python code."""
        lines: list[str] = []

        # Module docstring
        if config.include_docstrings:
            lines.extend([
                '"""Auto-generated validation suite.',
                "",
                f"Suite: {suite.name}",
                f"Strictness: {suite.strictness.value}",
                f"Total rules: {len(suite.rules)}",
                '"""',
                "",
            ])

        # Imports
        if config.include_imports:
            lines.extend(self._generate_imports(suite, config))
            lines.append("")

        # Type alias
        if config.include_type_hints:
            lines.append("ValidatorList = list[Validator]")
            lines.append("")
            lines.append("")

        # Main function
        lines.append("def create_validators()" + (
            " -> ValidatorList:" if config.include_type_hints else ":"
        ))
        if config.include_docstrings:
            lines.append('    """Create validation rules for the suite."""')
        lines.append("    validators = []")
        lines.append("")

        # Rules
        rules = self._sort_rules(suite.rules, config)
        for rule in rules:
            lines.extend(self._format_validator_append(rule, config))

        lines.append("    return validators")
        lines.append("")

        return "\n".join(lines)

    def _format_class_based(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        """Generate class-based Python code."""
        lines: list[str] = []

        # Module docstring
        if config.include_docstrings:
            lines.extend([
                '"""Auto-generated validation suite.',
                "",
                f"Suite: {suite.name}",
                '"""',
                "",
            ])

        # Imports
        if config.include_imports:
            lines.extend(self._generate_imports(suite, config))
            lines.append("from dataclasses import dataclass")
            lines.append("")

        # Suite class
        suite_name = self._to_class_name(suite.name)
        lines.append("@dataclass")
        lines.append(f"class {suite_name}ValidationSuite:")
        if config.include_docstrings:
            lines.append(f'    """Validation suite: {suite.name}."""')
        lines.append("")

        # Class attributes
        lines.append(f'    name: str = "{suite.name}"')
        lines.append(f'    strictness: str = "{suite.strictness.value}"')
        lines.append("")

        # Create validators method
        lines.append("    def create_validators(self)" + (
            " -> list[Validator]:" if config.include_type_hints else ":"
        ))
        if config.include_docstrings:
            lines.append('        """Create all validators for this suite."""')
        lines.append("        validators = []")
        lines.append("")

        rules = self._sort_rules(suite.rules, config)
        for rule in rules:
            lines.extend(self._format_validator_append(rule, config, indent=8))

        lines.append("        return validators")
        lines.append("")

        return "\n".join(lines)

    def _format_declarative(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        """Generate declarative-style Python code."""
        lines: list[str] = []

        # Module docstring
        if config.include_docstrings:
            lines.extend([
                '"""Auto-generated validation suite (declarative style).',
                "",
                f"Suite: {suite.name}",
                '"""',
                "",
            ])

        # Imports
        if config.include_imports:
            lines.append("from typing import Any")
            lines.append("")

        # Declarative config
        lines.append("SUITE_CONFIG = {")
        lines.append(f'    "name": "{suite.name}",')
        lines.append(f'    "strictness": "{suite.strictness.value}",')
        lines.append('    "rules": [')

        rules = self._sort_rules(suite.rules, config)
        for rule in rules:
            lines.append("        {")
            lines.append(f'            "name": "{rule.name}",')
            lines.append(f'            "validator": "{rule.validator_class}",')
            lines.append(f'            "category": "{rule.category.value}",')

            if rule.columns:
                lines.append(f'            "columns": {list(rule.columns)},')
            if rule.parameters:
                lines.append(f'            "parameters": {rule.parameters!r},')
            if rule.mostly is not None:
                lines.append(f'            "mostly": {rule.mostly},')

            lines.append("        },")

        lines.append("    ],")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _generate_imports(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> list[str]:
        """Generate import statements."""
        lines = []

        if config.include_type_hints:
            lines.append("from typing import Any")
            lines.append("")

        # Collect unique validator classes
        validators = sorted(set(r.validator_class for r in suite.rules))

        lines.append("from truthound.validators import (")
        for v in validators:
            lines.append(f"    {v},")
        lines.append(")")

        if config.include_type_hints:
            lines.append("from truthound.validators.base import Validator")

        return lines

    def _format_validator_append(
        self,
        rule: "GeneratedRule",
        config: ExportConfig,
        indent: int = 4,
    ) -> list[str]:
        """Format validator.append() call."""
        prefix = " " * indent
        lines = []

        # Comment
        if config.include_description and rule.description:
            lines.append(f"{prefix}# {rule.name}: {rule.description}")
        else:
            lines.append(f"{prefix}# {rule.name}")

        # Build parameters
        params: list[str] = []
        if rule.columns:
            params.append(f"columns={list(rule.columns)}")
        for k, v in rule.parameters.items():
            if isinstance(v, str):
                params.append(f'{k}="{v}"')
            else:
                params.append(f"{k}={v!r}")
        if rule.mostly is not None:
            params.append(f"mostly={rule.mostly}")

        param_str = ", ".join(params)

        # Check line length
        full_line = f"{prefix}validators.append({rule.validator_class}({param_str}))"
        if len(full_line) <= config.max_line_length:
            lines.append(full_line)
        else:
            # Multi-line format
            lines.append(f"{prefix}validators.append(")
            lines.append(f"{prefix}    {rule.validator_class}(")
            for i, param in enumerate(params):
                comma = "," if i < len(params) - 1 else ""
                lines.append(f"{prefix}        {param}{comma}")
            lines.append(f"{prefix}    )")
            lines.append(f"{prefix})")

        lines.append("")
        return lines

    def _to_class_name(self, name: str) -> str:
        """Convert name to valid Python class name."""
        # Remove special characters and convert to PascalCase
        parts = name.replace("-", "_").replace(".", "_").split("_")
        return "".join(p.capitalize() for p in parts if p)


class TOMLFormatter(SuiteFormatter):
    """TOML format output for validation suites."""

    format_name = "toml"
    file_extension = ".toml"

    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        lines: list[str] = []

        # Suite header
        lines.append("[suite]")
        lines.append(f'name = "{suite.name}"')
        lines.append(f'strictness = "{suite.strictness.value}"')

        if config.include_metadata:
            lines.append(f'source_profile = "{suite.source_profile}"')
            if suite.metadata:
                for key, value in suite.metadata.items():
                    if isinstance(value, str):
                        lines.append(f'{key} = "{value}"')
                    elif isinstance(value, list):
                        lines.append(f'{key} = {value}')
                    else:
                        lines.append(f'{key} = {value}')

        lines.append("")

        # Rules
        rules = self._sort_rules(suite.rules, config)
        for rule in rules:
            lines.append(f"[[rules]]")
            lines.append(f'name = "{rule.name}"')
            lines.append(f'validator = "{rule.validator_class}"')
            lines.append(f'category = "{rule.category.value}"')

            if config.include_confidence:
                lines.append(f'confidence = "{rule.confidence.value}"')

            if rule.columns:
                cols_str = ", ".join(f'"{c}"' for c in rule.columns)
                lines.append(f"columns = [{cols_str}]")

            if rule.parameters:
                lines.append("[rules.parameters]")
                for k, v in rule.parameters.items():
                    if isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    else:
                        lines.append(f"{k} = {v}")

            if rule.mostly is not None:
                lines.append(f"mostly = {rule.mostly}")

            if config.include_description and rule.description:
                lines.append(f'description = "{rule.description}"')

            lines.append("")

        return "\n".join(lines)


class CheckpointFormatter(SuiteFormatter):
    """Truthound Checkpoint format for CI/CD integration."""

    format_name = "checkpoint"
    file_extension = ".yaml"

    def format(
        self,
        suite: "ValidationSuite",
        config: ExportConfig,
    ) -> str:
        lines: list[str] = []

        # Header
        lines.append("# Auto-generated Truthound Checkpoint")
        lines.append(f"# Source: {suite.name}")
        lines.append("")
        lines.append("checkpoints:")
        lines.append(f"  - name: {suite.name}_validation")
        lines.append('    data_source: "${DATA_SOURCE}"  # Set via environment')
        lines.append("    validators:")

        # Convert rules to checkpoint validators
        rules = self._sort_rules(suite.rules, config)
        for rule in rules:
            lines.append(f"      - type: {rule.validator_class}")
            if rule.columns:
                lines.append(f"        columns: {list(rule.columns)}")
            if rule.parameters:
                for k, v in rule.parameters.items():
                    if isinstance(v, str):
                        lines.append(f'        {k}: "{v}"')
                    else:
                        lines.append(f"        {k}: {v}")
            if rule.mostly is not None:
                lines.append(f"        mostly: {rule.mostly}")

        lines.append("")
        lines.append("    # Severity thresholds")
        lines.append("    min_severity: warning")
        lines.append("")
        lines.append("    # Optional actions")
        lines.append("    actions:")
        lines.append("      - store_result")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# Formatter Registry
# =============================================================================


class FormatterRegistry:
    """Registry for suite formatters.

    Provides plugin-based formatter registration and lookup.
    """

    def __init__(self) -> None:
        self._formatters: dict[str, type[SuiteFormatter]] = {}

    def register(
        self,
        formatter_class: type[SuiteFormatter],
        name: str | None = None,
    ) -> None:
        """Register a formatter class."""
        key = name or formatter_class.format_name
        self._formatters[key] = formatter_class

    def get(self, name: str) -> type[SuiteFormatter]:
        """Get formatter by name."""
        if name not in self._formatters:
            available = list(self._formatters.keys())
            raise KeyError(
                f"Formatter '{name}' not found. Available: {available}"
            )
        return self._formatters[name]

    def list_all(self) -> dict[str, type[SuiteFormatter]]:
        """List all registered formatters."""
        return dict(self._formatters)

    def create(self, name: str, **kwargs: Any) -> SuiteFormatter:
        """Create formatter instance by name."""
        formatter_class = self.get(name)
        return formatter_class(**kwargs)

    def get_extension(self, name: str) -> str:
        """Get file extension for format."""
        return self.get(name).file_extension


# Global registry
formatter_registry = FormatterRegistry()

# Register built-in formatters
formatter_registry.register(YAMLFormatter)
formatter_registry.register(JSONFormatter)
formatter_registry.register(PythonFormatter)
formatter_registry.register(TOMLFormatter)
formatter_registry.register(CheckpointFormatter)


def register_formatter(
    name: str | None = None,
) -> Callable[[type[SuiteFormatter]], type[SuiteFormatter]]:
    """Decorator to register a formatter.

    Example:
        @register_formatter("custom")
        class CustomFormatter(SuiteFormatter):
            ...
    """
    def decorator(cls: type[SuiteFormatter]) -> type[SuiteFormatter]:
        formatter_registry.register(cls, name)
        return cls
    return decorator


# =============================================================================
# Post-Processors
# =============================================================================


class ExportPostProcessor(Protocol):
    """Protocol for export post-processors."""

    def process(self, content: str, suite: "ValidationSuite") -> str:
        """Process the exported content."""
        ...


class AddHeaderPostProcessor:
    """Add custom header to output."""

    def __init__(self, header: str):
        self.header = header

    def process(self, content: str, suite: "ValidationSuite") -> str:
        return f"{self.header}\n\n{content}"


class AddFooterPostProcessor:
    """Add custom footer to output."""

    def __init__(self, footer: str):
        self.footer = footer

    def process(self, content: str, suite: "ValidationSuite") -> str:
        return f"{content}\n\n{self.footer}"


class TemplatePostProcessor:
    """Apply template substitution."""

    def __init__(self, substitutions: dict[str, str]):
        self.substitutions = substitutions

    def process(self, content: str, suite: "ValidationSuite") -> str:
        result = content
        for key, value in self.substitutions.items():
            result = result.replace(f"${{{key}}}", value)
        return result


# =============================================================================
# Suite Exporter
# =============================================================================


@dataclass
class ExportResult:
    """Result of a suite export operation."""

    success: bool
    output_path: Path | None
    content: str
    format: str
    message: str = ""
    error: Exception | None = None


class SuiteExporter:
    """Main exporter for validation suites.

    Combines formatters, configuration, and post-processors
    to provide a flexible export pipeline.

    Example:
        exporter = SuiteExporter(format="yaml")
        exporter.export(suite, "rules.yaml")

        # With custom config
        exporter = SuiteExporter(
            format="python",
            config=ExportConfig(code_style=CodeStyle.CLASS_BASED),
        )
        result = exporter.export_to_string(suite)

        # With post-processors
        exporter.add_post_processor(
            AddHeaderPostProcessor("# Custom header")
        )
    """

    def __init__(
        self,
        format: str | ExportFormat = "yaml",
        config: ExportConfig | None = None,
        formatter: SuiteFormatter | None = None,
    ):
        """Initialize exporter.

        Args:
            format: Output format name or enum
            config: Export configuration
            formatter: Custom formatter instance (overrides format)
        """
        self.format_name = (
            format.value if isinstance(format, ExportFormat) else format
        )
        self.config = config or DEFAULT_CONFIG
        self._post_processors: list[ExportPostProcessor] = []

        if formatter:
            self._formatter = formatter
        else:
            self._formatter = formatter_registry.create(self.format_name)

    @property
    def formatter(self) -> SuiteFormatter:
        """Get the formatter instance."""
        return self._formatter

    def add_post_processor(self, processor: ExportPostProcessor) -> "SuiteExporter":
        """Add a post-processor to the pipeline."""
        self._post_processors.append(processor)
        return self

    def clear_post_processors(self) -> "SuiteExporter":
        """Clear all post-processors."""
        self._post_processors.clear()
        return self

    def export_to_string(
        self,
        suite: "ValidationSuite",
        config: ExportConfig | None = None,
    ) -> str:
        """Export suite to string.

        Args:
            suite: Validation suite to export
            config: Override configuration

        Returns:
            Formatted string output
        """
        cfg = config or self.config
        content = self._formatter.format(suite, cfg)

        # Apply post-processors
        for processor in self._post_processors:
            content = processor.process(content, suite)

        return content

    def export(
        self,
        suite: "ValidationSuite",
        output_path: str | Path,
        config: ExportConfig | None = None,
    ) -> ExportResult:
        """Export suite to file.

        Args:
            suite: Validation suite to export
            output_path: Output file path
            config: Override configuration

        Returns:
            Export result
        """
        path = Path(output_path)

        try:
            content = self.export_to_string(suite, config)

            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ExportResult(
                success=True,
                output_path=path,
                content=content,
                format=self.format_name,
                message=f"Successfully exported to {path}",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                output_path=path,
                content="",
                format=self.format_name,
                message=f"Export failed: {e}",
                error=e,
            )

    def export_to_stdout(
        self,
        suite: "ValidationSuite",
        config: ExportConfig | None = None,
    ) -> str:
        """Export suite and print to stdout.

        Args:
            suite: Validation suite to export
            config: Override configuration

        Returns:
            Formatted string output (also printed)
        """
        content = self.export_to_string(suite, config)
        print(content)
        return content


# =============================================================================
# Convenience Functions
# =============================================================================


def create_exporter(
    format: str = "yaml",
    config: ExportConfig | None = None,
    **formatter_options: Any,
) -> SuiteExporter:
    """Create a suite exporter with the given format.

    Args:
        format: Output format (yaml, json, python, toml, checkpoint)
        config: Export configuration
        **formatter_options: Options passed to formatter

    Returns:
        Configured SuiteExporter instance
    """
    formatter = formatter_registry.create(format, **formatter_options)
    return SuiteExporter(
        format=format,
        config=config,
        formatter=formatter,
    )


def export_suite(
    suite: "ValidationSuite",
    output_path: str | Path,
    format: str | None = None,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export a validation suite to file.

    Args:
        suite: Validation suite to export
        output_path: Output file path
        format: Output format (inferred from extension if not provided)
        config: Export configuration

    Returns:
        Export result
    """
    path = Path(output_path)

    # Infer format from extension if not provided
    if format is None:
        ext_to_format = {
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".py": "python",
            ".toml": "toml",
        }
        format = ext_to_format.get(path.suffix.lower(), "yaml")

    exporter = create_exporter(format=format, config=config)
    return exporter.export(suite, path)


def format_suite(
    suite: "ValidationSuite",
    format: str = "yaml",
    config: ExportConfig | None = None,
) -> str:
    """Format a validation suite to string.

    Args:
        suite: Validation suite to format
        format: Output format
        config: Export configuration

    Returns:
        Formatted string
    """
    exporter = create_exporter(format=format, config=config)
    return exporter.export_to_string(suite, config)


def get_available_formats() -> list[str]:
    """Get list of available export formats."""
    return list(formatter_registry.list_all().keys())


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ExportFormat",
    "CodeStyle",
    "OutputMode",
    # Configuration
    "ExportConfig",
    "DEFAULT_CONFIG",
    "MINIMAL_CONFIG",
    "VERBOSE_CONFIG",
    # Protocol and base
    "SuiteFormatterProtocol",
    "SuiteFormatter",
    # Built-in formatters
    "YAMLFormatter",
    "JSONFormatter",
    "PythonFormatter",
    "TOMLFormatter",
    "CheckpointFormatter",
    # Registry
    "FormatterRegistry",
    "formatter_registry",
    "register_formatter",
    # Post-processors
    "ExportPostProcessor",
    "AddHeaderPostProcessor",
    "AddFooterPostProcessor",
    "TemplatePostProcessor",
    # Exporter
    "ExportResult",
    "SuiteExporter",
    # Convenience
    "create_exporter",
    "export_suite",
    "format_suite",
    "get_available_formats",
]
