"""Automatic documentation generation for validators.

This module provides documentation generation:
- Markdown documentation
- Sphinx/MkDocs integration
- API reference generation
- Usage examples extraction

Example:
    from truthound.validators.sdk.enterprise.docs import (
        DocGenerator,
        DocConfig,
        DocFormat,
    )

    # Generate documentation
    generator = DocGenerator(config)
    docs = generator.generate(MyValidator)

    # Output to file
    generator.write(docs, Path("docs/validators"))
"""

from __future__ import annotations

import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocFormat(Enum):
    """Output format for documentation."""

    MARKDOWN = auto()
    RST = auto()          # reStructuredText (Sphinx)
    HTML = auto()
    JSON = auto()
    YAML = auto()


@dataclass(frozen=True)
class DocConfig:
    """Configuration for documentation generation.

    Attributes:
        format: Output format
        include_source: Whether to include source code
        include_examples: Whether to include usage examples
        include_changelog: Whether to include changelog
        template_dir: Directory containing custom templates
        output_dir: Directory for output files
        base_url: Base URL for links
        project_name: Name of the project
        version: Version string
    """

    format: DocFormat = DocFormat.MARKDOWN
    include_source: bool = False
    include_examples: bool = True
    include_changelog: bool = False
    template_dir: Path | None = None
    output_dir: Path | None = None
    base_url: str = ""
    project_name: str = "Truthound"
    version: str = "0.2.0"

    @classmethod
    def sphinx(cls, output_dir: Path) -> "DocConfig":
        """Create Sphinx-compatible configuration."""
        return cls(
            format=DocFormat.RST,
            include_source=True,
            include_examples=True,
            output_dir=output_dir,
        )

    @classmethod
    def mkdocs(cls, output_dir: Path) -> "DocConfig":
        """Create MkDocs-compatible configuration."""
        return cls(
            format=DocFormat.MARKDOWN,
            include_source=True,
            include_examples=True,
            output_dir=output_dir,
        )


@dataclass
class ParameterDoc:
    """Documentation for a parameter."""

    name: str
    type_hint: str
    description: str
    default: str | None = None
    required: bool = True


@dataclass
class MethodDoc:
    """Documentation for a method."""

    name: str
    signature: str
    description: str
    parameters: list[ParameterDoc] = field(default_factory=list)
    returns: str = ""
    raises: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class ValidatorDocumentation:
    """Complete documentation for a validator.

    Attributes:
        name: Validator name
        category: Validator category
        description: Full description
        version: Version string
        author: Author information
        license: License type
        parameters: Configuration parameters
        methods: Public methods
        examples: Usage examples
        see_also: Related validators
        changelog: Change history
        source_code: Source code (if included)
        generated_at: When documentation was generated
    """

    name: str
    category: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    parameters: list[ParameterDoc] = field(default_factory=list)
    methods: list[MethodDoc] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    changelog: list[dict[str, str]] = field(default_factory=list)
    source_code: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_hint,
                    "description": p.description,
                    "default": p.default,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "methods": [
                {
                    "name": m.name,
                    "signature": m.signature,
                    "description": m.description,
                    "parameters": [
                        {"name": p.name, "type": p.type_hint, "description": p.description}
                        for p in m.parameters
                    ],
                    "returns": m.returns,
                    "raises": m.raises,
                    "examples": m.examples,
                }
                for m in self.methods
            ],
            "examples": self.examples,
            "see_also": self.see_also,
            "changelog": self.changelog,
            "generated_at": self.generated_at.isoformat(),
        }


class DocFormatter(ABC):
    """Abstract base class for documentation formatters."""

    @abstractmethod
    def format(self, doc: ValidatorDocumentation) -> str:
        """Format documentation to string."""
        pass


class MarkdownFormatter(DocFormatter):
    """Formats documentation as Markdown."""

    def format(self, doc: ValidatorDocumentation) -> str:
        """Format documentation as Markdown."""
        lines = []

        # Header
        lines.append(f"# {doc.name}")
        lines.append("")
        lines.append(f"> {doc.category} validator")
        lines.append("")

        # Description
        if doc.description:
            lines.append("## Description")
            lines.append("")
            lines.append(doc.description)
            lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Version**: {doc.version}")
        if doc.author:
            lines.append(f"- **Author**: {doc.author}")
        lines.append(f"- **License**: {doc.license}")
        lines.append("")

        # Parameters
        if doc.parameters:
            lines.append("## Parameters")
            lines.append("")
            lines.append("| Name | Type | Required | Default | Description |")
            lines.append("|------|------|----------|---------|-------------|")
            for param in doc.parameters:
                required = "Yes" if param.required else "No"
                default = param.default or "-"
                lines.append(
                    f"| `{param.name}` | `{param.type_hint}` | {required} | "
                    f"{default} | {param.description} |"
                )
            lines.append("")

        # Methods
        if doc.methods:
            lines.append("## Methods")
            lines.append("")
            for method in doc.methods:
                lines.append(f"### `{method.name}`")
                lines.append("")
                lines.append(f"```python")
                lines.append(method.signature)
                lines.append("```")
                lines.append("")
                lines.append(method.description)
                lines.append("")

                if method.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    for param in method.parameters:
                        lines.append(f"- `{param.name}` ({param.type_hint}): {param.description}")
                    lines.append("")

                if method.returns:
                    lines.append(f"**Returns:** {method.returns}")
                    lines.append("")

                if method.raises:
                    lines.append("**Raises:**")
                    lines.append("")
                    for exc in method.raises:
                        lines.append(f"- {exc}")
                    lines.append("")

                if method.examples:
                    lines.append("**Examples:**")
                    lines.append("")
                    for example in method.examples:
                        lines.append("```python")
                        lines.append(example)
                        lines.append("```")
                    lines.append("")

        # Examples
        if doc.examples:
            lines.append("## Examples")
            lines.append("")
            for i, example in enumerate(doc.examples, 1):
                lines.append(f"### Example {i}")
                lines.append("")
                lines.append("```python")
                lines.append(example)
                lines.append("```")
                lines.append("")

        # See Also
        if doc.see_also:
            lines.append("## See Also")
            lines.append("")
            for ref in doc.see_also:
                lines.append(f"- [{ref}]({ref.lower()}.md)")
            lines.append("")

        # Source Code
        if doc.source_code:
            lines.append("## Source Code")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Click to expand</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(doc.source_code)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Generated at {doc.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC*")

        return "\n".join(lines)


class RSTFormatter(DocFormatter):
    """Formats documentation as reStructuredText."""

    def format(self, doc: ValidatorDocumentation) -> str:
        """Format documentation as RST."""
        lines = []

        # Title
        title = doc.name
        lines.append("=" * len(title))
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        lines.append(f"*{doc.category} validator*")
        lines.append("")

        # Description
        if doc.description:
            lines.append("Description")
            lines.append("-----------")
            lines.append("")
            lines.append(doc.description)
            lines.append("")

        # Metadata
        lines.append("Metadata")
        lines.append("--------")
        lines.append("")
        lines.append(f":Version: {doc.version}")
        if doc.author:
            lines.append(f":Author: {doc.author}")
        lines.append(f":License: {doc.license}")
        lines.append("")

        # Parameters
        if doc.parameters:
            lines.append("Parameters")
            lines.append("----------")
            lines.append("")
            for param in doc.parameters:
                lines.append(f"``{param.name}`` : {param.type_hint}")
                lines.append(f"    {param.description}")
                if param.default:
                    lines.append(f"    (default: {param.default})")
                lines.append("")

        # Examples
        if doc.examples:
            lines.append("Examples")
            lines.append("--------")
            lines.append("")
            for example in doc.examples:
                lines.append(".. code-block:: python")
                lines.append("")
                for line in example.split("\n"):
                    lines.append(f"    {line}")
                lines.append("")

        return "\n".join(lines)


class JSONFormatter(DocFormatter):
    """Formats documentation as JSON."""

    def format(self, doc: ValidatorDocumentation) -> str:
        """Format documentation as JSON."""
        return json.dumps(doc.to_dict(), indent=2)


class DocGenerator:
    """Generates documentation for validators."""

    def __init__(self, config: DocConfig | None = None):
        """Initialize generator.

        Args:
            config: Documentation configuration
        """
        self.config = config or DocConfig()
        self._formatter = self._create_formatter()

    def _create_formatter(self) -> DocFormatter:
        """Create appropriate formatter."""
        if self.config.format == DocFormat.MARKDOWN:
            return MarkdownFormatter()
        elif self.config.format == DocFormat.RST:
            return RSTFormatter()
        elif self.config.format == DocFormat.JSON:
            return JSONFormatter()
        else:
            return MarkdownFormatter()

    def _parse_docstring(self, docstring: str | None) -> dict[str, Any]:
        """Parse docstring into components."""
        if not docstring:
            return {"description": "", "params": {}, "returns": "", "raises": [], "examples": []}

        lines = docstring.strip().split("\n")
        result: dict[str, Any] = {
            "description": "",
            "params": {},
            "returns": "",
            "raises": [],
            "examples": [],
        }

        current_section = "description"
        description_lines = []
        current_param = ""

        for line in lines:
            stripped = line.strip()

            # Section headers
            if stripped.startswith("Args:") or stripped.startswith("Parameters:"):
                current_section = "params"
                continue
            elif stripped.startswith("Returns:"):
                current_section = "returns"
                continue
            elif stripped.startswith("Raises:"):
                current_section = "raises"
                continue
            elif stripped.startswith("Example:") or stripped.startswith("Examples:"):
                current_section = "examples"
                continue

            # Content parsing
            if current_section == "description":
                description_lines.append(stripped)
            elif current_section == "params":
                if ":" in stripped and not stripped.startswith(" "):
                    parts = stripped.split(":", 1)
                    current_param = parts[0].strip()
                    result["params"][current_param] = parts[1].strip() if len(parts) > 1 else ""
                elif current_param and stripped:
                    result["params"][current_param] += " " + stripped
            elif current_section == "returns":
                if result["returns"]:
                    result["returns"] += " " + stripped
                else:
                    result["returns"] = stripped
            elif current_section == "raises":
                if stripped:
                    result["raises"].append(stripped)
            elif current_section == "examples":
                result["examples"].append(stripped)

        result["description"] = " ".join(description_lines).strip()
        return result

    def _extract_parameters(self, validator_class: type) -> list[ParameterDoc]:
        """Extract configuration parameters from validator."""
        params = []

        # Check __init__ signature
        if hasattr(validator_class, "__init__"):
            sig = inspect.signature(validator_class.__init__)
            doc_info = self._parse_docstring(validator_class.__init__.__doc__)

            for name, param in sig.parameters.items():
                if name in ("self", "args", "kwargs"):
                    continue

                type_hint = ""
                if param.annotation != inspect.Parameter.empty:
                    type_hint = str(param.annotation)

                default = None
                required = True
                if param.default != inspect.Parameter.empty:
                    default = repr(param.default)
                    required = False

                description = doc_info["params"].get(name, "")

                params.append(ParameterDoc(
                    name=name,
                    type_hint=type_hint,
                    description=description,
                    default=default,
                    required=required,
                ))

        return params

    def _extract_methods(self, validator_class: type) -> list[MethodDoc]:
        """Extract public method documentation."""
        methods = []

        for name, method in inspect.getmembers(validator_class, predicate=inspect.isfunction):
            if name.startswith("_") and name not in ("__init__", "__call__"):
                continue

            try:
                sig = inspect.signature(method)
                sig_str = f"def {name}{sig}"
            except (ValueError, TypeError):
                sig_str = f"def {name}(...)"

            doc_info = self._parse_docstring(method.__doc__)

            params = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                type_hint = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                params.append(ParameterDoc(
                    name=param_name,
                    type_hint=type_hint,
                    description=doc_info["params"].get(param_name, ""),
                ))

            methods.append(MethodDoc(
                name=name,
                signature=sig_str,
                description=doc_info["description"],
                parameters=params,
                returns=doc_info["returns"],
                raises=doc_info["raises"],
                examples=doc_info["examples"],
            ))

        return methods

    def generate(self, validator_class: type) -> ValidatorDocumentation:
        """Generate documentation for a validator.

        Args:
            validator_class: Validator class to document

        Returns:
            ValidatorDocumentation
        """
        name = getattr(validator_class, "name", validator_class.__name__)
        category = getattr(validator_class, "category", "custom")
        version = getattr(validator_class, "version", "1.0.0")
        author = getattr(validator_class, "author", "")

        # Parse class docstring
        doc_info = self._parse_docstring(validator_class.__doc__)

        # Extract source if configured
        source_code = ""
        if self.config.include_source:
            try:
                source_code = inspect.getsource(validator_class)
            except (OSError, TypeError):
                pass

        # Extract examples from docstrings
        examples = []
        if self.config.include_examples:
            examples = doc_info["examples"]
            # Also check for examples attribute
            if hasattr(validator_class, "_validator_meta"):
                meta = validator_class._validator_meta
                examples.extend(list(getattr(meta, "examples", [])))

        return ValidatorDocumentation(
            name=name,
            category=category,
            description=doc_info["description"],
            version=version,
            author=author,
            license=getattr(validator_class, "license_type", "MIT"),
            parameters=self._extract_parameters(validator_class),
            methods=self._extract_methods(validator_class),
            examples=examples,
            see_also=getattr(validator_class, "see_also", []),
            changelog=getattr(validator_class, "changelog", []),
            source_code=source_code,
        )

    def format(self, doc: ValidatorDocumentation) -> str:
        """Format documentation to string.

        Args:
            doc: Documentation to format

        Returns:
            Formatted documentation string
        """
        return self._formatter.format(doc)

    def write(
        self,
        doc: ValidatorDocumentation,
        output_dir: Path | None = None,
    ) -> Path:
        """Write documentation to file.

        Args:
            doc: Documentation to write
            output_dir: Output directory (overrides config)

        Returns:
            Path to written file
        """
        output_dir = output_dir or self.config.output_dir
        if output_dir is None:
            output_dir = Path("docs/validators")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine extension
        extensions = {
            DocFormat.MARKDOWN: ".md",
            DocFormat.RST: ".rst",
            DocFormat.HTML: ".html",
            DocFormat.JSON: ".json",
            DocFormat.YAML: ".yaml",
        }
        ext = extensions.get(self.config.format, ".md")

        # Write file
        filename = f"{doc.name.lower().replace(' ', '_')}{ext}"
        filepath = output_dir / filename

        content = self.format(doc)
        with open(filepath, "w") as f:
            f.write(content)

        return filepath

    def generate_index(
        self,
        docs: list[ValidatorDocumentation],
        output_dir: Path | None = None,
    ) -> Path:
        """Generate index file for multiple validators.

        Args:
            docs: List of documentation
            output_dir: Output directory

        Returns:
            Path to index file
        """
        output_dir = output_dir or self.config.output_dir
        if output_dir is None:
            output_dir = Path("docs/validators")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by category
        by_category: dict[str, list[ValidatorDocumentation]] = {}
        for doc in docs:
            if doc.category not in by_category:
                by_category[doc.category] = []
            by_category[doc.category].append(doc)

        # Generate index
        lines = []
        lines.append("# Validator Reference")
        lines.append("")
        lines.append(f"*{len(docs)} validators in {len(by_category)} categories*")
        lines.append("")

        for category in sorted(by_category.keys()):
            lines.append(f"## {category.title()}")
            lines.append("")
            lines.append("| Validator | Version | Description |")
            lines.append("|-----------|---------|-------------|")

            for doc in sorted(by_category[category], key=lambda d: d.name):
                desc = doc.description[:50] + "..." if len(doc.description) > 50 else doc.description
                lines.append(f"| [{doc.name}]({doc.name.lower()}.md) | {doc.version} | {desc} |")

            lines.append("")

        filepath = output_dir / "index.md"
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return filepath


def generate_docs(
    validator_class: type,
    format: DocFormat = DocFormat.MARKDOWN,
) -> str:
    """Generate documentation for a validator.

    Args:
        validator_class: Validator to document
        format: Output format

    Returns:
        Documentation string
    """
    config = DocConfig(format=format)
    generator = DocGenerator(config)
    doc = generator.generate(validator_class)
    return generator.format(doc)
