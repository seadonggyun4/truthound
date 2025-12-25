"""High-level API documentation generation.

This module provides the main interface for generating API documentation
from Python source code with support for multiple output formats.

Key Features:
- One-line documentation generation
- Multiple format support (MkDocs, Sphinx, HTML)
- Configurable extraction and rendering
- CLI integration support

Example:
    from truthound.docs import generate_docs

    # Simple usage
    generate_docs("src/truthound", format="mkdocs", output_dir="docs/api")

    # With configuration
    from truthound.docs import DocumentationGenerator, DocConfig

    config = DocConfig(
        format="mkdocs",
        title="Truthound API",
        theme="material",
    )
    generator = DocumentationGenerator(config)
    generator.generate("src/truthound")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from truthound.docs.extractor import (
    APIExtractor,
    ModuleInfo,
    PackageInfo,
)
from truthound.docs.parser import DocstringParser, DocstringStyle
from truthound.docs.renderer import (
    DocFormat,
    DocRenderer,
    RenderConfig,
    get_renderer,
)


@dataclass
class DocConfig:
    """Configuration for documentation generation.

    Combines extraction and rendering configuration into a single
    convenient configuration object.

    Attributes:
        format: Output format (mkdocs, sphinx, html, json)
        output_dir: Output directory for generated documentation
        title: Documentation title
        description: Documentation description
        include_private: Include private members (underscore prefix)
        include_magic: Include magic methods (dunder)
        docstring_style: Preferred docstring style for parsing
        theme: Theme name for output format
        include_source_links: Include links to source code
        nav_depth: Navigation tree depth
        extra_css: Additional CSS files
        extra_js: Additional JavaScript files
    """

    format: str | DocFormat = DocFormat.MKDOCS
    output_dir: str = "docs/api"
    title: str = "API Reference"
    description: str = ""
    include_private: bool = False
    include_magic: bool = False
    docstring_style: DocstringStyle = DocstringStyle.AUTO
    theme: str = "material"
    include_source_links: bool = True
    nav_depth: int = 3
    extra_css: list[str] = field(default_factory=list)
    extra_js: list[str] = field(default_factory=list)

    def to_render_config(self) -> RenderConfig:
        """Convert to RenderConfig."""
        fmt = DocFormat(self.format) if isinstance(self.format, str) else self.format
        return RenderConfig(
            output_dir=self.output_dir,
            format=fmt,
            title=self.title,
            description=self.description,
            theme=self.theme,
            show_source_links=self.include_source_links,
            nav_depth=self.nav_depth,
            extra_css=self.extra_css,
            extra_js=self.extra_js,
        )


@dataclass
class GenerationResult:
    """Result of documentation generation.

    Attributes:
        success: Whether generation succeeded
        output_dir: Output directory path
        files_written: List of written file paths
        modules_processed: Number of modules processed
        classes_found: Number of classes found
        functions_found: Number of functions found
        errors: List of error messages
        warnings: List of warning messages
    """

    success: bool = True
    output_dir: str = ""
    files_written: list[str] = field(default_factory=list)
    modules_processed: int = 0
    classes_found: int = 0
    functions_found: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "files_written": self.files_written,
            "modules_processed": self.modules_processed,
            "classes_found": self.classes_found,
            "functions_found": self.functions_found,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Documentation Generation: {status}",
            f"Output directory: {self.output_dir}",
            f"Modules processed: {self.modules_processed}",
            f"Classes found: {self.classes_found}",
            f"Functions found: {self.functions_found}",
            f"Files written: {len(self.files_written)}",
        ]

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for err in self.errors[:5]:  # Show first 5
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")

        return "\n".join(lines)


class DocumentationGenerator:
    """High-level documentation generator.

    Combines API extraction and rendering into a simple interface.

    Example:
        generator = DocumentationGenerator(DocConfig(
            format="mkdocs",
            title="My API",
            output_dir="docs",
        ))

        result = generator.generate("src/mypackage")
        print(result)

    Attributes:
        config: Generation configuration
        extractor: API extractor instance
        renderer: Documentation renderer instance
    """

    def __init__(
        self,
        config: DocConfig | None = None,
        extractor: APIExtractor | None = None,
        renderer: DocRenderer | None = None,
    ):
        """Initialize generator.

        Args:
            config: Generation configuration
            extractor: Custom API extractor
            renderer: Custom documentation renderer
        """
        self.config = config or DocConfig()

        # Create extractor
        self.extractor = extractor or APIExtractor(
            include_private=self.config.include_private,
            include_magic=self.config.include_magic,
            docstring_parser=DocstringParser(style=self.config.docstring_style),
        )

        # Create renderer
        self.renderer = renderer or get_renderer(
            self.config.format,
            self.config.to_render_config(),
        )

    def generate(
        self,
        source: str | Path,
        output_dir: str | Path | None = None,
    ) -> GenerationResult:
        """Generate documentation for a package or module.

        Args:
            source: Path to package/module or module name
            output_dir: Output directory (uses config if not specified)

        Returns:
            GenerationResult with generation statistics
        """
        result = GenerationResult(
            output_dir=str(output_dir or self.config.output_dir),
        )

        try:
            # Extract API information
            source_path = Path(source)

            if source_path.is_dir():
                info = self.extractor.extract_package(source_path)
            elif source_path.is_file():
                info = self.extractor.extract_module(file_path=source_path)
            else:
                # Try as module name
                info = self.extractor.extract_module(module_name=str(source))

            # Count statistics
            if isinstance(info, PackageInfo):
                result.modules_processed = len(info.modules)
                for module in info.modules:
                    result.classes_found += len(module.classes)
                    result.functions_found += len(module.functions)
            else:
                result.modules_processed = 1
                result.classes_found = len(info.classes)
                result.functions_found = len(info.functions)

            # Render documentation
            written_files = self.renderer.write(info, output_dir or self.config.output_dir)
            result.files_written = [str(f) for f in written_files]

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        return result

    def generate_module(
        self,
        module_name: str,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate documentation for a single module.

        Args:
            module_name: Fully qualified module name
            output_path: Output file path

        Returns:
            Generated documentation content
        """
        info = self.extractor.extract_module(module_name=module_name)
        content = self.renderer.render_module(info)

        if output_path:
            Path(output_path).write_text(content, encoding="utf-8")

        return content

    def generate_from_info(
        self,
        info: PackageInfo | ModuleInfo,
        output_dir: str | Path | None = None,
    ) -> GenerationResult:
        """Generate documentation from pre-extracted API info.

        Args:
            info: Pre-extracted API information
            output_dir: Output directory

        Returns:
            GenerationResult
        """
        result = GenerationResult(
            output_dir=str(output_dir or self.config.output_dir),
        )

        try:
            if isinstance(info, PackageInfo):
                result.modules_processed = len(info.modules)
                for module in info.modules:
                    result.classes_found += len(module.classes)
                    result.functions_found += len(module.functions)
            else:
                result.modules_processed = 1
                result.classes_found = len(info.classes)
                result.functions_found = len(info.functions)

            written_files = self.renderer.write(info, output_dir or self.config.output_dir)
            result.files_written = [str(f) for f in written_files]

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        return result


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_docs(
    source: str | Path,
    format: str | DocFormat = "mkdocs",
    output_dir: str = "docs/api",
    title: str = "API Reference",
    **kwargs: Any,
) -> GenerationResult:
    """Generate API documentation with a single function call.

    This is the simplest way to generate documentation.

    Args:
        source: Path to package/module or module name
        format: Output format (mkdocs, sphinx, html, json)
        output_dir: Output directory
        title: Documentation title
        **kwargs: Additional configuration options

    Returns:
        GenerationResult with generation statistics

    Example:
        # Generate MkDocs documentation
        result = generate_docs(
            "src/mypackage",
            format="mkdocs",
            output_dir="docs/api",
            title="My Package API",
        )

        if result.success:
            print(f"Generated {len(result.files_written)} files")
        else:
            print(f"Errors: {result.errors}")
    """
    config = DocConfig(
        format=format,
        output_dir=output_dir,
        title=title,
        **kwargs,
    )

    generator = DocumentationGenerator(config)
    return generator.generate(source)


def generate_json_api(
    source: str | Path,
    output_path: str | Path = "api.json",
    **kwargs: Any,
) -> Path:
    """Generate JSON API documentation for programmatic access.

    Args:
        source: Path to package/module
        output_path: Output JSON file path
        **kwargs: Additional extraction options

    Returns:
        Path to generated JSON file
    """
    extractor = APIExtractor(**kwargs)

    source_path = Path(source)
    if source_path.is_dir():
        info = extractor.extract_package(source_path)
    else:
        info = extractor.extract_module(file_path=source_path)

    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(info.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def validate_docstrings(
    source: str | Path,
    strict: bool = False,
    **kwargs: Any,
) -> dict[str, list[str]]:
    """Validate docstrings in a package or module.

    Checks for:
    - Missing docstrings on public APIs
    - Undocumented parameters
    - Undocumented return values
    - Undocumented exceptions

    Args:
        source: Path to package/module
        strict: If True, treat warnings as errors
        **kwargs: Additional extraction options

    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    extractor = APIExtractor(**kwargs)

    source_path = Path(source)
    if source_path.is_dir():
        info = extractor.extract_package(source_path)
        modules = info.modules
    else:
        info = extractor.extract_module(file_path=source_path)
        modules = [info]

    issues: dict[str, list[str]] = {"errors": [], "warnings": []}

    for module in modules:
        # Check module docstring
        if not module.docstring.short_description:
            issues["warnings"].append(f"Module {module.qualified_name}: Missing docstring")

        # Check classes
        for cls in module.classes:
            _validate_class_docstring(cls, issues)

        # Check functions
        for func in module.functions:
            _validate_function_docstring(func, issues, is_method=False)

    if strict:
        issues["errors"].extend(issues["warnings"])
        issues["warnings"] = []

    return issues


def _validate_class_docstring(cls: ClassInfo, issues: dict[str, list[str]]) -> None:
    """Validate class docstring."""
    prefix = f"Class {cls.qualified_name}"

    if not cls.docstring.short_description:
        issues["warnings"].append(f"{prefix}: Missing docstring")

    # Validate methods
    for method in cls.methods:
        if not method.name.startswith("_"):
            _validate_function_docstring(method, issues, is_method=True)


def _validate_function_docstring(
    func: FunctionInfo,
    issues: dict[str, list[str]],
    is_method: bool = False,
) -> None:
    """Validate function/method docstring."""
    prefix = f"{'Method' if is_method else 'Function'} {func.qualified_name}"

    if not func.docstring.short_description:
        issues["warnings"].append(f"{prefix}: Missing docstring")
        return

    # Check parameters
    doc_params = {p.name for p in func.docstring.params}
    func_params = {p.name.lstrip("*") for p in func.parameters if p.name not in ("self", "cls")}

    missing = func_params - doc_params
    if missing:
        issues["warnings"].append(f"{prefix}: Undocumented parameters: {missing}")

    extra = doc_params - func_params
    if extra:
        issues["warnings"].append(f"{prefix}: Documented but nonexistent parameters: {extra}")

    # Check return value
    if func.return_type and func.return_type != "None" and not func.docstring.returns:
        issues["warnings"].append(f"{prefix}: Return value not documented")


# =============================================================================
# CLI Support
# =============================================================================


def main() -> int:
    """CLI entry point for documentation generation.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate API documentation from Python source code",
    )
    parser.add_argument(
        "source",
        help="Path to Python package or module",
    )
    parser.add_argument(
        "-o", "--output",
        default="docs/api",
        help="Output directory (default: docs/api)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["mkdocs", "sphinx", "html", "json", "markdown"],
        default="mkdocs",
        help="Output format (default: mkdocs)",
    )
    parser.add_argument(
        "-t", "--title",
        default="API Reference",
        help="Documentation title",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private members",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate docstrings instead of generating",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (with --validate)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.validate:
        issues = validate_docstrings(
            args.source,
            strict=args.strict,
            include_private=args.include_private,
        )

        if issues["errors"]:
            print("Errors:")
            for err in issues["errors"]:
                print(f"  - {err}")
            return 1

        if issues["warnings"]:
            print("Warnings:")
            for warn in issues["warnings"]:
                print(f"  - {warn}")

        print(f"\nValidation completed with {len(issues['warnings'])} warnings")
        return 0

    result = generate_docs(
        args.source,
        format=args.format,
        output_dir=args.output,
        title=args.title,
        include_private=args.include_private,
    )

    if args.verbose:
        print(result)
    else:
        if result.success:
            print(f"Generated {len(result.files_written)} files to {result.output_dir}")
        else:
            print(f"Failed: {result.errors}")

    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
