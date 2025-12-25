"""Truthound Documentation System.

This module provides a comprehensive API documentation generation system
with support for multiple output formats (MkDocs, Sphinx, HTML).

Key Features:
- Automatic docstring extraction and parsing
- Multiple output format support
- Cross-reference generation
- Usage example extraction
- API reference generation

Example:
    from truthound.docs import DocumentationGenerator, DocConfig

    # Generate API documentation
    generator = DocumentationGenerator(DocConfig(format="mkdocs"))
    generator.generate("src/truthound", output_dir="docs/api")

    # Or use convenience function
    from truthound.docs import generate_docs
    generate_docs("src/truthound", format="mkdocs")
"""

from truthound.docs.generator import (
    DocumentationGenerator,
    DocConfig,
    DocFormat,
    generate_docs,
)
from truthound.docs.parser import (
    DocstringParser,
    ParsedDocstring,
    DocstringStyle,
)
from truthound.docs.extractor import (
    APIExtractor,
    ModuleInfo,
    ClassInfo,
    FunctionInfo,
)
from truthound.docs.renderer import (
    DocRenderer,
    MkDocsRenderer,
    SphinxRenderer,
    HTMLRenderer,
)

__all__ = [
    # Generator
    "DocumentationGenerator",
    "DocConfig",
    "DocFormat",
    "generate_docs",
    # Parser
    "DocstringParser",
    "ParsedDocstring",
    "DocstringStyle",
    # Extractor
    "APIExtractor",
    "ModuleInfo",
    "ClassInfo",
    "FunctionInfo",
    # Renderer
    "DocRenderer",
    "MkDocsRenderer",
    "SphinxRenderer",
    "HTMLRenderer",
]
