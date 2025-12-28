"""Documentation extraction from plugin source code.

This module provides tools to extract documentation from plugins
by parsing source code, docstrings, and type annotations.
"""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.plugins.base import Plugin

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """Schema for plugin configuration.

    Attributes:
        name: Configuration name
        type_hint: Type annotation
        default: Default value
        description: Description from docstring
        required: Whether the config is required
    """

    name: str
    type_hint: str = "Any"
    default: Any = None
    description: str = ""
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type_hint,
            "default": repr(self.default) if self.default is not None else None,
            "description": self.description,
            "required": self.required,
        }


@dataclass
class HookDocumentation:
    """Documentation for a plugin hook.

    Attributes:
        name: Hook name
        hook_type: Type of hook (e.g., AFTER_VALIDATION)
        description: Hook description
        parameters: List of parameter docs
        returns: Return value documentation
        example: Example usage code
    """

    name: str
    hook_type: str
    description: str = ""
    parameters: list[tuple[str, str, str]] = field(default_factory=list)  # (name, type, desc)
    returns: str = ""
    example: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "hook_type": self.hook_type,
            "description": self.description,
            "parameters": [
                {"name": n, "type": t, "description": d}
                for n, t, d in self.parameters
            ],
            "returns": self.returns,
            "example": self.example,
        }


@dataclass
class CodeExample:
    """Code example for documentation.

    Attributes:
        title: Example title
        description: What the example demonstrates
        code: The example code
        language: Programming language
    """

    title: str
    description: str = ""
    code: str = ""
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "code": self.code,
            "language": self.language,
        }


@dataclass
class PluginDocumentation:
    """Complete documentation for a plugin.

    Attributes:
        name: Plugin name
        version: Plugin version
        plugin_type: Type of plugin
        description: Full description
        author: Plugin author
        license: License identifier
        hooks: List of hook documentation
        configuration: Configuration schema
        examples: Code examples
        changelog: List of changelog entries
    """

    name: str
    version: str
    plugin_type: str
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    hooks: list[HookDocumentation] = field(default_factory=list)
    configuration: list[ConfigSchema] = field(default_factory=list)
    examples: list[CodeExample] = field(default_factory=list)
    changelog: list[tuple[str, str]] = field(default_factory=list)  # (version, changes)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "hooks": [h.to_dict() for h in self.hooks],
            "configuration": [c.to_dict() for c in self.configuration],
            "examples": [e.to_dict() for e in self.examples],
            "changelog": [
                {"version": v, "changes": c} for v, c in self.changelog
            ],
            "dependencies": self.dependencies,
        }


class DocumentationExtractor:
    """Extracts documentation from plugin source code.

    Parses plugin classes to extract:
    - Plugin metadata (name, version, type)
    - Docstrings and descriptions
    - Hook definitions and documentation
    - Configuration schema
    - Code examples from docstrings

    Example:
        >>> extractor = DocumentationExtractor()
        >>> docs = extractor.extract(MyPlugin)
        >>> print(docs.name, docs.version)
    """

    def extract(
        self,
        plugin_class: type["Plugin"],
    ) -> PluginDocumentation:
        """Extract documentation from a plugin class.

        Args:
            plugin_class: Plugin class to document

        Returns:
            PluginDocumentation with extracted info
        """
        # Get basic info from plugin
        try:
            instance = plugin_class()
            info = instance.info
        except Exception:
            # Can't instantiate, extract from class
            info = None

        # Get docstring
        description = inspect.getdoc(plugin_class) or ""

        # Extract examples from docstring
        examples = self._extract_examples(description)

        # Extract hooks
        hooks = self._extract_hooks(plugin_class)

        # Extract configuration
        config = self._extract_config(plugin_class)

        return PluginDocumentation(
            name=info.name if info else plugin_class.__name__.lower(),
            version=info.version if info else "1.0.0",
            plugin_type=info.plugin_type.value if info else "custom",
            description=self._clean_description(description),
            author=info.author if info else "",
            license=info.license if info else "",
            homepage=info.homepage if info else "",
            hooks=hooks,
            configuration=config,
            examples=examples,
            dependencies=list(info.dependencies) if info else [],
        )

    def extract_from_file(
        self,
        file_path: Path,
    ) -> list[PluginDocumentation]:
        """Extract documentation from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of PluginDocumentation for each plugin class
        """
        with open(file_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)
        docs: list[PluginDocumentation] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a Plugin subclass
                if self._is_plugin_class(node):
                    doc = self._extract_from_ast(node, source)
                    if doc:
                        docs.append(doc)

        return docs

    def _extract_hooks(
        self,
        plugin_class: type["Plugin"],
    ) -> list[HookDocumentation]:
        """Extract hook definitions from plugin class."""
        hooks: list[HookDocumentation] = []

        # Look for get_hooks method
        if hasattr(plugin_class, "get_hooks"):
            try:
                instance = plugin_class()
                hook_dict = instance.get_hooks()

                for hook_name, handler in hook_dict.items():
                    doc = inspect.getdoc(handler) or ""
                    params = self._extract_parameters(handler)

                    hooks.append(HookDocumentation(
                        name=hook_name,
                        hook_type=self._infer_hook_type(hook_name),
                        description=doc,
                        parameters=params,
                    ))
            except Exception:
                pass

        # Look for decorated methods
        for name, method in inspect.getmembers(plugin_class, predicate=inspect.isfunction):
            if hasattr(method, "_hook_type"):
                doc = inspect.getdoc(method) or ""
                params = self._extract_parameters(method)

                hooks.append(HookDocumentation(
                    name=name,
                    hook_type=getattr(method, "_hook_type", "custom"),
                    description=doc,
                    parameters=params,
                ))

        return hooks

    def _extract_config(
        self,
        plugin_class: type["Plugin"],
    ) -> list[ConfigSchema]:
        """Extract configuration schema from plugin class."""
        config: list[ConfigSchema] = []

        # Check for config class
        for name, attr in inspect.getmembers(plugin_class):
            if name.endswith("Config") and isinstance(attr, type):
                # Extract fields from dataclass or class
                annotations = getattr(attr, "__annotations__", {})
                defaults = {}

                # Get default values
                for field_name in annotations:
                    if hasattr(attr, field_name):
                        defaults[field_name] = getattr(attr, field_name)

                for field_name, type_hint in annotations.items():
                    config.append(ConfigSchema(
                        name=field_name,
                        type_hint=str(type_hint),
                        default=defaults.get(field_name),
                        required=field_name not in defaults,
                    ))

        return config

    def _extract_examples(self, docstring: str) -> list[CodeExample]:
        """Extract code examples from docstring."""
        examples: list[CodeExample] = []

        if not docstring:
            return examples

        # Look for Example: or Examples: sections
        lines = docstring.split("\n")
        in_example = False
        example_lines: list[str] = []
        example_title = "Example"

        for line in lines:
            if line.strip().startswith("Example:") or line.strip().startswith("Examples:"):
                in_example = True
                example_title = line.strip().rstrip(":")
                continue

            if in_example:
                if line.strip().startswith(">>>"):
                    example_lines.append(line.strip()[4:])
                elif line.strip().startswith("..."):
                    example_lines.append(line.strip()[4:])
                elif line.strip() == "" and example_lines:
                    # End of example
                    examples.append(CodeExample(
                        title=example_title,
                        code="\n".join(example_lines),
                    ))
                    example_lines = []
                    in_example = False

        # Handle trailing example
        if example_lines:
            examples.append(CodeExample(
                title=example_title,
                code="\n".join(example_lines),
            ))

        return examples

    def _extract_parameters(
        self,
        func: Any,
    ) -> list[tuple[str, str, str]]:
        """Extract parameter documentation from function."""
        params: list[tuple[str, str, str]] = []

        try:
            sig = inspect.signature(func)
            annotations = func.__annotations__ if hasattr(func, "__annotations__") else {}

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                type_hint = str(annotations.get(param_name, "Any"))
                params.append((param_name, type_hint, ""))

        except Exception:
            pass

        return params

    def _extract_from_ast(
        self,
        node: ast.ClassDef,
        source: str,
    ) -> PluginDocumentation | None:
        """Extract documentation from AST class node."""
        docstring = ast.get_docstring(node) or ""

        return PluginDocumentation(
            name=node.name.lower().replace("plugin", ""),
            version="1.0.0",
            plugin_type="custom",
            description=docstring,
        )

    def _is_plugin_class(self, node: ast.ClassDef) -> bool:
        """Check if AST node is a Plugin subclass."""
        for base in node.bases:
            if isinstance(base, ast.Name) and "Plugin" in base.id:
                return True
            if isinstance(base, ast.Attribute) and "Plugin" in base.attr:
                return True
        return False

    def _infer_hook_type(self, hook_name: str) -> str:
        """Infer hook type from name."""
        hook_name_lower = hook_name.lower()

        if "before" in hook_name_lower and "validation" in hook_name_lower:
            return "BEFORE_VALIDATION"
        if "after" in hook_name_lower and "validation" in hook_name_lower:
            return "AFTER_VALIDATION"
        if "error" in hook_name_lower:
            return "ON_ERROR"
        if "report" in hook_name_lower:
            return "ON_REPORT_GENERATE"

        return "CUSTOM"

    def _clean_description(self, docstring: str) -> str:
        """Clean up docstring for use as description."""
        if not docstring:
            return ""

        # Remove example sections
        lines = docstring.split("\n")
        clean_lines: list[str] = []
        skip = False

        for line in lines:
            if line.strip().startswith("Example:"):
                skip = True
            elif skip and line.strip() == "":
                skip = False
            elif not skip:
                clean_lines.append(line)

        return "\n".join(clean_lines).strip()
