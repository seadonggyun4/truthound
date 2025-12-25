"""API information extraction from Python source code.

This module provides tools to extract API information from Python
modules, classes, and functions for documentation generation.

Key Features:
- AST-based extraction for accurate parsing
- Import analysis and dependency tracking
- Type annotation extraction
- Decorator recognition
- Source code context extraction

Example:
    from truthound.docs.extractor import APIExtractor

    extractor = APIExtractor()
    module_info = extractor.extract_module("truthound.api")

    for cls in module_info.classes:
        print(f"Class: {cls.name}")
        for method in cls.methods:
            print(f"  Method: {method.name}")
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from truthound.docs.parser import DocstringParser, ParsedDocstring


class Visibility(str, Enum):
    """Visibility level of API elements."""

    PUBLIC = "public"
    PROTECTED = "protected"  # Single underscore
    PRIVATE = "private"  # Double underscore


@dataclass
class SourceLocation:
    """Source code location information.

    Attributes:
        file_path: Path to source file
        line_start: Starting line number
        line_end: Ending line number
        column: Column offset
    """

    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    column: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column": self.column,
        }


@dataclass
class DecoratorInfo:
    """Information about a decorator.

    Attributes:
        name: Decorator name
        args: Positional arguments
        kwargs: Keyword arguments
        source: Original decorator source
    """

    name: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "args": self.args,
            "kwargs": self.kwargs,
            "source": self.source,
        }


@dataclass
class ParameterInfo:
    """Function parameter information.

    Attributes:
        name: Parameter name
        type_annotation: Type annotation string
        default: Default value string
        kind: Parameter kind (positional, keyword, etc.)
    """

    name: str
    type_annotation: str = ""
    default: str | None = None
    kind: str = "positional_or_keyword"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type_annotation": self.type_annotation,
            "default": self.default,
            "kind": self.kind,
        }


@dataclass
class FunctionInfo:
    """Information about a function or method.

    Attributes:
        name: Function name
        qualified_name: Fully qualified name
        signature: Function signature string
        parameters: List of parameters
        return_type: Return type annotation
        docstring: Parsed docstring
        decorators: List of decorators
        is_async: Whether function is async
        is_generator: Whether function is a generator
        is_classmethod: Whether it's a classmethod
        is_staticmethod: Whether it's a staticmethod
        is_property: Whether it's a property
        visibility: Visibility level
        location: Source location
    """

    name: str
    qualified_name: str = ""
    signature: str = ""
    parameters: list[ParameterInfo] = field(default_factory=list)
    return_type: str = ""
    docstring: ParsedDocstring = field(default_factory=ParsedDocstring)
    decorators: list[DecoratorInfo] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    visibility: Visibility = Visibility.PUBLIC
    location: SourceLocation = field(default_factory=SourceLocation)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "signature": self.signature,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "docstring": self.docstring.to_dict(),
            "decorators": [d.to_dict() for d in self.decorators],
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "is_classmethod": self.is_classmethod,
            "is_staticmethod": self.is_staticmethod,
            "is_property": self.is_property,
            "visibility": self.visibility.value,
            "location": self.location.to_dict(),
        }


@dataclass
class ClassInfo:
    """Information about a class.

    Attributes:
        name: Class name
        qualified_name: Fully qualified name
        bases: Base class names
        docstring: Parsed docstring
        methods: List of methods
        class_methods: List of class methods
        static_methods: List of static methods
        properties: List of properties
        attributes: List of class attributes
        decorators: List of decorators
        is_abstract: Whether class is abstract
        is_dataclass: Whether class is a dataclass
        is_enum: Whether class is an Enum
        visibility: Visibility level
        location: Source location
    """

    name: str
    qualified_name: str = ""
    bases: list[str] = field(default_factory=list)
    docstring: ParsedDocstring = field(default_factory=ParsedDocstring)
    methods: list[FunctionInfo] = field(default_factory=list)
    class_methods: list[FunctionInfo] = field(default_factory=list)
    static_methods: list[FunctionInfo] = field(default_factory=list)
    properties: list[FunctionInfo] = field(default_factory=list)
    attributes: list[dict[str, Any]] = field(default_factory=list)
    decorators: list[DecoratorInfo] = field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False
    is_enum: bool = False
    visibility: Visibility = Visibility.PUBLIC
    location: SourceLocation = field(default_factory=SourceLocation)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "bases": self.bases,
            "docstring": self.docstring.to_dict(),
            "methods": [m.to_dict() for m in self.methods],
            "class_methods": [m.to_dict() for m in self.class_methods],
            "static_methods": [m.to_dict() for m in self.static_methods],
            "properties": [p.to_dict() for p in self.properties],
            "attributes": self.attributes,
            "decorators": [d.to_dict() for d in self.decorators],
            "is_abstract": self.is_abstract,
            "is_dataclass": self.is_dataclass,
            "is_enum": self.is_enum,
            "visibility": self.visibility.value,
            "location": self.location.to_dict(),
        }


@dataclass
class ModuleInfo:
    """Information about a Python module.

    Attributes:
        name: Module name
        qualified_name: Fully qualified module name
        file_path: Path to module file
        docstring: Parsed module docstring
        classes: List of classes
        functions: List of functions
        constants: List of module-level constants
        imports: List of imports
        all_exports: Contents of __all__
        submodules: List of submodule names
    """

    name: str
    qualified_name: str = ""
    file_path: str = ""
    docstring: ParsedDocstring = field(default_factory=ParsedDocstring)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    constants: list[dict[str, Any]] = field(default_factory=list)
    imports: list[dict[str, str]] = field(default_factory=list)
    all_exports: list[str] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "docstring": self.docstring.to_dict(),
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "constants": self.constants,
            "imports": self.imports,
            "all_exports": self.all_exports,
            "submodules": self.submodules,
        }


@dataclass
class PackageInfo:
    """Information about a Python package.

    Attributes:
        name: Package name
        root_path: Root directory path
        modules: List of module information
        version: Package version
        description: Package description
    """

    name: str
    root_path: str = ""
    modules: list[ModuleInfo] = field(default_factory=list)
    version: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "root_path": self.root_path,
            "modules": [m.to_dict() for m in self.modules],
            "version": self.version,
            "description": self.description,
        }


class APIExtractor:
    """Extracts API information from Python source code.

    Uses AST parsing combined with runtime introspection to extract
    comprehensive API information for documentation.

    Example:
        extractor = APIExtractor()

        # Extract single module
        module_info = extractor.extract_module("truthound.api")

        # Extract entire package
        package_info = extractor.extract_package("src/truthound")

    Attributes:
        include_private: Whether to include private members
        include_magic: Whether to include dunder methods
        docstring_parser: Parser for docstrings
    """

    def __init__(
        self,
        include_private: bool = False,
        include_magic: bool = False,
        docstring_parser: DocstringParser | None = None,
    ):
        """Initialize extractor.

        Args:
            include_private: Include private (underscore) members
            include_magic: Include magic (__dunder__) methods
            docstring_parser: Custom docstring parser
        """
        self.include_private = include_private
        self.include_magic = include_magic
        self.docstring_parser = docstring_parser or DocstringParser()

    def extract_module(
        self,
        module_name: str | None = None,
        file_path: str | Path | None = None,
    ) -> ModuleInfo:
        """Extract API information from a module.

        Args:
            module_name: Fully qualified module name
            file_path: Path to module file

        Returns:
            ModuleInfo with extracted information

        Raises:
            ValueError: If neither module_name nor file_path provided
            ImportError: If module cannot be loaded
        """
        if not module_name and not file_path:
            raise ValueError("Either module_name or file_path must be provided")

        # Load module
        if file_path:
            file_path = Path(file_path)
            module = self._load_module_from_file(file_path)
            module_name = module.__name__
        else:
            module = importlib.import_module(module_name)
            file_path = Path(inspect.getfile(module)) if hasattr(module, "__file__") else None

        # Get source code
        try:
            source = inspect.getsource(module)
            tree = ast.parse(source)
        except (OSError, TypeError):
            # Can't get source, use introspection only
            return self._extract_module_introspection(module, module_name, file_path)

        # Extract using AST
        return self._extract_module_ast(
            module,
            tree,
            module_name,
            str(file_path) if file_path else "",
        )

    def extract_package(
        self,
        package_path: str | Path,
        package_name: str | None = None,
    ) -> PackageInfo:
        """Extract API information from an entire package.

        Args:
            package_path: Path to package directory
            package_name: Package name (inferred from path if not given)

        Returns:
            PackageInfo with all module information
        """
        package_path = Path(package_path)

        if not package_path.is_dir():
            raise ValueError(f"Package path must be a directory: {package_path}")

        if package_name is None:
            package_name = package_path.name

        # Add package to sys.path temporarily
        parent_path = str(package_path.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)

        package_info = PackageInfo(
            name=package_name,
            root_path=str(package_path),
        )

        # Find all Python files
        for py_file in package_path.rglob("*.py"):
            # Skip test files
            if "test" in py_file.parts:
                continue

            # Calculate module name
            rel_path = py_file.relative_to(package_path)
            parts = list(rel_path.parts)
            parts[-1] = parts[-1].replace(".py", "")

            if parts[-1] == "__init__":
                parts = parts[:-1]

            if parts:
                module_name = f"{package_name}." + ".".join(parts)
            else:
                module_name = package_name

            try:
                module_info = self.extract_module(file_path=py_file)
                module_info.qualified_name = module_name
                package_info.modules.append(module_info)
            except Exception as e:
                # Log and continue
                print(f"Warning: Failed to extract {py_file}: {e}")

        # Extract package info from __init__.py
        init_file = package_path / "__init__.py"
        if init_file.exists():
            try:
                init_module = self.extract_module(file_path=init_file)
                package_info.description = init_module.docstring.short_description
                if hasattr(init_module, "all_exports"):
                    package_info.version = self._extract_version(init_file)
            except Exception:
                pass

        return package_info

    def _load_module_from_file(self, file_path: Path) -> Any:
        """Load a module from file path."""
        module_name = file_path.stem

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            # Module might have import errors, still extract what we can
            pass

        return module

    def _extract_module_ast(
        self,
        module: Any,
        tree: ast.Module,
        module_name: str,
        file_path: str,
    ) -> ModuleInfo:
        """Extract module info using AST."""
        info = ModuleInfo(
            name=module_name.split(".")[-1],
            qualified_name=module_name,
            file_path=file_path,
        )

        # Extract module docstring
        if ast.get_docstring(tree):
            info.docstring = self.docstring_parser.parse(ast.get_docstring(tree))

        # Extract __all__
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            info.all_exports = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]

        # Extract imports
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append({
                        "module": alias.name,
                        "alias": alias.asname or "",
                    })
            elif isinstance(node, ast.ImportFrom):
                module_from = node.module or ""
                for alias in node.names:
                    info.imports.append({
                        "module": f"{module_from}.{alias.name}" if module_from else alias.name,
                        "from": module_from,
                        "name": alias.name,
                        "alias": alias.asname or "",
                    })

        # Extract classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if self._should_include(node.name):
                    class_info = self._extract_class_ast(node, module_name, file_path)
                    info.classes.append(class_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._should_include(node.name):
                    func_info = self._extract_function_ast(node, module_name, file_path)
                    info.functions.append(func_info)

            elif isinstance(node, ast.Assign):
                # Module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        info.constants.append({
                            "name": target.id,
                            "value": ast.unparse(node.value) if hasattr(ast, "unparse") else "",
                            "line": node.lineno,
                        })

        return info

    def _extract_class_ast(
        self,
        node: ast.ClassDef,
        parent_name: str,
        file_path: str,
    ) -> ClassInfo:
        """Extract class info from AST node."""
        qualified_name = f"{parent_name}.{node.name}"

        info = ClassInfo(
            name=node.name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(node.name),
            location=SourceLocation(
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                column=node.col_offset,
            ),
        )

        # Extract bases
        info.bases = [self._unparse_node(base) for base in node.bases]

        # Extract decorators
        for decorator in node.decorator_list:
            info.decorators.append(self._extract_decorator(decorator))

            # Check for special decorators
            dec_name = self._get_decorator_name(decorator)
            if dec_name == "dataclass":
                info.is_dataclass = True
            elif dec_name in ("abstractmethod", "ABC"):
                info.is_abstract = True

        # Check if enum
        for base in info.bases:
            if "Enum" in base:
                info.is_enum = True

        # Extract docstring
        if ast.get_docstring(node):
            info.docstring = self.docstring_parser.parse(ast.get_docstring(node))

        # Extract methods and attributes
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._should_include(child.name):
                    continue

                method_info = self._extract_function_ast(child, qualified_name, file_path)

                # Classify method
                if method_info.is_classmethod:
                    info.class_methods.append(method_info)
                elif method_info.is_staticmethod:
                    info.static_methods.append(method_info)
                elif method_info.is_property:
                    info.properties.append(method_info)
                else:
                    info.methods.append(method_info)

            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                # Class attribute with annotation
                info.attributes.append({
                    "name": child.target.id,
                    "type": self._unparse_node(child.annotation) if child.annotation else "",
                    "value": self._unparse_node(child.value) if child.value else None,
                })

        return info

    def _extract_function_ast(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_name: str,
        file_path: str,
    ) -> FunctionInfo:
        """Extract function info from AST node."""
        qualified_name = f"{parent_name}.{node.name}"

        info = FunctionInfo(
            name=node.name,
            qualified_name=qualified_name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            visibility=self._get_visibility(node.name),
            location=SourceLocation(
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                column=node.col_offset,
            ),
        )

        # Extract decorators
        for decorator in node.decorator_list:
            info.decorators.append(self._extract_decorator(decorator))

            dec_name = self._get_decorator_name(decorator)
            if dec_name == "classmethod":
                info.is_classmethod = True
            elif dec_name == "staticmethod":
                info.is_staticmethod = True
            elif dec_name == "property":
                info.is_property = True

        # Extract parameters
        args = node.args

        # Defaults are aligned to the end
        defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)

        for i, arg in enumerate(args.args):
            param = ParameterInfo(
                name=arg.arg,
                type_annotation=self._unparse_node(arg.annotation) if arg.annotation else "",
                default=self._unparse_node(defaults[i]) if defaults[i] else None,
                kind="positional_or_keyword",
            )
            info.parameters.append(param)

        # *args
        if args.vararg:
            info.parameters.append(ParameterInfo(
                name=f"*{args.vararg.arg}",
                type_annotation=self._unparse_node(args.vararg.annotation) if args.vararg.annotation else "",
                kind="var_positional",
            ))

        # Keyword-only args
        kw_defaults = list(args.kw_defaults)
        for i, arg in enumerate(args.kwonlyargs):
            info.parameters.append(ParameterInfo(
                name=arg.arg,
                type_annotation=self._unparse_node(arg.annotation) if arg.annotation else "",
                default=self._unparse_node(kw_defaults[i]) if kw_defaults[i] else None,
                kind="keyword_only",
            ))

        # **kwargs
        if args.kwarg:
            info.parameters.append(ParameterInfo(
                name=f"**{args.kwarg.arg}",
                type_annotation=self._unparse_node(args.kwarg.annotation) if args.kwarg.annotation else "",
                kind="var_keyword",
            ))

        # Return type
        if node.returns:
            info.return_type = self._unparse_node(node.returns)

        # Build signature
        info.signature = self._build_signature(info)

        # Check for generator
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                info.is_generator = True
                break

        # Extract docstring
        if ast.get_docstring(node):
            info.docstring = self.docstring_parser.parse(ast.get_docstring(node))

        return info

    def _extract_decorator(self, node: ast.expr) -> DecoratorInfo:
        """Extract decorator information."""
        if isinstance(node, ast.Name):
            return DecoratorInfo(name=node.id, source=node.id)

        elif isinstance(node, ast.Attribute):
            source = self._unparse_node(node)
            return DecoratorInfo(name=node.attr, source=source)

        elif isinstance(node, ast.Call):
            name = self._get_decorator_name(node)
            args = [self._unparse_node(arg) for arg in node.args]
            kwargs = {
                kw.arg: self._unparse_node(kw.value)
                for kw in node.keywords
                if kw.arg
            }
            source = self._unparse_node(node)
            return DecoratorInfo(name=name, args=args, kwargs=kwargs, source=source)

        return DecoratorInfo(name="unknown", source=self._unparse_node(node))

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""

    def _build_signature(self, func_info: FunctionInfo) -> str:
        """Build function signature string."""
        parts = []

        for param in func_info.parameters:
            part = param.name
            if param.type_annotation:
                part += f": {param.type_annotation}"
            if param.default is not None:
                part += f" = {param.default}"
            parts.append(part)

        sig = f"({', '.join(parts)})"

        if func_info.return_type:
            sig += f" -> {func_info.return_type}"

        return sig

    def _unparse_node(self, node: ast.expr | None) -> str:
        """Convert AST node back to source code."""
        if node is None:
            return ""

        if hasattr(ast, "unparse"):
            return ast.unparse(node)

        # Fallback for older Python
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._unparse_node(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._unparse_node(node.value)}[{self._unparse_node(node.slice)}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return f"{self._unparse_node(node.left)} | {self._unparse_node(node.right)}"
        elif isinstance(node, ast.List):
            items = ", ".join(self._unparse_node(elt) for elt in node.elts)
            return f"[{items}]"
        elif isinstance(node, ast.Tuple):
            items = ", ".join(self._unparse_node(elt) for elt in node.elts)
            return f"({items})"
        elif isinstance(node, ast.Dict):
            pairs = [
                f"{self._unparse_node(k)}: {self._unparse_node(v)}"
                for k, v in zip(node.keys, node.values)
                if k is not None
            ]
            return "{" + ", ".join(pairs) + "}"

        return str(node)

    def _get_visibility(self, name: str) -> Visibility:
        """Determine visibility from name."""
        if name.startswith("__") and not name.endswith("__"):
            return Visibility.PRIVATE
        elif name.startswith("_"):
            return Visibility.PROTECTED
        return Visibility.PUBLIC

    def _should_include(self, name: str) -> bool:
        """Check if member should be included based on settings."""
        if name.startswith("__"):
            if name.endswith("__"):
                return self.include_magic
            return self.include_private
        elif name.startswith("_"):
            return self.include_private
        return True

    def _extract_module_introspection(
        self,
        module: Any,
        module_name: str,
        file_path: Path | None,
    ) -> ModuleInfo:
        """Extract module info using runtime introspection."""
        info = ModuleInfo(
            name=module_name.split(".")[-1],
            qualified_name=module_name,
            file_path=str(file_path) if file_path else "",
        )

        # Get docstring
        if module.__doc__:
            info.docstring = self.docstring_parser.parse(module.__doc__)

        # Get __all__
        if hasattr(module, "__all__"):
            info.all_exports = list(module.__all__)

        # Iterate members
        for name, obj in inspect.getmembers(module):
            if not self._should_include(name):
                continue

            if inspect.isclass(obj) and obj.__module__ == module_name:
                info.classes.append(self._extract_class_introspection(obj, module_name))

            elif inspect.isfunction(obj) and obj.__module__ == module_name:
                info.functions.append(self._extract_function_introspection(obj, module_name))

        return info

    def _extract_class_introspection(self, cls: type, parent_name: str) -> ClassInfo:
        """Extract class info using introspection."""
        info = ClassInfo(
            name=cls.__name__,
            qualified_name=f"{parent_name}.{cls.__name__}",
            bases=[base.__name__ for base in cls.__bases__ if base is not object],
            visibility=self._get_visibility(cls.__name__),
        )

        if cls.__doc__:
            info.docstring = self.docstring_parser.parse(cls.__doc__)

        for name, obj in inspect.getmembers(cls):
            if not self._should_include(name):
                continue

            if isinstance(obj, property):
                info.properties.append(FunctionInfo(
                    name=name,
                    is_property=True,
                    docstring=self.docstring_parser.parse(obj.fget.__doc__) if obj.fget else ParsedDocstring(),
                ))

            elif inspect.ismethod(obj) or inspect.isfunction(obj):
                method_info = self._extract_function_introspection(obj, info.qualified_name)
                info.methods.append(method_info)

        return info

    def _extract_function_introspection(self, func: Any, parent_name: str) -> FunctionInfo:
        """Extract function info using introspection."""
        info = FunctionInfo(
            name=func.__name__,
            qualified_name=f"{parent_name}.{func.__name__}",
            visibility=self._get_visibility(func.__name__),
        )

        # Get signature
        try:
            sig = inspect.signature(func)
            info.signature = str(sig)

            for param_name, param in sig.parameters.items():
                param_info = ParameterInfo(
                    name=param_name,
                    kind=param.kind.name.lower(),
                )

                if param.annotation != inspect.Parameter.empty:
                    param_info.type_annotation = str(param.annotation)

                if param.default != inspect.Parameter.empty:
                    param_info.default = repr(param.default)

                info.parameters.append(param_info)

            if sig.return_annotation != inspect.Signature.empty:
                info.return_type = str(sig.return_annotation)

        except (ValueError, TypeError):
            pass

        if func.__doc__:
            info.docstring = self.docstring_parser.parse(func.__doc__)

        info.is_async = inspect.iscoroutinefunction(func)
        info.is_generator = inspect.isgeneratorfunction(func)

        return info

    def _extract_version(self, init_file: Path) -> str:
        """Try to extract version from __init__.py."""
        try:
            content = init_file.read_text()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        except Exception:
            pass
        return ""


# Convenience function
def extract_api(
    source: str | Path,
    **kwargs: Any,
) -> ModuleInfo | PackageInfo:
    """Extract API information from a module or package.

    Args:
        source: Module name, file path, or package directory
        **kwargs: Options passed to APIExtractor

    Returns:
        ModuleInfo for single modules, PackageInfo for packages
    """
    extractor = APIExtractor(**kwargs)
    source_path = Path(source) if not isinstance(source, Path) else source

    if source_path.is_dir():
        return extractor.extract_package(source_path)
    elif source_path.is_file():
        return extractor.extract_module(file_path=source_path)
    else:
        # Assume it's a module name
        return extractor.extract_module(module_name=str(source))
