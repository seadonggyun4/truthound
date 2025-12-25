"""Docstring parsing with support for multiple docstring styles.

This module provides robust docstring parsing capabilities supporting
Google, NumPy, Sphinx/reStructuredText, and Epytext docstring styles.

Key Features:
- Automatic style detection
- Consistent output format regardless of input style
- Parameter, return, exception, and example extraction
- Markdown and RST support

Example:
    from truthound.docs.parser import DocstringParser, DocstringStyle

    parser = DocstringParser(style=DocstringStyle.AUTO)
    parsed = parser.parse('''
        Calculate the sum of two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
    ''')

    print(parsed.description)
    print(parsed.params)
    print(parsed.returns)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class DocstringStyle(Enum):
    """Supported docstring styles."""

    AUTO = auto()  # Auto-detect style
    GOOGLE = auto()  # Google style
    NUMPY = auto()  # NumPy style
    SPHINX = auto()  # Sphinx/reST style
    EPYTEXT = auto()  # Epytext style


@dataclass
class ParamInfo:
    """Information about a function parameter.

    Attributes:
        name: Parameter name
        type_hint: Type annotation string
        description: Parameter description
        default: Default value if any
        optional: Whether parameter is optional
    """

    name: str
    type_hint: str = ""
    description: str = ""
    default: str | None = None
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type_hint": self.type_hint,
            "description": self.description,
            "default": self.default,
            "optional": self.optional,
        }


@dataclass
class ReturnInfo:
    """Information about function return value.

    Attributes:
        type_hint: Return type annotation
        description: Return value description
    """

    type_hint: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type_hint": self.type_hint,
            "description": self.description,
        }


@dataclass
class RaiseInfo:
    """Information about raised exceptions.

    Attributes:
        exception_type: Exception class name
        description: When/why exception is raised
    """

    exception_type: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "exception_type": self.exception_type,
            "description": self.description,
        }


@dataclass
class ExampleInfo:
    """Code example from docstring.

    Attributes:
        code: Example code
        description: Example description
        language: Code language (default: python)
    """

    code: str
    description: str = ""
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "description": self.description,
            "language": self.language,
        }


@dataclass
class AttributeInfo:
    """Information about class/module attributes.

    Attributes:
        name: Attribute name
        type_hint: Type annotation
        description: Attribute description
    """

    name: str
    type_hint: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type_hint": self.type_hint,
            "description": self.description,
        }


@dataclass
class ParsedDocstring:
    """Parsed docstring with structured information.

    Contains all extracted information from a docstring in a
    normalized format regardless of the original docstring style.

    Attributes:
        raw: Original raw docstring
        style: Detected docstring style
        short_description: First line summary
        long_description: Extended description
        params: List of parameter information
        returns: Return value information
        raises: List of exception information
        examples: List of code examples
        attributes: List of attribute information
        notes: Additional notes
        warnings: Warning notes
        see_also: Cross-references
        deprecated: Deprecation notice
        version: Version information
        todo: Todo items
    """

    raw: str = ""
    style: DocstringStyle = DocstringStyle.AUTO
    short_description: str = ""
    long_description: str = ""
    params: list[ParamInfo] = field(default_factory=list)
    returns: ReturnInfo | None = None
    yields: ReturnInfo | None = None
    raises: list[RaiseInfo] = field(default_factory=list)
    examples: list[ExampleInfo] = field(default_factory=list)
    attributes: list[AttributeInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    deprecated: str | None = None
    version: str | None = None
    todo: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        """Get full description (short + long)."""
        if self.long_description:
            return f"{self.short_description}\n\n{self.long_description}"
        return self.short_description

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "style": self.style.name,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "params": [p.to_dict() for p in self.params],
            "returns": self.returns.to_dict() if self.returns else None,
            "yields": self.yields.to_dict() if self.yields else None,
            "raises": [r.to_dict() for r in self.raises],
            "examples": [e.to_dict() for e in self.examples],
            "attributes": [a.to_dict() for a in self.attributes],
            "notes": self.notes,
            "warnings": self.warnings,
            "see_also": self.see_also,
            "deprecated": self.deprecated,
            "version": self.version,
            "todo": self.todo,
        }

    def to_markdown(self) -> str:
        """Render as Markdown."""
        lines = []

        if self.short_description:
            lines.append(self.short_description)
            lines.append("")

        if self.long_description:
            lines.append(self.long_description)
            lines.append("")

        if self.deprecated:
            lines.append(f"> **Deprecated:** {self.deprecated}")
            lines.append("")

        if self.params:
            lines.append("**Parameters:**")
            lines.append("")
            for p in self.params:
                type_str = f" ({p.type_hint})" if p.type_hint else ""
                default_str = f" = {p.default}" if p.default else ""
                lines.append(f"- `{p.name}`{type_str}{default_str}: {p.description}")
            lines.append("")

        if self.returns:
            lines.append("**Returns:**")
            lines.append("")
            type_str = f" ({self.returns.type_hint})" if self.returns.type_hint else ""
            lines.append(f"- {type_str}: {self.returns.description}")
            lines.append("")

        if self.yields:
            lines.append("**Yields:**")
            lines.append("")
            type_str = f" ({self.yields.type_hint})" if self.yields.type_hint else ""
            lines.append(f"- {type_str}: {self.yields.description}")
            lines.append("")

        if self.raises:
            lines.append("**Raises:**")
            lines.append("")
            for r in self.raises:
                lines.append(f"- `{r.exception_type}`: {r.description}")
            lines.append("")

        if self.examples:
            lines.append("**Examples:**")
            lines.append("")
            for ex in self.examples:
                if ex.description:
                    lines.append(ex.description)
                lines.append(f"```{ex.language}")
                lines.append(ex.code)
                lines.append("```")
                lines.append("")

        if self.notes:
            lines.append("**Notes:**")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        if self.warnings:
            lines.append("**Warnings:**")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"> ⚠️ {warning}")
            lines.append("")

        if self.see_also:
            lines.append("**See Also:**")
            lines.append("")
            for ref in self.see_also:
                lines.append(f"- {ref}")
            lines.append("")

        return "\n".join(lines)


class DocstringParser:
    """Parser for Python docstrings.

    Supports automatic detection and parsing of Google, NumPy, Sphinx,
    and Epytext docstring styles.

    Example:
        parser = DocstringParser()
        parsed = parser.parse(my_function.__doc__)
        print(parsed.short_description)
        print(parsed.params)

    Attributes:
        style: Preferred docstring style (AUTO for detection)
        trim: Whether to strip leading/trailing whitespace
    """

    # Section headers for Google style
    GOOGLE_SECTIONS = {
        "args": "params",
        "arguments": "params",
        "parameters": "params",
        "params": "params",
        "returns": "returns",
        "return": "returns",
        "yields": "yields",
        "yield": "yields",
        "raises": "raises",
        "raise": "raises",
        "exceptions": "raises",
        "example": "examples",
        "examples": "examples",
        "attributes": "attributes",
        "note": "notes",
        "notes": "notes",
        "warning": "warnings",
        "warnings": "warnings",
        "see also": "see_also",
        "deprecated": "deprecated",
        "todo": "todo",
    }

    # Section headers for NumPy style
    NUMPY_SECTIONS = {
        "parameters": "params",
        "params": "params",
        "returns": "returns",
        "yields": "yields",
        "raises": "raises",
        "examples": "examples",
        "attributes": "attributes",
        "notes": "notes",
        "warnings": "warnings",
        "see also": "see_also",
        "references": "see_also",
        "deprecated": "deprecated",
    }

    def __init__(
        self,
        style: DocstringStyle = DocstringStyle.AUTO,
        trim: bool = True,
    ):
        """Initialize parser.

        Args:
            style: Docstring style to use (AUTO for detection)
            trim: Whether to strip whitespace from parsed content
        """
        self.style = style
        self.trim = trim

    def parse(self, docstring: str | None) -> ParsedDocstring:
        """Parse a docstring.

        Args:
            docstring: Raw docstring to parse

        Returns:
            ParsedDocstring with extracted information
        """
        if not docstring:
            return ParsedDocstring()

        # Clean up the docstring
        docstring = self._clean_docstring(docstring)

        # Detect style if AUTO
        style = self.style
        if style == DocstringStyle.AUTO:
            style = self._detect_style(docstring)

        # Parse based on style
        if style == DocstringStyle.GOOGLE:
            return self._parse_google(docstring, style)
        elif style == DocstringStyle.NUMPY:
            return self._parse_numpy(docstring, style)
        elif style == DocstringStyle.SPHINX:
            return self._parse_sphinx(docstring, style)
        elif style == DocstringStyle.EPYTEXT:
            return self._parse_epytext(docstring, style)
        else:
            # Default to Google style parsing
            return self._parse_google(docstring, style)

    def _clean_docstring(self, docstring: str) -> str:
        """Clean and normalize docstring."""
        if not docstring:
            return ""

        # Handle indentation
        lines = docstring.expandtabs().splitlines()

        # Find minimum indentation (excluding first line)
        min_indent = float("inf")
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                min_indent = min(min_indent, len(line) - len(stripped))

        # Remove common indentation
        if min_indent < float("inf"):
            cleaned = [lines[0].strip()]
            for line in lines[1:]:
                if line.strip():
                    cleaned.append(line[int(min_indent):])
                else:
                    cleaned.append("")
            docstring = "\n".join(cleaned)

        if self.trim:
            docstring = docstring.strip()

        return docstring

    def _detect_style(self, docstring: str) -> DocstringStyle:
        """Detect docstring style.

        Args:
            docstring: Cleaned docstring

        Returns:
            Detected DocstringStyle
        """
        # Check for Sphinx style (:param, :returns:, etc.)
        if re.search(r":\w+:", docstring):
            return DocstringStyle.SPHINX

        # Check for Epytext style (@param, @return, etc.)
        if re.search(r"@\w+", docstring):
            return DocstringStyle.EPYTEXT

        # Check for NumPy style (underlined sections)
        if re.search(r"\n\s*-{3,}\s*\n", docstring):
            return DocstringStyle.NUMPY

        # Check for Google style (Section:)
        if re.search(r"\n\s*(Args|Returns|Raises|Example|Attributes):\s*\n", docstring):
            return DocstringStyle.GOOGLE

        # Default to Google
        return DocstringStyle.GOOGLE

    def _parse_google(self, docstring: str, style: DocstringStyle) -> ParsedDocstring:
        """Parse Google-style docstring."""
        result = ParsedDocstring(raw=docstring, style=style)

        # Split into sections
        sections = self._split_google_sections(docstring)

        # Parse description (content before first section)
        if "" in sections:
            desc = sections[""].strip()
            lines = desc.split("\n", 1)
            result.short_description = lines[0].strip()
            if len(lines) > 1:
                result.long_description = lines[1].strip()

        # Parse each section
        for section_name, content in sections.items():
            if section_name == "":
                continue

            section_type = self.GOOGLE_SECTIONS.get(section_name.lower())

            if section_type == "params":
                result.params = self._parse_google_params(content)
            elif section_type == "returns":
                result.returns = self._parse_google_return(content)
            elif section_type == "yields":
                result.yields = self._parse_google_return(content)
            elif section_type == "raises":
                result.raises = self._parse_google_raises(content)
            elif section_type == "examples":
                result.examples = self._parse_examples(content)
            elif section_type == "attributes":
                result.attributes = self._parse_google_attributes(content)
            elif section_type == "notes":
                result.notes = self._parse_notes(content)
            elif section_type == "warnings":
                result.warnings = self._parse_notes(content)
            elif section_type == "see_also":
                result.see_also = self._parse_see_also(content)
            elif section_type == "deprecated":
                result.deprecated = content.strip()
            elif section_type == "todo":
                result.todo = self._parse_notes(content)

        return result

    def _split_google_sections(self, docstring: str) -> dict[str, str]:
        """Split docstring into Google-style sections."""
        sections: dict[str, str] = {}

        # Pattern for section headers
        pattern = r"^(\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?):\s*$"

        current_section = ""
        current_content: list[str] = []
        base_indent = 0

        for line in docstring.split("\n"):
            match = re.match(pattern, line, re.MULTILINE)

            if match:
                # Save previous section
                if current_section or current_content:
                    sections[current_section] = "\n".join(current_content)

                current_section = match.group(2)
                current_content = []
                base_indent = len(match.group(1))
            else:
                current_content.append(line)

        # Save last section
        if current_section or current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_google_params(self, content: str) -> list[ParamInfo]:
        """Parse Google-style parameter documentation."""
        params = []

        # Pattern: name (type): description or name: description
        pattern = r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$"

        current_param: ParamInfo | None = None
        current_desc: list[str] = []

        for line in content.split("\n"):
            match = re.match(pattern, line)

            if match:
                # Save previous param
                if current_param:
                    current_param.description = " ".join(current_desc).strip()
                    params.append(current_param)

                name = match.group(1)
                type_hint = match.group(2) or ""
                desc = match.group(3).strip()

                # Check for optional
                optional = "optional" in type_hint.lower()
                if optional:
                    type_hint = re.sub(r",?\s*optional", "", type_hint, flags=re.I)

                # Check for default
                default = None
                default_match = re.search(r"default[s]?\s*[=:]\s*(.+)", desc, re.I)
                if default_match:
                    default = default_match.group(1).strip()

                current_param = ParamInfo(
                    name=name.lstrip("*"),
                    type_hint=type_hint.strip(),
                    optional=optional,
                    default=default,
                )
                current_desc = [desc] if desc else []
            else:
                stripped = line.strip()
                if stripped and current_param:
                    current_desc.append(stripped)

        # Save last param
        if current_param:
            current_param.description = " ".join(current_desc).strip()
            params.append(current_param)

        return params

    def _parse_google_return(self, content: str) -> ReturnInfo:
        """Parse Google-style return documentation."""
        content = content.strip()

        # Pattern: type: description or just description
        match = re.match(r"^([^:]+):\s*(.*)$", content, re.DOTALL)

        if match:
            type_hint = match.group(1).strip()
            description = match.group(2).strip()

            # Check if it's really a type or just the description
            if " " in type_hint and not any(c in type_hint for c in "[]|,()"):
                # Likely not a type
                return ReturnInfo(description=content)

            return ReturnInfo(type_hint=type_hint, description=description)

        return ReturnInfo(description=content)

    def _parse_google_raises(self, content: str) -> list[RaiseInfo]:
        """Parse Google-style raises documentation."""
        raises = []

        pattern = r"^\s*(\w+(?:\.\w+)*)\s*:\s*(.*)$"

        current_raise: RaiseInfo | None = None
        current_desc: list[str] = []

        for line in content.split("\n"):
            match = re.match(pattern, line)

            if match:
                if current_raise:
                    current_raise.description = " ".join(current_desc).strip()
                    raises.append(current_raise)

                current_raise = RaiseInfo(
                    exception_type=match.group(1),
                )
                current_desc = [match.group(2)] if match.group(2) else []
            else:
                stripped = line.strip()
                if stripped and current_raise:
                    current_desc.append(stripped)

        if current_raise:
            current_raise.description = " ".join(current_desc).strip()
            raises.append(current_raise)

        return raises

    def _parse_google_attributes(self, content: str) -> list[AttributeInfo]:
        """Parse Google-style attribute documentation."""
        attributes = []

        pattern = r"^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$"

        for line in content.split("\n"):
            match = re.match(pattern, line)
            if match:
                attributes.append(AttributeInfo(
                    name=match.group(1),
                    type_hint=match.group(2) or "",
                    description=match.group(3).strip(),
                ))

        return attributes

    def _parse_numpy(self, docstring: str, style: DocstringStyle) -> ParsedDocstring:
        """Parse NumPy-style docstring."""
        result = ParsedDocstring(raw=docstring, style=style)

        # Split into sections based on underlines
        sections = self._split_numpy_sections(docstring)

        # Parse description
        if "" in sections:
            desc = sections[""].strip()
            lines = desc.split("\n", 1)
            result.short_description = lines[0].strip()
            if len(lines) > 1:
                result.long_description = lines[1].strip()

        # Parse each section
        for section_name, content in sections.items():
            if section_name == "":
                continue

            section_type = self.NUMPY_SECTIONS.get(section_name.lower())

            if section_type == "params":
                result.params = self._parse_numpy_params(content)
            elif section_type == "returns":
                result.returns = self._parse_numpy_return(content)
            elif section_type == "yields":
                result.yields = self._parse_numpy_return(content)
            elif section_type == "raises":
                result.raises = self._parse_numpy_raises(content)
            elif section_type == "examples":
                result.examples = self._parse_examples(content)
            elif section_type == "attributes":
                result.attributes = self._parse_numpy_attributes(content)
            elif section_type == "notes":
                result.notes = [content.strip()]
            elif section_type == "warnings":
                result.warnings = [content.strip()]
            elif section_type == "see_also":
                result.see_also = self._parse_see_also(content)
            elif section_type == "deprecated":
                result.deprecated = content.strip()

        return result

    def _split_numpy_sections(self, docstring: str) -> dict[str, str]:
        """Split docstring into NumPy-style sections."""
        sections: dict[str, str] = {}

        # Find section headers (word followed by dashes on next line)
        lines = docstring.split("\n")
        current_section = ""
        current_content: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if next line is dashes
            if i + 1 < len(lines) and re.match(r"^-+$", lines[i + 1].strip()):
                # Save previous section
                if current_section or current_content:
                    sections[current_section] = "\n".join(current_content)

                current_section = stripped
                current_content = []
                i += 2  # Skip the dash line
                continue

            current_content.append(line)
            i += 1

        # Save last section
        if current_section or current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_numpy_params(self, content: str) -> list[ParamInfo]:
        """Parse NumPy-style parameter documentation."""
        params = []

        # Pattern: name : type
        pattern = r"^(\w+)\s*:\s*(.*)$"

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]
            match = re.match(pattern, line)

            if match:
                name = match.group(1)
                type_hint = match.group(2).strip()

                # Collect description from following indented lines
                desc_lines = []
                i += 1
                while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                    if lines[i].strip():
                        desc_lines.append(lines[i].strip())
                    i += 1

                params.append(ParamInfo(
                    name=name,
                    type_hint=type_hint,
                    description=" ".join(desc_lines),
                ))
            else:
                i += 1

        return params

    def _parse_numpy_return(self, content: str) -> ReturnInfo:
        """Parse NumPy-style return documentation."""
        lines = content.strip().split("\n")
        if not lines:
            return ReturnInfo()

        # First line might be type
        first = lines[0].strip()
        if ":" in first:
            type_hint, desc = first.split(":", 1)
            return ReturnInfo(type_hint=type_hint.strip(), description=desc.strip())

        # Check if it's just a type on the first line
        if len(lines) > 1:
            return ReturnInfo(
                type_hint=first,
                description=" ".join(line.strip() for line in lines[1:]),
            )

        return ReturnInfo(description=first)

    def _parse_numpy_raises(self, content: str) -> list[RaiseInfo]:
        """Parse NumPy-style raises documentation."""
        return self._parse_numpy_params_generic(content, RaiseInfo, "exception_type")

    def _parse_numpy_attributes(self, content: str) -> list[AttributeInfo]:
        """Parse NumPy-style attribute documentation."""
        return self._parse_numpy_params_generic(content, AttributeInfo, "name")

    def _parse_numpy_params_generic(
        self,
        content: str,
        cls: type,
        name_field: str,
    ) -> list:
        """Generic parser for NumPy-style param-like sections."""
        items = []
        pattern = r"^(\w+(?:\.\w+)*)\s*:\s*(.*)$"

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]
            match = re.match(pattern, line)

            if match:
                name = match.group(1)
                extra = match.group(2).strip()

                desc_lines = []
                i += 1
                while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                    if lines[i].strip():
                        desc_lines.append(lines[i].strip())
                    i += 1

                kwargs = {
                    name_field: name,
                    "description": " ".join(desc_lines) if desc_lines else extra,
                }
                if hasattr(cls, "type_hint"):
                    kwargs["type_hint"] = extra

                items.append(cls(**kwargs))
            else:
                i += 1

        return items

    def _parse_sphinx(self, docstring: str, style: DocstringStyle) -> ParsedDocstring:
        """Parse Sphinx/reST-style docstring."""
        result = ParsedDocstring(raw=docstring, style=style)

        # Extract description (everything before first :field:)
        desc_match = re.match(r"^(.*?)(?=\n\s*:|\Z)", docstring, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip()
            lines = desc.split("\n", 1)
            result.short_description = lines[0].strip()
            if len(lines) > 1:
                result.long_description = lines[1].strip()

        # Parse :param name: description
        for match in re.finditer(
            r":param\s+(?:(\w+)\s+)?(\w+):\s*(.+?)(?=\n\s*:|$)",
            docstring,
            re.DOTALL,
        ):
            result.params.append(ParamInfo(
                name=match.group(2),
                type_hint=match.group(1) or "",
                description=match.group(3).strip(),
            ))

        # Parse :type name: type
        for match in re.finditer(r":type\s+(\w+):\s*(.+?)(?=\n\s*:|$)", docstring, re.DOTALL):
            name = match.group(1)
            type_hint = match.group(2).strip()
            for param in result.params:
                if param.name == name:
                    param.type_hint = type_hint
                    break

        # Parse :returns: or :return:
        return_match = re.search(r":returns?:\s*(.+?)(?=\n\s*:|$)", docstring, re.DOTALL)
        if return_match:
            result.returns = ReturnInfo(description=return_match.group(1).strip())

        # Parse :rtype:
        rtype_match = re.search(r":rtype:\s*(.+?)(?=\n\s*:|$)", docstring, re.DOTALL)
        if rtype_match and result.returns:
            result.returns.type_hint = rtype_match.group(1).strip()

        # Parse :raises:
        for match in re.finditer(
            r":raises?\s+(\w+(?:\.\w+)*):\s*(.+?)(?=\n\s*:|$)",
            docstring,
            re.DOTALL,
        ):
            result.raises.append(RaiseInfo(
                exception_type=match.group(1),
                description=match.group(2).strip(),
            ))

        # Parse examples
        example_match = re.search(r":example:(.+?)(?=\n\s*:|$)", docstring, re.DOTALL)
        if example_match:
            result.examples = self._parse_examples(example_match.group(1))

        return result

    def _parse_epytext(self, docstring: str, style: DocstringStyle) -> ParsedDocstring:
        """Parse Epytext-style docstring."""
        result = ParsedDocstring(raw=docstring, style=style)

        # Extract description (everything before first @field)
        desc_match = re.match(r"^(.*?)(?=\n\s*@|\Z)", docstring, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip()
            lines = desc.split("\n", 1)
            result.short_description = lines[0].strip()
            if len(lines) > 1:
                result.long_description = lines[1].strip()

        # Parse @param name: description
        for match in re.finditer(
            r"@param\s+(\w+):\s*(.+?)(?=\n\s*@|$)",
            docstring,
            re.DOTALL,
        ):
            result.params.append(ParamInfo(
                name=match.group(1),
                description=match.group(2).strip(),
            ))

        # Parse @type name: type
        for match in re.finditer(r"@type\s+(\w+):\s*(.+?)(?=\n\s*@|$)", docstring, re.DOTALL):
            name = match.group(1)
            type_hint = match.group(2).strip()
            for param in result.params:
                if param.name == name:
                    param.type_hint = type_hint
                    break

        # Parse @return: description
        return_match = re.search(r"@returns?:\s*(.+?)(?=\n\s*@|$)", docstring, re.DOTALL)
        if return_match:
            result.returns = ReturnInfo(description=return_match.group(1).strip())

        # Parse @rtype: type
        rtype_match = re.search(r"@rtype:\s*(.+?)(?=\n\s*@|$)", docstring, re.DOTALL)
        if rtype_match and result.returns:
            result.returns.type_hint = rtype_match.group(1).strip()

        # Parse @raise exception: description
        for match in re.finditer(
            r"@raises?\s+(\w+(?:\.\w+)*):\s*(.+?)(?=\n\s*@|$)",
            docstring,
            re.DOTALL,
        ):
            result.raises.append(RaiseInfo(
                exception_type=match.group(1),
                description=match.group(2).strip(),
            ))

        return result

    def _parse_examples(self, content: str) -> list[ExampleInfo]:
        """Parse example code blocks."""
        examples = []

        # Look for doctest-style examples
        doctest_pattern = r">>>\s*(.+?)(?=\n>>>|\n\n|\Z)"
        matches = list(re.finditer(doctest_pattern, content, re.DOTALL))

        if matches:
            code_lines = []
            for match in matches:
                code_lines.append(match.group(0))
            examples.append(ExampleInfo(
                code="\n".join(code_lines).strip(),
                language="python",
            ))
        elif content.strip():
            # Treat entire content as example
            examples.append(ExampleInfo(
                code=content.strip(),
                language="python",
            ))

        return examples

    def _parse_notes(self, content: str) -> list[str]:
        """Parse notes section."""
        notes = []
        current_note: list[str] = []

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("*"):
                if current_note:
                    notes.append(" ".join(current_note))
                current_note = [stripped.lstrip("-* ")]
            elif stripped:
                current_note.append(stripped)

        if current_note:
            notes.append(" ".join(current_note))

        return notes if notes else [content.strip()]

    def _parse_see_also(self, content: str) -> list[str]:
        """Parse see also references."""
        refs = []

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped:
                # Remove list markers
                stripped = stripped.lstrip("-* ")
                if stripped:
                    refs.append(stripped)

        return refs
