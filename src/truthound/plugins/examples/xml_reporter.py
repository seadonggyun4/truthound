"""Example reporter plugin for XML output.

This plugin demonstrates how to create a custom reporter
that outputs validation results in XML format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.dom import minidom

from truthound.plugins import (
    ReporterPlugin,
    PluginConfig,
    PluginInfo,
    PluginType,
)
from truthound.reporters.base import ValidationReporter, ReporterConfig
from truthound.core import ValidationResult


@dataclass
class XMLReporterConfig(ReporterConfig):
    """Configuration for XML reporter.

    Attributes:
        pretty_print: Whether to format XML with indentation.
        include_samples: Whether to include sample values.
        xml_declaration: Whether to include XML declaration.
        encoding: XML encoding.
        root_element: Name of the root element.
    """

    pretty_print: bool = True
    include_samples: bool = True
    xml_declaration: bool = True
    encoding: str = "utf-8"
    root_element: str = "validation-report"


class XMLReporter(ValidationReporter[XMLReporterConfig]):
    """Reporter that outputs validation results as XML.

    This reporter creates well-structured XML documents suitable for:
    - Integration with enterprise systems
    - Data exchange with external tools
    - Archival purposes

    Example output:
        <?xml version="1.0" encoding="utf-8"?>
        <validation-report>
            <metadata>
                <source>data.csv</source>
                <timestamp>2024-01-15T10:30:00</timestamp>
                <row-count>10000</row-count>
            </metadata>
            <summary>
                <total-issues>5</total-issues>
                <passed>true</passed>
            </summary>
            <issues>
                <issue severity="high">
                    <column>email</column>
                    <type>invalid_format</type>
                    <count>150</count>
                    <details>Invalid email format</details>
                </issue>
            </issues>
        </validation-report>

    Example:
        >>> reporter = XMLReporter(XMLReporterConfig(pretty_print=True))
        >>> xml = reporter.render(validation_result)
        >>> print(xml)
    """

    name = "xml"
    file_extension = ".xml"
    content_type = "application/xml"

    def __init__(self, config: XMLReporterConfig | None = None, **kwargs: Any):
        if config is None:
            config = XMLReporterConfig()
        super().__init__(config, **kwargs)
        self._config: XMLReporterConfig = config

    def render(self, data: ValidationResult) -> str:
        """Render validation result as XML.

        Args:
            data: Validation result to render.

        Returns:
            XML string.
        """
        # Create root element
        root = ET.Element(self._config.root_element)

        # Add metadata
        self._add_metadata(root, data)

        # Add summary
        self._add_summary(root, data)

        # Add issues
        self._add_issues(root, data)

        # Convert to string
        if self._config.pretty_print:
            xml_str = minidom.parseString(
                ET.tostring(root, encoding="unicode")
            ).toprettyxml(indent="  ")

            # Remove extra blank lines from minidom
            lines = [line for line in xml_str.split("\n") if line.strip()]
            xml_str = "\n".join(lines)
        else:
            xml_str = ET.tostring(root, encoding="unicode")

        # Add XML declaration if needed
        if self._config.xml_declaration:
            declaration = f'<?xml version="1.0" encoding="{self._config.encoding}"?>\n'
            # Remove minidom's declaration if present
            if xml_str.startswith("<?xml"):
                xml_str = xml_str.split("?>", 1)[1].strip()
            xml_str = declaration + xml_str

        return xml_str

    def _add_metadata(self, root: ET.Element, data: ValidationResult) -> None:
        """Add metadata section to XML."""
        metadata = ET.SubElement(root, "metadata")

        ET.SubElement(metadata, "source").text = data.source
        ET.SubElement(metadata, "timestamp").text = datetime.now().isoformat()
        ET.SubElement(metadata, "row-count").text = str(data.row_count)
        ET.SubElement(metadata, "column-count").text = str(data.column_count)

        if hasattr(data, "validators_used") and data.validators_used:
            validators = ET.SubElement(metadata, "validators")
            for v in data.validators_used:
                ET.SubElement(validators, "validator").text = v

    def _add_summary(self, root: ET.Element, data: ValidationResult) -> None:
        """Add summary section to XML."""
        summary = ET.SubElement(root, "summary")

        total_issues = len(data.issues)
        ET.SubElement(summary, "total-issues").text = str(total_issues)
        ET.SubElement(summary, "passed").text = str(total_issues == 0).lower()

        # Count by severity
        severity_counts: dict[str, int] = {}
        for issue in data.issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if severity_counts:
            by_severity = ET.SubElement(summary, "by-severity")
            for severity, count in sorted(severity_counts.items()):
                sev_elem = ET.SubElement(by_severity, "severity")
                sev_elem.set("level", severity)
                sev_elem.text = str(count)

        # Count by type
        type_counts: dict[str, int] = {}
        for issue in data.issues:
            issue_type = issue.issue_type
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1

        if type_counts:
            by_type = ET.SubElement(summary, "by-type")
            for issue_type, count in sorted(type_counts.items()):
                type_elem = ET.SubElement(by_type, "type")
                type_elem.set("name", issue_type)
                type_elem.text = str(count)

    def _add_issues(self, root: ET.Element, data: ValidationResult) -> None:
        """Add issues section to XML."""
        issues_elem = ET.SubElement(root, "issues")

        for issue in data.issues:
            issue_elem = ET.SubElement(issues_elem, "issue")
            issue_elem.set("severity", issue.severity.value)

            ET.SubElement(issue_elem, "column").text = issue.column
            ET.SubElement(issue_elem, "type").text = issue.issue_type
            ET.SubElement(issue_elem, "count").text = str(issue.count)

            if issue.details:
                ET.SubElement(issue_elem, "details").text = issue.details

            if issue.expected is not None:
                ET.SubElement(issue_elem, "expected").text = str(issue.expected)

            if issue.actual is not None:
                ET.SubElement(issue_elem, "actual").text = str(issue.actual)

            if self._config.include_samples and issue.sample_values:
                samples = ET.SubElement(issue_elem, "samples")
                for sample in issue.sample_values[:5]:
                    ET.SubElement(samples, "sample").text = str(sample)

    def _default_config(self) -> XMLReporterConfig:
        """Return default XML reporter config."""
        return XMLReporterConfig()


# =============================================================================
# Plugin Class
# =============================================================================


class XMLReporterPlugin(ReporterPlugin):
    """Plugin that provides XML output format for reports.

    This plugin registers an XML reporter that can be used with:
    - truthound check --format xml
    - get_reporter("xml")

    Example:
        >>> from truthound.plugins import PluginManager
        >>> manager = PluginManager()
        >>> manager.load_from_class(XMLReporterPlugin)
        >>>
        >>> # Now XML format is available
        >>> from truthound.reporters.factory import get_reporter
        >>> reporter = get_reporter("xml")
    """

    def _get_plugin_name(self) -> str:
        return "xml-reporter"

    def _get_plugin_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return "XML output format for validation reports"

    def get_reporters(self) -> dict[str, type]:
        """Return reporter classes to register."""
        return {"xml": XMLReporter}

    def setup(self) -> None:
        """Plugin setup."""
        pass

    def teardown(self) -> None:
        """Plugin teardown."""
        pass
