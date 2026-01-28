"""Concrete quality reporter implementations.

This module provides the concrete reporter classes that generate
quality reports in various formats (Console, JSON, HTML, Markdown).
Each reporter combines a formatter with the base reporter functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from truthound.reporters.quality.base import (
    BaseQualityReporter,
    QualityReporterError,
    QualityRenderError,
)
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityDisplayMode,
)
from truthound.reporters.quality.protocols import QualityReportable
from truthound.reporters.quality.formatters import (
    ConsoleFormatter,
    JsonFormatter,
    MarkdownFormatter,
    HtmlFormatter,
)
from truthound.reporters.quality.filters import QualityFilter

if TYPE_CHECKING:
    from truthound.profiler.quality import RuleQualityScore


# =============================================================================
# Console Quality Reporter
# =============================================================================


class ConsoleQualityReporter(BaseQualityReporter[QualityReporterConfig]):
    """Quality reporter for console/terminal output.

    Uses Rich library markup for colorful terminal output.

    Example:
        >>> reporter = ConsoleQualityReporter()
        >>> print(reporter.render(scores))
    """

    name = "console"
    file_extension = ".txt"
    content_type = "text/plain"

    def __init__(
        self,
        config: QualityReporterConfig | None = None,
        color: bool = True,
        width: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._color = color
        self._width = width
        self._formatter = ConsoleFormatter(self._config)

    def render(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> str:
        """Render quality scores for console output.

        Args:
            data: Quality score(s) to render.

        Returns:
            Console-formatted output string.
        """
        try:
            scores = self.normalize_input(data)

            # Apply filters if configured
            if self._config.filters.has_filters():
                filter_obj = QualityFilter.from_config(self._config.filters)
                scores = filter_obj.apply(scores)

            # Sort scores
            scores = self.sort_scores(scores)

            # Limit if configured
            if self._config.max_scores:
                scores = scores[:self._config.max_scores]

            # Build output
            lines = []

            # Title
            lines.append(f"[bold cyan]{self._config.title}[/bold cyan]")
            lines.append("")

            # Summary
            if self._config.include_summary:
                summary = self._formatter.format_summary(
                    scores,
                    include_statistics=self._config.include_statistics,
                )
                lines.append(summary)
                lines.append("")

            # Main content
            if self._config.display_mode == QualityDisplayMode.COMPACT:
                # Table only
                content = self._formatter.format_scores(scores)
                lines.append(content)
            else:
                # Table
                content = self._formatter.format_scores(scores)
                lines.append(content)

                # Individual details for detailed/full mode
                if self._config.display_mode in (QualityDisplayMode.DETAILED, QualityDisplayMode.FULL):
                    lines.append("")
                    lines.append("[bold]Detailed Scores[/bold]")
                    lines.append("")
                    for score in scores:
                        lines.append(self._formatter.format_score(score))
                        lines.append("")

            return "\n".join(lines)

        except Exception as e:
            raise QualityRenderError(f"Failed to render console output: {e}")


# =============================================================================
# JSON Quality Reporter
# =============================================================================


class JsonQualityReporter(BaseQualityReporter[QualityReporterConfig]):
    """Quality reporter for JSON output.

    Produces structured JSON suitable for APIs or further processing.

    Example:
        >>> reporter = JsonQualityReporter(indent=2)
        >>> json_output = reporter.render(scores)
    """

    name = "json"
    file_extension = ".json"
    content_type = "application/json"

    def __init__(
        self,
        config: QualityReporterConfig | None = None,
        indent: int = 2,
        sort_keys: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._indent = indent
        self._sort_keys = sort_keys
        self._formatter = JsonFormatter(self._config, indent=indent, sort_keys=sort_keys)

    def render(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> str:
        """Render quality scores as JSON.

        Args:
            data: Quality score(s) to render.

        Returns:
            JSON string.
        """
        try:
            scores = self.normalize_input(data)

            # Apply filters if configured
            if self._config.filters.has_filters():
                filter_obj = QualityFilter.from_config(self._config.filters)
                scores = filter_obj.apply(scores)

            # Sort scores
            scores = self.sort_scores(scores)

            # Limit if configured
            if self._config.max_scores:
                scores = scores[:self._config.max_scores]

            return self._formatter.format_scores(scores)

        except Exception as e:
            raise QualityRenderError(f"Failed to render JSON output: {e}")


# =============================================================================
# Markdown Quality Reporter
# =============================================================================


class MarkdownQualityReporter(BaseQualityReporter[QualityReporterConfig]):
    """Quality reporter for Markdown output.

    Produces Markdown suitable for documentation or GitHub.

    Example:
        >>> reporter = MarkdownQualityReporter()
        >>> md_output = reporter.render(scores)
    """

    name = "markdown"
    file_extension = ".md"
    content_type = "text/markdown"

    def __init__(
        self,
        config: QualityReporterConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._formatter = MarkdownFormatter(self._config)

    def render(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> str:
        """Render quality scores as Markdown.

        Args:
            data: Quality score(s) to render.

        Returns:
            Markdown string.
        """
        try:
            scores = self.normalize_input(data)

            # Apply filters if configured
            if self._config.filters.has_filters():
                filter_obj = QualityFilter.from_config(self._config.filters)
                scores = filter_obj.apply(scores)

            # Sort scores
            scores = self.sort_scores(scores)

            # Limit if configured
            if self._config.max_scores:
                scores = scores[:self._config.max_scores]

            return self._formatter.format_scores(scores)

        except Exception as e:
            raise QualityRenderError(f"Failed to render Markdown output: {e}")


# =============================================================================
# HTML Quality Reporter
# =============================================================================


class HtmlQualityReporter(BaseQualityReporter[QualityReporterConfig]):
    """Quality reporter for HTML output.

    Produces styled HTML with optional charts (ApexCharts).

    Example:
        >>> reporter = HtmlQualityReporter(include_charts=True)
        >>> html_output = reporter.render(scores)
    """

    name = "html"
    file_extension = ".html"
    content_type = "text/html"

    def __init__(
        self,
        config: QualityReporterConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._formatter = HtmlFormatter(self._config)

    def render(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> str:
        """Render quality scores as HTML.

        Args:
            data: Quality score(s) to render.

        Returns:
            HTML string.
        """
        try:
            scores = self.normalize_input(data)

            # Apply filters if configured
            if self._config.filters.has_filters():
                filter_obj = QualityFilter.from_config(self._config.filters)
                scores = filter_obj.apply(scores)

            # Sort scores
            scores = self.sort_scores(scores)

            # Limit if configured
            if self._config.max_scores:
                scores = scores[:self._config.max_scores]

            return self._formatter.format_scores(scores)

        except Exception as e:
            raise QualityRenderError(f"Failed to render HTML output: {e}")


# =============================================================================
# JUnit Quality Reporter (for CI/CD integration)
# =============================================================================


class JUnitQualityReporter(BaseQualityReporter[QualityReporterConfig]):
    """Quality reporter for JUnit XML output.

    Produces JUnit-compatible XML for CI/CD integration.
    Quality scores below threshold are reported as failures.

    Example:
        >>> reporter = JUnitQualityReporter(min_f1=0.7)
        >>> junit_xml = reporter.render(scores)
    """

    name = "junit"
    file_extension = ".xml"
    content_type = "application/xml"

    def __init__(
        self,
        config: QualityReporterConfig | None = None,
        min_f1: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._min_f1 = min_f1

    def render(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> str:
        """Render quality scores as JUnit XML.

        Args:
            data: Quality score(s) to render.

        Returns:
            JUnit XML string.
        """
        try:
            import xml.etree.ElementTree as ET
            from datetime import datetime

            scores = self.normalize_input(data)
            scores = self.sort_scores(scores)

            # Calculate test stats
            total = len(scores)
            failures = sum(1 for s in scores if s.metrics.f1_score < self._min_f1)
            passed = total - failures

            # Build XML
            testsuite = ET.Element("testsuite")
            testsuite.set("name", "Quality Scores")
            testsuite.set("tests", str(total))
            testsuite.set("failures", str(failures))
            testsuite.set("errors", "0")
            testsuite.set("time", "0")
            testsuite.set("timestamp", datetime.now().isoformat())

            for score in scores:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", score.rule_name)
                testcase.set("classname", "quality.scores")

                f1 = score.metrics.f1_score
                if f1 < self._min_f1:
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("type", "QualityBelowThreshold")
                    failure.set("message", f"F1 score {f1:.2%} is below threshold {self._min_f1:.2%}")
                    failure.text = score.recommendation

            # Generate XML string
            return ET.tostring(testsuite, encoding="unicode", method="xml")

        except Exception as e:
            raise QualityRenderError(f"Failed to render JUnit XML output: {e}")
