"""Immutable context for the report generation pipeline.

This module defines the core data structures that flow through the pipeline,
ensuring immutability and clear state tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, TypeVar, Generic


T = TypeVar("T")


@dataclass(frozen=True)
class TranslatableString:
    """A string that can be translated.

    This is used to mark strings that should be looked up in the message catalog
    during the i18n transformation phase.

    Example:
        title = TranslatableString(
            key="report.quality_score.title",
            params={"score": 95.5}
        )
    """
    key: str
    params: dict[str, Any] = field(default_factory=dict)
    default: str | None = None

    def format(self, template: str) -> str:
        """Format the template with parameters.

        Args:
            template: Message template with {param} placeholders.

        Returns:
            Formatted string.
        """
        try:
            return template.format(**self.params)
        except KeyError:
            return self.default or self.key


@dataclass(frozen=True)
class ReportData:
    """Structured report data for pipeline processing.

    This is a more structured representation of profile data that
    is optimized for pipeline processing.

    Attributes:
        raw: Original raw data from the profiler.
        sections: Pre-processed section data.
        metadata: Report metadata (title, author, etc.).
        alerts: Collected alerts/warnings.
        recommendations: Generated recommendations.
        charts: Chart specifications.
        tables: Table specifications.
    """
    raw: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)

    def with_section(self, name: str, data: dict[str, Any]) -> "ReportData":
        """Create a new ReportData with an added/updated section.

        Args:
            name: Section name.
            data: Section data.

        Returns:
            New ReportData with the section added.
        """
        new_sections = dict(self.sections)
        new_sections[name] = data
        return replace(self, sections=new_sections)

    def with_alert(self, alert: dict[str, Any]) -> "ReportData":
        """Create a new ReportData with an added alert.

        Args:
            alert: Alert data.

        Returns:
            New ReportData with the alert added.
        """
        return replace(self, alerts=[*self.alerts, alert])

    def with_recommendation(self, recommendation: str) -> "ReportData":
        """Create a new ReportData with an added recommendation.

        Args:
            recommendation: Recommendation text.

        Returns:
            New ReportData with the recommendation added.
        """
        return replace(self, recommendations=[*self.recommendations, recommendation])

    def with_chart(self, chart: dict[str, Any]) -> "ReportData":
        """Create a new ReportData with an added chart.

        Args:
            chart: Chart specification.

        Returns:
            New ReportData with the chart added.
        """
        return replace(self, charts=[*self.charts, chart])

    def with_metadata(self, **kwargs: Any) -> "ReportData":
        """Create a new ReportData with updated metadata.

        Args:
            **kwargs: Metadata key-value pairs to add/update.

        Returns:
            New ReportData with updated metadata.
        """
        new_metadata = dict(self.metadata)
        new_metadata.update(kwargs)
        return replace(self, metadata=new_metadata)


@dataclass(frozen=True)
class ReportContext:
    """Immutable context passed through the pipeline.

    This is the main data carrier through all pipeline stages.
    Each stage receives a context and produces a new (possibly modified) context.

    The context is immutable - modifications create new instances.

    Attributes:
        data: Report data to process.
        locale: Target locale for i18n (e.g., "en", "ko", "ja").
        theme: Theme name or identifier.
        template: Template name to use for rendering.
        output_format: Target output format (html, pdf, markdown, json).
        options: Additional options for pipeline stages.
        created_at: When this context was created.
        version: Context version for tracking changes.
        trace: List of pipeline stages this context has passed through.

    Example:
        ctx = ReportContext(
            data=ReportData(raw=profile_dict),
            locale="ko",
            theme="enterprise",
            output_format="pdf",
        )

        # Modify immutably
        ctx = ctx.with_locale("ja").with_option("page_size", "A4")
    """
    data: ReportData
    locale: str = "en"
    theme: str = "default"
    template: str = "default"
    output_format: str = "html"
    options: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    trace: tuple[str, ...] = field(default_factory=tuple)

    # Convenience accessors

    @property
    def raw_data(self) -> dict[str, Any]:
        """Get the raw data dictionary."""
        return self.data.raw

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata dictionary."""
        return self.data.metadata

    @property
    def title(self) -> str:
        """Get the report title."""
        return self.data.metadata.get("title", "Data Quality Report")

    @property
    def subtitle(self) -> str:
        """Get the report subtitle."""
        return self.data.metadata.get("subtitle", "")

    # Fluent modification methods

    def with_data(self, data: ReportData) -> "ReportContext":
        """Create a new context with different data.

        Args:
            data: New report data.

        Returns:
            New context with the data replaced.
        """
        return replace(self, data=data, version=self.version + 1)

    def with_locale(self, locale: str) -> "ReportContext":
        """Create a new context with a different locale.

        Args:
            locale: New locale code.

        Returns:
            New context with the locale changed.
        """
        return replace(self, locale=locale, version=self.version + 1)

    def with_theme(self, theme: str) -> "ReportContext":
        """Create a new context with a different theme.

        Args:
            theme: New theme name.

        Returns:
            New context with the theme changed.
        """
        return replace(self, theme=theme, version=self.version + 1)

    def with_template(self, template: str) -> "ReportContext":
        """Create a new context with a different template.

        Args:
            template: New template name.

        Returns:
            New context with the template changed.
        """
        return replace(self, template=template, version=self.version + 1)

    def with_output_format(self, output_format: str) -> "ReportContext":
        """Create a new context with a different output format.

        Args:
            output_format: New output format.

        Returns:
            New context with the output format changed.
        """
        return replace(self, output_format=output_format, version=self.version + 1)

    def with_option(self, key: str, value: Any) -> "ReportContext":
        """Create a new context with an added/updated option.

        Args:
            key: Option key.
            value: Option value.

        Returns:
            New context with the option added.
        """
        new_options = dict(self.options)
        new_options[key] = value
        return replace(self, options=new_options, version=self.version + 1)

    def with_options(self, **options: Any) -> "ReportContext":
        """Create a new context with multiple options added/updated.

        Args:
            **options: Option key-value pairs.

        Returns:
            New context with the options added.
        """
        new_options = dict(self.options)
        new_options.update(options)
        return replace(self, options=new_options, version=self.version + 1)

    def with_trace(self, stage_name: str) -> "ReportContext":
        """Create a new context with a trace entry added.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            New context with the trace entry added.
        """
        return replace(
            self,
            trace=(*self.trace, stage_name),
            version=self.version + 1,
        )

    def get_option(self, key: str, default: T = None) -> T | Any:
        """Get an option value with default.

        Args:
            key: Option key.
            default: Default value if not found.

        Returns:
            Option value or default.
        """
        return self.options.get(key, default)

    # Factory methods

    @classmethod
    def from_profile(
        cls,
        profile: dict[str, Any] | Any,
        locale: str = "en",
        theme: str = "default",
        **options: Any,
    ) -> "ReportContext":
        """Create a context from profile data.

        Args:
            profile: Profile data dict or object with to_dict() method.
            locale: Target locale.
            theme: Theme name.
            **options: Additional options.

        Returns:
            New ReportContext.
        """
        if hasattr(profile, "to_dict"):
            raw_data = profile.to_dict()
        else:
            raw_data = dict(profile) if profile else {}

        # Extract metadata from profile
        metadata = {
            "title": raw_data.get("title", "Data Quality Report"),
            "subtitle": raw_data.get("subtitle", ""),
            "source": raw_data.get("source", ""),
            "row_count": raw_data.get("row_count", 0),
            "column_count": raw_data.get("column_count", 0),
        }

        data = ReportData(raw=raw_data, metadata=metadata)

        return cls(
            data=data,
            locale=locale,
            theme=theme,
            options=options,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "data": {
                "raw": self.data.raw,
                "sections": self.data.sections,
                "metadata": self.data.metadata,
                "alerts": self.data.alerts,
                "recommendations": self.data.recommendations,
                "charts": self.data.charts,
                "tables": self.data.tables,
            },
            "locale": self.locale,
            "theme": self.theme,
            "template": self.template,
            "output_format": self.output_format,
            "options": self.options,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "trace": list(self.trace),
        }
