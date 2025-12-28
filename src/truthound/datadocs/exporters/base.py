"""Base classes and protocols for exporters.

This module defines the core abstractions for output format exporters
in the report generation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.datadocs.engine.context import ReportContext


@dataclass
class ExportOptions:
    """Options for export operations.

    Attributes:
        page_size: Page size for PDF (A4, Letter, etc.).
        orientation: Page orientation (portrait, landscape).
        margin_top: Top margin.
        margin_right: Right margin.
        margin_bottom: Bottom margin.
        margin_left: Left margin.
        compress: Whether to compress output.
        include_metadata: Whether to include metadata.
        minify: Whether to minify output.
    """
    page_size: str = "A4"
    orientation: str = "portrait"
    margin_top: str = "1cm"
    margin_right: str = "1cm"
    margin_bottom: str = "1cm"
    margin_left: str = "1cm"
    compress: bool = True
    include_metadata: bool = True
    minify: bool = False


@dataclass
class ExportResult:
    """Result of an export operation.

    Attributes:
        content: Exported content (bytes or string).
        format: Output format.
        size_bytes: Size of the output in bytes.
        metadata: Additional metadata about the export.
        success: Whether export was successful.
        error: Error message if export failed.
    """
    content: bytes | str
    format: str
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        if self.size_bytes == 0 and self.content:
            if isinstance(self.content, bytes):
                self.size_bytes = len(self.content)
            else:
                self.size_bytes = len(self.content.encode("utf-8"))


@runtime_checkable
class Exporter(Protocol):
    """Protocol for output format exporters.

    Exporters receive rendered HTML and convert it to the target format.
    """

    @property
    def format(self) -> str:
        """Get the output format (e.g., 'html', 'pdf')."""
        ...

    def export(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> bytes | str:
        """Export content to the target format.

        Args:
            content: Rendered HTML content.
            ctx: Report context for additional options.

        Returns:
            Exported content (bytes for binary formats).
        """
        ...


class BaseExporter(ABC):
    """Abstract base class for exporters.

    Provides common functionality and ensures consistent interface.
    """

    def __init__(
        self,
        options: ExportOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            options: Export options.
            name: Optional name for this exporter.
        """
        self._options = options or ExportOptions()
        self._name = name or self.__class__.__name__

    @property
    @abstractmethod
    def format(self) -> str:
        """Get the output format."""
        pass

    @property
    def name(self) -> str:
        """Get the exporter name."""
        return self._name

    @property
    def options(self) -> ExportOptions:
        """Get export options."""
        return self._options

    def export(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> bytes | str:
        """Export content to the target format.

        Args:
            content: Rendered HTML content.
            ctx: Report context.

        Returns:
            Exported content.
        """
        try:
            result = self._do_export(content, ctx)

            if isinstance(result, ExportResult):
                if not result.success:
                    raise RuntimeError(result.error or "Export failed")
                return result.content

            return result

        except Exception as e:
            # Re-raise with context
            raise RuntimeError(
                f"Export to {self.format} failed: {str(e)}"
            ) from e

    @abstractmethod
    def _do_export(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> bytes | str | ExportResult:
        """Perform the actual export.

        Subclasses implement this method.

        Args:
            content: Rendered HTML content.
            ctx: Report context.

        Returns:
            Exported content or ExportResult.
        """
        pass

    def with_options(self, **kwargs: Any) -> "BaseExporter":
        """Create a new exporter with updated options.

        Args:
            **kwargs: Option values to update.

        Returns:
            New exporter instance.
        """
        from dataclasses import replace
        new_options = replace(self._options, **kwargs)
        return self.__class__(options=new_options, name=self._name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(format={self.format!r})"
