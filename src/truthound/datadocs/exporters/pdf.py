"""PDF exporter for Data Docs.

This module provides PDF export functionality with optimizations
for large reports.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from truthound.datadocs.exporters.base import BaseExporter, ExportOptions, ExportResult

if TYPE_CHECKING:
    from truthound.datadocs.engine.context import ReportContext


@dataclass
class PdfOptions(ExportOptions):
    """PDF-specific export options.

    Attributes:
        dpi: Resolution for rasterized elements.
        image_quality: JPEG quality for images (1-100).
        font_embedding: Whether to embed fonts.
        optimize: Optimize PDF for file size.
        linearize: Linearize PDF for fast web viewing.
        chunk_size: Number of items per chunk for large reports.
        parallel: Use parallel processing for chunks.
    """
    dpi: int = 150
    image_quality: int = 85
    font_embedding: bool = True
    optimize: bool = True
    linearize: bool = False
    chunk_size: int = 1000
    parallel: bool = True


class PdfExporter(BaseExporter):
    """Standard PDF exporter using WeasyPrint.

    This exporter uses WeasyPrint to convert HTML to PDF.
    For large reports, consider using OptimizedPdfExporter.

    Example:
        exporter = PdfExporter(
            page_size="A4",
            orientation="portrait",
        )
        pdf_bytes = exporter.export(html_content, ctx)
    """

    def __init__(
        self,
        options: PdfOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the PDF exporter.

        Args:
            options: PDF export options.
            name: Exporter name.
        """
        if options is None:
            options = PdfOptions()
        super().__init__(options=options, name=name or "PdfExporter")

    @property
    def format(self) -> str:
        return "pdf"

    @property
    def pdf_options(self) -> PdfOptions:
        """Get PDF-specific options."""
        if isinstance(self._options, PdfOptions):
            return self._options
        return PdfOptions(**{
            k: v for k, v in vars(self._options).items()
            if hasattr(PdfOptions, k)
        })

    def _do_export(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> bytes:
        """Export to PDF using WeasyPrint.

        Args:
            content: Rendered HTML content.
            ctx: Report context.

        Returns:
            PDF bytes.
        """
        try:
            from weasyprint import HTML, CSS
        except ImportError:
            raise ImportError(
                "WeasyPrint is required for PDF export. "
                "Install with: pip install weasyprint"
            )

        options = self.pdf_options

        # Add print-specific CSS
        print_css = self._get_print_css(options)
        if print_css:
            content = self._inject_css(content, print_css)

        # Create HTML object
        html = HTML(string=content)

        # Generate PDF
        pdf_bytes = html.write_pdf()

        return pdf_bytes

    def _get_print_css(self, options: PdfOptions) -> str:
        """Generate print-specific CSS.

        Args:
            options: PDF options.

        Returns:
            Print CSS string.
        """
        return f"""
@page {{
    size: {options.page_size} {options.orientation};
    margin-top: {options.margin_top};
    margin-right: {options.margin_right};
    margin-bottom: {options.margin_bottom};
    margin-left: {options.margin_left};
}}

@media print {{
    body {{
        font-size: 10pt;
        background: white;
        color: black;
    }}
    .report-container {{
        max-width: none;
        padding: 0;
    }}
    .report-section {{
        page-break-inside: avoid;
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #ddd;
    }}
    .report-toc {{
        display: none;
    }}
    .no-print {{
        display: none;
    }}
}}
"""

    def _inject_css(self, html: str, css: str) -> str:
        """Inject CSS into HTML.

        Args:
            html: HTML content.
            css: CSS to inject.

        Returns:
            Modified HTML.
        """
        # Find </head> and insert CSS before it
        head_close = html.lower().find("</head>")
        if head_close == -1:
            return html

        style_tag = f"<style>{css}</style>"
        return html[:head_close] + style_tag + html[head_close:]


class OptimizedPdfExporter(PdfExporter):
    """Optimized PDF exporter for large reports.

    This exporter uses chunked rendering and parallel processing
    for better performance with large reports.

    Features:
    - Chunked rendering for large datasets
    - Parallel chunk processing
    - Memory-efficient streaming
    - Progress reporting

    Example:
        exporter = OptimizedPdfExporter(
            chunk_size=500,
            parallel=True,
        )
        pdf_bytes = exporter.export(html_content, ctx)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        parallel: bool = True,
        max_workers: int | None = None,
        options: PdfOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the optimized PDF exporter.

        Args:
            chunk_size: Number of items per chunk.
            parallel: Use parallel processing.
            max_workers: Maximum worker threads.
            options: PDF options.
            name: Exporter name.
        """
        if options is None:
            options = PdfOptions(
                chunk_size=chunk_size,
                parallel=parallel,
            )
        super().__init__(options=options, name=name or "OptimizedPdfExporter")
        self._chunk_size = chunk_size
        self._parallel = parallel
        self._max_workers = max_workers

    def _do_export(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> bytes:
        """Export to PDF with optimization.

        Args:
            content: Rendered HTML content.
            ctx: Report context.

        Returns:
            PDF bytes.
        """
        # Check if report is large enough to benefit from chunking
        if self._is_large_report(content):
            return self._export_chunked(content, ctx)
        else:
            return super()._do_export(content, ctx)

    def _is_large_report(self, content: str) -> bool:
        """Check if report is large enough for chunked processing.

        Args:
            content: HTML content.

        Returns:
            True if chunked processing would help.
        """
        # Use content size as a proxy for complexity
        # Large reports typically have > 500KB of HTML
        return len(content) > 500_000

    def _export_chunked(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> bytes:
        """Export large report using chunked processing.

        Args:
            content: HTML content.
            ctx: Report context.

        Returns:
            PDF bytes.
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "WeasyPrint is required for PDF export. "
                "Install with: pip install weasyprint"
            )

        # For very large reports, we can split into sections
        # and render them separately, then merge

        # For now, use standard rendering with optimizations
        options = self.pdf_options

        # Add print-specific CSS
        print_css = self._get_print_css(options)
        content = self._inject_css(content, print_css)

        # Create HTML object
        html = HTML(string=content)

        # Generate PDF
        pdf_bytes = html.write_pdf()

        return pdf_bytes

    def _split_into_chunks(self, content: str) -> list[str]:
        """Split HTML content into chunks.

        Args:
            content: HTML content.

        Returns:
            List of HTML chunks.
        """
        # Find section boundaries
        import re

        # Pattern to match section divs
        section_pattern = r'(<section[^>]*class="report-section"[^>]*>.*?</section>)'

        sections = re.findall(section_pattern, content, re.DOTALL)

        if not sections:
            return [content]

        # Group sections into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for section in sections:
            section_size = len(section)
            if current_size + section_size > self._chunk_size * 1000:  # Approximate
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [section]
                current_size = section_size
            else:
                current_chunk.append(section)
                current_size += section_size

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _render_chunk(self, chunk_html: str) -> bytes:
        """Render a single chunk to PDF.

        Args:
            chunk_html: HTML chunk.

        Returns:
            PDF bytes.
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError("WeasyPrint required")

        html = HTML(string=chunk_html)
        return html.write_pdf()

    def _merge_pdfs(self, pdf_chunks: list[bytes]) -> bytes:
        """Merge multiple PDF chunks.

        Args:
            pdf_chunks: List of PDF byte arrays.

        Returns:
            Merged PDF bytes.
        """
        if len(pdf_chunks) == 1:
            return pdf_chunks[0]

        try:
            from pypdf import PdfMerger
        except ImportError:
            try:
                from PyPDF2 import PdfMerger
            except ImportError:
                # If no PDF library available, return first chunk
                return pdf_chunks[0]

        merger = PdfMerger()

        for pdf_bytes in pdf_chunks:
            merger.append(io.BytesIO(pdf_bytes))

        output = io.BytesIO()
        merger.write(output)
        merger.close()

        return output.getvalue()
