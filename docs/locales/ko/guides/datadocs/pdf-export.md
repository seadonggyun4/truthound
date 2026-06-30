# PDF Export

실무 운영 가이드에서 Data Docs, Truthound, HTML, PDF, Data, Docs, WeasyPrint을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설치

실무 운영 가이드에서 PDF, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 1. System Library 설치

#### macOS (Homebrew)

```bash
brew install pango cairo gdk-pixbuf libffi
```

#### Ubuntu/Debian

```bash
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

#### Fedora/RHEL

```bash
sudo dnf install pango gdk-pixbuf2 libffi-devel
```

#### Alpine Linux

```bash
apk add pango gdk-pixbuf libffi-dev
```

#### Windows

GTK3 런타임 is required:

1. 실무 운영 가이드에서 Download, GTK3, Windows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 Extract, PATH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Alternatively을(를) 다루는 항목입니다:
```bash
pip install weasyprint[gtk3]
```

### 2. Python Package 설치

```bash
pip install truthound[pdf]
```

### Docker

```dockerfile
# Debian/Ubuntu based
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*
RUN pip install truthound[pdf]
```

```dockerfile
# Alpine based
FROM python:3.11-alpine
RUN apk add --no-cache pango gdk-pixbuf libffi-dev
RUN pip install truthound[pdf]
```

## Basic Usage

### CLI

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

### Python API

```python
from truthound.datadocs import export_to_pdf

path = export_to_pdf(
    profile=profile_dict,
    output_path="report.pdf",
    title="Data Quality Report",
    subtitle="Q4 2025",
    theme="professional",
)
print(f"PDF saved to: {path}")
```

### export_report Function

```python
from truthound.datadocs import export_report

# HTML export
export_report(profile_dict, "report.html", format="html")

# PDF export
export_report(profile_dict, "report.pdf", format="pdf")
```

## PDF Exporter

### PdfExporter

실무 운영 가이드에서 PDF을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.exporters.pdf import PdfExporter, PdfOptions

options = PdfOptions(
    page_size="A4",           # Page size
    orientation="portrait",   # portrait or landscape
    margin_top="1in",
    margin_right="0.75in",
    margin_bottom="1in",
    margin_left="0.75in",
    dpi=150,                  # Rasterization resolution
    image_quality=85,         # JPEG quality (1-100)
    font_embedding=True,      # Font embedding
    optimize=True,            # File size optimization
    linearize=False,          # Web viewing optimization
)

exporter = PdfExporter(options=options)
result = exporter.export(html_content, report_context)
pdf_bytes = result.content
```

### OptimizedPdfExporter

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.exporters.pdf import OptimizedPdfExporter, PdfOptions

exporter = OptimizedPdfExporter(
    chunk_size=1000,       # Items per chunk
    parallel=True,         # Enable parallel processing
    max_workers=None,      # Number of worker threads (None=auto)
    options=PdfOptions(
        page_size="A4",
        optimize=True,
    ),
)

result = exporter.export(html_content, report_context)
```

실무 운영 가이드에서 Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Chunk, Processes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 PDF, Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Memory, Streaming-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 PDF, Chunk, PyPDF2을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SVG Chart Rendering

실무 운영 가이드에서 PDF, Charts, SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs import HTMLReportBuilder

# Builder for PDF (internally uses _use_svg=True)
builder = HTMLReportBuilder(theme="professional", _use_svg=True)
html = builder.build(profile_dict)

# export_to_pdf automatically uses SVG
from truthound.datadocs import export_to_pdf
export_to_pdf(profile_dict, "report.pdf")  # Uses SVG charts
```

실무 운영 가이드에서 SVG, Supported, Charts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Bar, Horizontal, Line을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Pie, Donut을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Unsupported, Charts, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Heatmap, Scatter, Box, Gauge, Radar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Print CSS

실무 운영 가이드에서 PDF, CSS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```css
@page {
    size: A4 portrait;
    margin-top: 1in;
    margin-right: 0.75in;
    margin-bottom: 1in;
    margin-left: 0.75in;
}

@media print {
    body {
        font-size: 10pt;
        background: white;
        color: black;
    }
    .report-container {
        max-width: none;
        padding: 0;
    }
    .report-section {
        page-break-inside: avoid;
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #ddd;
    }
    .report-toc {
        display: none;
    }
    .no-print {
        display: none;
    }
}
```

## Error Handling

### WeasyPrintDependencyError

실무 운영 가이드에서 Raised을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs import export_to_pdf
from truthound.datadocs.builder import WeasyPrintDependencyError

try:
    export_to_pdf(profile_dict, "report.pdf")
except WeasyPrintDependencyError as e:
    print("PDF export requires system dependencies.")
    print(e)  # Outputs installation guide
```

실무 운영 가이드에서 Common, Errors을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```
cannot load library 'libpango-1.0-0'
```

실무 운영 가이드에서 System, Refer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```
ModuleNotFoundError: No module named 'weasyprint'
```

실무 운영 가이드에서 `pip install truthound[pdf]`, Python, Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API 레퍼런스

### PdfOptions

```python
@dataclass
class PdfOptions(ExportOptions):
    dpi: int = 150                    # Rasterization resolution
    image_quality: int = 85           # JPEG quality (1-100)
    font_embedding: bool = True       # Font embedding
    optimize: bool = True             # File size optimization
    linearize: bool = False           # Linearization for web viewing
    chunk_size: int = 1000            # Chunk size
    parallel: bool = True             # Parallel processing
```

### ExportOptions (Base)

```python
@dataclass
class ExportOptions:
    page_size: str = "A4"             # Page size
    orientation: str = "portrait"     # portrait/landscape
    margin_top: str = "1in"
    margin_right: str = "0.75in"
    margin_bottom: str = "1in"
    margin_left: str = "0.75in"
    compress: bool = True             # Enable compression
    include_metadata: bool = True     # Include metadata
    minify: bool = False              # HTML minification
```

### ExportResult

```python
@dataclass
class ExportResult:
    content: bytes | str              # Exported content
    format: str                       # Format (pdf, html, etc.)
    size_bytes: int                   # Size in bytes
    metadata: dict[str, Any]          # Metadata
    success: bool = True              # Success status
    error: str | None = None          # Error message
```

### export_to_pdf

```python
def export_to_pdf(
    profile: dict[str, Any] | Any,
    output_path: str | Path,
    title: str = "Data Profile Report",
    subtitle: str = "",
    theme: ReportTheme | str = ReportTheme.PROFESSIONAL,
) -> Path:
    """
    Export profile to PDF.

    Args:
        profile: TableProfile dict or object
        output_path: Output PDF file path
        title: Report title
        subtitle: Subtitle
        theme: Theme

    Returns:
        PDF file path

    Raises:
        WeasyPrintDependencyError: When dependencies are not installed
    """
```

## 함께 보기

- 실무 운영 가이드에서 HTML, Reports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Charts, Chart을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Themes, Theme을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
