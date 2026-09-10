# PDF Export

Truthound Data Docs는 WeasyPrint를 사용하여 HTML 보고서를 PDF로 내보냅니다.

인쇄용 알림 목록은 블록 레이아웃을 사용합니다. 한 페이지에 들어가는 짧은
알림 카드의 제목·본문·제안을 함께 배치하여 페이지 경계에서 잘리지 않게 합니다.
화면용 레이아웃, 보고서 테마, 입력 데이터와 품질 계산은 변경하지 않습니다.

## 설치

PDF 내보내기에는 시스템 라이브러리와 Python 패키지가 모두 필요합니다.

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
    theme="light",
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
builder = HTMLReportBuilder(theme="light", _use_svg=True)
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

## Visual Smoke Testing

Truthound의 A4 보고서 테마는 visual smoke test로 보호됩니다. 보고서 변경 중 핵심 레이아웃 규칙이 조용히 사라지지 않도록 deterministic sample report를 `light`, `dark`, `minimal` 테마별로 생성하고 HTML/PDF-ready 산출물에 다음 항목이 포함되는지 검증합니다.

- A4 portrait print 규칙
- 210mm 문서지 shell 크기
- 한국어 보고서용 font stack
- 인쇄 header 반복이 가능한 collapsed report table
- summary box, caption, page-break 제어
- PDF-ready 렌더링을 위한 SVG chart output

PDF export smoke test는 WeasyPrint와 시스템 라이브러리를 사용할 수 있는 환경에서 실행됩니다. 이 smoke는 실제 PDF를 생성한 뒤 `%PDF` header와 최소 파일 크기를 확인하고, 가능하면 `pypdf` 또는 `pdfplumber`로 보고서 텍스트를 추출하며, Poppler `pdftoppm`이 있으면 첫 페이지 PNG 렌더링까지 확인합니다. 가벼운 개발 환경에 의존성이 없으면 PDF test는 명시적으로 skip되며, HTML과 PDF-ready HTML 검증은 계속 실행됩니다.

PDF export 검증을 반드시 수행해야 하는 CI 환경에서는 `truthound[pdf]`와 플랫폼별 WeasyPrint/Pango/Cairo 라이브러리를 설치한 뒤 다음처럼 실행합니다.

```bash
TRUTHOUND_DATADOCS_REQUIRE_PDF_SMOKE=1 pytest tests/datadocs/test_report_visual_smoke.py
```

Poppler가 설치되어 있고 첫 페이지 렌더링까지 필수로 검증하려면 다음 flag를 함께 사용합니다.

```bash
TRUTHOUND_DATADOCS_REQUIRE_PDF_RENDER=1 pytest tests/datadocs/test_report_visual_smoke.py
```

이 flag들은 PDF 의존성 부재를 skip이 아니라 실패로 처리하므로, CI에서 PDF coverage가 조용히 사라지는 일을 막습니다.

WeasyPrint를 사용할 수 있는 환경에서는 PDF smoke가 공개 보고서 테마인
`light`, `dark`, `minimal`을 모두 실제 PDF로 export합니다. 이를 통해 테마별
PDF 렌더링 회귀를 막으면서도, 가벼운 기본 개발 환경에서는 시스템 PDF 의존성
없이 HTML 및 PDF-ready structural smoke를 계속 실행할 수 있습니다.

PDF coverage를 주장하는 release 또는 benchmark gate는 smoke test를 non-skip
mode로 실행해야 합니다. macOS runner에서 Homebrew로 Pango, Cairo, GLib을
설치했지만 WeasyPrint가 `libgobject` 등 동적 라이브러리를 찾지 못하면
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`를 지정합니다. 이 smoke test는
생성된 PDF에서 텍스트도 추출하므로, A4 보고서 목차, 장별 참조, 진단 기준 부록,
한국어 localization marker가 source HTML뿐 아니라 PDF artifact에도 남아 있는지
확인할 수 있습니다.

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
    theme: ReportTheme | str = ReportTheme.LIGHT,
    language: str = "en",
) -> Path:
    """
    Export profile to PDF.

    Args:
        profile: TableProfile dict or object
        output_path: Output PDF file path
        title: Report title
        subtitle: Subtitle
        theme: Theme
        language: 보고서 locale. 한국어 라벨은 "ko"를 사용합니다.

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
