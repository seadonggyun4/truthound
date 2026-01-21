# PDF Export

Truthound Data Docs는 WeasyPrint를 사용하여 HTML 리포트를 PDF로 내보낼 수 있습니다.

## 설치

PDF 내보내기는 **시스템 라이브러리**와 **Python 패키지** 모두 필요합니다.

### 1. 시스템 라이브러리 설치

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

GTK3 런타임이 필요합니다:

1. [GTK3 for Windows](https://github.com/nickvidal/weasyprint/releases) 다운로드
2. 압축 해제 후 PATH에 추가

또는 bundled 버전:
```bash
pip install weasyprint[gtk3]
```

### 2. Python 패키지 설치

```bash
pip install truthound[pdf]
```

### Docker

```dockerfile
# Debian/Ubuntu 기반
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
# Alpine 기반
FROM python:3.11-alpine
RUN apk add --no-cache pango gdk-pixbuf libffi-dev
RUN pip install truthound[pdf]
```

## 기본 사용법

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

### export_report 함수

```python
from truthound.datadocs import export_report

# HTML 내보내기
export_report(profile_dict, "report.html", format="html")

# PDF 내보내기
export_report(profile_dict, "report.pdf", format="pdf")
```

## PDF Exporter

### PdfExporter

기본 PDF 익스포터입니다.

```python
from truthound.datadocs.exporters.pdf import PdfExporter, PdfOptions

options = PdfOptions(
    page_size="A4",           # 페이지 크기
    orientation="portrait",   # portrait 또는 landscape
    margin_top="1in",
    margin_right="0.75in",
    margin_bottom="1in",
    margin_left="0.75in",
    dpi=150,                  # 래스터화 해상도
    image_quality=85,         # JPEG 품질 (1-100)
    font_embedding=True,      # 폰트 임베딩
    optimize=True,            # 파일 크기 최적화
    linearize=False,          # 웹 뷰잉 최적화
)

exporter = PdfExporter(options=options)
result = exporter.export(html_content, report_context)
pdf_bytes = result.content
```

### OptimizedPdfExporter

대용량 리포트를 위한 최적화된 익스포터입니다.

```python
from truthound.datadocs.exporters.pdf import OptimizedPdfExporter, PdfOptions

exporter = OptimizedPdfExporter(
    chunk_size=1000,       # 청크당 아이템 수
    parallel=True,         # 병렬 처리 활성화
    max_workers=None,      # 워커 스레드 수 (None=자동)
    options=PdfOptions(
        page_size="A4",
        optimize=True,
    ),
)

result = exporter.export(html_content, report_context)
```

**특징:**
- 청크 렌더링: 대용량 데이터셋을 분할 처리
- 병렬 처리: 청크별 병렬 PDF 생성
- 메모리 효율: 스트리밍 방식 처리
- PDF 병합: pypdf/PyPDF2를 사용한 청크 병합

## SVG 차트 렌더링

PDF 내보내기 시 차트는 자동으로 SVG로 렌더링됩니다.

```python
from truthound.datadocs import HTMLReportBuilder

# PDF용 빌더 (내부적으로 _use_svg=True)
builder = HTMLReportBuilder(theme="professional", _use_svg=True)
html = builder.build(profile_dict)

# export_to_pdf는 자동으로 SVG 사용
from truthound.datadocs import export_to_pdf
export_to_pdf(profile_dict, "report.pdf")  # SVG 차트 사용
```

**SVG 지원 차트:**
- Bar, Horizontal Bar, Line
- Pie, Donut

**미지원 차트 (Bar로 대체):**
- Heatmap, Scatter, Box, Gauge, Radar

## Print CSS

PDF 출력에 최적화된 CSS가 자동으로 적용됩니다.

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

## 에러 처리

### WeasyPrintDependencyError

시스템 라이브러리가 설치되지 않은 경우 발생합니다.

```python
from truthound.datadocs import export_to_pdf
from truthound.datadocs.builder import WeasyPrintDependencyError

try:
    export_to_pdf(profile_dict, "report.pdf")
except WeasyPrintDependencyError as e:
    print("PDF export requires system dependencies.")
    print(e)  # 설치 가이드 출력
```

**일반적인 에러:**

```
cannot load library 'libpango-1.0-0'
```

→ 시스템 라이브러리가 설치되지 않음. 위의 설치 가이드 참조.

```
ModuleNotFoundError: No module named 'weasyprint'
```

→ Python 패키지 미설치. `pip install truthound[pdf]` 실행.

## API Reference

### PdfOptions

```python
@dataclass
class PdfOptions(ExportOptions):
    dpi: int = 150                    # 래스터화 해상도
    image_quality: int = 85           # JPEG 품질 (1-100)
    font_embedding: bool = True       # 폰트 임베딩
    optimize: bool = True             # 파일 크기 최적화
    linearize: bool = False           # 웹 뷰잉용 선형화
    chunk_size: int = 1000            # 청크 크기
    parallel: bool = True             # 병렬 처리
```

### ExportOptions (Base)

```python
@dataclass
class ExportOptions:
    page_size: str = "A4"             # 페이지 크기
    orientation: str = "portrait"     # portrait/landscape
    margin_top: str = "1in"
    margin_right: str = "0.75in"
    margin_bottom: str = "1in"
    margin_left: str = "0.75in"
    compress: bool = True             # 압축 활성화
    include_metadata: bool = True     # 메타데이터 포함
    minify: bool = False              # HTML 최소화
```

### ExportResult

```python
@dataclass
class ExportResult:
    content: bytes | str              # 내보낸 콘텐츠
    format: str                       # 포맷 (pdf, html 등)
    size_bytes: int                   # 바이트 크기
    metadata: dict[str, Any]          # 메타데이터
    success: bool = True              # 성공 여부
    error: str | None = None          # 에러 메시지
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
    프로파일을 PDF로 내보냅니다.

    Args:
        profile: TableProfile dict 또는 객체
        output_path: 출력 PDF 파일 경로
        title: 리포트 제목
        subtitle: 부제목
        theme: 테마

    Returns:
        PDF 파일 경로

    Raises:
        WeasyPrintDependencyError: 의존성 미설치 시
    """
```

## See Also

- [HTML Reports](html-reports.md) - HTML 리포트 생성
- [Charts](charts.md) - 차트 렌더링
- [Themes](themes.md) - 테마 커스터마이징
