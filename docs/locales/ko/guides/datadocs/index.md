# Data Docs Guide

실무 운영 가이드에서 Truthound, HTML, PDF, API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
from truthound.datadocs import generate_html_report

# Generate report from profile
html = generate_html_report(
    profile=profile_dict,
    title="Data Quality Report",
    theme="professional",
)

# Save to file
with open("report.html", "w") as f:
    f.write(html)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common 워크플로우s

### 워크플로우 1: 검증 Run + 프로파일 Docs

```python
import truthound as th
from truthound.datadocs import generate_html_report, generate_validation_report
from pathlib import Path

# Step 1: Validate data
run = th.check("data.csv")

# Step 2: Profile data for profile-focused docs
profile = th.profile("data.csv")

# Step 3a: Generate profile docs
profile_html = generate_html_report(
    profile=profile.to_dict(),
    title="Daily Data Quality Report",
    theme="professional",
)

# Step 3b: Generate validation docs from the canonical ValidationRunResult
validation_html = generate_validation_report(
    run,
    title="Daily Validation Run",
    theme="professional",
)

# Step 4: Save both artifacts
Path("profile-report.html").write_text(profile_html)
Path("validation-report.html").write_text(validation_html)
```

### 워크플로우 2: Custom Themed 리포트

```python
from truthound.datadocs import generate_html_report
from truthound.datadocs.themes import ThemeConfig, ThemeColors

# Create custom theme
custom_theme = ThemeConfig(
    name="corporate",
    colors=ThemeColors(
        primary="#0066CC",
        secondary="#004499",
        success="#28A745",
        warning="#FFC107",
        error="#DC3545",
        background="#FFFFFF",
        text="#333333",
    ),
    logo_url="https://company.com/logo.png",
    font_family="'Segoe UI', Tahoma, sans-serif",
)

# Generate with custom theme
html = generate_html_report(
    profile=profile_dict,
    theme=custom_theme,
    title="Q4 Data Quality Report",
    subtitle="Finance Department",
)
```

### 워크플로우 3: PDF Export for Distribution

```python
from truthound.datadocs import generate_html_report
from truthound.datadocs.exporters import PDFExporter

# Generate HTML report
html = generate_html_report(
    profile=profile_dict,
    theme="professional",
    chart_library="svg",  # Use SVG for PDF compatibility
)

# Export to PDF
exporter = PDFExporter()
exporter.export(html, output_path="report.pdf")
```

### 워크플로우 4: Multilingual 리포트

```python
from truthound.datadocs import generate_html_report, HTMLReportBuilder, ReportConfig

# Use ReportConfig to set language
config = ReportConfig(
    theme="light",
    language="ko",  # Korean
)

builder = HTMLReportBuilder(config=config)
html = builder.build(
    profile=profile_dict,
    title="데이터 품질 보고서",
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Full Documentation

실무 운영 가이드에서 Data Docs, Truthound, HTML, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Sub-documents

| 실무 운영 가이드에서 Document을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| [HTML 리포트](html-reports.md) | Static HTML 리포트 generation |
| 실무 운영 가이드에서 Themes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Charts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ApexCharts, SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Sections을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 9 section type 설정 |
| [버전 관리](versioning.md) | 리포트 version management (4 strategies) |
| 실무 운영 가이드에서 Custom, Renderers을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 PDF, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PDF, WeasyPrint을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Data Docs, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Functionality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dependencies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|---------------|--------------|
| **HTML 리포트** | Self-contained HTML 리포트 generation | 실무 운영 가이드에서 None, CDN-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| **PDF 리포트** | Printable distribution 아티팩트 | 실무 운영 가이드에서 `truthound[pdf]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| **Markdown 리포트** | Review-friendly text 아티팩트 | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 주요 기능

- 실무 운영 가이드에서 Zero, Dependencies, CDN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 HTML, Self-Contained, Single을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Built-in, Themes, Default, Light, Dark, Professional, Minimal, Modern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 HTML, PDF, Automatic, Chart, Rendering, ApexCharts, SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Responsive, Design, Mobile/tablet/desktop을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Print, Optimization, Print-friendly, CSS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 PDF, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Multilingual, Support을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Report, Versioning, Incremental, Semantic, Timestamp, GitLike을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설치

```bash
# Basic installation (HTML report support - requires Jinja2)
pip install truthound[reports]

# PDF export support
pip install truthound[pdf]

# Full installation
pip install truthound[all]
```

> 실무 운영 가이드에서 HTML, `truthound[reports]`, `truthound[all]`, Note, Jinja2, Install을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 PDF, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

### 1. Generate 프로파일

```bash
truthound auto-profile data.csv -o profile.json
```

### 2. Generate HTML 리포트

```bash
truthound docs generate profile.json -o report.html
```

### 3. Python API

```python
from truthound.datadocs import generate_html_report

html = generate_html_report(
    profile=profile_dict,
    title="Data Quality Report",
    theme="professional",
    output_path="report.html",
)
```

실무 운영 가이드에서 HTML, Reports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CLI Commands

### `truthound docs generate`

```bash
truthound docs generate <profile_file> [OPTIONS]

# Options:
#   -o, --output TEXT    Output file path
#   -t, --title TEXT     Report title
#   -s, --subtitle TEXT  Subtitle
#   --theme TEXT         Theme (light, dark, professional, minimal, modern)
#   -f, --format TEXT    Output format (html, pdf)
```

### `truthound docs themes`

```bash
truthound docs themes  # List available themes
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 아키텍처

```
src/truthound/datadocs/
├── __init__.py          # Module exports
├── base.py              # Types, Enums, Protocols
├── builder.py           # HTMLReportBuilder, ProfileDataConverter
├── charts.py            # ApexChartsRenderer, SVGChartRenderer
├── sections.py          # 9 section renderers
├── styles.py            # CSS stylesheets
│
├── engine/              # Pipeline engine
│   ├── context.py       # ReportContext, ReportData
│   ├── pipeline.py      # ReportPipeline
│   └── registry.py      # ComponentRegistry
│
├── themes/              # Theme system
│   ├── base.py          # ThemeConfig, ThemeColors
│   ├── default.py       # 6 built-in themes
│   ├── enterprise.py    # EnterpriseTheme
│   └── loader.py        # YAML/JSON loader
│
├── renderers/           # Custom renderers
│   ├── base.py          # BaseRenderer
│   ├── jinja.py         # JinjaRenderer
│   └── custom.py        # String/File/Callable renderers
│
├── exporters/           # Output formats
│   ├── base.py          # BaseExporter
│   ├── html_reporter.py # HtmlExporter
│   ├── pdf.py           # PdfExporter, OptimizedPdfExporter
│   └── markdown.py      # MarkdownExporter
│
├── versioning/          # Version management
│   ├── version.py       # 4 version strategies
│   ├── storage.py       # InMemory, File storage
│   └── diff.py          # ReportDiffer
│
├── i18n/                # Multilingual support
│   ├── catalog.py       # 15 languages
│   ├── plurals.py       # CLDR plural rules
│   └── formatting.py    # Number/date formatting
```

### Class Hierarchy

```
BaseChartRenderer (ABC)
├── ApexChartsRenderer  # HTML reports (default)
└── SVGChartRenderer    # PDF export

BaseSectionRenderer (ABC)
├── OverviewSection
├── ColumnsSection
├── QualitySection
├── PatternsSection
├── DistributionSection
├── CorrelationsSection
├── RecommendationsSection
├── AlertsSection
└── CustomSection
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CI/CD 통합

### GitHub Actions

```yaml
name: Data Quality Report

on:
  schedule:
    - cron: '0 6 * * *'

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install truthound
      - run: truthound auto-profile data.csv -o profile.json
      - run: truthound docs generate profile.json -o report.html
      - uses: actions/upload-artifact@v4
        with:
          name: data-quality-report
          path: report.html
```

### GitLab CI

```yaml
data-quality-report:
  stage: report
  image: python:3.11
  script:
    - pip install truthound
    - truthound auto-profile data.csv -o profile.json
    - truthound docs generate profile.json -o report.html
  artifacts:
    paths:
      - report.html
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 문제 해결

### Charts not rendering

실무 운영 가이드에서 PDF, JavaScript, CDN을(를) 다루는 항목입니다:

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

### PDF export fails

실무 운영 가이드에서 PDF, System, Refer, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- 실무 운영 가이드에서 HTML, Reports, Static을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Themes, Theme을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Charts, Chart을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 PDF, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
