# Data Docs - HTML Report Generation (Phase 8)

The Truthound Data Docs module transforms data profile results into visually appealing and interactive HTML reports.

## Sub-documents

| Document | Description |
|----------|-------------|
| [HTML Reports](html-reports.md) | Static HTML report generation |
| [Themes](themes.md) | 6 built-in themes + customization |
| [Charts](charts.md) | ApexCharts and SVG chart rendering |
| [Sections](sections.md) | 9 section type configuration |
| [Versioning](versioning.md) | Report version management (4 strategies) |
| [Custom Renderers](custom-renderers.md) | Custom renderer development |
| [PDF Export](pdf-export.md) | PDF export (WeasyPrint) |
| [Dashboard](dashboard.md) | Stage 2 interactive dashboard |

---

## Overview

Data Docs consists of two stages:

| Stage | Functionality | Dependencies |
|-------|---------------|--------------|
| **Stage 1: Static HTML Report** | Self-contained HTML report generation | None (CDN-based) |
| **Stage 2: Interactive Dashboard** | Reflex-based interactive dashboard | `truthound[dashboard]` |

### Key Features

- **Zero Dependencies**: No npm/node build required, JS loaded from CDN
- **Self-Contained**: Single HTML file works offline
- **6 Built-in Themes**: Default, Light, Dark, Professional, Minimal, Modern + Enterprise customization
- **Automatic Chart Rendering**: ApexCharts for HTML, SVG for PDF (auto-selected)
- **Responsive Design**: Mobile/tablet/desktop support
- **Print Optimization**: Print-friendly CSS included
- **PDF Export**: Uses weasyprint (optional)
- **Multilingual Support**: 15 languages (en, ko, ja, zh, de, fr, es, pt, it, ru, ar, th, vi, id, tr)
- **Report Versioning**: 4 strategies (Incremental, Semantic, Timestamp, GitLike)

---

## Installation

```bash
# Basic installation (Stage 1: HTML Report - requires Jinja2)
pip install truthound[reports]

# PDF export support
pip install truthound[pdf]

# Dashboard support (Stage 2)
pip install truthound[dashboard]

# Full installation
pip install truthound[all]
```

> **Note**: HTML report generation requires Jinja2. Install `truthound[reports]` or `truthound[all]`.

For PDF export system dependencies, refer to the [PDF Export](pdf-export.md) documentation.

---

## Quick Start

### 1. Generate Profile

```bash
truthound auto-profile data.csv -o profile.json
```

### 2. Generate HTML Report

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

For detailed information, refer to the [HTML Reports](html-reports.md) documentation.

---

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

### `truthound dashboard`

```bash
truthound dashboard --profile profile.json --port 8080
```

---

## Architecture

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
│
└── dashboard/           # Stage 2: Dashboard
    ├── app.py           # DashboardApp
    ├── state.py         # Reflex state
    └── components.py    # UI components
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

---

## CI/CD Integration

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

---

## Troubleshooting

### Charts not rendering

In environments where JavaScript cannot be loaded from CDN, use the PDF format:

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

### PDF export fails

System libraries are required. Refer to the [PDF Export](pdf-export.md) documentation.

### Dashboard import error

```bash
pip install truthound[dashboard]
```

---

## See Also

- [HTML Reports](html-reports.md) - Static HTML report generation
- [Themes](themes.md) - Theme customization
- [Charts](charts.md) - Chart rendering
- [PDF Export](pdf-export.md) - PDF export
- [Dashboard](dashboard.md) - Interactive dashboard
