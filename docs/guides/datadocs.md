# Data Docs - HTML Report Generation (Phase 8)

Truthound's Data Docs module transforms data profile results into beautiful and interactive HTML reports. It generates self-contained HTML files that can be stored as artifacts in CI/CD pipelines or shared via email/Slack.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Themes](#themes)
7. [Chart Libraries](#chart-libraries)
8. [Report Sections](#report-sections)
9. [Customization](#customization)
10. [Dashboard (Stage 2)](#dashboard-stage-2)
11. [CI/CD Integration](#cicd-integration)
12. [Architecture](#architecture)

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
- **6 Built-in Themes**: Default, Light, Dark, Professional, Minimal, Modern (+ Enterprise)
- **Automatic Chart Rendering**: ApexCharts for HTML, SVG for PDF (auto-selected)
- **Responsive Design**: Mobile/tablet/desktop compatible
- **Print Optimized**: Print-friendly CSS included
- **PDF Export**: Uses WeasyPrint (optional)
- **Multilingual Support**: English (en), Korean (ko)
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

!!! warning "HTML Report Dependency"
    HTML report generation requires Jinja2. Make sure to install `truthound[reports]` or `truthound[all]`.

### PDF Export System Dependencies

PDF export uses WeasyPrint and requires **system libraries**.
Python packages alone are not sufficient, so you must install the system dependencies listed below first.

#### macOS (Homebrew)

```bash
brew install pango cairo gdk-pixbuf libffi
pip install truthound[pdf]
```

#### Ubuntu/Debian

```bash
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
pip install truthound[pdf]
```

#### Fedora/RHEL

```bash
sudo dnf install pango gdk-pixbuf2 libffi-devel
pip install truthound[pdf]
```

#### Alpine Linux

```bash
apk add pango gdk-pixbuf libffi-dev
pip install truthound[pdf]
```

#### Windows

Windows requires the GTK3 runtime:

1. Download [GTK3 for Windows](https://github.com/nickvidal/weasyprint/releases)
2. Extract and add to PATH
3. `pip install truthound[pdf]`

Or use the bundled version:
```bash
pip install weasyprint[gtk3]
```

#### Docker

When using Docker:

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

> **Note**: `pip install truthound[pdf]` only installs the Python package (weasyprint).
> Without the system libraries listed above, you will get the `cannot load library 'libpango-1.0-0'` error during PDF generation.

For detailed information, refer to the [WeasyPrint Installation Guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation).

---

## Quick Start

### 1. Generate Profile

First, generate a data profile:

```bash
truthound auto-profile data.csv -o profile.json
```

### 2. Generate HTML Report

```bash
truthound docs generate profile.json -o report.html
```

### 3. Open in Browser

Open the generated `report.html` file in a browser to view the complete data quality report.

---

## CLI Commands

### `truthound docs generate`

Generates an HTML report from a profile JSON file.

```bash
truthound docs generate <profile_file> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `<input>.html` | Output file path |
| `--title` | `-t` | "Data Profile Report" | Report title |
| `--subtitle` | `-s` | "" | Report subtitle |
| `--theme` | | "professional" | Theme (light, dark, professional, minimal, modern) |
| `--format` | `-f` | "html" | Output format (html, pdf) |

**Examples:**

```bash
# Basic usage
truthound docs generate profile.json -o report.html

# Custom title and dark theme
truthound docs generate profile.json -o report.html \
    --title "Q4 Data Quality Report" \
    --subtitle "Customer Dataset" \
    --theme dark

# PDF export (requires weasyprint)
truthound docs generate profile.json -o report.pdf --format pdf
```

### `truthound docs themes`

Displays the list of available themes.

```bash
truthound docs themes
```

**Output:**

```
Available report themes:

  light          - Clean and bright, suitable for most use cases
  dark           - Dark mode with vibrant colors, easy on the eyes
  professional   - Corporate style, subdued colors (default)
  minimal        - Minimalist design with monochrome accents
  modern         - Contemporary design with vibrant gradients
```

### `truthound dashboard`

Runs the interactive dashboard (Stage 2).

```bash
truthound dashboard [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--profile` | `-p` | None | Profile JSON file path |
| `--port` | | 8080 | Server port |
| `--host` | | "localhost" | Server host |
| `--title` | `-t` | "Truthound Dashboard" | Dashboard title |
| `--debug` | | False | Enable debug mode |

**Examples:**

```bash
# Run with profile
truthound dashboard --profile profile.json

# Custom port
truthound dashboard --profile profile.json --port 3000

# Allow external access
truthound dashboard --profile profile.json --host 0.0.0.0
```

---

## Python API

### HTMLReportBuilder

Use `HTMLReportBuilder` directly when fine-grained control is needed.

```python
from truthound.datadocs import (
    HTMLReportBuilder,
    ReportTheme,
    ChartLibrary,
    ReportConfig,
)

# Basic usage
builder = HTMLReportBuilder(theme=ReportTheme.PROFESSIONAL)
html = builder.build(profile, title="My Data Report")
builder.save(html, "report.html")

# Custom configuration
config = ReportConfig(
    theme=ReportTheme.DARK,
    include_toc=True,
    include_timestamp=True,
    footer_text="Generated by Data Team",
)
builder = HTMLReportBuilder(config=config)
html = builder.build(profile)
```

### Convenience Functions

Functions for simple usage:

```python
from truthound.datadocs import (
    generate_html_report,
    generate_report_from_file,
    export_report,
    export_to_pdf,
)

# Generate directly from profile dict
html = generate_html_report(
    profile=profile_dict,
    title="Data Quality Report",
    theme="professional",
    output_path="report.html",
)

# Generate from file
html = generate_report_from_file(
    profile_path="profile.json",
    output_path="report.html",
    title="My Report",
    theme="dark",
)

# Export in various formats
export_report(profile, "report.html", format="html")
export_report(profile, "report.pdf", format="pdf")

# Direct PDF export
export_to_pdf(profile, "report.pdf", title="PDF Report")
```

### Complete Workflow Example

```python
import truthound as th
from truthound.datadocs import generate_html_report

# 1. Load data
df = th.load("data.csv")

# 2. Generate profile
from truthound.profiler import DataProfiler
profiler = DataProfiler()
profile = profiler.profile(df)

# 3. Generate HTML report
html = generate_html_report(
    profile=profile.to_dict(),
    title="Customer Data Quality Report",
    subtitle="Q4 2025 Analysis",
    theme="professional",
    output_path="customer_report.html",
)

print(f"Report generated: {len(html):,} bytes")
```

---

## Themes

### Available Themes

| Theme | Description | Best For |
|-------|-------------|----------|
| `light` | Bright and clean design | General use, printing |
| `dark` | Dark mode, vibrant colors | Night work, presentations |
| `professional` | Corporate style, calm colors | Business reports (default) |
| `minimal` | Minimalist, monotone | Simple documentation |
| `modern` | Contemporary, gradients | Marketing, demos |

### Theme Preview

#### Professional Theme (Default)
- Background: Light Gray (#fafbfc)
- Primary: Blue (#2563eb)
- Surface: White (#ffffff)
- Calm and professional feel

#### Dark Theme
- Background: Dark (#0f172a)
- Primary: Blue (#60a5fa)
- Surface: Dark Gray (#1e293b)
- Dark mode that reduces eye strain

### Custom Theme

You can create custom themes:

```python
from truthound.datadocs import (
    HTMLReportBuilder,
    ReportConfig,
)
from truthound.datadocs.themes import (
    ThemeConfig,
    ThemeColors,
    ThemeTypography,
    ThemeSpacing,
    ThemeAssets,
)

# Define custom colors
custom_colors = ThemeColors(
    background="#fafafa",
    surface="#ffffff",
    text_primary="#333333",
    primary="#ff6b6b",
    secondary="#4ecdc4",
    accent="#ffe66d",
)

# Create custom theme (ThemeConfig from themes module)
custom_theme = ThemeConfig(
    name="my_brand",
    display_name="My Brand Theme",
    description="Custom theme for my brand",
    colors=custom_colors,
    # Additional options (optional)
    footer_text="Generated by My Company",
    show_toc=True,
)

# Apply to builder
config = ReportConfig(custom_theme=custom_theme)
builder = HTMLReportBuilder(config=config)
```

---

## Chart Rendering

Truthound automatically selects the optimal chart renderer based on output format:

| Output Format | Chart Renderer | Description |
|---------------|----------------|-------------|
| **HTML** | ApexCharts | Modern, interactive, tooltips/animations supported |
| **PDF** | SVG | No JavaScript required, optimized for PDF rendering |

### Theme-Aware Chart Colors

Charts automatically adjust colors to match the selected theme. In dark mode, all chart text elements are displayed in bright colors:

| Element | Light Theme | Dark Theme |
|---------|-------------|------------|
| Axis Labels | Dark gray | Light gray |
| Legend Text | Dark gray | White |
| Tooltip Text | Black | White |
| Data Labels | Auto contrast | Auto contrast |
| Chart Title | Dark | Light |

!!! tip "Dark Mode Optimization"
    In dark themes, charts use CSS variables to automatically adjust text colors.
    All text elements of ApexCharts (axis labels, legends, tooltips, data labels) are displayed according to the theme.

```bash
# Generate dark mode report
truthound docs generate profile.json -o report.html --theme dark
```

### Supported Chart Types

| Chart Type | HTML (ApexCharts) | PDF (SVG) |
|------------|-------------------|-----------|
| Bar | ✅ | ✅ |
| Horizontal Bar | ✅ | ✅ |
| Line | ✅ | ✅ |
| Pie | ✅ | ✅ |
| Donut | ✅ | ✅ |
| Histogram | ✅ | ✅ (bar fallback) |
| Heatmap | ✅ | ❌ |
| Scatter | ✅ | ❌ |
| Box Plot | ✅ | ❌ |
| Gauge | ✅ | ❌ |
| Radar | ✅ | ❌ |

> **Note**: Chart types not supported in PDF output are replaced with Bar charts.

### CDN URLs

ApexCharts is loaded from CDN:

```python
from truthound.datadocs import CDN_URLS, ChartLibrary

# ApexCharts (HTML reports)
CDN_URLS[ChartLibrary.APEXCHARTS]
# ['https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js']

# SVG (no dependencies - for PDF)
CDN_URLS[ChartLibrary.SVG]
# []
```

---

## Report Sections

Generated reports consist of 8 sections:

### 1. Overview

Displays key metrics of the dataset in card format:

- **Row Count**: Total number of rows
- **Column Count**: Total number of columns
- **Memory**: Estimated memory usage
- **Duplicates**: Number of duplicate rows
- **Missing**: Total null cells
- **Quality Score**: Overall quality score (0-100)

Data type distribution chart is also included.

### 2. Data Quality

Displays quality dimension scores as circular gauges:

- **Completeness**: Data completeness (null ratio)
- **Uniqueness**: Uniqueness (unique ratio)
- **Validity**: Validity (format match rate)
- **Consistency**: Consistency

Missing value distribution chart and warning list are also included.

### 3. Column Details

Detailed information for each column:

- **Summary Table**: Summary table of all columns
- **Column Cards**: Detailed cards per column
  - Data type badge
  - Null/Unique ratio
  - Descriptive statistics (numeric)
  - Detected patterns
  - Value distribution chart

### 4. Detected Patterns

Automatically detected data patterns:

- **Pattern Name**: Pattern type (Email, Phone, UUID, etc.)
- **Match Ratio**: Match rate
- **Sample Matches**: Sample values

### 5. Value Distribution

Value distribution analysis:

- Uniqueness distribution chart
- Top value frequency
- Histograms

### 6. Correlations

Correlations between columns:

- Correlation coefficient list
- Strong/medium/weak correlation visualization
- Positive/negative correlation distinction

### 7. Recommendations

Auto-generated improvement suggestions:

- Recommended Validator list
- Data quality improvement suggestions
- Pipeline recommendations

### 8. Alerts

Data quality issue warnings:

| Severity | Color | Example |
|----------|-------|---------|
| Info | Blue | Constant column found |
| Warning | Yellow | Over 50% missing |
| Error | Red | Over 80% missing |
| Critical | Dark Red | Data integrity violation |

---

## Customization

### Report Configuration

Customize reports with `ReportConfig`:

```python
from truthound.datadocs import ReportConfig, SectionType

config = ReportConfig(
    # Theme
    theme=ReportTheme.PROFESSIONAL,

    # Sections to include (in order)
    sections=[
        SectionType.OVERVIEW,
        SectionType.QUALITY,
        SectionType.COLUMNS,
        SectionType.PATTERNS,
        SectionType.DISTRIBUTION,
        SectionType.CORRELATIONS,
        SectionType.RECOMMENDATIONS,
        SectionType.ALERTS,
    ],

    # Layout options
    include_toc=True,           # Include table of contents
    include_header=True,        # Include header
    include_footer=True,        # Include footer
    include_timestamp=True,     # Show generation time

    # Custom content
    custom_css="",              # Additional CSS
    custom_js="",               # Additional JavaScript
    logo_url=None,              # Logo URL
    logo_base64=None,           # Logo Base64
    footer_text="Generated by Truthound",

    # Localization
    language="en",
    date_format="%Y-%m-%d %H:%M:%S",
)
```

### Custom CSS

You can inject additional CSS:

```python
config = ReportConfig(
    custom_css="""
    .report-title {
        color: #ff6b6b;
    }
    .metric-card {
        border: 2px solid #4ecdc4;
    }
    """,
)
```

### Custom JavaScript

You can inject additional JavaScript:

```python
config = ReportConfig(
    custom_js="""
    document.addEventListener('DOMContentLoaded', function() {
        console.log('Report loaded!');
    });
    """,
)
```

### Logo

You can add a company logo:

```python
# Add logo via URL
config = ReportConfig(
    logo_url="https://example.com/logo.png",
)

# Add logo via Base64 (offline support)
import base64
with open("logo.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

config = ReportConfig(
    logo_base64=f"data:image/png;base64,{logo_b64}",
)
```

---

## Dashboard (Stage 2)

Stage 2 provides a Reflex-based interactive dashboard.

### Installation

```bash
pip install truthound[dashboard]
```

### Features

- **Real-time Data Exploration**: Filtering, sorting, search
- **Column Drilldown**: Detailed analysis
- **Live Profiling**: Real-time data analysis
- **Interactive Charts**: Zoom, pan, hover

### Usage

```python
from truthound.datadocs import launch_dashboard

# Run with profile
launch_dashboard(
    profile_path="profile.json",
    port=8080,
    host="localhost",
    title="My Dashboard",
)
```

### CLI

```bash
truthound dashboard --profile profile.json --port 8080
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Data Quality Report

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Truthound
        run: pip install truthound

      - name: Generate Profile
        run: truthound auto-profile data.csv -o profile.json

      - name: Generate Report
        run: |
          truthound docs generate profile.json \
            -o report.html \
            --title "Daily Data Quality Report" \
            --theme professional

      - name: Upload Report
        uses: actions/upload-artifact@v4
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
    expire_in: 30 days
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    stages {
        stage('Generate Report') {
            steps {
                sh 'pip install truthound'
                sh 'truthound auto-profile data.csv -o profile.json'
                sh 'truthound docs generate profile.json -o report.html'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'report.html'
            publishHTML([
                reportName: 'Data Quality Report',
                reportDir: '.',
                reportFiles: 'report.html'
            ])
        }
    }
}
```

### Slack Notification

Send notification to Slack after report generation:

```bash
# Generate report
truthound docs generate profile.json -o report.html

# Send to Slack (using curl)
curl -F file=@report.html \
     -F channels=data-quality \
     -F title="Daily Data Quality Report" \
     -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
     https://slack.com/api/files.upload
```

---

## Architecture

### Module Structure

```
src/truthound/datadocs/
├── __init__.py          # Module exports & lazy imports
├── base.py              # Base types, Enums, Protocols, Registry
├── charts.py            # 2 chart renderers (ApexCharts, SVG)
├── sections.py          # 8 section renderers
├── styles.py            # CSS stylesheets
├── builder.py           # HTMLReportBuilder
│
├── engine/              # Pipeline engine
│   ├── context.py       # ReportContext, ReportData
│   ├── pipeline.py      # ReportPipeline, PipelineBuilder
│   └── registry.py      # ComponentRegistry
│
├── themes/              # Theme system
│   ├── base.py          # ThemeConfig, ThemeColors, ThemeAssets
│   ├── default.py       # 6 built-in themes (Default, Light, Dark, Minimal, Modern, Professional)
│   ├── enterprise.py    # EnterpriseTheme (white-labeling)
│   └── loader.py        # YAML/JSON loader
│
├── renderers/           # Template rendering
│   ├── jinja.py         # JinjaRenderer
│   └── custom.py        # StringTemplate, FileTemplate, Callable
│
├── exporters/           # Output formats
│   ├── html.py          # HtmlExporter
│   ├── pdf.py           # OptimizedPdfExporter
│   ├── markdown.py      # MarkdownExporter
│   └── json_exporter.py # JsonExporter
│
├── versioning/          # Version management
│   ├── version.py       # 4 version strategies
│   ├── storage.py       # InMemory, File storage
│   └── diff.py          # ReportDiffer
│
├── i18n/                # Multilingual support
│   ├── catalog.py       # 2 languages (en, ko)
│   ├── plurals.py       # CLDR plural rules
│   └── formatting.py    # Number/date formatting
│
└── dashboard/           # Stage 2: Dashboard
    ├── state.py         # Reflex state management
    ├── components.py    # UI components
    └── app.py           # Reflex app
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

### Registry Pattern

Renderers are automatically registered via decorators:

```python
from truthound.datadocs.base import (
    register_chart_renderer,
    register_section_renderer,
)

@register_chart_renderer(ChartLibrary.APEXCHARTS)
class ApexChartsRenderer(BaseChartRenderer):
    ...

@register_section_renderer(SectionType.OVERVIEW)
class OverviewSection(BaseSectionRenderer):
    ...
```

### Extensibility

Adding a custom chart renderer:

```python
from truthound.datadocs import (
    BaseChartRenderer,
    ChartLibrary,
    ChartSpec,
    register_chart_renderer,
)

# Register new chart library
@register_chart_renderer(ChartLibrary.CUSTOM)
class CustomChartRenderer(BaseChartRenderer):
    library = ChartLibrary.CUSTOM

    def render(self, spec: ChartSpec) -> str:
        # Custom rendering logic
        return "<div>Custom Chart</div>"

    def get_dependencies(self) -> list[str]:
        return ["https://example.com/chart-lib.js"]
```

---

## Troubleshooting

### Common Issues

#### 1. Wrong input file type

The `docs generate` command takes a **profile JSON file** as input. Passing data files (CSV, Parquet, etc.) directly will result in an error:

```
Error: 'data.csv' appears to be a data file, not a profile JSON.

This command requires a profile JSON file from 'auto-profile'.

To generate a report from your data:
  1. First, create a profile:
     truthound auto-profile data.csv -o profile.json

  2. Then, generate the report:
     truthound docs generate profile.json -o report.html
```

**Solution:**

```bash
# 1. First, generate a profile
truthound auto-profile data.csv -o profile.json

# 2. Then, generate the report
truthound docs generate profile.json -o report.html
```

!!! tip "Remember the Workflow"
    Remember the order: Data → Profile → Report!

    ```mermaid
    graph LR
        A[data.csv] --> B[auto-profile]
        B --> C[profile.json]
        C --> D[docs generate]
        D --> E[report.html]
    ```

#### 2. Charts not rendering

If JavaScript cannot be loaded from CDN:

```bash
# Export to PDF for SVG charts (no JavaScript required)
truthound docs generate profile.json -o report.pdf --format pdf
```

> **Note**: HTML reports use ApexCharts. SVG charts are automatically used for PDF export.

#### 3. PDF export fails

**Error: `cannot load library 'libpango-1.0-0'`**

This error occurs when system libraries are not installed.
`pip install truthound[pdf]` only installs the Python package; system libraries must be installed separately.

```bash
# macOS
brew install pango cairo gdk-pixbuf libffi

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# Fedora/RHEL
sudo dnf install pango gdk-pixbuf2 libffi-devel

# Alpine
apk add pango gdk-pixbuf libffi-dev
```

After installing system libraries, try again:

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

**Error: `ModuleNotFoundError: No module named 'weasyprint'`**

If the Python package is not installed:

```bash
pip install truthound[pdf]
```

> **Tip**: If PDF is not urgent, you can use HTML format first:
> ```bash
> truthound docs generate profile.json -o report.html
> ```

#### 4. Dashboard import error

If dashboard dependencies are not installed:

```bash
pip install truthound[dashboard]
```

#### 5. Large profile file

If the profile file is too large, use sampling:

```bash
truthound auto-profile data.csv -o profile.json --sample-size 100000
```

---

## API Reference

### Enums

```python
class ReportTheme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    MODERN = "modern"

class ChartLibrary(str, Enum):
    APEXCHARTS = "apexcharts"  # Default for HTML
    SVG = "svg"                # Used for PDF export

class ChartType(str, Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    BOX = "box"
    GAUGE = "gauge"
    RADAR = "radar"
    TABLE = "table"

class SectionType(str, Enum):
    OVERVIEW = "overview"
    COLUMNS = "columns"
    QUALITY = "quality"
    PATTERNS = "patterns"
    DISTRIBUTION = "distribution"
    CORRELATIONS = "correlations"
    RECOMMENDATIONS = "recommendations"
    ALERTS = "alerts"
    CUSTOM = "custom"

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### Data Classes

```python
@dataclass
class ReportConfig:
    theme: ReportTheme = ReportTheme.PROFESSIONAL
    custom_theme: ThemeConfig | None = None
    sections: list[SectionType] = ...
    include_toc: bool = True
    include_header: bool = True
    include_footer: bool = True
    include_timestamp: bool = True
    custom_css: str = ""
    custom_js: str = ""
    logo_url: str | None = None
    logo_base64: str | None = None
    footer_text: str = "Generated by Truthound"
    language: str = "en"
    date_format: str = "%Y-%m-%d %H:%M:%S"

@dataclass
class ChartSpec:
    chart_type: ChartType
    title: str = ""
    subtitle: str = ""
    labels: list[str] = field(default_factory=list)
    values: list[float | int] = field(default_factory=list)
    series: list[dict] | None = None
    colors: list[str] | None = None
    height: int = 300
    width: int | None = None
    show_legend: bool = True
    show_labels: bool = True
    show_grid: bool = True
    animation: bool = True
    options: dict[str, Any] = field(default_factory=dict)  # Additional options

@dataclass
class AlertSpec:
    title: str
    message: str
    severity: SeverityLevel = SeverityLevel.INFO
    column: str | None = None
    suggestion: str | None = None
```

---

## See Also

- [Auto-Profiling (docs/PROFILER.md)](PROFILER.md) - Data profiling
- [Reporters (docs/REPORTERS.md)](REPORTERS.md) - Other report formats
- [Examples (docs/EXAMPLES.md)](EXAMPLES.md) - Usage examples
