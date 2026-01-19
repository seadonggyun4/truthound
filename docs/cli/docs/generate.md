# truthound docs generate

Generate HTML or PDF reports from profile data.

## Synopsis

```bash
truthound docs generate <profile_file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `profile_file` | Yes | Path to the profile file (JSON) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | None | Output file path |
| `--title` | `-t` | `Data Profile Report` | Report title |
| `--subtitle` | `-s` | None | Report subtitle |
| `--theme` | | `professional` | Theme (light, dark, professional, minimal, modern) |
| `--charts` | `-c` | `apexcharts` | Chart engine (apexcharts, chartjs, plotly, svg) |
| `--format` | `-f` | `html` | Output format (html, pdf) |

## Description

The `docs generate` command creates human-readable reports from profile data:

1. **Parses** profile JSON data
2. **Applies** selected theme and styling
3. **Renders** charts with chosen engine
4. **Outputs** standalone HTML or PDF file

### Generated Report Features

- **Self-contained** - Single file with embedded styles and scripts
- **Offline viewing** - No server or internet required
- **CI/CD ready** - Store as build artifact
- **Shareable** - Email, Slack, or host on web

## Examples

### Basic HTML Report

```bash
truthound docs generate profile.json -o report.html
```

Output: `report.html` - Standalone HTML file viewable in any browser.

### Custom Title and Theme

```bash
truthound docs generate profile.json -o report.html --title "Q4 Data Quality Report" --subtitle "Production Database" --theme dark
```

### Different Chart Engine

```bash
# Lightweight Chart.js
truthound docs generate profile.json -o report.html --charts chartjs

# Interactive Plotly
truthound docs generate profile.json -o report.html --charts plotly

# Zero-dependency SVG
truthound docs generate profile.json -o report.html --charts svg
```

### PDF Output

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

!!! note "PDF Dependency"
    PDF export requires weasyprint:
    ```bash
    pip install truthound[pdf]
    ```

### Complete Example

```bash
truthound docs generate profile.json \
  -o quarterly_report.html \
  --title "Q4 2025 Data Quality Report" \
  --subtitle "Customer Analytics Pipeline" \
  --theme professional \
  --charts apexcharts
```

## Themes

| Theme | Description | Best For |
|-------|-------------|----------|
| `light` | Clean, bright style for general use | Day viewing, printing |
| `dark` | Dark mode for reduced eye strain | Night viewing, presentations |
| `professional` | Corporate style with calm colors (default) | Business reports |
| `minimal` | Minimalist design, monochrome accents | Simple documentation |
| `modern` | Contemporary design, vibrant gradients | Marketing, demos |

## Chart Engines

| Engine | Description | Pros | Cons |
|--------|-------------|------|------|
| `apexcharts` | Modern interactive charts | Beautiful, feature-rich | Larger file size |
| `chartjs` | Lightweight charts | Small, fast | Fewer features |
| `plotly` | Advanced interactive | Zooming, tooltips | Largest file size |
| `svg` | Pure SVG rendering | Zero dependencies, works offline | Static only |

### Engine Selection Guide

```bash
# For web viewing (default)
truthound docs generate profile.json -o report.html --charts apexcharts

# For email/simple sharing
truthound docs generate profile.json -o report.html --charts svg

# For data exploration
truthound docs generate profile.json -o report.html --charts plotly

# For lightweight reports
truthound docs generate profile.json -o report.html --charts chartjs
```

## Report Contents

Generated reports include:

### Overview Section
- Dataset summary (rows, columns, size)
- Data quality score
- Generation timestamp

### Column Analysis
- Data types and statistics
- Null/missing value rates
- Unique value counts
- Distribution charts

### Quality Metrics
- Completeness percentages
- Validity checks
- Pattern detection results

### Visualizations
- Distribution histograms
- Correlation heatmaps
- Missing value patterns

## Use Cases

### 1. CI/CD Integration

```yaml
# GitHub Actions
- name: Generate Profile
  run: truthound profile data.csv --format json -o profile.json

- name: Generate Report
  run: truthound docs generate profile.json -o report.html --theme professional

- name: Upload Artifact
  uses: actions/upload-artifact@v4
  with:
    name: data-quality-report
    path: report.html
```

### 2. Scheduled Reports

```bash
#!/bin/bash
# daily_report.sh
DATE=$(date +%Y%m%d)
truthound profile /data/daily_export.csv --format json -o profile.json
truthound docs generate profile.json \
  -o "/reports/daily_${DATE}.html" \
  --title "Daily Data Report - ${DATE}"
```

### 3. Multiple Themes

```bash
# Generate reports in multiple themes
for theme in light dark professional; do
  truthound docs generate profile.json \
    -o "report_${theme}.html" \
    --theme $theme
done
```

### 4. PDF for Distribution

```bash
# Generate PDF for email distribution
truthound docs generate profile.json \
  -o report.pdf \
  --format pdf \
  --theme professional \
  --title "Monthly Data Quality Summary"
```

## Workflow: Profile to Report

```bash
# Step 1: Generate profile
truthound auto-profile data.csv -o profile.json --format json

# Step 2: Generate report
truthound docs generate profile.json -o report.html

# Step 3: Open in browser
open report.html  # macOS
# or: xdg-open report.html  # Linux
# or: start report.html  # Windows
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Generation error |
| 2 | Invalid arguments or file not found |

## Related Commands

- [`docs themes`](themes.md) - List available themes
- [`profile`](../core/profile.md) - Generate basic profile
- [`auto-profile`](../profiler/auto-profile.md) - Generate detailed profile

## See Also

- [Data Docs Guide](../../guides/datadocs.md)
- [White-labeling](../../concepts/advanced.md)
