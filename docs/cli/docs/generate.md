# truthound docs generate

Generate HTML or PDF reports from profile data.

!!! warning "Dependencies"
    - HTML reports require Jinja2: `pip install truthound[reports]`
    - PDF reports require WeasyPrint: `pip install truthound[pdf]`

## Synopsis

```bash
truthound docs generate <profile_file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `profile_file` | Yes | Path to the profile file (JSON from auto-profile) |

!!! warning "Common Mistake"
    This command requires a **profile JSON file**, not a data file.
    If you pass a data file (CSV, Parquet, etc.), you'll see a helpful error:

    ```
    Error: 'data.csv' appears to be a data file, not a profile JSON.

    This command requires a profile JSON file from 'auto-profile'.

    To generate a report from your data:
      1. First, create a profile:
         truthound auto-profile data.csv -o profile.json

      2. Then, generate the report:
         truthound docs generate profile.json -o report.html
    ```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | None | Output file path |
| `--title` | `-t` | `Data Profile Report` | Report title |
| `--subtitle` | `-s` | `""` | Report subtitle |
| `--theme` | | `professional` | Theme (light, dark, professional, minimal, modern) |
| `--format` | `-f` | `html` | Output format (html, pdf) |

## Description

The `docs generate` command creates human-readable reports from profile data:

1. **Parses** profile JSON data
2. **Applies** selected theme and styling
3. **Renders** charts using ApexCharts (HTML) or SVG (PDF)
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

### PDF Output

```bash
truthound docs generate profile.json -o report.pdf --format pdf
```

!!! warning "PDF System Dependencies"
    PDF export requires WeasyPrint **and** system libraries (Pango, Cairo, etc.).

    `pip install truthound[pdf]` only installs the Python package. You must also install system libraries:

    ```bash
    # macOS
    brew install pango cairo gdk-pixbuf libffi

    # Ubuntu/Debian
    sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev

    # Then install Python package
    pip install truthound[pdf]
    ```

    See [Data Docs Guide](../../guides/datadocs.md#pdf-export-system-dependencies) for full installation instructions.

### Complete Example

```bash
truthound docs generate profile.json \
  -o quarterly_report.html \
  --title "Q4 2025 Data Quality Report" \
  --subtitle "Customer Analytics Pipeline" \
  --theme professional
```

## Themes

| Theme | Description | Best For |
|-------|-------------|----------|
| `light` | Clean, bright style for general use | Day viewing, printing |
| `dark` | Dark mode for reduced eye strain | Night viewing, presentations |
| `professional` | Corporate style with calm colors (default) | Business reports |
| `minimal` | Minimalist design, monochrome accents | Simple documentation |
| `modern` | Contemporary design, vibrant gradients | Marketing, demos |

## Chart Rendering

- **HTML reports**: Use ApexCharts for modern, interactive charts with tooltips and animations
- **PDF reports**: Use SVG rendering (no JavaScript dependency) for best compatibility

Chart library selection is automatic based on output format - no configuration needed.

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
