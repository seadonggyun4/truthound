# Quality Reporter Guide

The Quality Reporter system provides comprehensive reporting and filtering capabilities for rule quality scores in Truthound. It enables you to generate detailed reports, filter scores by various criteria, and compare rules by quality metrics.

## Overview

The Quality Reporter module consists of several components:

| Component | Description |
|-----------|-------------|
| **Reporters** | Generate reports in various formats (Console, JSON, HTML, Markdown, JUnit) |
| **Filters** | Composable filter system for selecting scores based on criteria |
| **Engine** | Pipeline-based report generation with caching and parallel processing |
| **CLI Commands** | Command-line tools for reporting, filtering, and comparing |

## Quick Start

### Python API

```python
from truthound.profiler.quality import RuleQualityScorer, ValidationRule, RuleType
from truthound.reporters.quality import (
    get_quality_reporter,
    QualityFilter,
    QualityReportEngine,
)

# Score some rules
scorer = RuleQualityScorer()
scores = scorer.score_all(rules, data)

# Generate a console report
reporter = get_quality_reporter("console")
print(reporter.render(scores))

# Generate an HTML report with charts
reporter = get_quality_reporter("html", include_charts=True)
reporter.write(scores, "quality_report.html")

# Filter and report
good_scores = QualityFilter.by_level(min_level="good").apply(scores)
reporter.render(good_scores)
```

### CLI

```bash
# Generate console report
th quality report scores.json

# Generate HTML report
th quality report scores.json -f html -o report.html

# Filter scores
th quality filter scores.json --min-level good -o filtered.json

# Compare rules by quality
th quality compare scores.json --sort-by f1

# View summary statistics
th quality summary scores.json
```

## Reporters

### Available Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `console` | `.txt` | Rich terminal output with colors and tables |
| `json` | `.json` | Structured JSON for APIs and processing |
| `html` | `.html` | Styled HTML with optional interactive charts |
| `markdown` | `.md` | Markdown for documentation and GitHub |
| `junit` | `.xml` | JUnit XML for CI/CD integration |

### Console Reporter

Produces colorful terminal output using Rich library markup.

```python
from truthound.reporters.quality import ConsoleQualityReporter

reporter = ConsoleQualityReporter(
    color=True,       # Enable colors
    width=120,        # Terminal width
)
print(reporter.render(scores))
```

Output example:
```
Quality Score Report

┌────────────────┬───────────┬─────────┬───────────┬────────┬──────────┬──────┐
│ Rule Name      │ Level     │ F1      │ Precision │ Recall │ Confid.  │ Use? │
├────────────────┼───────────┼─────────┼───────────┼────────┼──────────┼──────┤
│ email_format   │ excellent │ 95.50%  │ 96.00%    │ 95.00% │ 90.00%   │ ✓    │
│ age_range      │ good      │ 87.00%  │ 88.00%    │ 86.00% │ 85.00%   │ ✓    │
│ phone_pattern  │ poor      │ 53.40%  │ 55.00%    │ 52.00% │ 50.00%   │ ✗    │
└────────────────┴───────────┴─────────┴───────────┴────────┴──────────┴──────┘
```

### JSON Reporter

Produces structured JSON output.

```python
from truthound.reporters.quality import JsonQualityReporter

reporter = JsonQualityReporter(
    indent=2,        # JSON indentation
    sort_keys=True,  # Sort object keys
)
json_output = reporter.render(scores)
```

Output structure:
```json
{
  "scores": [
    {
      "rule_name": "email_format",
      "rule_type": "pattern",
      "column": "email",
      "metrics": {
        "f1_score": 0.955,
        "precision": 0.96,
        "recall": 0.95,
        "accuracy": 0.97,
        "confidence": 0.9,
        "quality_level": "excellent"
      },
      "recommendation": "Excellent quality. Safe to use.",
      "should_use": true
    }
  ],
  "count": 3,
  "statistics": {...},
  "generated_at": "2026-01-28T10:30:00"
}
```

### HTML Reporter

Produces styled HTML with optional interactive charts (ApexCharts).

```python
from truthound.reporters.quality import HtmlQualityReporter
from truthound.reporters.quality.config import QualityReporterConfig, ChartType

config = QualityReporterConfig(
    title="My Quality Report",
    include_charts=True,
    chart_types=[ChartType.BAR, ChartType.GAUGE, ChartType.RADAR],
    theme="light",  # light, dark, professional
)

reporter = HtmlQualityReporter(config=config)
reporter.write(scores, "report.html")
```

### Markdown Reporter

Produces Markdown suitable for documentation.

```python
from truthound.reporters.quality import MarkdownQualityReporter

reporter = MarkdownQualityReporter(
    title="Quality Report",
    description="Automated rule quality assessment",
)
md_output = reporter.render(scores)
```

### JUnit Reporter

Produces JUnit XML for CI/CD integration. Scores below the threshold are reported as failures.

```python
from truthound.reporters.quality.reporters import JUnitQualityReporter

reporter = JUnitQualityReporter(min_f1=0.7)
xml_output = reporter.render(scores)
```

## Filters

The filter system provides a composable way to select quality scores based on various criteria.

### Basic Filters

```python
from truthound.reporters.quality.filters import QualityFilter

# Filter by quality level
excellent_only = QualityFilter.by_level("excellent")
good_or_better = QualityFilter.by_level(min_level="good")

# Filter by metric values
high_f1 = QualityFilter.by_metric("f1_score", ">=", 0.9)
low_precision = QualityFilter.by_metric("precision", "<", 0.7)

# Filter by confidence
high_confidence = QualityFilter.by_confidence(min_value=0.8)

# Filter by column
specific_columns = QualityFilter.by_column(include=["email", "phone"])
exclude_columns = QualityFilter.by_column(exclude=["internal_id"])

# Filter by rule type
pattern_rules = QualityFilter.by_rule_type(include=["pattern"])

# Filter by recommendation
recommended = QualityFilter.by_recommendation(should_use=True)
```

### Combining Filters

Filters can be combined using AND, OR, and NOT operators.

```python
# Using methods
combined = (
    QualityFilter.by_level(min_level="good")
    .and_(QualityFilter.by_confidence(min_value=0.8))
)

# Using operators
combined = (
    QualityFilter.by_level("excellent")
    | QualityFilter.by_level("good")
) & QualityFilter.by_recommendation(should_use=True)

# Negation
not_poor = QualityFilter.by_level("poor").not_()
# or
not_poor = ~QualityFilter.by_level("poor")

# Complex combinations
filter_obj = QualityFilter.all_of(
    QualityFilter.by_level(min_level="acceptable"),
    QualityFilter.by_confidence(min_value=0.6),
    QualityFilter.by_recommendation(should_use=True),
)
```

### Custom Filters

Create custom filters with predicate functions.

```python
# Custom predicate filter
long_names = QualityFilter.custom(
    predicate=lambda s: len(s.rule_name) > 10,
    name="long_names",
    description="Rules with names longer than 10 characters",
)

# Apply filter
result = long_names.apply(scores)
```

### Filter from Configuration

Create filters from configuration objects (useful for YAML/JSON config).

```python
from truthound.reporters.quality.config import QualityFilterConfig

config = QualityFilterConfig(
    min_level="acceptable",
    min_f1=0.7,
    min_confidence=0.5,
    should_use_only=True,
    include_columns=["email", "phone"],
)

filter_obj = QualityFilter.from_config(config)
filtered_scores = filter_obj.apply(scores)
```

## Report Engine

The Report Engine provides a pipeline-based approach for generating reports.

### Basic Usage

```python
from truthound.reporters.quality.engine import (
    QualityReportEngine,
    generate_quality_report,
)

# Using the engine
engine = QualityReportEngine()
result = engine.generate(
    scores,
    format="html",
    output_path="report.html",
    filter=QualityFilter.by_level(min_level="good"),
    sort_order=ReportSortOrder.F1_DESC,
    max_scores=20,
)

print(f"Generated in {result.generation_time_ms}ms")
print(f"Scores included: {result.scores_count}")

# Convenience function
result = generate_quality_report(scores, format="json")
```

### Pipeline API

For more control, use the pipeline API directly.

```python
from truthound.reporters.quality.engine import QualityReportPipeline
from truthound.reporters.quality.config import ReportSortOrder

pipeline = (
    QualityReportPipeline()
    .filter(QualityFilter.by_level(min_level="acceptable"))
    .sort(ReportSortOrder.F1_DESC)
    .limit(10)
    .statistics()
    .render("html")
    .write("report.html")
)

context = pipeline.execute(scores)

print(f"Filtered count: {context.filtered_count}")
print(f"Statistics: {context.statistics.to_dict()}")
```

### Convenience Functions

```python
from truthound.reporters.quality.engine import (
    filter_quality_scores,
    compare_quality_scores,
)

# Filter scores
filtered = filter_quality_scores(
    scores,
    min_level="acceptable",
    min_f1=0.7,
    should_use_only=True,
)

# Compare and rank scores
ranked = compare_quality_scores(
    scores,
    sort_by="f1_score",
    descending=True,
)
```

## CLI Commands

### quality report

Generate quality reports in various formats.

```bash
# Basic console report
th quality report scores.json

# HTML report with charts
th quality report scores.json -f html -o report.html --charts

# JSON report filtered by level
th quality report scores.json -f json --min-level good

# Markdown with custom title
th quality report scores.json -f markdown -o QUALITY.md --title "Rule Quality Assessment"

# Filter and limit
th quality report scores.json -f html -o report.html --min-f1 0.8 --max 20

# Only recommended rules
th quality report scores.json --should-use-only
```

### quality filter

Filter quality scores by criteria.

```bash
# Filter by minimum level
th quality filter scores.json --min-level good

# Filter by F1 score range
th quality filter scores.json --min-f1 0.7 --max-f1 0.95

# Filter by specific columns
th quality filter scores.json --columns email,phone,name

# Filter by rule types
th quality filter scores.json --rule-types pattern,range

# Invert filter (show non-matching)
th quality filter scores.json --min-level good --invert

# Save filtered results
th quality filter scores.json --min-level acceptable -o filtered.json
```

### quality compare

Compare and rank rules by quality metrics.

```bash
# Rank by F1 score (default)
th quality compare scores.json

# Rank by precision
th quality compare scores.json --sort-by precision

# Top 10 rules
th quality compare scores.json --max 10

# Ascending order
th quality compare scores.json --asc

# Group by column
th quality compare scores.json --group-by column

# Group by quality level
th quality compare scores.json --group-by level

# Save comparison
th quality compare scores.json -o comparison.json
```

### quality summary

Display summary statistics.

```bash
th quality summary scores.json
```

Output:
```
Quality Score Summary

Total Rules: 25

Quality Level Distribution:
┌──────────────┬───────┬────────────┐
│ Level        │ Count │ Percentage │
├──────────────┼───────┼────────────┤
│ Excellent    │ 8     │ 32.0%      │
│ Good         │ 10    │ 40.0%      │
│ Acceptable   │ 4     │ 16.0%      │
│ Poor         │ 2     │ 8.0%       │
│ Unacceptable │ 1     │ 4.0%       │
└──────────────┴───────┴────────────┘

Recommendations:
  Should Use: 22
  Should Not Use: 3

Metric Averages:
┌────────────┬─────────┬─────────┬─────────┐
│ Metric     │ Average │ Min     │ Max     │
├────────────┼─────────┼─────────┼─────────┤
│ F1 Score   │ 83.25%  │ 45.00%  │ 98.50%  │
│ Precision  │ 85.10%  │ 48.00%  │ 99.00%  │
│ Recall     │ 81.40%  │ 42.00%  │ 98.00%  │
│ Confidence │ 78.00%  │ 35.00%  │ 95.00%  │
└────────────┴─────────┴─────────┴─────────┘
```

## Configuration

### QualityReporterConfig

Main configuration class for reporters.

```python
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityDisplayMode,
    ReportSortOrder,
    ChartType,
)

config = QualityReporterConfig(
    # Output settings
    output_path="report.html",
    title="Quality Score Report",
    description="Automated assessment of validation rules",

    # Content settings
    include_metrics=True,
    include_confusion_matrix=True,
    include_confidence_intervals=True,
    include_trend_analysis=False,
    include_recommendations=True,
    include_statistics=True,
    include_summary=True,
    include_charts=True,

    # Formatting
    metric_precision=4,           # Decimal places
    percentage_format=True,       # Display as percentages
    timestamp_format="%Y-%m-%d %H:%M:%S",

    # Display
    display_mode=QualityDisplayMode.NORMAL,  # COMPACT, NORMAL, DETAILED, FULL
    sort_order=ReportSortOrder.F1_DESC,
    max_scores=None,              # None for all

    # HTML-specific
    theme="light",                # light, dark, professional
    chart_library="apexcharts",   # apexcharts, chartjs
    chart_types=[ChartType.BAR, ChartType.GAUGE],
    custom_css=None,
)
```

### Preset Configurations

```python
# Compact (minimal details)
config = QualityReporterConfig.compact()

# Detailed (all metrics)
config = QualityReporterConfig.detailed()

# Full (everything enabled)
config = QualityReporterConfig.full()
```

## Custom Reporters

Create custom reporters by extending the base class.

```python
from truthound.reporters.quality.base import BaseQualityReporter
from truthound.reporters.quality.config import QualityReporterConfig
from truthound.reporters.quality.factory import register_quality_reporter

@register_quality_reporter("csv")
class CsvQualityReporter(BaseQualityReporter[QualityReporterConfig]):
    name = "csv"
    file_extension = ".csv"
    content_type = "text/csv"

    def render(self, data):
        scores = self.normalize_input(data)
        scores = self.sort_scores(scores)

        lines = ["rule_name,level,f1,precision,recall,should_use"]
        for score in scores:
            m = score.metrics
            lines.append(
                f"{score.rule_name},{m.quality_level.value},"
                f"{m.f1_score},{m.precision},{m.recall},{score.should_use}"
            )
        return "\n".join(lines)

# Use it
reporter = get_quality_reporter("csv")
csv_output = reporter.render(scores)
```

## Integration Examples

### CI/CD Pipeline

```yaml
# GitHub Actions
steps:
  - name: Run quality checks
    run: |
      th quality report scores.json -f junit -o quality.xml
      th quality report scores.json -f html -o quality.html

  - name: Upload quality report
    uses: actions/upload-artifact@v3
    with:
      name: quality-report
      path: quality.html

  - name: Publish test results
    uses: EnricoMi/publish-unit-test-result-action@v2
    with:
      files: quality.xml
```

### Automated Alerting

```python
from truthound.reporters.quality import QualityFilter, generate_quality_report

# Score rules
scores = scorer.score_all(rules, data)

# Check for degraded rules
degraded = QualityFilter.all_of(
    QualityFilter.by_level(max_level="poor"),
    QualityFilter.by_recommendation(should_use=False),
).apply(scores)

if degraded:
    # Generate alert report
    result = generate_quality_report(degraded, format="markdown")
    send_slack_alert(f"Quality Alert: {len(degraded)} rules degraded\n{result.content}")
```

### Dashboard Integration

```python
from truthound.reporters.quality.base import QualityStatistics

# Calculate statistics for dashboard
stats = QualityStatistics.from_scores(scores)

# Send to monitoring system
metrics = {
    "quality.total_rules": stats.total_count,
    "quality.excellent_count": stats.excellent_count,
    "quality.should_use_count": stats.should_use_count,
    "quality.avg_f1": stats.avg_f1,
    "quality.min_f1": stats.min_f1,
}
prometheus_client.push_metrics(metrics)
```

## Best Practices

1. **Use appropriate formats** for your use case:
   - Console for quick checks
   - JSON for API/programmatic use
   - HTML for stakeholder reports
   - Markdown for documentation
   - JUnit for CI/CD

2. **Filter early** to reduce report size and focus on relevant scores.

3. **Set quality thresholds** appropriate for your domain:
   ```python
   from truthound.reporters.quality.config import QualityThresholds

   thresholds = QualityThresholds(
       excellent=0.95,
       good=0.85,
       acceptable=0.70,
       poor=0.50,
   )
   ```

4. **Monitor trends** over time by storing and comparing reports.

5. **Integrate with CI/CD** to catch quality regressions early.

## API Reference

### Modules

- `truthound.reporters.quality.protocols` - Protocol definitions
- `truthound.reporters.quality.config` - Configuration classes
- `truthound.reporters.quality.base` - Base classes and statistics
- `truthound.reporters.quality.filters` - Filter system
- `truthound.reporters.quality.formatters` - Output formatters
- `truthound.reporters.quality.reporters` - Concrete reporters
- `truthound.reporters.quality.factory` - Factory functions
- `truthound.reporters.quality.engine` - Report engine and pipeline

### Key Classes

| Class | Description |
|-------|-------------|
| `BaseQualityReporter` | Abstract base for all reporters |
| `QualityReporterConfig` | Main configuration class |
| `QualityFilter` | Factory for creating filters |
| `QualityReportEngine` | High-level report generation |
| `QualityReportPipeline` | Composable pipeline for reports |
| `QualityStatistics` | Aggregate statistics calculator |

### Key Functions

| Function | Description |
|----------|-------------|
| `get_quality_reporter(format, **kwargs)` | Create reporter by format name |
| `generate_quality_report(scores, format, ...)` | Generate report directly |
| `filter_quality_scores(scores, ...)` | Filter scores by criteria |
| `compare_quality_scores(scores, ...)` | Sort and compare scores |
