# Sections

Truthound Data Docs reports consist of 9 section types.

## Section Types

```python
from truthound.datadocs import SectionType

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
```

## Configuring Sections

### Default Section Order

```python
from truthound.datadocs import ReportConfig, SectionType

config = ReportConfig(
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
)
```

### Including Specific Sections Only

```python
# Simple report with summary information only
config = ReportConfig(
    sections=[
        SectionType.OVERVIEW,
        SectionType.QUALITY,
        SectionType.ALERTS,
    ],
)
```

## Section Details

### 1. Overview

Displays key dataset metrics in card format.

**Displayed Items:**
- **Row Count**: Total number of rows
- **Column Count**: Total number of columns
- **Memory**: Estimated memory usage
- **Duplicates**: Number of duplicate rows (if present)
- **Missing**: Total null cells (if present)
- **Quality Score**: Overall quality score (0-100)

**Charts:**
- Data type distribution (Donut chart)

### 2. Quality (Data Quality)

Displays scores by quality dimension.

**Metrics:**
- **Overall**: Overall quality score
- **Completeness**: Data completeness (based on null ratio)
- **Uniqueness**: Uniqueness (based on unique ratio)

**Charts:**
- Missing value distribution by column (Horizontal Bar)

**Warnings:**
- High missing value column alerts
- Constant column notifications
- Duplicate row warnings

### 3. Columns (Column Details)

Provides detailed information for each column.

**Table:**
| Column | Type | Null % | Unique % | Distinct |
|--------|------|--------|----------|----------|
| name | string | 0.0% | 95.2% | 952 |
| age | integer | 2.5% | 8.3% | 83 |

**Metadata:**
- Detailed profile data for each column

### 4. Patterns (Detected Patterns)

Lists automatically detected data patterns.

**Displayed Items:**
- **Column**: Column where pattern was detected
- **Pattern**: Pattern type (Email, Phone, UUID, etc.)
- **Regex**: Matching regular expression
- **Match Ratio**: Match percentage
- **Sample Matches**: Sample values

> This section is hidden if no patterns are detected.

### 5. Distribution (Value Distribution)

Displays value distribution analysis results.

**Charts:**
- Uniqueness distribution by column (Horizontal Bar)

### 6. Correlations

Displays correlations between columns.

**Display Format:**
```
column_a ↔ column_b: 0.85 (strong positive correlation)
column_c ↔ column_d: -0.72 (strong negative correlation)
```

> This section is hidden if no correlation data is available.

### 7. Recommendations

Lists auto-generated improvement suggestions.

**Examples:**
- Add NotNullValidator for column 'email'
- Add EmailFormatValidator for column 'email'
- Consider implementing duplicate row detection
- Review data collection for columns with high missing values

> This section is hidden if there are no recommendations.

### 8. Alerts

Displays data quality issue alerts.

**Severity Levels:**

| Severity | Color | Description |
|----------|-------|-------------|
| Info | Blue | Informational |
| Warning | Yellow | Attention needed |
| Error | Red | Problem found |
| Critical | Dark Red | Serious issue |

**AlertSpec Structure:**

```python
from truthound.datadocs import AlertSpec, SeverityLevel

alert = AlertSpec(
    title="High Missing Values in 'email'",
    message="Column has 65.3% missing values",
    severity=SeverityLevel.WARNING,
    column="email",
    metric="null_ratio",
    value=0.653,
    threshold=0.5,
    suggestion="Consider imputation or removal",
)
```

> This section is hidden if there are no alerts.

### 9. Custom

Allows inclusion of user-defined content.

## SectionSpec

A data class for section configuration.

```python
from truthound.datadocs import SectionSpec, SectionType, ChartSpec, AlertSpec

section = SectionSpec(
    section_type=SectionType.OVERVIEW,
    title="Overview",
    subtitle="Dataset summary and key metrics",
    charts=[ChartSpec(...)],      # Chart list
    tables=[{"headers": [...], "rows": [...]}],  # Table list
    metrics={"row_count": 1000},  # Metrics dictionary
    alerts=[AlertSpec(...)],      # Alert list
    text_blocks=["recommendation 1", "recommendation 2"],  # Text list
    custom_html="",               # Custom HTML
    collapsible=False,            # Whether collapsible
    collapsed_default=False,      # Default collapsed state
    priority=0,                   # Sort priority
    visible=True,                 # Visibility
    metadata={},                  # Additional metadata
)
```

## Section Renderers

### Basic Usage

```python
from truthound.datadocs import get_section_renderer, SectionType

# Get section renderer
renderer = get_section_renderer(SectionType.OVERVIEW)

# Render section
html = renderer.render(section_spec, chart_renderer, theme_config)
```

### Available Renderers

```python
from truthound.datadocs import (
    OverviewSection,
    ColumnsSection,
    QualitySection,
    PatternsSection,
    DistributionSection,
    CorrelationsSection,
    RecommendationsSection,
    AlertsSection,
    CustomSection,
)

# Direct usage
renderer = OverviewSection()
html = renderer.render(section_spec, chart_renderer, theme_config)
```

## Custom Section Renderers

### Registering a New Section Renderer

```python
from truthound.datadocs import (
    BaseSectionRenderer,
    SectionSpec,
    register_section_renderer,
)

@register_section_renderer("executive_summary")
class ExecutiveSummarySection(BaseSectionRenderer):
    """Executive summary section."""

    section_type = "executive_summary"

    def render(
        self,
        spec: SectionSpec,
        chart_renderer,
        theme_config,
    ) -> str:
        metrics = spec.metrics
        return f"""
        <section class="executive-summary">
            <h2>{spec.title}</h2>
            <div class="key-metrics">
                <div class="metric">
                    <span class="value">{metrics.get('quality_score', 0)}</span>
                    <span class="label">Quality Score</span>
                </div>
                <div class="metric">
                    <span class="value">{len(spec.alerts)}</span>
                    <span class="label">Issues Found</span>
                </div>
            </div>
            <div class="recommendation">
                {spec.text_blocks[0] if spec.text_blocks else ''}
            </div>
        </section>
        """
```

### Using Custom Sections

```python
from truthound.datadocs import ReportConfig, SectionType

config = ReportConfig(
    sections=[
        "executive_summary",  # Custom section (string)
        SectionType.OVERVIEW,
        SectionType.QUALITY,
        SectionType.COLUMNS,
    ]
)
```

## API Reference

### SectionType Enum

```python
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
```

### SeverityLevel Enum

```python
class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### AlertSpec

```python
@dataclass
class AlertSpec:
    title: str                           # Alert title
    message: str                         # Alert message
    severity: SeverityLevel = SeverityLevel.INFO  # Severity
    column: str | None = None            # Related column
    metric: str | None = None            # Related metric name
    value: float | None = None           # Current value
    threshold: float | None = None       # Threshold value
    suggestion: str | None = None        # Improvement suggestion
```

### SectionSpec

```python
@dataclass
class SectionSpec:
    section_type: SectionType            # Section type
    title: str                           # Section title
    subtitle: str = ""                   # Subtitle
    charts: list[ChartSpec]              # Chart list
    tables: list[dict[str, Any]]         # Table list
    metrics: dict[str, Any]              # Metrics dictionary
    alerts: list[AlertSpec]              # Alert list
    text_blocks: list[str]               # Text block list
    custom_html: str = ""                # Custom HTML
    collapsible: bool = False            # Whether collapsible
    collapsed_default: bool = False      # Default collapsed state
    priority: int = 0                    # Sort priority
    visible: bool = True                 # Visibility
    metadata: dict[str, Any]             # Additional metadata
```

## See Also

- [HTML Reports](html-reports.md) - HTML report generation
- [Charts](charts.md) - Chart rendering
- [Custom Renderers](custom-renderers.md) - Custom renderer development
