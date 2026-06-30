# Sections

실무 운영 가이드에서 Data Docs, Truthound, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

### 1. 개요

실무 운영 가이드에서 Displays을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Displayed, Items을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Row, Count, Total을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- **컬럼 Count**: Total number of 컬럼
- 실무 운영 가이드에서 Memory, Estimated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Duplicates, Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Missing, Total을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Quality, Score, Overall을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Charts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Data, Donut을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 2. Quality (데이터 품질)

실무 운영 가이드에서 Displays을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

**메트릭:**
- 실무 운영 가이드에서 Overall을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Completeness, Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Uniqueness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Charts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Missing, Horizontal, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Warnings을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- High missing value 컬럼 알림
- Constant 컬럼 notifications
- 실무 운영 가이드에서 Duplicate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 3. 컬럼 (컬럼 Details)

실무 운영 가이드에서 Provides을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

**테이블:**
| 컬럼 | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unique을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Distinct을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|--------|----------|----------|
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Metadata을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Detailed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 4. Patterns (Detected Patterns)

실무 운영 가이드에서 Lists을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Displayed, Items을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Column을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Pattern, Email, Phone, UUID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Regex, Matching을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Match, Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Sample, Matches을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

> 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 5. Distribution (Value Distribution)

실무 운영 가이드에서 Displays을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Charts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Uniqueness, Horizontal, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 6. Correlations

Displays correlations between 컬럼.

실무 운영 가이드에서 Display, Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
column_a ↔ column_b: 0.85 (strong positive correlation)
column_c ↔ column_d: -0.72 (strong negative correlation)
```

> 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 7. Recommendations

실무 운영 가이드에서 Lists을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

**예시:**
- 실무 운영 가이드에서 Add, NotNullValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Add, EmailFormatValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Consider을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Review을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

> 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 8. 알림

Displays 데이터 품질 issue 알림.

실무 운영 가이드에서 Severity, Levels을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Color을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------|-------------|
| 실무 운영 가이드에서 Info을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Blue을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Informational을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Warning을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yellow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Attention을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Red을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Problem을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Critical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dark, Red을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Serious을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 AlertSpec, Structure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

> 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 9. Custom

실무 운영 가이드에서 Allows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SectionSpec

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

## API 레퍼런스

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

## 함께 보기

- 실무 운영 가이드에서 HTML, Reports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Charts, Chart을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Custom, Renderers을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
