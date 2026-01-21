# Sections

Truthound Data Docs 리포트는 9가지 섹션으로 구성됩니다.

## 섹션 타입

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

## 섹션 구성하기

### 기본 섹션 순서

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

### 특정 섹션만 포함

```python
# 요약 정보만 포함하는 간단한 리포트
config = ReportConfig(
    sections=[
        SectionType.OVERVIEW,
        SectionType.QUALITY,
        SectionType.ALERTS,
    ],
)
```

## 섹션 상세

### 1. Overview (개요)

데이터셋의 핵심 메트릭을 카드 형태로 표시합니다.

**표시 항목:**
- **Row Count**: 전체 행 수
- **Column Count**: 전체 컬럼 수
- **Memory**: 추정 메모리 사용량
- **Duplicates**: 중복 행 수 (있는 경우)
- **Missing**: 전체 null 셀 수 (있는 경우)
- **Quality Score**: 종합 품질 점수 (0-100)

**차트:**
- 데이터 타입 분포 (Donut 차트)

### 2. Quality (데이터 품질)

품질 차원별 점수를 표시합니다.

**메트릭:**
- **Overall**: 종합 품질 점수
- **Completeness**: 데이터 완전성 (null 비율 기반)
- **Uniqueness**: 유일성 (unique 비율 기반)

**차트:**
- 컬럼별 결측치 분포 (Horizontal Bar)

**경고:**
- 고결측 컬럼 경고
- 상수 컬럼 알림
- 중복 행 경고

### 3. Columns (컬럼 상세)

각 컬럼에 대한 상세 정보를 제공합니다.

**테이블:**
| Column | Type | Null % | Unique % | Distinct |
|--------|------|--------|----------|----------|
| name | string | 0.0% | 95.2% | 952 |
| age | integer | 2.5% | 8.3% | 83 |

**메타데이터:**
- 각 컬럼의 상세 프로파일 데이터

### 4. Patterns (탐지된 패턴)

자동으로 탐지된 데이터 패턴을 나열합니다.

**표시 항목:**
- **Column**: 패턴이 탐지된 컬럼
- **Pattern**: 패턴 유형 (Email, Phone, UUID 등)
- **Regex**: 매칭 정규식
- **Match Ratio**: 일치율
- **Sample Matches**: 샘플 값

> 패턴이 없으면 이 섹션은 숨겨집니다.

### 5. Distribution (값 분포)

값 분포 분석 결과를 표시합니다.

**차트:**
- 컬럼별 유일성 분포 (Horizontal Bar)

### 6. Correlations (상관관계)

컬럼 간 상관관계를 표시합니다.

**표시 형식:**
```
column_a ↔ column_b: 0.85 (강한 양의 상관관계)
column_c ↔ column_d: -0.72 (강한 음의 상관관계)
```

> 상관관계 데이터가 없으면 이 섹션은 숨겨집니다.

### 7. Recommendations (추천)

자동 생성된 개선 제안을 나열합니다.

**예시:**
- Add NotNullValidator for column 'email'
- Add EmailFormatValidator for column 'email'
- Consider implementing duplicate row detection
- Review data collection for columns with high missing values

> 추천 사항이 없으면 이 섹션은 숨겨집니다.

### 8. Alerts (경고)

데이터 품질 이슈 경고를 표시합니다.

**심각도 레벨:**

| Severity | Color | 설명 |
|----------|-------|------|
| Info | Blue | 참고 정보 |
| Warning | Yellow | 주의 필요 |
| Error | Red | 문제 발견 |
| Critical | Dark Red | 심각한 문제 |

**AlertSpec 구조:**

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

> 경고가 없으면 이 섹션은 숨겨집니다.

### 9. Custom (커스텀)

사용자 정의 콘텐츠를 포함할 수 있습니다.

## SectionSpec

섹션 설정을 위한 데이터 클래스입니다.

```python
from truthound.datadocs import SectionSpec, SectionType, ChartSpec, AlertSpec

section = SectionSpec(
    section_type=SectionType.OVERVIEW,
    title="Overview",
    subtitle="Dataset summary and key metrics",
    charts=[ChartSpec(...)],      # 차트 목록
    tables=[{"headers": [...], "rows": [...]}],  # 테이블 목록
    metrics={"row_count": 1000},  # 메트릭 딕셔너리
    alerts=[AlertSpec(...)],      # 경고 목록
    text_blocks=["recommendation 1", "recommendation 2"],  # 텍스트 목록
    custom_html="",               # 커스텀 HTML
    collapsible=False,            # 접을 수 있는지
    collapsed_default=False,      # 기본 접힘 상태
    priority=0,                   # 정렬 우선순위
    visible=True,                 # 표시 여부
    metadata={},                  # 추가 메타데이터
)
```

## 섹션 렌더러

### 기본 사용법

```python
from truthound.datadocs import get_section_renderer, SectionType

# 섹션 렌더러 가져오기
renderer = get_section_renderer(SectionType.OVERVIEW)

# 섹션 렌더링
html = renderer.render(section_spec, chart_renderer, theme_config)
```

### 사용 가능한 렌더러

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

# 직접 사용
renderer = OverviewSection()
html = renderer.render(section_spec, chart_renderer, theme_config)
```

## 커스텀 섹션 렌더러

### 새 섹션 렌더러 등록

```python
from truthound.datadocs import (
    BaseSectionRenderer,
    SectionSpec,
    register_section_renderer,
)

@register_section_renderer("executive_summary")
class ExecutiveSummarySection(BaseSectionRenderer):
    """경영진용 요약 섹션."""

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

### 커스텀 섹션 사용

```python
from truthound.datadocs import ReportConfig, SectionType

config = ReportConfig(
    sections=[
        "executive_summary",  # 커스텀 섹션 (문자열)
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
    title: str                           # 경고 제목
    message: str                         # 경고 메시지
    severity: SeverityLevel = SeverityLevel.INFO  # 심각도
    column: str | None = None            # 관련 컬럼
    metric: str | None = None            # 관련 메트릭 이름
    value: float | None = None           # 현재 값
    threshold: float | None = None       # 임계값
    suggestion: str | None = None        # 개선 제안
```

### SectionSpec

```python
@dataclass
class SectionSpec:
    section_type: SectionType            # 섹션 유형
    title: str                           # 섹션 제목
    subtitle: str = ""                   # 부제목
    charts: list[ChartSpec]              # 차트 목록
    tables: list[dict[str, Any]]         # 테이블 목록
    metrics: dict[str, Any]              # 메트릭 딕셔너리
    alerts: list[AlertSpec]              # 경고 목록
    text_blocks: list[str]               # 텍스트 블록 목록
    custom_html: str = ""                # 커스텀 HTML
    collapsible: bool = False            # 접을 수 있는지
    collapsed_default: bool = False      # 기본 접힘 상태
    priority: int = 0                    # 정렬 우선순위
    visible: bool = True                 # 표시 여부
    metadata: dict[str, Any]             # 추가 메타데이터
```

## See Also

- [HTML Reports](html-reports.md) - HTML 리포트 생성
- [Charts](charts.md) - 차트 렌더링
- [Custom Renderers](custom-renderers.md) - 커스텀 렌더러 개발
