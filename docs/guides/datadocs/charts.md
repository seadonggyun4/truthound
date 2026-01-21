# Charts

Truthound Data Docs는 출력 형식에 따라 자동으로 최적의 차트 렌더러를 선택합니다.

## 차트 라이브러리

| 출력 형식 | 차트 렌더러 | 특징 |
|-----------|-------------|------|
| **HTML** | ApexCharts | 인터랙티브, 툴팁, 애니메이션, 줌 |
| **PDF** | SVG | JavaScript 불필요, 정적 렌더링 |

## 지원 차트 유형

```python
from truthound.datadocs import ChartType

# 지원되는 모든 차트 유형
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
```

### 렌더러별 지원 현황

| Chart Type | ApexCharts (HTML) | SVG (PDF) |
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

> PDF 출력에서 지원되지 않는 차트 유형은 Bar 차트로 대체됩니다.

## ChartSpec

차트 설정을 위한 데이터 클래스입니다.

```python
from truthound.datadocs import ChartSpec, ChartType

chart = ChartSpec(
    chart_type=ChartType.DONUT,
    title="Data Types Distribution",
    subtitle="Column types breakdown",
    labels=["string", "integer", "float", "datetime"],
    values=[10, 5, 3, 2],
    series=None,              # 멀티 시리즈용
    colors=["#2563eb", "#7c3aed", "#db2777", "#ea580c"],
    height=300,
    width=None,               # None = 반응형
    show_legend=True,
    show_labels=True,
    show_grid=True,
    animation=True,
    options={},               # 차트별 추가 옵션
)
```

## 차트 렌더러 사용하기

### 기본 사용법

```python
from truthound.datadocs import (
    get_chart_renderer,
    ChartLibrary,
    ChartSpec,
    ChartType,
)

# ApexCharts 렌더러 (HTML용)
renderer = get_chart_renderer(ChartLibrary.APEXCHARTS)

# SVG 렌더러 (PDF용)
renderer = get_chart_renderer(ChartLibrary.SVG)

# 차트 렌더링
chart_spec = ChartSpec(
    chart_type=ChartType.BAR,
    title="Top Columns by Null Ratio",
    labels=["col_a", "col_b", "col_c"],
    values=[45.5, 23.1, 12.8],
)

html = renderer.render(chart_spec)
```

### CDN 의존성 확인

```python
from truthound.datadocs import CDN_URLS, ChartLibrary

# ApexCharts CDN URL
print(CDN_URLS[ChartLibrary.APEXCHARTS])
# ['https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js']

# SVG는 외부 의존성 없음
print(CDN_URLS[ChartLibrary.SVG])
# []
```

## ApexCharts 렌더러

HTML 리포트의 기본 차트 렌더러입니다.

```python
from truthound.datadocs.charts import ApexChartsRenderer

renderer = ApexChartsRenderer()

# 의존성 확인
deps = renderer.get_dependencies()
# ['https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js']

# 차트 렌더링
chart_spec = ChartSpec(
    chart_type=ChartType.DONUT,
    title="Data Types",
    labels=["string", "integer", "float"],
    values=[10, 5, 3],
)
html = renderer.render(chart_spec)
```

### ApexCharts 옵션 커스터마이징

```python
chart_spec = ChartSpec(
    chart_type=ChartType.LINE,
    title="Quality Trend",
    labels=["Jan", "Feb", "Mar", "Apr"],
    values=[85, 87, 82, 90],
    options={
        "stroke": {"curve": "smooth"},
        "markers": {"size": 4},
        "yaxis": {"min": 0, "max": 100},
    },
)
```

## SVG 렌더러

PDF 출력을 위한 JavaScript-free 렌더러입니다.

```python
from truthound.datadocs.charts import SVGChartRenderer

renderer = SVGChartRenderer()

# 의존성 없음
deps = renderer.get_dependencies()  # []

# 차트 렌더링
html = renderer.render(chart_spec)  # 순수 SVG HTML 반환
```

### SVG 지원 차트

SVG 렌더러는 다음 차트 유형을 네이티브로 지원합니다:

- Bar
- Horizontal Bar
- Line
- Pie
- Donut

다른 차트 유형은 Bar 차트로 대체됩니다.

## 커스텀 차트 렌더러

### 렌더러 등록

```python
from truthound.datadocs import (
    BaseChartRenderer,
    ChartSpec,
    register_chart_renderer,
)

@register_chart_renderer("plotly")
class PlotlyChartRenderer(BaseChartRenderer):
    """Plotly 기반 차트 렌더러."""

    library = "plotly"

    def render(self, spec: ChartSpec) -> str:
        import json

        trace = {
            "type": spec.chart_type.value,
            "x": spec.labels,
            "y": spec.values,
        }

        chart_id = f"chart-{id(spec)}"
        return f"""
        <div id="{chart_id}"></div>
        <script>
            Plotly.newPlot('{chart_id}', [{json.dumps(trace)}]);
        </script>
        """

    def get_dependencies(self) -> list[str]:
        return ["https://cdn.plot.ly/plotly-latest.min.js"]
```

### 커스텀 렌더러 사용

```python
from truthound.datadocs import get_chart_renderer

# 등록된 커스텀 렌더러 가져오기
renderer = get_chart_renderer("plotly")
html = renderer.render(chart_spec)
```

## HTMLReportBuilder에서 차트 라이브러리 지정

기본적으로 HTML 리포트는 ApexCharts를, PDF는 SVG를 사용합니다.

```python
from truthound.datadocs import HTMLReportBuilder

# HTML 리포트 (ApexCharts 자동 선택)
builder = HTMLReportBuilder(theme="professional")
html = builder.build(profile)

# PDF 내보내기 (SVG 자동 선택)
from truthound.datadocs import export_to_pdf
export_to_pdf(profile, "report.pdf")  # 내부적으로 _use_svg=True 사용
```

## API Reference

### ChartLibrary Enum

```python
from truthound.datadocs import ChartLibrary

class ChartLibrary(str, Enum):
    APEXCHARTS = "apexcharts"  # HTML 리포트용 (기본값)
    SVG = "svg"                # PDF 내보내기용
```

### ChartSpec 전체 필드

```python
@dataclass
class ChartSpec:
    chart_type: ChartType        # 차트 유형 (필수)
    title: str = ""              # 차트 제목
    subtitle: str = ""           # 부제목
    labels: list[str]            # X축 레이블
    values: list[float | int]    # 데이터 값
    series: list[dict] | None = None  # 멀티 시리즈 데이터
    colors: list[str] | None = None   # 색상 팔레트
    height: int = 300            # 차트 높이 (px)
    width: int | None = None     # 차트 너비 (None=반응형)
    show_legend: bool = True     # 범례 표시
    show_labels: bool = True     # 레이블 표시
    show_grid: bool = True       # 그리드 표시
    animation: bool = True       # 애니메이션 활성화
    options: dict[str, Any]      # 추가 옵션
```

### BaseChartRenderer Protocol

```python
class BaseChartRenderer(ABC):
    library: str  # 차트 라이브러리 이름

    @abstractmethod
    def render(self, spec: ChartSpec) -> str:
        """차트를 HTML로 렌더링."""
        ...

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """필요한 CDN URL 목록 반환."""
        ...
```

## See Also

- [HTML Reports](html-reports.md) - HTML 리포트 생성
- [Sections](sections.md) - 섹션 렌더러
- [PDF Export](pdf-export.md) - PDF 내보내기
