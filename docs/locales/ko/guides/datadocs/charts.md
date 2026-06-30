# Charts

실무 운영 가이드에서 Data Docs, Truthound, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Chart Libraries

| 실무 운영 가이드에서 Output, Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Chart, Renderer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Characteristics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------------|----------------|-----------------|
| 실무 운영 가이드에서 HTML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ApexCharts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Interactive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 PDF을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JavaScript을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Supported Chart Types

```python
from truthound.datadocs import ChartType

# All supported chart types
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

### Support Status by Renderer

| 실무 운영 가이드에서 Chart, Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HTML, ApexCharts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PDF, SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|-------------------|-----------|
| 실무 운영 가이드에서 Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Horizontal, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Line을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Pie을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Donut을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Histogram을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Heatmap을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Scatter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Box, Plot을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Gauge을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Radar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

> 실무 운영 가이드에서 PDF, Chart, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ChartSpec

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs import ChartSpec, ChartType

chart = ChartSpec(
    chart_type=ChartType.DONUT,
    title="Data Types Distribution",
    subtitle="Column types breakdown",
    labels=["string", "integer", "float", "datetime"],
    values=[10, 5, 3, 2],
    series=None,              # For multi-series
    colors=["#2563eb", "#7c3aed", "#db2777", "#ea580c"],
    height=300,
    width=None,               # None = responsive
    show_legend=True,
    show_labels=True,
    show_grid=True,
    animation=True,
    options={},               # Additional chart-specific options
)
```

## Using Chart Renderers

### Basic Usage

```python
from truthound.datadocs import (
    get_chart_renderer,
    ChartLibrary,
    ChartSpec,
    ChartType,
)

# ApexCharts renderer (for HTML)
renderer = get_chart_renderer(ChartLibrary.APEXCHARTS)

# SVG renderer (for PDF)
renderer = get_chart_renderer(ChartLibrary.SVG)

# Render chart
chart_spec = ChartSpec(
    chart_type=ChartType.BAR,
    title="Top Columns by Null Ratio",
    labels=["col_a", "col_b", "col_c"],
    values=[45.5, 23.1, 12.8],
)

html = renderer.render(chart_spec)
```

### CDN Dependency Verification

```python
from truthound.datadocs import CDN_URLS, ChartLibrary

# ApexCharts CDN URL
print(CDN_URLS[ChartLibrary.APEXCHARTS])
# ['https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js']

# SVG has no external dependencies
print(CDN_URLS[ChartLibrary.SVG])
# []
```

## ApexCharts Renderer

실무 운영 가이드에서 HTML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.charts import ApexChartsRenderer

renderer = ApexChartsRenderer()

# Check dependencies
deps = renderer.get_dependencies()
# ['https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js']

# Render chart
chart_spec = ChartSpec(
    chart_type=ChartType.DONUT,
    title="Data Types",
    labels=["string", "integer", "float"],
    values=[10, 5, 3],
)
html = renderer.render(chart_spec)
```

### Customizing ApexCharts Options

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

## SVG Renderer

실무 운영 가이드에서 PDF, JavaScript-free을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.charts import SVGChartRenderer

renderer = SVGChartRenderer()

# No dependencies
deps = renderer.get_dependencies()  # []

# Render chart
html = renderer.render(chart_spec)  # Returns pure SVG HTML
```

### SVG Supported Charts

실무 운영 가이드에서 SVG을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Horizontal, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Line을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Pie을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Donut을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Other, Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Custom Chart Renderers

### Registering a Renderer

```python
from truthound.datadocs import (
    BaseChartRenderer,
    ChartSpec,
    register_chart_renderer,
)

@register_chart_renderer("plotly")
class PlotlyChartRenderer(BaseChartRenderer):
    """Plotly-based chart renderer."""

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

### Using Custom Renderers

```python
from truthound.datadocs import get_chart_renderer

# Get registered custom renderer
renderer = get_chart_renderer("plotly")
html = renderer.render(chart_spec)
```

## Specifying Chart Library in HTMLReportBuilder

실무 운영 가이드에서 HTML, PDF, ApexCharts, PDFs, SVG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs import HTMLReportBuilder

# HTML report (ApexCharts auto-selected)
builder = HTMLReportBuilder(theme="professional")
html = builder.build(profile)

# PDF export (SVG auto-selected)
from truthound.datadocs import export_to_pdf
export_to_pdf(profile, "report.pdf")  # Internally uses _use_svg=True
```

## API 레퍼런스

### ChartLibrary Enum

```python
from truthound.datadocs import ChartLibrary

class ChartLibrary(str, Enum):
    APEXCHARTS = "apexcharts"  # For HTML reports (default)
    SVG = "svg"                # For PDF export
```

### ChartSpec Complete Fields

```python
@dataclass
class ChartSpec:
    chart_type: ChartType        # Chart type (required)
    title: str = ""              # Chart title
    subtitle: str = ""           # Subtitle
    labels: list[str]            # X-axis labels
    values: list[float | int]    # Data values
    series: list[dict] | None = None  # Multi-series data
    colors: list[str] | None = None   # Color palette
    height: int = 300            # Chart height (px)
    width: int | None = None     # Chart width (None=responsive)
    show_legend: bool = True     # Display legend
    show_labels: bool = True     # Display labels
    show_grid: bool = True       # Display grid
    animation: bool = True       # Enable animation
    options: dict[str, Any]      # Additional options
```

### BaseChartRenderer Protocol

```python
class BaseChartRenderer(ABC):
    library: str  # Chart library name

    @abstractmethod
    def render(self, spec: ChartSpec) -> str:
        """Render chart to HTML."""
        ...

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Return list of required CDN URLs."""
        ...
```

## 함께 보기

- 실무 운영 가이드에서 HTML, Reports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Sections, Section을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 PDF, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
