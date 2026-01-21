# Charts

Truthound Data Docs automatically selects the optimal chart renderer based on the output format.

## Chart Libraries

| Output Format | Chart Renderer | Characteristics |
|---------------|----------------|-----------------|
| **HTML** | ApexCharts | Interactive, tooltips, animations, zoom |
| **PDF** | SVG | No JavaScript required, static rendering |

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

> Chart types not supported in PDF output are substituted with Bar charts.

## ChartSpec

A data class for chart configuration.

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

The default chart renderer for HTML reports.

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

A JavaScript-free renderer for PDF output.

```python
from truthound.datadocs.charts import SVGChartRenderer

renderer = SVGChartRenderer()

# No dependencies
deps = renderer.get_dependencies()  # []

# Render chart
html = renderer.render(chart_spec)  # Returns pure SVG HTML
```

### SVG Supported Charts

The SVG renderer natively supports the following chart types:

- Bar
- Horizontal Bar
- Line
- Pie
- Donut

Other chart types are substituted with Bar charts.

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

By default, HTML reports use ApexCharts, and PDFs use SVG.

```python
from truthound.datadocs import HTMLReportBuilder

# HTML report (ApexCharts auto-selected)
builder = HTMLReportBuilder(theme="professional")
html = builder.build(profile)

# PDF export (SVG auto-selected)
from truthound.datadocs import export_to_pdf
export_to_pdf(profile, "report.pdf")  # Internally uses _use_svg=True
```

## API Reference

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

## See Also

- [HTML Reports](html-reports.md) - HTML report generation
- [Sections](sections.md) - Section renderers
- [PDF Export](pdf-export.md) - PDF export
