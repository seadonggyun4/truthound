# Visualization and HTML Reports

`truthound.profiler.visualization` provides native SVG, Plotly and ECharts chart
renderers and a section-based HTML report generator. SVG needs no additional chart
package. Install `truthound[profiler-visualization]` for Plotly; the supported
package range is `plotly>=5.24,<7`.

## Generate a Report

```python
from truthound.profiler.visualization import (
    HTMLReportGenerator, ProfileData, ReportConfig, ReportTheme,
)
data = ProfileData(
    table_name="example", row_count=3, column_count=1,
    columns=[{"name": "amount", "inferred_type": "integer",
              "null_ratio": 0, "unique_ratio": 1}],
)
generator = HTMLReportGenerator(ReportConfig(renderer="svg", theme=ReportTheme.LIGHT))
html = generator.generate(data)
generator.save(data, "profile.html")
```

`ProfileDataConverter.from_dict(profile.to_dict())` accepts a native profile
payload, including ISO `profiled_at`/`timestamp`. `generate_report(data, title="",
theme="light", renderer="auto", config=None, output_path=None, **options)` accepts
`ProfileData` or a dictionary; it returns HTML or the saved path. A supplied config
takes precedence over convenience arguments.

A per-call `generate(data, config=...)` override uses a separate renderer instance
and does not change later calls, including when rendering fails. An explicitly
injected chart renderer remains selected for that override. Table names are HTML
escaped; chart JSON escapes script delimiters before embedding.

## Configuration and Resources

`ReportConfig` supports title/subtitle, theme, logo path/base64, TOC/timestamp/summary
flags, sections, custom CSS/JS, `embed_resources`, language and renderer. Default
sections are overview, data quality, column details, patterns and recommendations.
Built-in themes are light, dark, professional, minimal and colorful. TOC links
match the generated section anchors.

`renderer="auto"` chooses Plotly when installed, otherwise SVG.
`embed_resources=True` (default) embeds Plotly's actual installed JavaScript and
adds no external script URL. This can add several MiB to a report; artifact budgets
still apply. ECharts embedding requires a factory-supplied trusted JavaScript
string via `EChartsChartRenderer(javascript=...)`. Missing embedded resources fail
rather than silently loading a remote script. For explicitly selected CDN loading,
use `embed_resources=False` with a renderer whose `use_cdn=True`. Opening that HTML
requires access to its external script host. Native generation itself does not
fetch those URLs.

Custom CSS/JS, renderer implementations, section templates, themes and resource
URLs are trusted application configuration. Escaping data does not sandbox this
operator-supplied code. Do not put secrets or raw private records into public
reports. Full profiles can contain values and recommendations; disclosure consent
is distinct from rendering.

## Charts and Sections

```python
from truthound.profiler.visualization import SVGChartRenderer, ChartData, ChartConfig, ChartType
chart = SVGChartRenderer().render(
    ChartData(labels=["A", "B"], values=[10, 20], title="Counts"),
    ChartConfig(chart_type=ChartType.BAR, width=400, height=300),
)
```

Check `renderer.supports_chart_type(type)` for the selected backend; enum membership
does not mean every renderer implements every chart. Plotly and ECharts use
script-safe JSON for chart data. Label/value pairs and table row lengths must
align; interactive rendering rejects mismatches instead of silently truncating.
Chart rendering and HTML generation are different from executing the page in a
browser. A native Plotly 6.9.0 report with bundled JavaScript has been checked in a
local browser; this is scoped evidence, not verification of every backend/version.

`section_registry` accepts trusted `SectionRenderer` implementations and
`CustomSectionRenderer` callbacks. `ReportTemplate` provides template/CSS/JS helpers;
it is not a wire-selected template engine.

## Export

`HTMLReportGenerator.save()`, `ReportExporter.save_html()`, `to_file()` for
HTML/JSON/PDF, and `to_pdf()` publish through a private sibling file, atomic
replacement and file/directory flush. Rendering finishes before replacement.
A failure after replacement may leave new content present; inspect the destination
before retry. `render_pdf(profile, config=None)` and `to_pdf_bytes(html)` return
native PDF bytes without writing a path.

Install `truthound[pdf]` and the native WeasyPrint libraries. The legacy pdfkit
fallback is used only if WeasyPrint cannot be imported; native rendering errors
propagate instead of switching providers. On macOS, the calling process may need
its installed native library directory in `DYLD_FALLBACK_LIBRARY_PATH` (for an
Apple Silicon Homebrew installation, `/opt/homebrew/lib`). An isolated MCP factory
must configure its own native library discovery. No system library paths are
changed by Core.

Profile PDF export selects static SVG automatically, even for a Plotly HTML
configuration. SVG dimensions are explicit and fit A4 content width; JavaScript
charts are not executed. Direct `to_pdf_bytes(html)` accepts caller-owned HTML,
so its resources and native provider behavior remain the caller's responsibility.
SVG bar charts support positive and negative values. Renderers reject mismatched
label/value lengths, non-finite values and negative pie values instead of silently
losing data or emitting invalid geometry.

## Approved MCP Visualization

A private `ProfilerRuntime.visualizers` map binds native `HTMLReportGenerator`
objects and `visualization_slots` selects private output paths. `inspect` lists
aliases. `visualization_export(profile_id, visualizer_id)` requires a complete
retained native profile and returns a real `text/html` artifact under the normal
owner budget. Set `export_format="pdf"` for an `application/pdf` artifact; HTML
remains the default. The same format field applies to `visualization_save`. `visualization_save(profile_id, visualizer_id, visualization_slot_id)`
checks rendered size before publishing to the selected slot and returns byte count
and SHA-256, omitting the path. Both require exact approval and sensitive-result
consent. Wire requests cannot supply CSS/JS, output paths or renderer imports.
The native factory owns resource/CDN choices. Installed-worker tests cover HTML
and native PDF artifact reading and exact saved-byte hashes. Save slots are fixed
paths: the selected format determines bytes, not the path suffix.

## Next Steps

- [ML Inference](ml-inference.md)
- [Quality Scoring](quality-scoring.md)
